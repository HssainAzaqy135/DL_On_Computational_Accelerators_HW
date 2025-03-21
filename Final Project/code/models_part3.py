# --------- Imports -------------------
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models
from torchvision.models import resnet18
# Needed for training
import time
from torch.optim.lr_scheduler import StepLR
from models_part1 import FinalClassifier
# --------- Loss ----------------------
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5,eps = 1e-12):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, z1, z2):
        """
        Compute the NT-Xent loss for two sets of augmented embeddings.

        Args:
            z1 (torch.Tensor): First set of augmented embeddings [batch_size, out_dim].
            z2 (torch.Tensor): Second set of augmented embeddings [batch_size, out_dim].

        Returns:
            torch.Tensor: Scalar loss value (mean over the batch).
        """
        batch_size = z1.size(0)
        
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z = torch.cat([z1, z2], dim=0)  # [2N, out_dim]

        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # [2N, 2N]
        
        # Mask self-similarities
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)
        
        # numerical stability
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix_stable = sim_matrix - sim_max
        
        # Extract positive pair sims
        pos_sim = torch.cat([torch.diag(sim_matrix_stable, batch_size), 
                             torch.diag(sim_matrix_stable, -batch_size)])
        
        exp_sim = torch.exp(sim_matrix_stable)
        denom = exp_sim.sum(dim=1) + self.eps  # Add epsilon to avoid log(0)
        
        # Compute loss in log domain
        loss = -pos_sim + torch.log(denom)
        return loss.mean()

# --------- Auxiliry ------------------
# Define the augmentation function that will be applied during training
class SimCLRTransform:
    def __init__(self,size = None):
        # Assumes the data is not normalized yet
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p = 0.3),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8)
        ])

    def __call__(self, image):
        return self.transform(image)

# -------------------------------------
# --------- MNIST ---------------------

class MnistSimCLR(nn.Module):
    def __init__(self, latent_dim=128,dropout_prob  = 0.1,temperature = 0.5):
        super().__init__()
        self.temperature = temperature
          # Augmentation function
        self.aug_func = SimCLRTransform(size = 28)
        self.data_norm_func = transforms.Normalize(mean=[0.1307], std=[0.3081])

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 14x14x32
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 7x7x64
            nn.LeakyReLU(negative_slope=0.01),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout_prob) 
        )
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, 2*latent_dim),  # Project to a larger dimensional space before the final projection
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(2*latent_dim, latent_dim)  # Final feature dimension (e.g., 128)
        )

    def forward(self, x):
        """Forward pass through the encoder and projector."""
        return self.projection(self.encoder(x))

    def get_device(self):
        """Returns the current device of the model"""
        return next(self.parameters()).device 
        
    def train_autoencoder(self, train_loader, val_loader, num_epochs=20, learning_rate=1e-4):
        device = self.get_device()
        criterion = NTXentLoss(temperature=self.temperature).to(device)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            start_time = time.time()
            self.train()  # Set the model to training mode
            total_train_loss = 0.0

            # Training loop
            for batch_idx, (images, _) in enumerate(train_loader):
                images = images.to(device)  # Move the data to the device
                
                # Create augmented pairs (two augmented versions of the same image)
                aug1 = self.aug_func(images)
                aug2 = self.aug_func(images)
        
                # Forward pass (compute embeddings for augmented images)
                z_i = self(self.data_norm_func(aug1))
                z_j = self(self.data_norm_func(aug2))
                # print(f"z_i shape {z_i.shape}, z_j shape {z_j.shape}")
                # Compute loss
                loss = criterion(z_i, z_j)
        
                # Backward pass and optimization
                optimizer.zero_grad() 
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  # Clip gradients
                optimizer.step() 
        
                total_train_loss += loss.item()   
            
             
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            scheduler.step()  # Step the scheduler

            # Validation loop
            self.eval()   
            total_val_loss = 0.0
            with torch.no_grad(): 
                for images, _ in val_loader:
                    images = images.to(device)
                    # No augmentation for val set
                    # aug1 = self.aug_func(images)
                    # aug2 = self.aug_func(images)
                    z_i = self(self.data_norm_func(images))
                    z_j = self(self.data_norm_func(images))
                    loss = criterion(z_i, z_j)
                    total_val_loss += loss.item()

            # Average validation loss
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Time taken for the epoch
            epoch_time = time.time() - start_time

            # Print the progress
            print(f'Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f}s, '
                  f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        return train_losses, val_losses
# ------- CIFAR10 ---------------------
class Cifar10SimCLR(nn.Module):
    def __init__(self, latent_dim=128,dropout_prob  = 0.1,temperature = 0.5):
        super().__init__()
        self.temperature = temperature
        # Augmentation function
        self.aug_func = SimCLRTransform(size = 32)
        self.data_norm_func = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),    # 32x32x3 -> 32x32x64
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Dropout(p=dropout_prob),                  
        
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32x32x64 -> 16x16x128
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(128),
            nn.Dropout(p=dropout_prob),                  
        
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 16x16x128 -> 8x8x256
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(256),
            nn.Dropout(p=dropout_prob),                  
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2,padding=1),  # 8x8x256 -> 4x4x512
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(512),
            nn.Dropout(p=dropout_prob),                 

            nn.Conv2d(512, 128, kernel_size=3, stride=2,padding=1),  # 4x4x512 -> 2x2x128
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(128),
            nn.Dropout(p=dropout_prob),
            
            nn.Flatten(),                                # 2x2×128 = 512
            nn.Linear(2 * 2 * 128, latent_dim),          # 512 -> latent_dim            
        )
            
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, 2*latent_dim),  # Project to a larger dimensional space before the final projection
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(2*latent_dim, latent_dim)
        )
        print("Initializing weights ....")
        self.initialize_weights()
        print("Initializing weights DONE")
        
    def initialize_weights(self):
        # Initialize convolutional and batchnorm layers
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
                if m.bias is not None:  # Bias exists unless explicitly disabled
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
                
    def forward(self, x):
        """Forward pass through the encoder and projector."""
        return self.projection(self.encoder(x))

    def get_device(self):
        """Returns the current device of the model"""
        return next(self.parameters()).device 
        
    def train_autoencoder(self, train_loader, val_loader, num_epochs=20, learning_rate=1e-3,weight_decay = 1e-3):
        device = self.get_device()
        criterion = NTXentLoss(temperature=self.temperature).to(device)
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate,weight_decay= weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            start_time = time.time()
            self.train()  # Set the model to training mode
            total_train_loss = 0.0

            # Training loop
            for batch_idx, (images, _) in enumerate(train_loader):
                images = images.to(device)  # Move the data to the device
                
                # Create augmented pairs (two augmented versions of the same image)
                aug1 = self.aug_func(images)#.clamp(min=-1,max =1)
                aug2 = self.aug_func(images)#.clamp(min=-1,max =1)

                # Forward pass (compute embeddings for augmented images)
                z_i = self(self.data_norm_func(aug1))
                z_j = self(self.data_norm_func(aug2))

                # Compute loss
                loss = criterion(z_i, z_j)
        
                # Backward pass and optimization
                optimizer.zero_grad() 
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
        
                total_train_loss += loss.item()   
            
             
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            scheduler.step()  # Step the scheduler

            # Validation loop
            self.eval()   
            total_val_loss = 0.0
            with torch.no_grad(): 
                for images, _ in val_loader:
                    images = images.to(device)
                    # No augmentation for val set
                    # aug1 = self.aug_func(images)
                    # aug2 = self.aug_func(images)
                    
                    z_i = self(self.data_norm_func(images))
                    z_j = self(self.data_norm_func(images))
                    loss = criterion(z_i, z_j)
                    total_val_loss += loss.item()

            # Average validation loss
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Time taken for the epoch
            epoch_time = time.time() - start_time

            # Print the progress
            print(f'Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f}s, '
                  f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        return train_losses, val_losses