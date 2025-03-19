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
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip()           
        ])

    def __call__(self, image):
        return self.transform(image)

# -------------------------------------
# --------- MNIST ---------------------

class MnistSimCLR(nn.Module):
    def __init__(self, latent_dim=128,dropout_prob  = 0.1,temperature = 0.5):
        super().__init__()
        self.temperature = temperature
        self.aug_func = SimCLRTransform(size = 28)  # Augmentation function
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
                z_i = self(aug1)  # Get the embeddings for the first augmentation
                z_j = self(aug2)  # Get the embeddings for the second augmentation
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
                    aug1 = self.aug_func(images)
                    aug2 = self.aug_func(images)

                    z_i = self(aug1)
                    z_j = self(aug2)
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
    def __init__(self, latent_dim=128,dropout_prob  = 0.35,temperature = 0.5,resnet = True):
        super().__init__()
        self.temperature = temperature
        self.aug_func = SimCLRTransform(size = 32)  # Augmentation function
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),    # 32x32x3 -> 32x32x64
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 32x32x64 -> 16x16x128
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 16x16x128 -> 8x8x256
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 128, kernel_size=1, stride=1),  # Reduce channels 256 -> 128
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(128),
            
            nn.Flatten(),                                 # 8x8×128 = 8192
            nn.Linear(8 * 8 * 128, latent_dim),                 # 8192 -> latent_dim
            # nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout_prob)
        )

        if(resnet):
            self.encoder = resnet18(pretrained = False)
            self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.encoder.maxpool = nn.Identity()
            self.encoder.fc = nn.Linear(512, latent_dim)
            
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, 2*latent_dim),  # Project to a larger dimensional space before the final projection
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(2*latent_dim, latent_dim)
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
                aug1 = self.aug_func(images)#.clamp(min=-1,max =1)
                aug2 = self.aug_func(images)#.clamp(min=-1,max =1)
                # # Debug
                # if torch.isnan(aug1).any() or torch.isnan(aug2).any():
                #     print("NaN detected in augmented images!")
                #     print("aug1 min:", aug1.min().item(), "aug1 max:", aug1.max().item())
                #     print("aug2 min:", aug2.min().item(), "aug2 max:", aug2.max().item())
                #     exit()  # Stop training to investigate
                # Forward pass (compute embeddings for augmented images)
                z_i = self(aug1)
                z_j = self(aug2)
                # # Debug
                # if torch.isnan(z_i).any() or torch.isnan(z_j).any():
                #     print("NaN detected in embeddings before loss calculation!")
                #     print("z_i min:", z_i.min().item(), "z_i max:", z_i.max().item())
                #     print("z_j min:", z_j.min().item(), "z_j max:", z_j.max().item())
                #     exit()

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
                    aug1 = self.aug_func(images)
                    aug2 = self.aug_func(images)

                    z_i = self(aug1)
                    z_j = self(aug2)
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