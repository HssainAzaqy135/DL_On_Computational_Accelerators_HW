# --------- Imports -------------------
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
# Needed for training
import time
from torch.optim.lr_scheduler import StepLR
from models_part1 import FinalClassifier
# --------- Loss ----------------------
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        
    def forward(self, z_i, z_j):
        # Flatten spatial and channel dimensions
        z_i_flat = z_i.reshape(self.batch_size, -1)
        z_j_flat = z_j.reshape(self.batch_size, -1)
        # print(f"shape of z_i = {z_i.shape} and flat is {z_i_flat.shape}")
        N = 2 * self.batch_size
        z = torch.cat((z_i_flat, z_j_flat), dim=0)
        z = F.normalize(z, dim=1)
        
        device = z.device

        similarity_matrix = torch.mm(z, z.T) / self.temperature

        indices = torch.arange(0, N, device=device)
        labels = torch.cat([torch.arange(self.batch_size, device=device), 
                           torch.arange(self.batch_size, device=device)])
                           
        positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        # no self-similarities in positive mask
        identity_mask = torch.eye(N, device=device)
        positive_mask = positive_mask - identity_mask
        
        # numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / positive_mask.sum(1)
        loss = -mean_log_prob_pos.mean()
        
        return loss

# --------- Auxiliry ------------------
# Define the augmentation function that will be applied during training
class SimCLRTransform:
    def __init__(self,size = None):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
            transforms.RandomAffine(degrees=15, translate=(0.2, 0.2)),
            transforms.RandomApply([
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
            ], p=0.6),
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
        criterion = NTXentLoss(batch_size=train_loader.batch_size, temperature=self.temperature).to(device)  # NT-Xent loss
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
                optimizer.zero_grad()  #  
                loss.backward()  #  
                optimizer.step()  #  
        
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
    def __init__(self, latent_dim=128,dropout_prob  = 0.35,temperature = 0.5):
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
        criterion = NTXentLoss(batch_size=train_loader.batch_size, temperature=self.temperature).to(device)  # NT-Xent loss
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
                z_i = self(aug1)
                z_j = self(aug2)

                # Compute loss
                loss = criterion(z_i, z_j)
        
                # Backward pass and optimization
                optimizer.zero_grad() 
                loss.backward()
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