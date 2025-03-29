# --------- Imports -------------------
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models
# Needed for training
import time
from torch.optim.lr_scheduler import StepLR
from models_part1 import FinalClassifier
# --------- Loss ----------------------
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, eps=1e-12):
        super().__init__()
        self.temperature = temperature
        self.eps = eps  # Small constant for numerical stability

    def forward(self, z1, z2, z3, z4):
        """
        Compute the NT-Xent loss for contrastive learning.
    
        Args:
            z1 (torch.Tensor): First set of augmented embeddings [batch_size, out_dim].
            z2 (torch.Tensor): Second set of augmented embeddings [batch_size, out_dim].
    
        Returns:
            torch.Tensor: Scalar loss value (mean over the batch).
        """
        batch_size = z1.size(0)
            
        # Normalize embeddings to unit norm
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z3 = F.normalize(z3, dim=1)
        z4 = F.normalize(z4, dim=1)

        # Concatenate the embeddings from the 4 augmentations
        z = torch.cat([z1, z2, z3, z4], dim=0)  # Shape: [4N, out_dim]
        # Compute cosine similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # Shape: [4N, 4N]
    
        # Create mask to ignore self-similarities
        mask = torch.eye(4 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix.masked_fill_(mask, float('-inf'))  # Prevent self-similarity from affecting loss
    
    
        pos_sim = torch.cat([
        sim_matrix[range(batch_size), range(1 * batch_size, 2 * batch_size)],  # z1 <-> z2
        sim_matrix[range(1 * batch_size, 2 * batch_size), range(2 * batch_size, 3 * batch_size)],  # z2 <-> z3
        sim_matrix[range(2 * batch_size, 3 * batch_size), range(3 * batch_size, 4 * batch_size)],  # z3 <-> z4
        sim_matrix[range(3 * batch_size, 4 * batch_size), range(batch_size)]   # z4 <-> z1
        
        ])

        # Ensure that pos_sim has the correct shape, which should be of length equal to the number of positive pairs
        assert pos_sim.size(0) == 4 * batch_size  # 6 positive pairs for 4 sets of embeddings
        
        # Compute the denominator (sum over negative samples)
        exp_sim = torch.exp(sim_matrix)  # Exponentiate all values
        denom = exp_sim.sum(dim=1, keepdim=True) + self.eps  # Avoid log(0) by adding epsilon

        
        # Compute contrastive loss
        loss = -pos_sim + torch.log(denom.squeeze())
        
        return loss.mean()

# --------- Auxiliry ------------------
# Define the augmentation function that will be applied during training
class SimCLRTransform:
    def __init__(self,size = None):
        # Assumes the data is not normalized yet
        self.transform = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p = 0.3),
        transforms.RandomApply([transforms.ColorJitter(0.25, 0.25, 0.25, 0.1)], p=0.6),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(1, 3))], p=0.5),
        transforms.RandomAutocontrast(p=0.5)])

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

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 28x28x1 -> 14x14x32
            nn.BatchNorm2d(32),
            nn.GELU(),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14x14x32 -> 7x7x64
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim), # 64*7*7 -> latent_dim
        )
        
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),  # Project to a larger dimensional space before the final projection
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(latent_dim, latent_dim)  # Final feature dimension (e.g., 128)
        )

    def forward(self, x):
        """Forward pass through the encoder and projector."""
        return self.projection(self.encoder(x))

    def get_device(self):
        """Returns the current device of the model"""
        return next(self.parameters()).device 
        
    def train_autoencoder(self, train_loader, val_loader, num_epochs=20, learning_rate=1e-3,weight_decay= 1e-3):
        device = self.get_device()
        criterion = NTXentLoss(temperature=self.temperature).to(device)
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate,weight_decay= 1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.125,patience=5,min_lr=1e-7)

        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            start_time = time.time()
            self.train()  # Set the model to training mode
            total_train_loss = 0  # Initialize the loss accumulator

            # Training loop
            for batch_idx, (images, _) in enumerate(train_loader):
                batch_size = images.shape[0]
                images = images.to(device)  # Move the images to the device
        
                # Step 1: Augment the images
                aug1 = images  # ORIGINAL IMAGE
                aug2 = self.aug_func(images)  # Second augmented version
                aug3 = self.aug_func(images)  # Third augmented version
                aug4 = self.aug_func(images)  # Fourth augmented version
        
                # Step 2: Compute embeddings for all augmented images
                z1 = self(aug1)  # Embedding for augmented image 1
                z2 = self(aug2)  # Embedding for augmented image 2
                z3 = self(aug3)  # Embedding for augmented image 3
                z4 = self(aug4)  # Embedding for augmented image 4
        
    
                # Step 3: Concatenate all embeddings into a single tensor
                embedded_pair = torch.cat([z1, z2, z3, z4], dim=0)  # Shape: [4N, out_dim]
        
                # Step 4: Split the concatenated embeddings into four parts
                z1, z2, z3, z4 = torch.split(embedded_pair, batch_size, dim=0)
        
                # Step 5: Compute the NT-Xent loss
                loss = criterion(z1, z2, z3, z4)  # Compute the contrastive loss
        
                # Step 6: Backward pass and optimization
                optimizer.zero_grad()  # Clear the previous gradients
                loss.backward()  # Backpropagate the loss
                optimizer.step()  # Update the model weights
                total_train_loss += loss.item()  # Accumulate the loss

            avg_train_loss = total_train_loss / len(train_loader)  # Average loss for the epoch
            train_losses.append(avg_train_loss)

            

            # Validation loop
            self.eval()   
            total_val_loss = 0.0
            with torch.no_grad(): 
                for images, _ in val_loader:
                    batch_size= images.shape[0]
                    images = images.to(device)
                    # No augmentation for val set
                    aug_pair = torch.cat([images, images, images, images], dim=0)  # Concatenate 4 images
                    embedded_pair = self(aug_pair)
                    # Forward pass (compute embeddings for augmented images)
                    z1, z2, z3, z4 = embedded_pair[:batch_size], embedded_pair[batch_size:2*batch_size], embedded_pair[2*batch_size:3*batch_size], embedded_pair[3*batch_size:]
                    loss = criterion(z1, z2, z3, z4)
                    total_val_loss += loss.item()

            # Average validation loss
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            scheduler.step(avg_val_loss)
            # Time taken for the epoch
            epoch_time = time.time() - start_time

            curr_lr = optimizer.param_groups[0]['lr']
            # Print the progress
            print(f'Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f}s, '
                  f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
                  f'LR: {curr_lr:.6f}')
        
        return train_losses, val_losses
# ------- CIFAR10 ---------------------
class Cifar10SimCLR(nn.Module):
    def __init__(self, latent_dim=128,dropout_prob  = 0.1,temperature = 0.5):
        super().__init__()
        self.temperature = temperature
        # Augmentation function
        self.aug_func = SimCLRTransform(size = 32)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),    # 32x32x3 -> 16x16x32
            nn.BatchNorm2d(32),
            nn.GELU(),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),    # 16x16x32 -> 8x8x64
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),    # 8x8x64 -> 4x4x128
            nn.BatchNorm2d(128),
            nn.GELU(),
            
            nn.Flatten(),                                # 4x4×128 = 2048
            nn.Linear(4 * 4 * 128, latent_dim),          # 2048 -> latent_dim            
        )
            
        self.projection = nn.Sequential(
            nn.Linear(latent_dim,latent_dim),  # Project to a larger dimensional space before the final projection
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(latent_dim, latent_dim)
        )
        print("Initializing weights ....")
        self.initialize_weights()
        print("Initializing weights DONE")
        
    def initialize_weights(self):
        # Initialize convolutional and batchnorm layers
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:  # Bias exists unless explicitly disabled
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
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
        
    def train_autoencoder(self, train_loader, val_loader, num_epochs=20, learning_rate=1e-3,weight_decay= 1e-3):
        device = self.get_device()
        criterion = NTXentLoss(temperature=self.temperature).to(device)
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate,weight_decay= 1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.125,patience=5,min_lr=1e-7)

        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            start_time = time.time()
            self.train()  # Set the model to training mode
            total_train_loss = 0.0

            # Training loop
            for batch_idx, (images, _) in enumerate(train_loader):
                batch_size= images.shape[0]
                images = images.to(device)  # Move the data to the device

                # Step 1: Augment the images
                aug1 = images  # ORIGINAL IMAGE
                aug2 = self.aug_func(images)  # Second augmented version
                aug3 = self.aug_func(images)  # Third augmented version
                aug4 = self.aug_func(images)  # Fourth augmented version
        
                # Step 2: Compute embeddings for all augmented images
                z1 = self(aug1)  # Embedding for augmented image 1
                z2 = self(aug2)  # Embedding for augmented image 2
                z3 = self(aug3)  # Embedding for augmented image 3
                z4 = self(aug4)  # Embedding for augmented image 4
        
    
                # Step 3: Concatenate all embeddings into a single tensor
                embedded_pair = torch.cat([z1, z2, z3, z4], dim=0)  # Shape: [4N, out_dim]
        
                # Step 4: Split the concatenated embeddings into four parts
                z1, z2, z3, z4 = torch.split(embedded_pair, batch_size, dim=0)
        
                # Step 5: Compute the NT-Xent loss
                loss = criterion(z1, z2, z3, z4)  # Compute the contrastive loss
        
                # Step 6: Backward pass and optimization
                optimizer.zero_grad()  # Clear the previous gradients
                loss.backward()  # Backpropagate the loss
                optimizer.step()  # Update the model weights
                total_train_loss += loss.item()  # Accumulate the loss

            avg_train_loss = total_train_loss / len(train_loader)  # Average loss for the epoch
            train_losses.append(avg_train_loss)

            

            # Validation loop
            self.eval()   
            total_val_loss = 0.0
            with torch.no_grad(): 
                for images, _ in val_loader:
                    batch_size= images.shape[0]
                    images = images.to(device)
                    # No augmentation for val set
                    aug_pair = torch.cat([images, images, images, images], dim=0)  # Concatenate 4 images
                    embedded_pair = self(aug_pair)
                    # Forward pass (compute embeddings for augmented images)
                    z1, z2, z3, z4 = embedded_pair[:batch_size], embedded_pair[batch_size:2*batch_size], embedded_pair[2*batch_size:3*batch_size], embedded_pair[3*batch_size:]
                    loss = criterion(z1, z2, z3, z4)
                    total_val_loss += loss.item()


            # Average validation loss
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            scheduler.step(avg_val_loss)
            # Time taken for the epoch
            epoch_time = time.time() - start_time

            curr_lr = optimizer.param_groups[0]['lr']
            # Print the progress
            print(f'Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f}s, '
                  f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
                  f'LR: {curr_lr:.6f}')
        
        return train_losses, val_losses