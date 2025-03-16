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
# Define NT-Xent Loss
# class NTXentLoss(nn.Module):
#     """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
#     It also supports the unsupervised contrastive loss in SimCLR"""
#     def __init__(self, temperature=0.07,base_temperature=0.07):
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature
#         self.base_temperature = base_temperature

#     def forward(self, features):
#         """ SimCLR unsupervised loss:


#         Args:
#             features: hidden vector of shape [bsz, n_views, ...].
#             labels: ground truth of shape [bsz].
#             mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
#                 has the same class as sample i. Can be asymmetric.
#         Returns:
#             A loss scalar.
#         """

#         features = features.view(features.shape[0], features.shape[1], -1)

#         mask = torch.eye(batch_size, dtype=torch.float32).to(device)

#         contrast_count = features.shape[1]
#         contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
#         anchor_feature = contrast_feature
#         anchor_count = contrast_count

#         # compute logits
#         anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T)/self.temperature
#         # for numerical stability
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach()

#         # tile mask
#         mask = mask.repeat(anchor_count, contrast_count)
#         # mask-out self-contrast cases
#         logits_mask = torch.scatter(torch.ones_like(mask),1,torch.arange(batch_size * anchor_count).view(-1, 1).to(device),0)
#         mask = mask * logits_mask

#         # compute log_prob
#         exp_logits = torch.exp(logits) * logits_mask
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

#         # compute mean of log-likelihood over positive
#         # modified to handle edge cases when there is no positive pair
#         # for an anchor point. 
#         # Edge case e.g.:- 
#         # features of shape: [4,1,...]
#         # labels:            [0,1,1,2]
#         # loss before mean:  [nan, ..., ..., nan] 
#         mask_pos_pairs = mask.sum(1)
#         mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

#         # loss
#         loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
#         loss = loss.view(anchor_count, batch_size).mean()

#         return loss

# class NTXentLoss(nn.Module):
#     def __init__(self, batch_size, temperature=0.5):
#         super(NTXentLoss, self).__init__()
#         self.temperature = temperature
#         self.batch_size = batch_size
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, z_i, z_j):
#         N = 2 * self.batch_size
#         z = torch.cat((z_i, z_j), dim=0)  # Concatenate the embeddings of the augmented images
#         z = F.normalize(z, dim=1)  # Normalize the embeddings
#         device = z.device
#         similarity_matrix = torch.mm(z, z.T)  # Compute similarity matrix
#         labels = torch.cat([torch.arange(self.batch_size),torch.arange(self.batch_size)], dim=0)  # Create labels for contrastive loss
#         labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)  # Binary matrix for positive pairs
#         logits = similarity_matrix / self.temperature  # Scale by temperature

#         # Mask out the diagonal (i.e., self-similarities)
#         mask = torch.eye(N, dtype=torch.bool).to(device)  # Identity matrix (diagonal is 1)
#         print(f"N = {N}, z shape  = {z.shape}, mask shape = {mask.shape}, logits shape = {logits.shape}")
        
#         logits = logits[~mask].view(N, N - 1)  # Remove diagonal elements (self-similarities)
#         labels = labels[~mask].view(N, N - 1)  # Remove diagonal elements from labels
#         loss = self.criterion(logits, labels.argmax(dim=1))  # Compute the cross-entropy loss
        
#         return loss

    
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        
    def forward(self, z_i, z_j):
        # Flatten spatial and channel dimensions
        z_i_flat = z_i.reshape(self.batch_size, -1)
        z_j_flat = z_j.reshape(self.batch_size, -1)
        
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
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2)),
            transforms.ColorJitter(contrast=0.5)
        ])

    def __call__(self, image):
        return self.transform(image)

# -------------------------------------
# --------- MNIST ---------------------

class MnistSimCLR(nn.Module):
    def __init__(self, latent_dim=128,dropout_prob  = 0.1,temperature = 0.5):
        super().__init__()
        self.temperature = temperature
        self.aug_func = SimCLRTransform()  # Augmentation function
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
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)

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
                optimizer.zero_grad()  # Zero the gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update weights
        
                total_train_loss += loss.item()  # Accumulate the loss
            
            # Average training loss
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            scheduler.step()  # Step the scheduler

            # Validation loop
            self.eval()  # Set the model to evaluation mode
            total_val_loss = 0.0
            with torch.no_grad():  # No gradients are needed for validation
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
