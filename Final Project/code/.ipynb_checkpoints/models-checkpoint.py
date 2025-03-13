# ------ Imports -------
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from matplotlib import pyplot as plt
# Needed for training
from torch.optim.lr_scheduler import StepLR
# ----------------------
class FinalClassifier(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        return self.classifier(x)

    def get_device(self):
        """Returns the current device of the model"""
        return next(self.parameters()).device 
        
    def fit_classifier(self,encoder, train_loader, val_loader, num_epochs=50, learning_rate=1e-3):
        device = self.get_device()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        
        encoder = encoder.to(device)
        encoder.eval()
        
        train_losses = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            self.train()
            total_train_loss = 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                with torch.no_grad():
                    latent = encoder(data)
                optimizer.zero_grad()
                output = self.forward(latent)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            scheduler.step()
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            self.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    latent = encoder(data)
                    output = self.forward(latent)
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            val_accuracy = 100 * correct / total
            val_accuracies.append(val_accuracy)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()


def test_classifier(encoder, classifier, test_loader):
    device = classifier.get_device()
    classifier.eval()
    encoder.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            latent = encoder(data)
            output = classifier(latent)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# ---------- AutoEncoders ---------------
# ---------- MNIST ----------------------
class MNISTAutoencoder(nn.Module):
    def __init__(self, latent_dim=128,dropout_prob = 0.1):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # for output to be  in [0,1]
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def get_device(self):
        """Returns the current device of the model"""
        return next(self.parameters()).device 

    def train_autoencoder(self, train_loader, val_loader, num_epochs=20, learning_rate=1e-4):
        device = self.get_device()
        
        criterion = nn.MSELoss()  # Reconstruction loss
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Lists to store losses
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training phase
            self.train()
            total_train_loss = 0
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(device)
                optimizer.zero_grad()
                reconstructed, _ = self.forward(data)
                loss = criterion(reconstructed, data)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.eval()
            total_val_loss = 0
            with torch.no_grad():  # No gradient computation for validation
                for data, _ in val_loader:
                    data = data.to(device)
                    reconstructed, _ = self.forward(data)
                    loss = criterion(reconstructed, data)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
        plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()


# ---------- CIFAR10 ----------------------
class CIFAR10Autoencoder(nn.Module):
    def __init__(self, latent_dim=128,dropout_prob = 0.1):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),    # 32x32x3 -> 32x32x64
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 32x32x64 -> 16x16x128
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 16x16x128 -> 8x8x256
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, stride=1, padding=1), # Refine: 8x8x256 -> 8x8x256
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Flatten(),                                # 8x8x256 = 16384
            nn.Linear(256 * 8 * 8, latent_dim),          # 16384 -> 128
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)                   # Reduced dropout
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),          # 128 -> 16384
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Unflatten(1, (256, 8, 8)),               # Reshape to 8x8x256
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # 8x8x256 -> 16x16x128
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 16x16x128 -> 32x32x64
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),   # Refine: 32x32x64 -> 32x32x64
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),    # 32x32x64 -> 32x32x3
            nn.Tanh()                                    # Output [-1, 1]
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def get_device(self):
        """Returns the current device of the model"""
        return next(self.parameters()).device 
        
    def train_autoencoder(self, train_loader, val_loader, num_epochs=30, learning_rate=5e-4):
        device = self.get_device()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=15, gamma=0.1)  # Reduce LR every 15 epochs
        
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            self.train()
            total_train_loss = 0
            for data, _ in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                reconstructed, _ = self.forward(data)
                loss = criterion(reconstructed, data)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            scheduler.step()
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for data, _ in val_loader:
                    data = data.to(device)
                    reconstructed, _ = self.forward(data)
                    loss = criterion(reconstructed, data)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
        plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()