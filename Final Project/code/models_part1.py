# ------ Imports -------
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import time
from torchvision.models import resnet18
# Needed for training
from torch.optim.lr_scheduler import StepLR
# ----------------------
class FinalClassifier(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(256),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        return self.classifier(x)

    def get_device(self):
        """Returns the current device of the model"""
        return next(self.parameters()).device 
        
    def fit_classifier(self, encoder, train_loader, val_loader, num_epochs=50, learning_rate=1e-3):
        device = self.get_device()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
            
        encoder = encoder.to(device)
        encoder.eval()
        
        train_losses = []
        val_accuracies = []
        train_accuracies = []
            
        for epoch in range(num_epochs):
            start_time = time.time()
            self.train()
            total_train_loss = 0
            correct_train = 0
            total_train = 0
            
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
                _, predicted = torch.max(output.data, 1)
                total_train += target.size(0)
                correct_train += (predicted == target).sum().item()
            
            scheduler.step()
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            train_accuracy = 100 * correct_train / total_train
            train_accuracies.append(train_accuracy)
                
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

            epoch_time = time.time() - start_time
            print(f'Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f}s, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%, LR: {scheduler.get_last_lr()[0]:.6f}')
            
        return train_losses, train_accuracies,val_accuracies


# ---------- AutoEncoders ---------------
# ---------- MNIST ----------------------
class MNISTAutoencoder(nn.Module):
    def __init__(self, latent_dim=128,dropout_prob = 0.1):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 14x14
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 7x7
            nn.LeakyReLU(negative_slope=0.01),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout_prob)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout_prob),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.01),
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
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            start_time = time.time()
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

            scheduler.step()
            
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
            epoch_time = time.time() - start_time
            print(f'Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f}s, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        return train_losses, val_losses
# ---------- CIFAR10 ----------------------
class CIFAR10Autoencoder(nn.Module):
    def __init__(self, latent_dim=128,dropout_prob = 0.35,resnet  = True):
        super().__init__()
        
        # Encoder
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
        
        # **Smaller but logically same Decoder**
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 512),  # latent_dim -> 8192
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout_prob),
        
            nn.Unflatten(1, (512, 4, 4)),  # Reshape to 8x8x128
        
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),  # 4x4x512 -> 8x8x256
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(256), 
        
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), # 8x8x256 ->16x16x128
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(128),
        
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # 16x16x128 -> 32x32x64
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1, output_padding=0), # 32x32x64 -> 32x32x32
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            
            nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1),  # 32x32x32 -> 32x32x3
            nn.Sigmoid()  # Output [0, 1]
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
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            start_time = time.time()
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
            
            scheduler.step()
            
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
            epoch_time = time.time() - start_time
            print(f'Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f}s, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} , LR: {scheduler.get_last_lr()[0]:.6f}')
        
        return train_losses, val_losses
