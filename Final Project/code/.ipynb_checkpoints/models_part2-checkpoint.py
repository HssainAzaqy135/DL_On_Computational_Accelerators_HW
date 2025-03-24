# --------- Imports -------------------
import torch
import torch.nn as nn
import torch.optim as optim
import time
# Needed for training
from torch.optim.lr_scheduler import StepLR
from models_part1 import FinalClassifier
# -------------------------------------
# --------- MNIST ---------------------
class MNISTClassifyingAutoencoder(nn.Module):
    def __init__(self, latent_dim=128,dropout_prob = 0.1):
        super().__init__()
        # Encoder
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

        self.classifier = FinalClassifier(latent_dim)
    
    def forward(self, x):
        latent = self.encoder(x)
        pred = self.classifier(latent)
        return pred

    def get_device(self):
        """Returns the current device of the model"""
        return next(self.parameters()).device 

    def train_autoencoder(self, train_loader, val_loader, num_epochs=20, learning_rate=1e-3,weight_decay=1e-3):
        device = self.get_device()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate,weight_decay=1e-3)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        
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
                optimizer.zero_grad()
                output = self.forward(data)
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
                    output = self.forward(data)
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            val_accuracy = 100 * correct / total
            val_accuracies.append(val_accuracy)

            epoch_time = time.time() - start_time
            print(f'Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f}s, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%, LR: {scheduler.get_last_lr()[0]:.6f}')
            
        return train_losses, train_accuracies,val_accuracies

# --------- CIFAR --------------------- 
class CIFAR10ClassifyingAutoencoder(nn.Module):
    def __init__(self, latent_dim=128,dropout_prob = 0.1):
        super().__init__()
        # Encoder
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
        
        self.classifier = FinalClassifier(latent_dim)
        
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
        latent = self.encoder(x)
        pred = self.classifier(latent)
        return pred

    def get_device(self):
        """Returns the current device of the model"""
        return next(self.parameters()).device 

    def train_autoencoder(self, train_loader, val_loader, num_epochs=20, learning_rate=1e-3,weight_decay = 1e-3):
        device = self.get_device()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate,weight_decay= weight_decay)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        
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
                optimizer.zero_grad()
                output = self.forward(data)
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
                    output = self.forward(data)
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            val_accuracy = 100 * correct / total
            val_accuracies.append(val_accuracy)

            epoch_time = time.time() - start_time
            print(f'Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f}s, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%, LR: {scheduler.get_last_lr()[0]:.6f}')
            
        return train_losses, train_accuracies,val_accuracies