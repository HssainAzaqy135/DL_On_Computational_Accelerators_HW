# ---------- Imports ------------------
import torch
import torch.nn as nn
import time
from matplotlib import pyplot as plt
# Needed for training
from torch.optim.lr_scheduler import StepLR
# -------------------------------------
# ---------- Plotting -----------------
def plot_losses(train_losses, val_losses=None):
    """Plots training and validation losses over epochs."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    
    if val_losses is not None:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    title =""
    if val_losses is not None:
        title = 'Training and Validation Loss'
    else:
        title = 'Training Loss'
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracies(train_accuracies, val_accuracies):
    """Plots training and validation accuracies over epochs."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

# ---------- Tests --------------------
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

def test_classifyingAutoEncoder( classifier, test_loader):
    device = classifier.get_device()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = classifier(data)
            _,predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')