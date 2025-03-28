# ---------- Imports ------------------
import torch
import torch.nn as nn
import time
import os
from matplotlib import pyplot as plt
from data_loading import load_and_prep_data
# Needed for training
from torch.optim.lr_scheduler import StepLR
from utils import plot_tsne
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
    return accuracy

def test_classifyingAutoEncoder(classifier, test_loader):
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
    return accuracy
# --------- Model container class for svaing and loading -----------------
class PretrainedModel(nn.Module):
    def __init__(self, encoder,classifier,decoder=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        # freeze all weights
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False
        if self.decoder is not None:
            for param in self.decoder.parameters():
                param.requires_grad = False
        
    def get_device(self):
        """Returns the current device of the model"""
        return next(self.parameters()).device
        
    def forward(self, x):
        return self.classifier(self.encoder(x))

    def reconstruct_image(self,x):
        return self.decoder(self.encoder(x))


def save_pretrained_model(path,encoder,classifier,decoder = None):
    model_to_save = PretrainedModel(encoder = encoder,classifier=classifier,decoder = decoder)
    torch.save(model_to_save,path)
    
# ------------ Folder creation function ----------------------------------
def create_model_folders():
    # Define the base directory for trained models
    base_dir = "trained_models"
    subfolders = ["part_1", "part_2", "part_3"]
    
    # Define the additional data folders
    data_folders = ["mnist_data", "cifar10_data","tsne_plots"]
    
    # Create the base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created directory: {base_dir}")
    else:
        print(f"Directory already exists: {base_dir}")
    
    # Create each subfolder under trained_models
    for subfolder in subfolders:
        folder_path = os.path.join(base_dir, subfolder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created directory: {folder_path}")
        else:
            print(f"Directory already exists: {folder_path}")
    
    # Create mnist_data and cifar_data folders at the root level
    for data_folder in data_folders:
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
            print(f"Created directory: {data_folder}")
        else:
            print(f"Directory already exists: {data_folder}")

# ------------ bulk testing for all models -------------------------------

def test_accuracy_all_models():
    models = ['mnist','cifar']
    parts = [1,2,3]
    for part in parts:
        for model_name in models:
            print(f"Testing : ** Part ** {part}, {model_name} model ...")
            # load relevant data loaders
            train_loader,val_loader,test_loader = load_and_prep_data(part = part,dataset=model_name)
            # load model
            pretrained_model = torch.load(f"trained_models/part_{part}/{model_name}.pth")

            print("------------------ Accuracy ------------------")
            if part != 2:
                pretrained_encoder = pretrained_model.encoder
                classifier = pretrained_model.classifier
                test_classifier(encoder=pretrained_encoder,
                                classifier=classifier,
                                test_loader=test_loader)
            else:
                test_classifyingAutoEncoder(classifier=pretrained_model,
                            test_loader=test_loader)
            print("------------------- DONE ---------------------")

