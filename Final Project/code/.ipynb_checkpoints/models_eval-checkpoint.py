# ---------- Imports ------------------
import torch
import torch.nn as nn
import time
import os
from matplotlib import pyplot as plt
from data_loading import load_and_prep_data
from torchvision import datasets, transforms
from models_testing import  test_classifier,test_classifyingAutoEncoder
import pandas as pd
import numpy as np
from utils import plot_tsne
# -------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Accuracy_report():
    models = ['mnist','cifar']
    parts = [1,2,3]
    results_dictionary = {        
        (1,"mnist"): -1,
        (2,"mnist"): -1,
        (3,"mnist"): -1,
        (1,"cifar"): -1,
        (2,"cifar"): -1,
        (3,"cifar"): -1,
    }
    for part in parts:
        for model_name in models:
            exp_results = {}
            print(f"Testing : ** Part ** {part}, {model_name} model ...")
            # load relevant data loaders
            train_loader,val_loader,test_loader = load_and_prep_data(part = part,dataset=model_name)
            # load model
            pretrained_model = torch.load(f"trained_models/part_{part}/{model_name}.pth")
            keys_loaders = [ ("train",train_loader),("val",val_loader),("test",test_loader)] 
            pretrained_encoder = pretrained_model.encoder
            classifier = pretrained_model.classifier
            for key,loader in keys_loaders:
                print(f"Computing {key} accuracy ...")
                if part != 2:
                    res = test_classifier(encoder=pretrained_encoder,
                                    classifier=classifier,
                                    test_loader=loader)
                else:
                    res = test_classifyingAutoEncoder(classifier=pretrained_model,
                                test_loader=loader)
                print("------------------------------")
                exp_results[key] = res
                
            results_dictionary[(part,model_name)] = exp_results
    
    # Convert directly to DataFrame
    data = []
    for (part, model), results in results_dictionary.items():
        data.append({
            'Part': part,
            'Model': model,
            'Train': results['train'],  # Use get() to handle missing keys
            'Val': results['val'],
            'Test': results['test']
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(data)
    
    # Sort and round for better presentation
    results_df = results_df.round(2) 
    return results_df 


def test_reconstruction_loss(model,loader):
    model.eval()
    model.to(device)
    criterion = nn.L1Loss()
    total_loss = 0
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            reconstructed = model.reconstruct_image(data)
            loss = criterion(reconstructed, data)
            total_loss += loss.item()
            
    avg_loss = total_loss / len(loader)
    return avg_loss

def Reconstruction_report():
    models = ['mnist','cifar']
    parts = [1]
    results_dictionary = {        
        (1,"mnist"): -1,
        (1,"cifar"): -1,
    }
    for part in parts:
        for model_name in models:
            exp_results = {}
            print(f"Testing Reconstruction: ** Part ** {part}, {model_name} model ...")
            # load relevant data loaders
            train_loader,val_loader,test_loader = load_and_prep_data(part = part,dataset=model_name)
            # load model
            pretrained_model = torch.load(f"trained_models/part_{part}/{model_name}.pth")
            keys_loaders = [("train",train_loader),("val",val_loader),("test",test_loader)] 

            for key,loader in keys_loaders:
                print(f"Computing {key} Reconstrucion loss (MAE) ...")
                res = test_reconstruction_loss(model = pretrained_model,loader = loader)
                print(f"Reconstruction loss: {res}")
                print("------------------------------")
                exp_results[key] = res
                
            results_dictionary[(part,model_name)] = exp_results

        # Convert directly to DataFrame
    data = []
    for (part, model), results in results_dictionary.items():
        data.append({
            'Part': part,
            'Model': model,
            'Train': results['train'],  # Use get() to handle missing keys
            'Val': results['val'],
            'Test': results['test']
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(data)
    
    # Sort and round for better presentation
    results_df = results_df.round(4) 
    return results_df 


def denormalize(image, mean, std):
    return image * std + mean

def showcase_interpolation():
    part = 1
    model_name = 'mnist'
    
    train_loader, val_loader, test_loader = load_and_prep_data(part=part, dataset=model_name)
    
    # Load model
    pretrained_model = torch.load(f"trained_models/part_{part}/{model_name}.pth")
    pretrained_model.eval()  # Set to evaluation mode
    
    # Get device and move model
    device = pretrained_model.get_device()
    pretrained_model.to(device)
    
    # Get a batch of images from train loader
    data_iter = iter(train_loader)
    images, _ = next(data_iter)
    
    # Select 2 images
    images = images[[0,3]].to(device)
    
    # Reconstruct images
    with torch.no_grad():
        reconstructed_images = pretrained_model.reconstruct_image(images)
    
    # Encode images
    with torch.no_grad():
        encodings = pretrained_model.encoder(images)
    
    # Generate interpolation
    steps = 10
    interpolated_encodings = [
        (1 - t) * encodings[0] + t * encodings[1] for t in np.linspace(0, 1, steps)
    ]
    interpolated_encodings = torch.stack(interpolated_encodings)
    
    # Decode interpolated encodings
    with torch.no_grad():
        interpolated_images = pretrained_model.decoder(interpolated_encodings)
    
    # Denormalization parameters
    mean, std = 0.5, 0.5
    
    # Move tensors to CPU for visualization
    images = images.cpu()
    reconstructed_images = reconstructed_images.cpu()
    interpolated_images = interpolated_images.cpu()
    
    # Denormalize images
    images = denormalize(images, mean, std).clamp(0, 1)
    reconstructed_images = denormalize(reconstructed_images, mean, std).clamp(0, 1)
    interpolated_images = denormalize(interpolated_images, mean, std).clamp(0, 1)
    
    # Plot original, reconstructed, and interpolated images
    fig, axes = plt.subplots(1, 12, figsize=(15, 2))
    fig.suptitle("Interpolation between two images", fontsize=14)
    
    axes[0].imshow(images[0].permute(1, 2, 0).squeeze(), cmap='gray')
    axes[0].axis('off')
    
    for i in range(10):
        axes[i + 1].imshow(interpolated_images[i].permute(1, 2, 0).squeeze(), cmap='gray')
        axes[i + 1].axis('off')
    
    axes[11].imshow(images[1].permute(1, 2, 0).squeeze(), cmap='gray')
    axes[11].axis('off')
    
    plt.show()




# -------------------------------------
def denormalize(image, mean, std):
    return image * std + mean

def showcase_reconstruction():
    parts = [1]
    models = ['mnist', 'cifar']
    
    for part in parts:
        for model_name in models:
            train_loader, val_loader, test_loader = load_and_prep_data(part=part, dataset=model_name)
            
            # Load model
            pretrained_model = torch.load(f"trained_models/part_{part}/{model_name}.pth")
            pretrained_model.eval()  # Set to evaluation mode
            
            # Get device and move model
            device = pretrained_model.get_device()
            pretrained_model.to(device)
            
            # Get a batch of images from train loader
            data_iter = iter(train_loader)
            images, _ = next(data_iter)
            
            # Select 5 images
            images = images[:5].to(device)
            
            # Reconstruct images
            with torch.no_grad():
                reconstructed_images = pretrained_model.reconstruct_image(images)
            
            # Denormalization parameters
            if model_name == 'mnist':
                mean, std = 0.5, 0.5
            else:  # cifar
                mean, std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1), torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            
            # Move tensors to CPU for visualization
            images = images.cpu()
            reconstructed_images = reconstructed_images.cpu()
            
            # Denormalize images
            images = denormalize(images, mean, std).clamp(0, 1)
            reconstructed_images = denormalize(reconstructed_images, mean, std).clamp(0, 1)
            
            # Plot original and reconstructed images
            fig, axes = plt.subplots(2, 5, figsize=(10, 4))
            fig.suptitle(f"Reconstruction of {model_name} dataset", fontsize=14)
            
            for i in range(5):
                # Original images
                axes[0, i].imshow(images[i].permute(1, 2, 0).squeeze(), cmap='gray' if model_name == 'mnist' else None)
                axes[0, i].axis('off')
                
                # Reconstructed images
                axes[1, i].imshow(reconstructed_images[i].permute(1, 2, 0).squeeze(), cmap='gray' if model_name == 'mnist' else None)
                axes[1, i].axis('off')
            
            plt.show()


# ------------ tSNE plotting ---------------------------------------------
def plot_all_tsne_plots():
    models = ['mnist','cifar']
    parts = [1,2,3]
    for part in parts:
        for model_name in models:
            title = f"({model_name}_part_{part})"
            train_loader,val_loader,test_loader = load_and_prep_data(part = part,dataset=model_name)
            # load model
            pretrained_model = torch.load(f"trained_models/part_{part}/{model_name}.pth")
            plot_tsne(model = pretrained_model.encoder,
                      dataloader= test_loader,
                      device = pretrained_model.get_device(),
                      title=title)