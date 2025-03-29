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
# -------------------------------------
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
            keys_loaders = [ ("train",train_loader),("val",val_loader),("test",test_loader)] 

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
    results_df = results_df.round(2) 
    return results_df 

def showcase_interpolation():
    pass




# -------------------------------------
def showcase_reconstruction(autoencoder, classifier, val_loader, num_images=5):
    autoencoder.eval()
    classifier.eval()
    
    # Randomly select indices
    indices = np.random.choice(len(val_dataset), num_images, replace=False)
    images = torch.stack([val_dataset[i][0] for i in indices])
    labels = torch.tensor([val_dataset[i][1] for i in indices])
    
    images = images.to(device)
    labels = labels.to(device)
    
    # 1. Reconstruct images through autoencoder
    with torch.no_grad():
        reconstructed, latent = autoencoder(images)
    
    # 2. Get classifications from classifier
    with torch.no_grad():
        class_outputs = classifier(latent)
        _, predictions = torch.max(class_outputs, 1)
    
    # 3. Prepare for visualization
    images = images.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # 4. Plot original vs reconstructed with predictions
    fig, axes = plt.subplots(2, num_images, figsize=(2*num_images, 4))
    
    for i in range(num_images):
        # Original images (top row)
        axes[0, i].imshow(images[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'True: {labels[i]}')
        
        # Reconstructed images (bottom row)
        axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Pred: {predictions[i]}')
    
    plt.tight_layout()
    plt.show()
    
    # 5. Print results
    print("\nResults:")
    print("Image | True Label | Predicted Label | Correct")
    print("-" * 45)
    for i in range(num_images):
        correct = "Yes" if labels[i] == predictions[i] else "No"
        print(f"{i:5d} | {labels[i]:11d} | {predictions[i]:14d} | {correct}")

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