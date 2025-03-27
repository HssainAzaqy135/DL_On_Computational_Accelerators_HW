# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
from matplotlib import pyplot as plt
# Needed for training
from torch.utils.data import random_split, DataLoader
# Models
from models_part1 import FinalClassifier , MNISTAutoencoder, CIFAR10Autoencoder
from models_part2 import MNISTClassifyingAutoencoder,CIFAR10ClassifyingAutoencoder
from models_part3 import MnistSimCLR,Cifar10SimCLR
from models_testing import  plot_accuracies,plot_losses,test_classifier,test_classifyingAutoEncoder
from models_testing import create_model_folders,PretrainedModel,save_pretrained_model,test_accuracy_all_models
from data_loading import load_and_prep_data
from models_testing import save_pretrained_model

base_save_dir = "trained_models/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_save_path(part,dataset):
    return f"{base_save_dir}part_{part}/{dataset}.pth"

def print_bar():
    print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")

def train_all_models():
    parts = [1,2,3]
    datasets = ['mnist','cifar']
    ret_dict = {}
    for part in parts:
        for dataset in datasets:
            ret_dict[f"part_{part}_{dataset}"] = train_model(part=part,dataset=dataset)

    return ret_dict

def train_model(part,dataset):
    train_function = {
        (1,"mnist"): train_part1_mnist,
        (2,"mnist"): train_part2_mnist,
        (3,"mnist"): train_part3_mnist,
        (1,"cifar"): train_part1_cifar,
        (2,"cifar"): train_part2_cifar,
        (3,"cifar"): train_part3_cifar,
    }
    return train_function[(part,dataset)](part,dataset)


def train_part1_mnist(part,dataset):
    print_bar()
    print(f"Part {part} ,Training {dataset} model ...")
    print_bar()
    train_loader,val_loader,test_loader = load_and_prep_data(part = part,dataset=dataset)

    # Model initialization
    model = MNISTAutoencoder(latent_dim=128).to(device)
    
    model_train_losses,model_val_losses = model.train_autoencoder(train_loader= train_loader,
                           val_loader=val_loader,
                           num_epochs=40,
                           learning_rate=1e-3,
                           weight_decay= 1e-3)
    print_bar()
    print(f"Part {part} , Training classifier for the {dataset} encoder")
    print_bar()
    pretrained_encoder = model.encoder
    for param in pretrained_encoder.parameters():
        param.requires_grad = False  # Ensure encoder is frozen
    classifier = FinalClassifier(latent_dim=128)
    classifier_train_losses, classifier_train_accuracies, classifier_val_accuracies = classifier.fit_classifier(encoder = pretrained_encoder,
                                                                               train_loader =  train_loader,
                                                                               val_loader = val_loader,
                                                                               num_epochs= 30, 
                                                                               learning_rate= 1e-3)
    save_path = compute_save_path(part= part,dataset= dataset)
    print_bar()
    print(f"Part {part} , Saving pretrained model and classifier to {save_path} ...")
    print_bar()
    save_pretrained_model(path = save_path,
                      encoder = model.encoder,
                      classifier = classifier,
                      decoder = model.decoder)
    print_bar()
    print("---------------------------- model saved -----------------------------")
    print_bar()
    return {"model_train_losses":model_train_losses,
            "model_val_losses":model_val_losses,
            "classifier_train_accuracies":classifier_train_accuracies,
            "classifier_val_accuracies":classifier_val_accuracies
           }
    

    

def train_part2_mnist(part,dataset):
    print_bar()
    print(f"Part {part} ,Training {dataset} model  WITH CLASSIFIER ...")
    print_bar()
    train_loader,val_loader,test_loader = load_and_prep_data(part = part,dataset=dataset)

    # Model initialization
    model = MNISTClassifyingAutoencoder(latent_dim=128).to(device)
    
    train_losses, classifier_train_accuracies, classifier_val_accuracies = model.train_autoencoder(train_loader= train_loader,
                       val_loader=val_loader,
                       num_epochs=40,
                       learning_rate=1e-3,
                       weight_decay= 1e-3)
    save_path = compute_save_path(part= part,dataset= dataset)
    print_bar()
    print(f"Part {part} , Saving pretrained model and classifier to {save_path} ...")
    print_bar()
    save_pretrained_model(path = save_path,
                      encoder = model.encoder,
                      classifier = model.classifier,
                      decoder = None)
    print("---------------------------- model saved -----------------------------")
    return {"model_train_losses":None,
            "model_val_losses":None,
            "classifier_train_accuracies":classifier_train_accuracies,
            "classifier_val_accuracies":classifier_val_accuracies
           }
    

def train_part3_mnist(part,dataset):
    print_bar()
    print(f"Part {part} ,Training {dataset} model ...")
    print_bar()
    train_loader,val_loader,test_loader = load_and_prep_data(part = part,dataset=dataset)

    # Model initialization
    model = MnistSimCLR(latent_dim=128,dropout_prob=0.1,temperature = 0.1).to(device)
    
    model_train_losses,model_val_losses = model.train_autoencoder(train_loader= train_loader,
                           val_loader=val_loader,
                           num_epochs=40,
                           learning_rate=1e-3,
                           weight_decay= 1e-3)
    
    print_bar()
    print(f"Part {part} , Training classifier for the {dataset} encoder")
    print_bar()
    pretrained_encoder = model.encoder
    for param in pretrained_encoder.parameters():
        param.requires_grad = False  # Ensure encoder is frozen
    classifier = FinalClassifier(latent_dim=128)
    classifier_train_losses, classifier_train_accuracies, classifier_val_accuracies = classifier.fit_classifier(encoder = pretrained_encoder,
                                                                               train_loader =  train_loader,
                                                                               val_loader = val_loader,
                                                                               num_epochs= 30, 
                                                                               learning_rate= 1e-3)
    save_path = compute_save_path(part= part,dataset= dataset)
    print_bar()
    print(f"Part {part} , Saving pretrained model and classifier to {save_path} ...")
    print_bar()
    save_pretrained_model(path = save_path,
                      encoder = model.encoder,
                      classifier = classifier,
                      decoder = None)
    print_bar()
    print("---------------------------- model saved -----------------------------")
    print_bar()
    return {"model_train_losses":model_train_losses,
            "model_val_losses":model_val_losses,
            "classifier_train_accuracies":classifier_train_accuracies,
            "classifier_val_accuracies":classifier_val_accuracies
           }
    

def train_part1_cifar(part,dataset):
    print_bar()
    print(f"Part {part} ,Training {dataset} model ...")
    print_bar()
    train_loader,val_loader,test_loader = load_and_prep_data(part = part,dataset=dataset)

    # Model initialization
    model = CIFAR10Autoencoder(latent_dim=128).to(device)
    
    model_train_losses,model_val_losses = model.train_autoencoder(train_loader= train_loader,
                           val_loader=val_loader,
                           num_epochs=40,
                           learning_rate=1e-3,
                           weight_decay= 1e-3)
    print_bar()
    print(f"Part {part} , Training classifier for the {dataset} encoder")
    print_bar()
    pretrained_encoder = model.encoder
    for param in pretrained_encoder.parameters():
        param.requires_grad = False  # Ensure encoder is frozen
    classifier = FinalClassifier(latent_dim=128)
    classifier_train_losses, classifier_train_accuracies, classifier_val_accuracies = classifier.fit_classifier(encoder = pretrained_encoder,
                                                                               train_loader =  train_loader,
                                                                               val_loader = val_loader,
                                                                               num_epochs= 30, 
                                                                               learning_rate= 1e-3)
    save_path = compute_save_path(part= part,dataset= dataset)
    print_bar()
    print(f"Part {part} , Saving pretrained model and classifier to {save_path} ...")
    print_bar()
    save_pretrained_model(path = save_path,
                      encoder = model.encoder,
                      classifier = classifier,
                      decoder = model.decoder)
    print_bar()
    print("---------------------------- model saved -----------------------------")
    print_bar()
    return {"model_train_losses":model_train_losses,
            "model_val_losses":model_val_losses,
            "classifier_train_accuracies":classifier_train_accuracies,
            "classifier_val_accuracies":classifier_val_accuracies
           }
    

def train_part2_cifar(part,dataset):
    print_bar()
    print(f"Part {part} ,Training {dataset} model  WITH CLASSIFIER ...")
    print_bar()
    train_loader,val_loader,test_loader = load_and_prep_data(part = part,dataset=dataset)

    # Model initialization
    model = CIFAR10ClassifyingAutoencoder(latent_dim=128).to(device)
    
    train_losses, classifier_train_accuracies, classifier_val_accuracies = model.train_autoencoder(train_loader= train_loader,
                       val_loader=val_loader,
                       num_epochs=40,
                       learning_rate=1e-3,
                       weight_decay= 1e-3)
    save_path = compute_save_path(part= part,dataset= dataset)
    print_bar()
    print(f"Part {part} , Saving pretrained model and classifier to {save_path} ...")
    print_bar()
    save_pretrained_model(path = save_path,
                      encoder = model.encoder,
                      classifier = model.classifier,
                      decoder = None)
    print("---------------------------- model saved -----------------------------")
    return {"model_train_losses":None,
            "model_val_losses":None,
            "classifier_train_accuracies":classifier_train_accuracies,
            "classifier_val_accuracies":classifier_val_accuracies
           }
    

def train_part3_cifar(part,dataset):
    print_bar()
    print(f"Part {part} ,Training {dataset} model ...")
    print_bar()
    train_loader,val_loader,test_loader = load_and_prep_data(part = part,dataset=dataset)

    # Model initialization
    model = Cifar10SimCLR(latent_dim=128,dropout_prob=0.1,temperature = 0.1).to(device)
    
    model_train_losses,model_val_losses = model.train_autoencoder(train_loader= train_loader,
                           val_loader=val_loader,
                           num_epochs=60,
                           learning_rate=1e-3,
                           weight_decay= 1e-3)
    
    print_bar()
    print(f"Part {part} , Training classifier for the {dataset} encoder")
    print_bar()
    pretrained_encoder = model.encoder
    for param in pretrained_encoder.parameters():
        param.requires_grad = False  # Ensure encoder is frozen
    classifier = FinalClassifier(latent_dim=128)
    classifier_train_losses, classifier_train_accuracies, classifier_val_accuracies = classifier.fit_classifier(encoder = pretrained_encoder,
                                                                               train_loader =  train_loader,
                                                                               val_loader = val_loader,
                                                                               num_epochs= 40, 
                                                                               learning_rate= 1e-3)
    save_path = compute_save_path(part= part,dataset= dataset)
    print_bar()
    print(f"Part {part} , Saving pretrained model and classifier to {save_path} ...")
    print_bar()
    save_pretrained_model(path = save_path,
                      encoder = model.encoder,
                      classifier = classifier,
                      decoder = None)
    print_bar()
    print("---------------------------- model saved -----------------------------")
    print_bar()
    return {"model_train_losses":model_train_losses,
            "model_val_losses":model_val_losses,
            "classifier_train_accuracies":classifier_train_accuracies,
            "classifier_val_accuracies":classifier_val_accuracies
           }
    