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

# ------------ Data Selction function ---------------------
def load_and_prep_data(part,dataset):
    prep_function = {
        (1,"mnist"): prep_part1_mnist,
        (2,"mnist"): prep_part2_mnist,
        (3,"mnist"): prep_part3_mnist,
        (1,"cifar"): prep_part1_cifar,
        (2,"cifar"): prep_part2_cifar,
        (3,"cifar"): prep_part3_cifar,
    }
    return prep_function[(part,dataset)]()

# ------------ Mnist prep functions -----------------------
def prep_part1_mnist():
    mnist_path = "./mnist_data"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root=mnist_path,  
        train=True,       
        transform=transform,  # Apply transformations here
        download=True     
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root=mnist_path,
        train=False,  
        transform=transform,  # Apply same transformations for test data
        download=True
    )
    
    print("MNIST dataset downloaded successfully!")

    # Define loaders
    train_size = 50_000
    val_size = 10_000
    batch_size = 64
    
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    # Create DataLoaders
    mnist_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    mnist_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    mnist_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("DataLoaders created successfully!")
    return mnist_train_loader,mnist_val_loader,mnist_test_loader

def prep_part2_mnist():
    return prep_part1_mnist()

def prep_part3_mnist():
    mnist_path = "./mnist_data"

    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root=mnist_path,  
        train=True,       
        transform=train_transform,  # Apply train transformations
        download=True     
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root=mnist_path,
        train=False,  
        transform=test_transform,  # Apply test transformations
        download=True
    )
    
    print("MNIST dataset downloaded successfully!")
    train_size = 50_000
    val_size = 10_000
    batch_size = 256
    
    
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    # Create DataLoaders
    mnist_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    mnist_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    mnist_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("DataLoaders created successfully!")
    return mnist_train_loader,mnist_val_loader,mnist_test_loader


# ------------ Cifar prep functions -----------------------
def prep_part1_cifar():
    cifar10_path = "./cifar10_data"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    
    c10_full_train_dataset = torchvision.datasets.CIFAR10(
        root=cifar10_path,  
        train=True,       
        transform=transform,
        download=True     
    )
    
    c10_test_dataset = torchvision.datasets.CIFAR10(
        root=cifar10_path,
        train=False,  
        transform=transform,
        download=True
    )

    print("CIFAR-10 dataset downloaded successfully!")
    train_size = 40_000
    val_size = 10_000
    
    c10_train_dataset, c10_val_dataset = random_split(c10_full_train_dataset, [train_size, val_size])
    
    # Check dataset sizes
    print(f"Train size: {len(c10_train_dataset)}, Validation size: {len(c10_val_dataset)}, Test size: {len(c10_test_dataset)}")
    
    # Define batch size
    batch_size = 64
    
    # Create DataLoaders
    c10_train_loader = DataLoader(c10_train_dataset, batch_size=batch_size, shuffle=True)
    c10_val_loader = DataLoader(c10_val_dataset, batch_size=batch_size, shuffle=False)
    c10_test_loader = DataLoader(c10_test_dataset, batch_size=batch_size, shuffle=False)
    
    print("DataLoaders created successfully!")
    return c10_train_loader,c10_val_loader,c10_test_loader

def prep_part2_cifar():
    return prep_part1_cifar()

def prep_part3_cifar():
    cifar10_path = "./cifar10_data"

    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    c10_full_train_dataset = torchvision.datasets.CIFAR10(
        root=cifar10_path,  
        train=True,       
        transform=train_transform,
        download=True     
    )
    
    c10_test_dataset = torchvision.datasets.CIFAR10(
        root=cifar10_path,
        train=False,  
        transform=test_transform,
        download=True
    )
    
    print("CIFAR-10 dataset downloaded successfully!")
    train_size = 40_000
    val_size = 10_000
    batch_size = 256
    
    c10_train_dataset, c10_val_dataset = random_split(c10_full_train_dataset, [train_size, val_size])
    
    # Check dataset sizes
    print(f"Train size: {len(c10_train_dataset)}, Validation size: {len(c10_val_dataset)}, Test size: {len(c10_test_dataset)}")
    
    
    
    # Create DataLoaders
    c10_train_loader = DataLoader(c10_train_dataset, batch_size=batch_size, shuffle=True)
    c10_val_loader = DataLoader(c10_val_dataset, batch_size=batch_size, shuffle=False)
    c10_test_loader = DataLoader(c10_test_dataset, batch_size=batch_size, shuffle=False)
    
    print("DataLoaders created successfully!")
    return c10_train_loader,c10_val_loader,c10_test_loader
    