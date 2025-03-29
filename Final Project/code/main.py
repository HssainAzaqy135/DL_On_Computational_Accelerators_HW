import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from models_testing import  plot_accuracies,plot_losses,test_classifier,test_classifyingAutoEncoder
from models_testing import create_model_folders,PretrainedModel,save_pretrained_model,test_accuracy_all_models
from models_eval import plot_all_tsne_plots
from data_loading import load_and_prep_data
from models_training import train_model,train_all_models
import utils

#logging results
import json

    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Folders to save models for later evaluation
    create_model_folders()
    # training (about 1.5 hours)
    training_results = train_all_models()
    
    file_path = "training_results.json"

    # Write dictionary to JSON file
    with open(file_path, 'w') as json_file:
        json.dump(training_results, json_file)

    test_accuracy_all_models()