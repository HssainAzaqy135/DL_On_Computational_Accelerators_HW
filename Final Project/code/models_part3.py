# --------- Imports -------------------
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# Needed for training
import time
from torch.optim.lr_scheduler import StepLR
from models_part1 import FinalClassifier
# --------- Loss ----------------------
class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.5, device='cuda'):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, z1, z2):
        # Normalize the embeddings (L2 normalization)
        z1 = F.normalize(z1, dim=-1, p=2)
        z2 = F.normalize(z2, dim=-1, p=2)

        # cosine similarity
        similarity_matrix = torch.matmul(z1, z2.T) / self.temperature

        # Create labels for the positive pairs (diagonal elements)
        batch_size = z1.size(0)
        labels = torch.arange(batch_size).to(self.device)

        # Compute the loss using cross-entropy
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss
# --------- Auxiliry ------------------
class RandomNoise(object):
    def __init__(self, mean=0.0, std=0.1, p=0.5):
        """
        Add random noise to an image.
        
        :param mean: Mean of the Gaussian noise
        :param std: Standard deviation of the Gaussian noise
        :param p: Probability of applying the noise
        """
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, image):
        """
        Apply the noise to the image with probability p.
        
        :param image: Input image (tensor)
        :return: Noisy image (tensor)
        """
        if np.random.rand() < self.p:
            # Gaussian noise (generated with the same shape as the image tensor)
            noise = torch.randn_like(image) * self.std + self.mean  # Add Gaussian noise
            image = image + noise
            image = torch.clamp(image, 0.0, 1.0)  # Ensure pixel values remain in [0, 1] range
        return image

# -------------------------------------
# --------- MNIST ---------------------
