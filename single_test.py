import os
from os import walk
import cv2
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms, datasets
from torch.utils.data import Dataset, DataLoader, random_split


class GraphAnalysisModel(nn.Module):
    def __init__(self, num_nodes):
        super(GraphAnalysisModel, self).__init__()
        self.num_nodes = num_nodes
        
        # Use a pre-trained ResNet as the backbone
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Additional layers for graph analysis
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_nodes * num_nodes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, self.num_nodes, self.num_nodes)
        return torch.sigmoid(x)

def test_single_image(model_path, image_path, matrix_path, output_size=25):
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphAnalysisModel(output_size).to(device)
    
    # Load the checkpoint
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Print the keys in the checkpoint for debugging
    print("Checkpoint keys:", checkpoint.keys())
    
    # Check if the checkpoint contains a 'model' key (common in YOLOv5 saves)
    if 'model' in checkpoint:
        state_dict = checkpoint['model'].state_dict()
        print("Using 'model' key from checkpoint")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("Using 'state_dict' key from checkpoint")
    else:
        state_dict = checkpoint
        print("Using entire checkpoint as state_dict")
    
    # Remove 'module.' prefix if it exists
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Print the keys in the state_dict for debugging
    print("State dict keys:", state_dict.keys())
    
    # Load the state dict
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Loaded state_dict with strict=True")
    except RuntimeError as e:
        print(f"Strict loading failed: {e}")
        print("Attempting to load with strict=False")
        model.load_state_dict(state_dict, strict=False)
        print("Loaded state_dict with strict=False")
    
    model.eval()

    # Prepare the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(image)
    
    # Convert output to numpy array
    predicted_matrix = output.squeeze().cpu().numpy()

    # Load the actual adjacency matrix
    actual_matrix = np.load(matrix_path)

    return predicted_matrix, actual_matrix

# Example usage
model_path = 'graph_analysis_model.pth'
image_path = 'image_0.png'
matrix_path = 'adjacency_matrix_0.npy'
predicted_matrix, actual_matrix = test_single_image(model_path, image_path, matrix_path)

print("Predicted Adjacency Matrix:")
print(predicted_matrix)

print("\nActual Adjacency Matrix:")
print(actual_matrix)


# Calculate and print the Mean Squared Error
mse = np.mean((predicted_matrix - actual_matrix)**2)
print(f"\nMean Squared Error: {mse}")