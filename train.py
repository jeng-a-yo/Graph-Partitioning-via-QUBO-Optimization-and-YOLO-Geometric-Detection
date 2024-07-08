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

from tqdm import tqdm

num_nodes = 25

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 64
epochs = 30
learning_rate = 0.001
momentum = 0.9

# Define the transforms for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the ratio for training, validation, and test sets
train_ratio = 0.7
val_ratio = 0.15

class GraphDataset(Dataset):
    def __init__(self, images_dir, matrices_dir, transform=None):
        self.images_dir = images_dir
        self.matrices_dir = matrices_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
        self.matrices = sorted([f for f in os.listdir(matrices_dir) if f.endswith('.npy')])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.images[idx])
        matrix_path = os.path.join(self.matrices_dir, self.matrices[idx])
        
        image = Image.open(image_path).convert('RGB')
        adjacency_matrix = np.load(matrix_path)
        
        if self.transform:
            image = self.transform(image)
        
        adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
        
        return image, adjacency_matrix

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

def train(model, train_loader, val_loader, optimizer, criterion):
    """Train the model and evaluate on the validation set"""
    train_acc, train_loss = [], []
    val_acc, val_loss = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        train_running_loss = 0.
        train_correct_predictions = 0
        train_total_predictions = 0

        # Training loop
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        for batch_idx, (data, target) in progress_bar:
            optimizer.zero_grad()  # Zero the gradients
            predict = model(data.to(device))  # Forward pass
            loss = criterion(predict, target.to(device))  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update parameters

            train_running_loss += loss.item()
            train_correct_predictions += (predict.round() == target.to(device)).float().sum().item()
            train_total_predictions += target.numel()

            progress_bar.set_postfix({'loss': round(loss.item(), 6)})
        
        train_loss.append(train_running_loss / len(train_loader))
        train_acc.append(train_correct_predictions / train_total_predictions)

        print(f"[Info] Epoch {epoch}: Training Loss: {round(train_loss[-1], 4)}, Training Accuracy: {round(train_acc[-1], 4)}")

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0

        progress_bar_val = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validation {epoch}")
        with torch.no_grad():  # No need to compute gradients during validation
            for batch_idx, (data, target) in progress_bar_val:
                predict = model(data.to(device))  # Forward pass
                loss = criterion(predict, target.to(device))  # Compute loss
                val_running_loss += loss.item()
                val_correct_predictions += (predict.round() == target.to(device)).float().sum().item()
                val_total_predictions += target.numel()

                progress_bar_val.set_postfix({'val_loss': round(loss.item(), 6)})

        val_loss.append(val_running_loss / len(val_loader))
        val_acc.append(val_correct_predictions / val_total_predictions)

        print(f"[Info] Epoch {epoch}: Validation Loss: {round(val_loss[-1], 4)}, Validation Accuracy: {round(val_acc[-1], 4)}")
        print("----------------------------------------------------------------")

    print("[Info] Training completed")
    return train_acc, train_loss, val_acc, val_loss

def test(model, test_loader):
    """Test the model on the test set"""
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        for data, target in progress_bar:
            predict = model(data.to(device))  # Forward pass
            correct_predictions += (predict.round() == target.to(device)).float().sum().item()
            total_predictions += target.numel()
            accuracy = correct_predictions / total_predictions
            progress_bar.set_postfix({'accuracy': f'{accuracy:.4f}'})
    
    print(f"[Info] Test Results: Accuracy: {accuracy:.4f}")

def draw_plot(train_acc, train_loss, val_acc, val_loss) -> None:
    # Plot training and validation loss and accuracy
    plt.figure(figsize=(12, 8))
    # Loss
    plt.subplot(2, 1, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    # Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    # Save figure
    plt.tight_layout()
    plt.savefig('training_graph.png')

def main():
    start_time = time.time()

    # Create the dataset and dataloader
    dataset = GraphDataset(images_dir='dataset/images', matrices_dir='dataset/adjacency_matrices', transform=transform)
    
    # Define the sizes for training, validation, and test sets
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Split the dataset
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Build the model
    model = GraphAnalysisModel(num_nodes).to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Train the model
    train_acc, train_loss, val_acc, val_loss = train(model, train_loader, val_loader, optimizer, criterion)

    # Evaluate the model
    test(model, test_loader)

    # Save the trained model
    torch.save(model.state_dict(), 'graph_analysis_model.pth')

    # Plot training and validation loss and accuracy
    draw_plot(train_acc, train_loss, val_acc, val_loss)

    print(f"[Info] Spent Time: {round(time.time() - start_time, 4)} seconds")

if __name__ == '__main__':
    main()