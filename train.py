import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

num_nodes = 25

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

# Define the transforms for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the dataset and dataloader
dataset = GraphDataset(images_dir='dataset/images', matrices_dir='dataset/adjacency_matrices', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GraphAnalysisModel(num_nodes).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, matrices in dataloader:
        images = images.to(device)
        matrices = matrices.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, matrices)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

# Save the trained model
torch.save(model.state_dict(), 'graph_analysis_model.pth')

# Function to predict adjacency matrix for a single image
def predict_adjacency_matrix(image_path):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
    
    return output.squeeze().cpu().numpy()

# Example usage
# predicted_matrix = predict_adjacency_matrix('path_to_test_image.png')
# print(predicted_matrix)