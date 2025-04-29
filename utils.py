# utils.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.models as models
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Simple baseline CNN
class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super(BaselineCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Training function
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

# Evaluation function
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return total_loss / len(test_loader), accuracy

# Plot function
def plot_loss(train_losses, test_losses, accuracies, epochs):
  plt.figure(figsize=(12, 4))

  # Plot loss
  plt.subplot(1, 2, 1)
  plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o')
  plt.plot(range(1, epochs + 1), test_losses, label='Test Loss', marker='o')
  plt.title("Loss over Epochs")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()

  # Plot accuracy
  plt.subplot(1, 2, 2)
  plt.plot(range(1, epochs + 1), accuracies, label='Test Accuracy', color='green', marker='o')
  plt.title("Accuracy over Epochs")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy (%)")
  plt.ylim(0, 100)
  plt.legend()

  plt.tight_layout()
  plt.show()
  
    
# Pre-trained model (ResNet50)
def create_pretrained_model(num_classes):
    # Load pre-trained ResNet50
    model = models.resnet50(weights='IMAGENET1K_V2')
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def plot_confusion_matrix(model, test_loader, classes, device):
    # Set model to evaluation mode
    model.eval()
    
    # Initialize lists to store predictions and ground truth
    all_preds = []
    all_labels = []
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Normalize confusion matrix (optional)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    # Calculate per-class accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    # Print per-class accuracy
    print("Per-class accuracy:")
    for i, class_name in enumerate(classes):
        print(f"{class_name}: {class_accuracy[i]*100:.2f}%")
    
    return cm, class_accuracy
