# project.py

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

def load_data(data_dir='/content/drive/MyDrive/dl4m/dl4m-group7-main/data', batch_size=32, image_size=(224, 224)):
    # Transformations for the dataset (resize, convert to tensor, normalize)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load dataset using ImageFolder
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Split into train/test (80/20 split)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, dataset.classes
