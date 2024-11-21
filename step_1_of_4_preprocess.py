print("Processing...")

import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory where the script is located
base_dir = os.path.join(script_dir, "gender-training-dataset")  # Relative path to the dataset


# Custom Dataset
class GenderDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.base_dir = base_dir
        self.transform = transform
        self.data = []
        self.labels = []

        # Load images and labels
        for label, folder in enumerate(["female", "male"]):
            folder_path = os.path.join(base_dir, folder)
            if not os.path.exists(folder_path):  # Check if folder exists
                print(f"Directory does not exist: {folder_path}")
                continue
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.data.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:  # Handle failed image load
            raise ValueError(f"Failed to load image: {img_path}")

        img = cv2.resize(img, (28, 28))  # Resize to 28x28

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return img, label


# Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor and scale [0, 255] to [0, 1]
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Create dataset and dataloader
dataset = GenderDataset(base_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Check the dataset
for images, labels in dataloader:
    print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
    break

# Save images and labels with progress indication
images = []
labels = []

for img, label in tqdm(dataset, desc="Processing images", unit="image"):
    images.append(img.numpy())
    labels.append(label)

images = np.array(images)
labels = np.array(labels)

# Save the dataset in the script directory
np.save(os.path.join(script_dir, 'images.npy'), images)
np.save(os.path.join(script_dir, 'labels.npy'), labels)

# Debugging: Print a few labels and corresponding file paths
for i in range(10):
    print(f"Image Path: {dataset.data[i]}, Label: {dataset.labels[i]}")

print(f"Success! Images shape: {images.shape}, Labels shape: {labels.shape}")
print("Proceed to Step 2.")
