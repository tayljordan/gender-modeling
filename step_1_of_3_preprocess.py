print('Processing...')

import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, "gender-training-dataset")


class GenderDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.base_dir = base_dir
        self.transform = transform
        self.data = []
        self.labels = []

        # Load images and labels
        for image_label, folder in enumerate(["female", "male"]):
            folder_path = os.path.join(base_dir, folder)
            if not os.path.exists(folder_path):  # Check if folder exists
                print(f"Directory does not exist: {folder_path}")
                continue
            if len(os.listdir(folder_path)) == 0:  # Skip empty folders
                print(f"Skipping empty directory: {folder_path}")
                continue
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.data.append(img_path)
                    self.labels.append(image_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image_label = self.labels[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:  # Handle failed image load
            raise ValueError(f"Failed to load image: {img_path}")

        # Options: 28X28, 64X64, 128X128
        image = cv2.resize(image, (28, 28))

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return image, image_label


# Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor and scale [0, 255] to [0, 1]
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Create dataset and dataloader
dataset = GenderDataset(base_dir, transform=transform)

# Common batch sizes: 16, 32, 64, 128
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


def visualize_data(dataset, num_samples=10):
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset))  # Randomly select an index
        img, label = dataset[idx]
        img = img.squeeze(0).numpy()  # Remove channel dimension

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title("Female" if label == 0 else "Male")
        plt.axis("off")
    plt.show()


# Visualize some samples
visualize_data(dataset)

# Check the dataset
for images, labels in dataloader:
    print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")

    # Options: 28X28, 64X64, 128X128
    assert images.shape[1:] == (1, 28, 28), "Dataloader image shape mismatch."
    break

# Save images and labels with progress indication
images = []
labels = []

for img, label in tqdm(dataset, desc="Processing images", unit="image"):
    images.append(img.numpy())
    labels.append(label)

images = np.array(images)
labels = np.array(labels)

# Sanity check on saved data
print(f"Final images shape: {images.shape}, Labels shape: {labels.shape}")

# Options: 28X28, 64X64, 128X128
assert images.shape[1:] == (1, 28, 28), "Image shape mismatch. Expected (1, 28, 28)."

# Save the dataset in the script directory
np.save(os.path.join(script_dir, 'images.npy'), images)
np.save(os.path.join(script_dir, 'labels.npy'), labels)

# Debugging: Print a few labels and corresponding file paths
for i in range(10):
    print(f"Image Path: {dataset.data[i]}, Label: {dataset.labels[i]}")

print(f"Success! Images shape: {images.shape}, Labels shape: {labels.shape}")
print("Proceed to Step 2.")
