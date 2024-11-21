import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt

# Set the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Paths
model_path = os.path.join(base_dir, "gender_recognition_model_transfer_learning.keras")
test_images_path = os.path.join(base_dir, "images.npy")
test_labels_path = os.path.join(base_dir, "labels.npy")

# Load the model
model = tf.keras.models.load_model(model_path)

# Load test data
test_images = np.load(test_images_path)  # Shape: (num_samples, height, width)
test_labels = np.load(test_labels_path)  # Shape: (num_samples,)

# Normalize test images (same as training)
test_images = test_images / 255.0

# Convert grayscale to RGB
test_images = np.stack([test_images] * 3, axis=-1)

# Resize images to match model input size (96x96)
test_images_resized = tf.image.resize(test_images, [96, 96]).numpy()

# Confirm preprocessing
print(
    f"Test images shape: {test_images_resized.shape}, range: {test_images_resized.min()} to {test_images_resized.max()}")

# Make predictions
predictions = model.predict(test_images_resized, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)  # Convert probabilities to class labels

# Ensure predictions and labels align
assert predicted_labels.shape[0] == test_labels.shape[0], "Mismatch between predictions and labels!"

# Calculate metrics
accuracy = accuracy_score(test_labels, predicted_labels)
classification_rep = classification_report(test_labels, predicted_labels, target_names=["Female", "Male"])
conf_matrix = confusion_matrix(test_labels, predicted_labels)
roc_auc = roc_auc_score(test_labels, predictions[:, 1])  # Use probabilities for "Male"

# Print Metrics
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_rep)
print("\nConfusion Matrix:")
print(conf_matrix)
print(f"\nROC-AUC: {roc_auc:.2f}")

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Female", "Male"],
            yticklabels=["Female", "Male"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(test_labels, predictions[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Identify indices of false negatives (actual: Female (0), predicted: Male (1))
false_negatives_idx = np.where((test_labels == 0) & (predicted_labels == 1))[0]

# Extract the false negative images
false_negatives_images = test_images_resized[false_negatives_idx]

# Save or visualize the false negative images
false_negatives_dir = os.path.join(base_dir, "false_negatives")
os.makedirs(false_negatives_dir, exist_ok=True)

for i, idx in enumerate(false_negatives_idx):
    img = false_negatives_images[i]
    plt.imsave(os.path.join(false_negatives_dir, f"false_negative_{idx}.png"), img)

print(f"False negative images saved to: {false_negatives_dir}")
