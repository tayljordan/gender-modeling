import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Get the base directory (gender-modeling)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Paths
model_path = os.path.join(base_dir, "gender_recognition_model.keras")
test_images_path = os.path.join(base_dir, "images.npy")
test_labels_path = os.path.join(base_dir, "labels.npy")

# Load the model
model = tf.keras.models.load_model(model_path)

# Load test data
test_images = np.load(test_images_path)  # Shape: (num_samples, height, width, channels)
test_labels = np.load(test_labels_path)  # Shape: (num_samples,)

# Normalize test images (ensure the same preprocessing as during training)
test_images = test_images / 255.0

# Make predictions
predictions = model.predict(test_images, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)  # Convert probabilities to class labels

# Calculate metrics
accuracy = accuracy_score(test_labels, predicted_labels)
classification_rep = classification_report(test_labels, predicted_labels, target_names=["Female", "Male"])
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Optional: ROC-AUC (requires binary probabilities)
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
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Female", "Male"], yticklabels=["Female", "Male"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Plot ROC Curve
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(test_labels, predictions[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
