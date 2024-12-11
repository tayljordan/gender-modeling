import os
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Absolute paths for data directory
data_dir = "/Users/jordantaylor/PycharmProjects/gender-modeling/gender-training-dataset"

# Parameters
validation_split = 0.2
downscaled_image_size = (64, 64)  # Smaller resolution for memory efficiency
batch_size = 16

# Ensure the dataset directory exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory not found: {data_dir}")

# Data generators for loading images
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=validation_split
)

train_generator = datagen.flow_from_directory(
    directory=data_dir,
    target_size=downscaled_image_size,  # Use downscaled resolution
    batch_size=batch_size,  # Process images in batches
    class_mode='binary',
    subset='training',
    shuffle=False
)

val_generator = datagen.flow_from_directory(
    directory=data_dir,
    target_size=downscaled_image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Extract features incrementally
def extract_features(generator):
    features, labels = [], []
    total_batches = len(generator)
    for img_batch, label_batch in tqdm(generator, total=total_batches, desc="Extracting features"):
        batch_features = img_batch.reshape(img_batch.shape[0], -1)  # Flatten batch
        features.append(batch_features)
        labels.append(label_batch)
    return np.vstack(features), np.hstack(labels)

# Extract features for training and validation sets
X_train, y_train = extract_features(train_generator)
X_val, y_val = extract_features(val_generator)

# Train a Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Evaluate the model
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)

# Print accuracies
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Classification report for validation set
print("\nValidation Set Classification Report:")
print(classification_report(y_val, y_val_pred))
