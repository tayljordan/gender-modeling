print('Processing...')

import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Set directories
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, "gender-training-dataset")


# Preprocessing function
def preprocess_image(img_path, target_size=(96, 96)):
    """Load, resize, and convert grayscale image to RGB."""
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {img_path}")

    # Resize to target size
    image_resized = cv2.resize(image, target_size)

    # Convert to RGB by duplicating grayscale channel
    image_rgb = np.stack([image_resized] * 3, axis=-1)

    return image_rgb / 255.0  # Normalize to [0, 1]


# Load dataset
data = []
labels = []

for label, folder in enumerate(["female", "male"]):
    folder_path = os.path.join(base_dir, folder)
    if not os.path.exists(folder_path):
        print(f"Directory does not exist: {folder_path}")
        continue
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                processed_image = preprocess_image(img_path, target_size=(96, 96))
                data.append(processed_image)
                labels.append(label)
            except ValueError as e:
                print(e)

# Convert to numpy arrays
data = np.array(data, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

# Split into training and test sets
train_images, test_images, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Sanity check
print(f"Training images shape: {train_images.shape}")
print(f"Test images shape: {test_images.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Test labels shape: {test_labels.shape}")

# Build transfer learning model with MobileNetV2
base_model = tf.keras.applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze pre-trained layers

# Add classification head
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=10, batch_size=32)

# Fine-tune the base model
base_model.trainable = True  # Unfreeze for fine-tuning
for layer in base_model.layers[:-50]:  # Freeze earlier layers
    layer.trainable = False

# Recompile with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fine-tune the model
model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=10, batch_size=32)

# Save the model
model.save(os.path.join(script_dir, 'gender_recognition_model_transfer_learning.keras'))

print("Model training complete and saved!")
