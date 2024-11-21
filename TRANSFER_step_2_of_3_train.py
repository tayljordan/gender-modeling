print('Processing...')

import tensorflow as tf
import numpy as np
import os


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95:
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


# Callback instance
callbacks = myCallback()

# Set the base directory relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load preprocessed data
image_path = os.path.join(script_dir, 'images.npy')
label_path = os.path.join(script_dir, 'labels.npy')

images = np.load(image_path)
labels = np.load(label_path)

# Normalize the images to [0, 1] (required for MobileNetV2)
images = images / 255.0

# Convert grayscale images to RGB
images = np.stack([images] * 3, axis=-1).reshape(-1, 28, 28, 3)

# Resize images to MobileNetV2 input size (96, 96)
images = tf.image.resize(images, [96, 96]).numpy()

# Split data into training and test sets
split_idx = int(0.8 * len(images))
training_images, test_images = images[:split_idx], images[split_idx:]
training_labels, test_labels = labels[:split_idx], labels[split_idx:]

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1)
])

# Confirm shapes and preprocessing ranges
print(f"Training images shape: {training_images.shape}, range: {training_images.min()} to {training_images.max()}")
print(f"Test images shape: {test_images.shape}, range: {test_images.min()} to {test_images.max()}")

# Build transfer learning model using MobileNetV2
base_model = tf.keras.applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze pre-trained layers

# Add classification head
model = tf.keras.Sequential([
    data_augmentation,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Prepare data using tf.data
train_dataset = (
    tf.data.Dataset.from_tensor_slices((training_images, training_labels))
    .shuffle(buffer_size=1024)  # Shuffle dataset
    .batch(32)  # Batch size
    .prefetch(buffer_size=tf.data.AUTOTUNE)  # Prefetch to improve performance
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    .batch(32)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Train the model
model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=[callbacks])

# Fine-tune the base model
base_model.trainable = True  # Unfreeze the base model for fine-tuning
for layer in base_model.layers[:-50]:  # Freeze earlier layers
    layer.trainable = False

# Recompile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fine-tune the model
model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=[callbacks])

# Save the model
model.save(os.path.join(script_dir, 'gender_recognition_model_transfer_learning.keras'))

print("Model training complete and saved!")
