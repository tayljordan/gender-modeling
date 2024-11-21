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

# Normalize the images (to [-1, 1])
images = (images / 255.0 - 0.5) / 0.5

# Reshape images to include the channel dimension
images = images.reshape(-1, 28, 28, 1)

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
print(f"Training images range: {training_images.min()} to {training_images.max()}")
print(f"Test images range: {test_images.min()} to {test_images.max()}")

'''
Small Dataset: Use a simple CNN or pretrained model (e.g., MobileNetV2).
Large Dataset: Deep CNN or pretrained models with fine-tuning.
Performance Boost: Add techniques like Batch Normalization or Dropout.
Cutting-Edge: Explore Vision Transformers for state-of-the-art performance.
'''

# Build and compile model - CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Use tf.data for efficient data handling with augmentation
train_dataset = (
    tf.data.Dataset.from_tensor_slices((training_images, training_labels))
    .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)  # Apply augmentation
    .shuffle(buffer_size=1024)  # Shuffle dataset
    .batch(32)  # Batch size
    .prefetch(buffer_size=tf.data.AUTOTUNE)  # Prefetch to improve performance
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    .batch(32)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Train model
model.fit(train_dataset, epochs=20, callbacks=[callbacks], validation_data=val_dataset)

# Save the model
model.save(os.path.join(script_dir, 'gender_recognition_model_28.keras'))

print("Success! Model saved in .keras format!")
print("You can go to test_individual_image if you want to test an individual image.")
print("Otherwise proceed to step 3.")
