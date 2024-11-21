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

# Normalize the images
images = images / 255.0

# Reshape images to include the channel dimension (assuming grayscale)
images = images.reshape(-1, 28, 28, 1)  # Ensure shape is (samples, height, width, channels)

# Split the data into training and test sets
split_idx = int(0.8 * len(images))  # 80-20 split
training_images, test_images = images[:split_idx], images[split_idx:]
training_labels, test_labels = labels[:split_idx], labels[split_idx:]

# Build the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),  # Explicit input layer
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 0 = Female, 1 = Male.
class_weights = {0: 1, 1: 1}
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks], validation_data=(test_images, test_labels),
          class_weight=class_weights)

# Save the model in the .keras format
model.save(os.path.join(script_dir, 'gender_recognition_model.keras'))
print("Success! Model saved in .keras format!")
