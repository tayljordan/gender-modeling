
import tensorflow as tf
import numpy as np
import ssl
import certifi
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs (INFO, WARNING, and ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl.create_default_context

# Load your dataset
images = np.load('images.npy')  # Your custom images (already resized to 28x28)
labels = np.load('labels.npy')  # Your custom labels

# Normalize images and remove the extra channel dimension
images = images / 255.0  # Normalize pixel values to [0, 1]
images = np.squeeze(images)  # Remove channel dimension, making it (28, 28)

# Split into training and testing sets
split_index = int(len(images) * 0.8)  # 80% training, 20% testing
training_images, test_images = images[:split_index], images[split_index:]
training_labels, test_labels = labels[:split_index], labels[split_index:]


# Design the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images
    tf.keras.layers.Dense(128, activation='relu'),  # Dense hidden layer with ReLU activation
    tf.keras.layers.Dense(2, activation='softmax')  # Output layer for 2 classes (male, female)
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(training_images, training_labels, epochs=5)

# Test the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_accuracy}")
