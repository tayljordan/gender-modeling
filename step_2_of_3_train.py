import tensorflow as tf
import pandas as pd
import numpy as np

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95:
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True

# Callback instance
callbacks = myCallback()

# Load dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the data
training_images = training_images / 255.0
test_images = test_images / 255.0

# Build the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Convolutional layer
    tf.keras.layers.MaxPooling2D(2, 2),  # Max pooling
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer
    tf.keras.layers.MaxPooling2D(2, 2),  # Second max pooling
    tf.keras.layers.Flatten(),  # Flatten for dense layers
    tf.keras.layers.Dense(128, activation='relu'),  # Fully connected layer
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer for 10 classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Ensure input data shape includes the channel dimension (grayscale)
training_images = np.expand_dims(training_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# Train the model
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

# Save the model in the .keras format
model.save('maritime_gender_recognition.keras')
print("Model saved successfully in .keras format!")
