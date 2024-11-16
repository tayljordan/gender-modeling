import tensorflow as tf

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

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),  # Explicitly use 'relu'
    tf.keras.layers.Dense(10, activation='softmax')  # Explicitly use 'softmax'
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

# Save the model in the .keras format
model.save('model.keras')
print("Model saved successfully in .keras format!")
