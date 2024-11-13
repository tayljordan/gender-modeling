import os
import tensorflow as tf
import ssl
import certifi

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl.create_default_context

# Sample data
# https://github.com/zalandoresearch/fashion-mnist?tab=readme-ov-file
mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Normalize images
training_images  = training_images / 255.0
test_images = test_images / 255.0

# Design the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# 4 - Compile and train the model
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

# 5 - Test the Model
model.evaluate(test_images, test_labels)

# Proceed here: https://developers.google.com/codelabs/tensorflow-2-computervision#5

