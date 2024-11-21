print('Processing...')

import os
import cv2
import numpy as np
import tensorflow as tf

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the model once
model = tf.keras.models.load_model('gender_recognition_model.keras')

def preprocess_image(image_path):
    """Preprocess the image to match the model's input."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    img = cv2.resize(img, (28, 28))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_gender(image_path):
    """Predict gender using the trained model."""
    preprocessed_img = preprocess_image(image_path)
    predictions = model.predict(preprocessed_img, verbose=0)  # Suppress prediction logs
    gender_label = np.argmax(predictions)  # Get the highest probability class
    return "Female" if gender_label == 0 else "Male"

# Test the prediction
image_path = "/gender-training-dataset/female/0 (9).png"
result = predict_gender(image_path)
print(f"The predicted gender is: {result}")
