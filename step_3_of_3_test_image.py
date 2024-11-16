import os
import cv2
import numpy as np
import tensorflow as tf

# Load the model saved in .keras format
model = tf.keras.models.load_model('model.keras')

def preprocess_image(image_path):
    """Preprocess the image to match the model's input."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_gender(image_path):
    """Predict gender using the trained model."""
    preprocessed_img = preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    gender_label = np.argmax(predictions)  # Get the highest probability class
    return "Female" if gender_label == 0 else "Male"

# Example usage
image_path = '/Users/jordantaylor/PycharmProjects/gender-modeling/gender-dataset/male/1 (2).jpeg'
result = predict_gender(image_path)
print(f"The predicted gender is: {result}")
