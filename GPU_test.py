import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import yaml
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load configuration from config.yaml
config_path = os.path.join(os.getcwd(), "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Extract parameters from config
params = config['parameters']
image_size = tuple(params['image_size'])  # Input image size (height, width)
rescale = params['rescale']  # Rescaling factor
test_directory_path_male = os.path.join(os.getcwd(), config['directories']['test_set_male'])
test_directory_path_female = os.path.join(os.getcwd(), config['directories']['test_set_female'])
model_path = os.path.join(os.getcwd(), config['model']['path'])

# Check and print GPU usage
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs are available: {len(gpus)} GPU(s) detected.")
else:
    print("No GPUs detected. Using CPU.")

# Load the trained model
model = load_model(model_path)

# Function to preprocess the image
def preprocess_image(image_path, target_size, rescale_factor):
    try:
        # Load and resize the image
        img = load_img(image_path, target_size=target_size)
        # Convert to array and normalize pixel values
        img_array = img_to_array(img) * rescale_factor
        # Add batch dimension
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Function to process all images in a directory and calculate accuracy
def process_directory(directory_path, actual_label, target_size, rescale_factor):
    correct_predictions = 0
    total_images = 0

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Only process image files
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Preprocess the image
            processed_image = preprocess_image(file_path, target_size, rescale_factor)

            if processed_image is not None:
                # Make a prediction
                prediction = model.predict(processed_image, verbose=0)

                # Interpret the prediction
                predicted_label = 1 if prediction[0][0] > 0.5 else 0  # Male=1, Female=0

                # Compare predicted label with the actual label
                if predicted_label == actual_label:
                    correct_predictions += 1

                total_images += 1

    # Calculate accuracy
    accuracy = correct_predictions / total_images * 100 if total_images > 0 else 0
    print(f"Processed {total_images} images in {directory_path}. Accuracy: {accuracy:.2f}%")
    return accuracy

# Process male test set (label=1)
male_accuracy = process_directory(test_directory_path_male, actual_label=1, target_size=image_size, rescale_factor=rescale)

# Process female test set (label=0)
female_accuracy = process_directory(test_directory_path_female, actual_label=0, target_size=image_size, rescale_factor=rescale)

# Calculate average accuracy
average_accuracy = (male_accuracy + female_accuracy) / 2

# Print the final accuracy results
print(f"Male Test Set Accuracy: {male_accuracy:.2f}%")
print(f"Female Test Set Accuracy: {female_accuracy:.2f}%")
print(f"Average Accuracy: {average_accuracy:.2f}%")
