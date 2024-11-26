import os
import yaml
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load configuration from config.yaml
config_path = os.path.join(os.getcwd(), "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Extract parameters from config
params = config['parameters']
image_size = tuple(params['image_size'])  # Input image size (height, width)
rescale = params['rescale']  # Rescaling factor
model_path = os.path.join(os.getcwd(), config['model']['path'])
test_directory_path_female = '/Users/jordantaylor/PycharmProjects/gender-modeling/test-set-large-female'

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

# Process test-set-large-female
def process_female_test_set(directory_path, target_size, rescale_factor):
    total_images = 0
    correct_predictions = 0

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Only process image files
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            processed_image = preprocess_image(file_path, target_size, rescale_factor)

            if processed_image is not None:
                # Make a prediction
                prediction = model.predict(processed_image, verbose=0)
                predicted_label = 1 if prediction[0][0] > 0.5 else 0  # Male=1, Female=0

                # Print the prediction for debugging
                print(f"Prediction: {prediction[0][0]:.4f}, Predicted Label: {predicted_label}, File: {file_path}")

                # Female is labeled as 0, so we check against 0
                if predicted_label == 0:
                    correct_predictions += 1

                total_images += 1

    # Calculate accuracy
    accuracy = correct_predictions / total_images * 100 if total_images > 0 else 0
    print(f"Processed {total_images} images in {directory_path}. Accuracy: {accuracy:.2f}%")

# Run the test
process_female_test_set(test_directory_path_female, image_size, rescale)
