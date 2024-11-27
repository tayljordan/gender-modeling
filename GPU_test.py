import os
import yaml
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load configuration
config_path = os.path.join(os.getcwd(), "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

params = config['parameters']
rescale = params['rescale']  # Rescaling factor
test_directory_path_male = os.path.join(os.getcwd(), config['directories']['test_set_male'])
test_directory_path_female = os.path.join(os.getcwd(), config['directories']['test_set_female'])
model_path = os.path.join(os.getcwd(), config['model']['path'])

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs are available: {len(gpus)} GPU(s) detected.")
else:
    print("No GPUs detected. Using CPU.")

# Load the trained model
model = load_model(model_path)

# Retrieve the model's expected input size
model_input_size = model.input_shape[1:3]  # Height and width
print(f"Model expected input size: {model_input_size}")

# Preprocessing function
def preprocess_image(image_path, rescale_factor):
    try:
        # Resize image to match model input size
        img = load_img(image_path, target_size=model_input_size)
        img_array = img_to_array(img) * rescale_factor  # Normalize pixel values
        return np.expand_dims(img_array, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Process images in a directory
def process_directory(directory_path, actual_label, rescale_factor):
    correct_predictions = 0
    total_images = 0
    debug_info = []

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            processed_image = preprocess_image(file_path, rescale_factor)
            if processed_image is not None:
                prediction = model.predict(processed_image, verbose=0)
                predicted_label = 1 if prediction[0][0] > 0.5 else 0  # Male=1, Female=0
                debug_info.append((filename, prediction[0][0], predicted_label, actual_label))
                if predicted_label == actual_label:
                    correct_predictions += 1
                total_images += 1

    # Print debugging info
    print(f"Debugging Predictions for {directory_path}:")
    for info in debug_info[:10]:  # Limit to first 10 for readability
        print(f"File: {info[0]}, Predicted Score: {info[1]:.4f}, Predicted Label: {info[2]}, Actual Label: {info[3]}")

    accuracy = correct_predictions / total_images * 100 if total_images > 0 else 0
    print(f"Processed {total_images} images in {directory_path}. Accuracy: {accuracy:.2f}%")
    return accuracy

# Process test sets
male_accuracy = process_directory(test_directory_path_male, actual_label=1, rescale_factor=rescale)
female_accuracy = process_directory(test_directory_path_female, actual_label=0, rescale_factor=rescale)

# Calculate average accuracy
average_accuracy = (male_accuracy + female_accuracy) / 2
print(f"Male Test Set Accuracy: {male_accuracy:.2f}%")
print(f"Female Test Set Accuracy: {female_accuracy:.2f}%")
print(f"Average Accuracy: {average_accuracy:.2f}%")
