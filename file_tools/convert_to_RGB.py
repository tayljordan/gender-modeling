# import os
# from PIL import Image
#
# # Base directory containing the dataset
# base_dir = "/Users/jordantaylor/PycharmProjects/gender-modeling/gender-master-dataset"
#
# # Function to process images recursively
# def convert_to_rgb(directory):
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             file_path = os.path.join(root, file)
#             try:
#                 # Open the image
#                 with Image.open(file_path) as img:
#                     # Check and convert to RGB
#                     if img.mode != 'RGB':
#                         print(f"Converting {file_path} to RGB.")
#                         img = img.convert('RGB')
#                         img.save(file_path)  # Overwrite the original file
#             except Exception as e:
#                 print(f"Failed to process {file_path}: {e}")
#
# # Process the base directory
# convert_to_rgb(base_dir)

import os
from PIL import Image

# Base directory containing the dataset
base_dir = "/Users/jordantaylor/PycharmProjects/gender-modeling/gender-master-dataset"

# Function to verify all images are in RGB mode
def verify_images(directory):
    all_rgb = True  # Flag to track if all images are RGB
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Open the image
                with Image.open(file_path) as img:
                    if img.mode != 'RGB':
                        print(f"Non-RGB image found: {file_path} (Mode: {img.mode})")
                        all_rgb = False
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
    if all_rgb:
        print("All images are in RGB mode.")
    else:
        print("Some images are not in RGB mode.")

# Run the verification
verify_images(base_dir)
