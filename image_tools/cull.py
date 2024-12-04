import os
import random
from shutil import copyfile

# List of directories
directories = [
    "/Users/jordantaylor/PycharmProjects/gender-modeling/gender-training-dataset/female_augmented",
    "/Users/jordantaylor/PycharmProjects/gender-modeling/gender-training-dataset/male_augmented",
    "/Users/jordantaylor/PycharmProjects/gender-modeling/test-set-female",
    "/Users/jordantaylor/PycharmProjects/gender-modeling/test-set-male",
]

# Percentage of images to retain
retain_percentage = 60  # e.g., retain 50% of images


# Function to reduce images in a directory
def reduce_images_in_directory(directory, retain_percentage):
    # List all files in the directory
    all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    total_files = len(all_files)

    # Calculate the number of files to retain
    retain_count = int(total_files * (retain_percentage / 100))

    # Randomly select files to retain
    files_to_retain = random.sample(all_files, retain_count)

    # Remove files not in the retain list
    for file in all_files:
        if file not in files_to_retain:
            os.remove(os.path.join(directory, file))
    print(f"Reduced {directory}: Retained {retain_count}/{total_files} images.")


# Process each directory
for directory in directories:
    reduce_images_in_directory(directory, retain_percentage)
