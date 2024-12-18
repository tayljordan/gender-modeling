import os
import shutil

# List of directories to clear and recreate
directories = [
    "/Users/jordantaylor/PycharmProjects/gender-modeling/gender-training-dataset/female_augmented",
    "/Users/jordantaylor/PycharmProjects/gender-modeling/gender-training-dataset/male_augmented",
    "/Users/jordantaylor/PycharmProjects/gender-modeling/test-set-female",
    "/Users/jordantaylor/PycharmProjects/gender-modeling/test-set-male",
]

def clear_and_recreate_directory(directory):
    try:
        # Delete the directory if it exists
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"Deleted directory: {directory}")

        # Recreate the directory
        os.makedirs(directory)
        print(f"Recreated directory: {directory}")
    except Exception as e:
        print(f"Error processing directory {directory}: {e}")

# Process each directory
for directory in directories:
    clear_and_recreate_directory(directory)
