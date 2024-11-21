import os

# Specify the directory path
directory = "/Users/jordantaylor/Desktop/gender-modeling/false_negatives"


# Delete
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    try:
        if os.path.isfile(file_path):  # Check if it is a file
            os.unlink(file_path)  # Delete the file
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")

print("All files deleted.")


import os
# Female: 2698
# Male: 2720
directory = '/Users/jordantaylor/Desktop/gender-modeling/gender-training-dataset/male_augmented'
num_files = len([file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))])

print(f"Number of files: {num_files}")


# import os
# import shutil
#
# # Source and target directories
# source_dir = "/Users/jordantaylor/Desktop/gender-modeling/gender-training-dataset/demog_male"
# target_dir = "/Users/jordantaylor/Desktop/gender-modeling/gender-training-dataset/male"
#
# # Ensure the target directory exists
# os.makedirs(target_dir, exist_ok=True)
#
# # Move files from the source directory to the target directory
# for file_name in os.listdir(source_dir):
#     file_path = os.path.join(source_dir, file_name)
#     target_path = os.path.join(target_dir, file_name)
#
#     # Check if it's a file before moving
#     if os.path.isfile(file_path):
#         shutil.move(file_path, target_path)
#
# print(f"All files from {source_dir} have been moved to {target_dir}.")
