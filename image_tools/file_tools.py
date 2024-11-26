# import os
# import shutil
# import random
#
# def ensure_target_images(source_dir, destination_dir, target_count):
#     # Ensure destination directory exists
#     os.makedirs(destination_dir, exist_ok=True)
#
#     # Get current number of files in the destination directory
#     current_files = [file for file in os.listdir(destination_dir) if os.path.isfile(os.path.join(destination_dir, file))]
#     current_count = len(current_files)
#
#     # Check the number of files needed
#     files_needed = target_count - current_count
#
#     if files_needed > 0:
#         # Get all files from the source directory
#         source_files = [file for file in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, file))]
#
#         # Randomly select files
#         selected_files = random.choices(source_files, k=files_needed)
#
#         # Copy the selected files to the destination directory
#         for file_name in selected_files:
#             source_file = os.path.join(source_dir, file_name)
#             destination_file = os.path.join(destination_dir, file_name)
#
#             # Rename file if it already exists
#             if os.path.exists(destination_file):
#                 base, ext = os.path.splitext(file_name)
#                 destination_file = os.path.join(destination_dir, f"{base}_{random.randint(1000, 9999)}{ext}")
#
#             shutil.copy(source_file, destination_file)
#
#         print(f"Added {files_needed} files to {destination_dir}. Total files: {target_count}.")
#     else:
#         print(f"Destination already has {current_count} files. No files were added.")
#
# # Directories for male images
# male_source = '/Users/jordantaylor/PycharmProjects/gender-modeling/gender-training-dataset/male'
# male_destination = '/Users/jordantaylor/PycharmProjects/gender-modeling/gender-training-dataset/male_augmented'
#
# # Ensure 50,000 images in the destination directory
# ensure_target_images(male_source, male_destination, 50000)
#
# # Verify the number of files in the destination directory
# directory = male_destination
# num_files = len([file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))])
#
# print(f"Number of files: {num_files}")



















import os

directory = '/Users/jordantaylor/PycharmProjects/gender-modeling/gender-training-dataset/male_augmented'
num_files = len([file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))])

print(f"Number of files: {num_files}")




import os

directory = '/Users/jordantaylor/PycharmProjects/gender-modeling/gender-training-dataset/female_augmented'
num_files = len([file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))])

print(f"Number of files: {num_files}")
