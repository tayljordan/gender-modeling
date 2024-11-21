import os

# Specify the directory path
directory = "/Users/jordantaylor/PycharmProjects/gender-modeling/gender-training-dataset/11Nov24_filtered"


# Delete all files in the directory
# for filename in os.listdir(directory):
#     file_path = os.path.join(directory, filename)
#     try:
#         if os.path.isfile(file_path):  # Check if it is a file
#             os.unlink(file_path)  # Delete the file
#     except Exception as e:
#         print(f"Error deleting file {file_path}: {e}")
#
# print("All files deleted.")


import os
# Female: 2698
# Male: 2720
directory = '/Users/jordantaylor/Desktop/gender-modeling/gender-training-dataset/male'
num_files = len([file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))])

print(f"Number of files: {num_files}")
