import os

# Specify the directory path
directory = "/Users/jordantaylor/PycharmProjects/gender-modeling/gender-dataset/11Nov24_filtered"


# Delete all files in the directory
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    try:
        if os.path.isfile(file_path):  # Check if it is a file
            os.unlink(file_path)  # Delete the file
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")

print("All files deleted.")
