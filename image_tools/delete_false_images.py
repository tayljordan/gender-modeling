import os

# Directories
female_false_dir = "/Users/jordantaylor/PycharmProjects/gender-modeling/false-images/female_false"
male_false_dir = "/Users/jordantaylor/PycharmProjects/gender-modeling/false-images/male_false"

def delete_all_files(directory):
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

        else:
            pass

# Delete files in both directories
delete_all_files(female_false_dir)
delete_all_files(male_false_dir)

print("All files deleted.")
