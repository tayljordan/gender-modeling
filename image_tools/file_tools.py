import os

directory = '/Users/jordantaylor/PycharmProjects/gender-modeling/test-set-female'
num_files = len([file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))])

print(f"Number of files: {num_files}")

import os

directory = '/Users/jordantaylor/PycharmProjects/gender-modeling/test-set-male'
num_files = len([file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))])

print(f"Number of files: {num_files}")