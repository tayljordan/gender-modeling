import shutil
import os

def copy_images(src_dir, dest_dir):
    """
    Copy all images from the source directory to the destination directory.

    Args:
        src_dir (str): Source directory path.
        dest_dir (str): Destination directory path.
    """
    if not os.path.exists(src_dir):
        print(f"Source directory not found: {src_dir}")
        return

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for file_name in os.listdir(src_dir):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
            src_path = os.path.join(src_dir, file_name)
            dest_path = os.path.join(dest_dir, file_name)
            shutil.copy(src_path, dest_path)

    print(f"Copied images from {src_dir} to {dest_dir}")

# Directories
female_false_dir = '/Users/jordantaylor/PycharmProjects/gender-modeling/false-images/female_false'
female_augmented_dir = '/Users/jordantaylor/PycharmProjects/gender-modeling/gender-training-dataset/female_augmented'
male_false_dir = '/Users/jordantaylor/PycharmProjects/gender-modeling/false-images/male_false'
male_augmented_dir = '/Users/jordantaylor/PycharmProjects/gender-modeling/gender-training-dataset/male_augmented'

# Copy female false images to female augmented
copy_images(female_false_dir, female_augmented_dir)

# Copy male false images to male augmented
copy_images(male_false_dir, male_augmented_dir)

# Verify the results
directories = [
    female_augmented_dir,
    male_augmented_dir,
    female_false_dir,
    male_false_dir
]

def count_images(directory):
    return sum(
        1 for file in os.listdir(directory)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))
    )

print("\nFinal Image Counts:")
for directory in directories:
    if os.path.exists(directory):
        print(f"{directory}: {count_images(directory)} images")
    else:
        print(f"{directory}: Directory not found")
