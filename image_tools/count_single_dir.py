import os

def count_images_in_directory(directory):
    """
    Count total images in a single directory.

    Args:
        directory (str): Path to the directory.

    Returns:
        int: Total number of images in the directory.
    """
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return 0

    return sum(
        1 for file in os.listdir(directory)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))
    )

# Directory to count images in
directory = "/Users/jordantaylor/PycharmProjects/gender-modeling/gender-master-dataset/demog_male_no_caucasian"

# Count images
image_count = count_images_in_directory(directory)

print(f"Total images in {directory}: {image_count}")
