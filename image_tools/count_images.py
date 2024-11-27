# gender-master-dataset

import os

def count_images_by_category(base_dir):
    """
    Count total images in each subdirectory and categorize as male, female, or total.

    Args:
        base_dir (str): Path to the base directory containing subdirectories.

    Returns:
        dict: Counts for males, females, and total images.
    """
    total_males = 0
    total_females = 0
    total_images = 0

    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):

            # Count images in this subdirectory
            image_count = sum(
                1 for file in os.listdir(subdir_path)
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))
            )

            # Check for "female" or "male" in the subdirectory name
            if "female" in subdir.lower():  # Check "female" first

                total_females += image_count
            elif "male" in subdir.lower():  # Check "male" afterward

                total_males += image_count
            total_images += image_count

    return {
        "total_males": total_males,
        "total_females": total_females,
        "total_images": total_images
    }


# Example usage
base_directory = "/Users/jordantaylor/PycharmProjects/gender-modeling/gender-master-dataset"
counts = count_images_by_category(base_directory)

print(f"Total males: {counts['total_males']}")
print(f"Total females: {counts['total_females']}")
print(f"Total images: {counts['total_images']}")




import os

def count_images_in_directories(directories):
    """
    Count total images in each specified directory.

    Args:
        directories (list): List of directory paths to count images.

    Returns:
        dict: Dictionary with directory names as keys and image counts as values.
    """
    image_counts = {}
    for directory in directories:
        if os.path.exists(directory):
            count = sum(
                1 for file in os.listdir(directory)
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))
            )
            image_counts[directory] = count
        else:
            image_counts[directory] = "Directory not found"
    return image_counts


# Directories to count images in
directories = [
    "/Users/jordantaylor/PycharmProjects/gender-modeling/gender-training-dataset/female_augmented",
    "/Users/jordantaylor/PycharmProjects/gender-modeling/gender-training-dataset/male_augmented",
    "/Users/jordantaylor/PycharmProjects/gender-modeling/test-set-female",
    "/Users/jordantaylor/PycharmProjects/gender-modeling/test-set-male",
]

# Count images in the specified directories
image_counts = count_images_in_directories(directories)

# Print results
for directory, count in image_counts.items():
    print(f"{directory}: {count} images")

