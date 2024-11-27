import os
import random
import shutil

# Paths
master_dataset_dir = "/Users/jordantaylor/PycharmProjects/gender-modeling/gender-master-dataset"
train_val_female_dir = "/Users/jordantaylor/PycharmProjects/gender-modeling/gender-training-dataset/female_augmented"
train_val_male_dir = "/Users/jordantaylor/PycharmProjects/gender-modeling/gender-training-dataset/male_augmented"
test_female_dir = "/Users/jordantaylor/PycharmProjects/gender-modeling/test-set-female"
test_male_dir = "/Users/jordantaylor/PycharmProjects/gender-modeling/test-set-male"

# Split ratio
test_ratio = 0.2  # 20% for testing

# Function to copy equal numbers of images for both categories
def copy_images_balanced(master_dir, train_val_female, train_val_male, test_female, test_male, num_images, test_ratio):
    # Ensure target directories exist
    os.makedirs(train_val_female, exist_ok=True)
    os.makedirs(train_val_male, exist_ok=True)
    os.makedirs(test_female, exist_ok=True)
    os.makedirs(test_male, exist_ok=True)

    # Collect all female and male images
    female_images = []
    male_images = []
    for subdir in os.listdir(master_dir):
        subdir_path = os.path.join(master_dir, subdir)
        if os.path.isdir(subdir_path):
            images = [os.path.join(subdir_path, file) for file in os.listdir(subdir_path)
                      if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))]
            if "female" in subdir.lower():
                female_images.extend(images)
            elif "male" in subdir.lower():
                male_images.extend(images)

    # Compute the limit for equal distribution
    min_images = min(len(female_images), len(male_images), num_images)
    female_images = random.sample(female_images, min_images)
    male_images = random.sample(male_images, min_images)

    # Split into train/validation and test sets
    split_point = int(min_images * (1 - test_ratio))
    train_val_female_images = female_images[:split_point]
    test_female_images = female_images[split_point:]
    train_val_male_images = male_images[:split_point]
    test_male_images = male_images[split_point:]

    # Copy images to their respective directories
    for img in train_val_female_images:
        shutil.copy(img, train_val_female)
    for img in test_female_images:
        shutil.copy(img, test_female)
    for img in train_val_male_images:
        shutil.copy(img, train_val_male)
    for img in test_male_images:
        shutil.copy(img, test_male)

    # Print summary
    print(f"Copied {len(train_val_female_images)} images to {train_val_female}")
    print(f"Copied {len(test_female_images)} images to {test_female}")
    print(f"Copied {len(train_val_male_images)} images to {train_val_male}")
    print(f"Copied {len(test_male_images)} images to {test_male}")

# Example usage
num_images_to_pull = 2000  # Specify the maximum number of images to pull
copy_images_balanced(
    master_dir=master_dataset_dir,
    train_val_female=train_val_female_dir,
    train_val_male=train_val_male_dir,
    test_female=test_female_dir,
    test_male=test_male_dir,
    num_images=num_images_to_pull,
    test_ratio=test_ratio
)
