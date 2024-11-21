import albumentations as A
import cv2
import os

# Define the augmentations with random probabilities, excluding bbox-specific transformations
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.Perspective(scale=(0.05, 0.1), p=0.5),
    A.MotionBlur(blur_limit=5, p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.CLAHE(clip_limit=4.0, p=0.3),
    A.CoarseDropout(max_holes=8, max_height=10, max_width=10, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
    A.Resize(224, 224),
])


def augment_and_save_images(input_dir, output_dir, augmentation, num_augmentations=3):
    """
    Apply augmentations to all images in a directory and save the augmented images.

    Args:
    input_dir (str): Directory containing input images.
    output_dir (str): Directory to save augmented images.
    augmentation (albumentations.Compose): Augmentation pipeline.
    num_augmentations (int): Number of augmented images per original image.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more formats if needed
            img_path = os.path.join(input_dir, filename)
            image = cv2.imread(img_path)

            # Apply augmentations multiple times, with each augmentation applied randomly
            for i in range(num_augmentations):
                augmented = augmentation(image=image)
                augmented_image = augmented['image']

                # Save augmented image
                output_path = os.path.join(output_dir, f"{filename.split('.')[0]}_aug_{i + 1}.jpg")
                cv2.imwrite(output_path, augmented_image)
                print(f"Saved augmented image {output_path}")


# Example usage
input_directory = "./input"
output_directory = "./output"
augment_and_save_images(input_directory, output_directory, augmentation, num_augmentations=10)
