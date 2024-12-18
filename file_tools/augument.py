import albumentations as A
import cv2
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Define the augmentations
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


def augment_and_save_image(filename, input_dir, output_dir, augmentation, num_augmentations=3):
    img_path = os.path.join(input_dir, filename)
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return  # Skip invalid images

    for i in range(num_augmentations):
        augmented = augmentation(image=image)
        augmented_image = augmented['image']

        # Save augmented image
        output_path = os.path.join(output_dir, f"{filename.split('.')[0]}_aug_{i + 1}.jpg")
        cv2.imwrite(output_path, augmented_image)


def augment_and_save_images_parallel(input_dir, output_dir, augmentation, num_augmentations=3, max_workers=8):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.endswith(".jpg") or f.endswith(".png")]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                augment_and_save_image,
                filename, input_dir, output_dir, augmentation, num_augmentations
            )
            for filename in files
        ]
        for _ in tqdm(as_completed(futures), total=len(files), desc="Augmenting Images"):
            pass


# Example usage
input_directory = "/Users/jordantaylor/PycharmProjects/gender-modeling/gender-training-dataset/male_augmented"
output_directory = "/Users/jordantaylor/PycharmProjects/gender-modeling/gender-training-dataset/male_augmented"
augment_and_save_images_parallel(input_directory, output_directory, augmentation, num_augmentations=10, max_workers=8)

input_directory = "/Users/jordantaylor/PycharmProjects/gender-modeling/gender-training-dataset/female_augmented"
output_directory = "/Users/jordantaylor/PycharmProjects/gender-modeling/gender-training-dataset/female_augmented"
augment_and_save_images_parallel(input_directory, output_directory, augmentation, num_augmentations=10, max_workers=8)
