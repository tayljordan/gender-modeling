
import os
import psutil
from PIL import Image
from tqdm import tqdm
import time


def resize_images_with_report(paths, target_size=(224, 224), output_base_dir="/Users/jordantaylor/PycharmProjects/gender-modeling/gender-master-dataset", report_path="resize_report.txt"):
    """
    Resize images in specified directories while tracking memory usage, progress, and errors.

    Args:
        paths (list): List of directories containing images.
        target_size (tuple): Target resolution (width, height).
        output_base_dir (str): Base directory to save resized images.
        report_path (str): Path to save the summary report.
    """
    start_time = time.time()
    total_files = 0
    resized_count = 0
    skipped_count = 0
    error_count = 0
    memory_usage = []
    errors = []

    try:
        for path in paths:
            if not os.path.exists(path):
                print(f"Directory not found: {path}")
                continue

            # Maintain subdirectory structure
            sub_dir = os.path.basename(path)
            output_dir = os.path.join(output_base_dir, sub_dir)
            os.makedirs(output_dir, exist_ok=True)

            files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))]
            total_files += len(files)

            for filename in tqdm(files, desc=f"Processing {path}", unit="file"):
                img_path = os.path.join(path, filename)
                try:
                    with Image.open(img_path) as img:
                        # Skip images with resolution < 224x224
                        if img.size[0] < 224 or img.size[1] < 224:
                            skipped_count += 1
                            continue

                        # Resize image
                        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                        img_resized.save(os.path.join(output_dir, filename))
                        resized_count += 1

                except Exception as e:
                    error_count += 1
                    errors.append((filename, str(e)))

                # Track memory usage
                memory_info = psutil.virtual_memory()
                memory_usage.append(memory_info.percent)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    finally:
        # Write report
        end_time = time.time()
        duration = end_time - start_time
        avg_memory_usage = sum(memory_usage) / len(memory_usage) if memory_usage else 0

        with open(report_path, "w") as report:
            report.write(f"Total images processed: {total_files}\n")
            report.write(f"Images resized: {resized_count}\n")
            report.write(f"Images skipped (low resolution): {skipped_count}\n")
            report.write(f"Errors: {error_count}\n")
            report.write(f"Average memory usage: {avg_memory_usage:.2f}%\n")
            report.write(f"Total time taken: {duration:.2f} seconds\n")
            if errors:
                report.write("\nErrors:\n")
                for filename, error in errors:
                    report.write(f"{filename}: {error}\n")

        print(f"\nReport saved at: {report_path}")


paths = [
    "/Users/jordantaylor/Desktop/female_maritime_curated_399",
    "/Users/jordantaylor/Desktop/female_maritime_curated_270"
]

resize_images_with_report(paths, target_size=(224, 224), report_path="resize_report.txt")
