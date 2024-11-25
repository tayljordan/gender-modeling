import os
import cv2
import numpy as np
import insightface

# Define directories
source_dir = "/Users/jordantaylor/PycharmProjects/gender-modeling/test-set"




target_dir = "/test-set-female"

# Create target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Load the InsightFace model
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=-1)  # Use -1 for CPU; 0 for GPU

# Set parameters
margin_ratio = 0.4  # 40% margin for bounding box

# Counter for saved faces
faces_saved = 0

# Iterate over all images in the source directory
for filename in os.listdir(source_dir):
    file_path = os.path.join(source_dir, filename)

    # Skip non-image files
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    try:
        # Load the image
        img = cv2.imread(file_path)
        if img is None:
            print(f"Failed to load image: {file_path}")
            continue

        # Convert image to RGB (InsightFace expects RGB format)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = model.get(img_rgb)
        if not faces:
            print(f"No faces detected in {file_path}")
            continue

        # Process each detected face
        for i, face in enumerate(faces):
            try:
                bbox = face.bbox.astype(int)

                # Calculate the margin for the bounding box
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                x_margin = int(width * margin_ratio)
                y_margin = int(height * margin_ratio)

                # Expand the bounding box
                expanded_bbox = [
                    max(0, bbox[0] - x_margin),  # Left
                    max(0, bbox[1] - y_margin),  # Top
                    min(img.shape[1], bbox[2] + x_margin),  # Right
                    min(img.shape[0], bbox[3] + y_margin)   # Bottom
                ]

                # Crop the image to the expanded bounding box
                cropped_face = img[expanded_bbox[1]:expanded_bbox[3], expanded_bbox[0]:expanded_bbox[2]]

                # Save the cropped face image
                output_filename = f"{os.path.splitext(filename)[0]}_face_{i + 1}.jpg"
                output_path = os.path.join(target_dir, output_filename)
                cv2.imwrite(output_path, cropped_face)

                # Increment counter and print update
                faces_saved += 1
                print(f"Saved: {output_path}")
            except Exception as face_error:
                print(f"Error processing face in {file_path}: {face_error}")

    except Exception as img_error:
        print(f"Error processing image {file_path}: {img_error}")

# Print final summary
print(f"Processing complete. Total faces saved: {faces_saved}")
