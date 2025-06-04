import glob
import os
from roboflow import Roboflow
from API import *

# Initialize Roboflow client
rf = Roboflow(api_key=api_key)

# Directory paths
images_dir = "Google_Images/images"
labels_dir = "Google_Images/labels"
classes_file = "Google_Images/classes.txt"

# Get the upload project
project = rf.workspace(workspace_id).project(project_id)

# Read class names for debugging
with open(classes_file, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]
print(f"Class mapping: {class_names}")

# Get all images
image_files = glob.glob(os.path.join(images_dir, "*.jpg"))

for image_path in image_files:
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]
    annotation_path = os.path.join(labels_dir, f"{base_name}.txt")

    if os.path.exists(annotation_path):
        print(f"Uploading {image_name} with label mapping...")
        try:
            result = project.single_upload(
                image_path=image_path,
                annotation_path=annotation_path,
                label_map_path=classes_file,  # This should map 0 -> vehicle
                is_prediction=False  # Set as ground truth annotations
            )
            print(f"✓ Success: {result}")
        except Exception as e:
            print(f"✗ Failed {image_name}: {e}")
    else:
        print(f"⚠ Missing annotation: {annotation_path}")