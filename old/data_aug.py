import os
import random
import json
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A

# Configuration
# Path to your downloaded Roboflow dataset (COCO format)
INPUT_DATASET_PATH = "C:/NTNU/Custom_Obj_Det/My-First-Project-1"
OUTPUT_DIR = "vehicle_dataset_split"

# Number of augmentations per image
NUM_AUGMENTATIONS = 15  # Increase this for more augmented images

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

# Path to annotations file
annotation_file = os.path.join(INPUT_DATASET_PATH, "train", "_annotations.coco.json")

# Read annotation file
print(f"Reading annotations from {annotation_file}")
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# Get all images
all_images = annotations['images']

# Find which images have annotations
images_with_annotations_ids = set()
image_id_to_annotations = {}

for ann in annotations['annotations']:
    img_id = ann['image_id']
    images_with_annotations_ids.add(img_id)
    if img_id not in image_id_to_annotations:
        image_id_to_annotations[img_id] = []
    image_id_to_annotations[img_id].append(ann)

# Create lists of images with and without annotations
images_with_annotations = [img for img in all_images if img['id'] in images_with_annotations_ids]
images_without_annotations = [img for img in all_images if img['id'] not in images_with_annotations_ids]

print(f"Total images: {len(all_images)}")
print(f"Images with annotations: {len(images_with_annotations)}")
print(f"Images without annotations: {len(images_without_annotations)}")

# Shuffle the images with annotations for random selection
random.shuffle(images_with_annotations)

# Select images for each split
test_images = images_with_annotations[:10]
validation_images = images_with_annotations[10:30]
training_annotated_images = images_with_annotations[30:]

# All images without annotations go to training
training_images = training_annotated_images + images_without_annotations

# Map image IDs to splits for easy lookup
image_id_to_split = {}
for img in test_images:
    image_id_to_split[img['id']] = 'test'
for img in validation_images:
    image_id_to_split[img['id']] = 'val'
for img in training_images:
    image_id_to_split[img['id']] = 'train'

# Keep track of processed annotations for each split
train_annotations = []
val_annotations = []
test_annotations = []

# Define multiple augmentation pipelines for variety
# Pipeline 1: Basic transformations (safer)
augmentation_1 = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.GaussNoise(p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.5),
    A.Blur(blur_limit=3, p=0.3),
    A.Rotate(limit=30, p=0.7),
    A.RandomScale(scale_limit=0.15, p=0.5),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

# Pipeline 2: More aggressive transformations
augmentation_2 = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=30, p=0.5),
    A.Blur(blur_limit=5, p=0.3),
    A.Rotate(limit=45, p=0.7),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.Perspective(p=0.3),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

# Pipeline 3: Color transforms focused
augmentation_3 = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
    A.GaussNoise(p=0.3),
    A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=40, p=0.7),
    A.CLAHE(clip_limit=4.0, p=0.7),
    A.ChannelShuffle(p=0.3),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

# Pipeline 4: Geometric transforms focused
augmentation_4 = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.7,
                       border_mode=cv2.BORDER_CONSTANT),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5,
                       border_mode=cv2.BORDER_CONSTANT),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5,
                     border_mode=cv2.BORDER_CONSTANT),
    A.OpticalDistortion(distort_limit=0.2, shift_limit=0.15, p=0.5,
                        border_mode=cv2.BORDER_CONSTANT),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

# Copy images and annotations to their respective directories
print("\nCopying and organizing dataset...")
for img in tqdm(all_images):
    img_id = img['id']
    img_filename = img['file_name']
    img_path = os.path.join(INPUT_DATASET_PATH, "train", img_filename)

    # Determine the split this image belongs to
    split = image_id_to_split[img_id]

    # Create the destination path
    dest_img_path = os.path.join(OUTPUT_DIR, split, "images", img_filename)

    # Copy the image
    shutil.copy(img_path, dest_img_path)

    # Process annotations if this image has any
    if img_id in images_with_annotations_ids:
        # Get annotations for this image
        img_anns = image_id_to_annotations[img_id]

        # Generate YOLO format labels
        img_width = img['width']
        img_height = img['height']

        # Create a label file in YOLO format
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(OUTPUT_DIR, split, "labels", label_filename)

        with open(label_path, 'w') as f:
            for ann in img_anns:
                # COCO format: [x, y, width, height]
                x, y, w, h = ann['bbox']

                # Convert to YOLO format: [class_id, x_center, y_center, width, height] (normalized)
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height

                # Write to file (class id is 0 for "vehicle")
                f.write(f"0 {x_center} {y_center} {width} {height}\n")

        # Track annotations by split
        if split == 'train':
            train_annotations.extend(img_anns)
        elif split == 'val':
            val_annotations.extend(img_anns)
        else:  # test
            test_annotations.extend(img_anns)

# Apply augmentations to training images with annotations
if training_annotated_images:
    print("\nApplying augmentations to training images...")
    successful_augmentations = 0

    for img in tqdm(training_annotated_images):
        img_id = img['id']
        img_filename = img['file_name']
        img_path = os.path.join(OUTPUT_DIR, 'train', 'images', img_filename)

        # Read the image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get annotations for this image in COCO format
        img_anns = image_id_to_annotations[img_id]
        bboxes = [ann['bbox'] for ann in img_anns]
        category_ids = [0] * len(bboxes)  # All are "vehicle" class

        # Skip if no valid bounding boxes
        if len(bboxes) == 0:
            continue

        # Apply different augmentation pipelines
        for aug_idx in range(NUM_AUGMENTATIONS):
            try:
                # Choose a random augmentation pipeline
                pipeline_choice = random.randint(1, 4)

                if pipeline_choice == 1:
                    augmented = augmentation_1(image=image, bboxes=bboxes, category_ids=category_ids)
                elif pipeline_choice == 2:
                    augmented = augmentation_2(image=image, bboxes=bboxes, category_ids=category_ids)
                elif pipeline_choice == 3:
                    augmented = augmentation_3(image=image, bboxes=bboxes, category_ids=category_ids)
                else:
                    augmented = augmentation_4(image=image, bboxes=bboxes, category_ids=category_ids)

                aug_image = augmented['image']
                aug_bboxes = augmented['bboxes']

                # Skip if augmentation resulted in no valid bounding boxes
                if len(aug_bboxes) == 0:
                    continue

                # Create new filename for augmented image
                base_name, ext = os.path.splitext(img_filename)
                aug_img_filename = f"{base_name}_aug{aug_idx}{ext}"
                aug_label_filename = f"{base_name}_aug{aug_idx}.txt"

                # Save augmented image
                aug_img_path = os.path.join(OUTPUT_DIR, 'train', 'images', aug_img_filename)
                aug_img_rgb = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(aug_img_path, aug_img_rgb)

                # Save augmented labels in YOLO format
                aug_label_path = os.path.join(OUTPUT_DIR, 'train', 'labels', aug_label_filename)
                with open(aug_label_path, 'w') as f:
                    for bbox in aug_bboxes:
                        x, y, w, h = bbox

                        # Convert to YOLO format
                        x_center = (x + w / 2) / img['width']
                        y_center = (y + h / 2) / img['height']
                        width = w / img['width']
                        height = h / img['height']

                        f.write(f"0 {x_center} {y_center} {width} {height}\n")

                successful_augmentations += 1

            except Exception as e:
                print(f"Warning: Failed to augment image {img_filename}: {str(e)}")
                continue

# Create dataset.yaml file for YOLOv5/YOLOv8 compatibility
yaml_content = f"""
path: {os.path.abspath(OUTPUT_DIR)}
train: train/images
val: val/images
test: test/images

nc: 1
names: ['vehicle']
"""

with open(os.path.join(OUTPUT_DIR, 'dataset.yaml'), 'w') as f:
    f.write(yaml_content)

print("\nDataset organization complete!")
print(f"Dataset saved to: {os.path.abspath(OUTPUT_DIR)}")
print("\nSummary of the split:")
print(f"Train: {len(training_images)} original images ({len(training_annotated_images)} with annotations)")
print(f"    + Approximately {len(training_annotated_images) * NUM_AUGMENTATIONS} augmented images")
print(f"Validation: {len(validation_images)} images (all with annotations)")
print(f"Test: {len(test_images)} images (all with annotations)")
print("\nThe dataset is organized in YOLO format with a dataset.yaml file for easy use with YOLOv5/YOLOv8.")