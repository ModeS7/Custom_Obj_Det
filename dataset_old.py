"""
Clean Dataset Creation with Random Crop Augmentation Geometric Transforms

METHODOLOGY:
1. Crop all images into 640x640 tiles
2. Random split: 42 val, 21 test, rest train
3. Augment train data 3x with geometric augmentations from random_crop_augmentation.py

GEOMETRIC AUGMENTATIONS REPLACED:
- Mirror (flip): 50% chance of either horizontal OR vertical flip (mutually exclusive)
- Rotation: 100% chance, random 0-360 degrees
- Shear: 100% chance, both directions, normal distribution (μ=0°, σ=8°), clipped to ±15°
- Crop/Zoom: 0-14% using normal distribution

COLOR/EFFECT AUGMENTATIONS (unchanged):
- Grayscale: Apply to 22% of images
- Hue: Between -34° and +34°
- Saturation: Between -34% and +34%
- Exposure: Between -15% and +15%
- Blur: Up to 3.1px
- Noise: Up to 1.64% of pixels
- Brightness: Between -10% and +10% (conservative)
"""

import os
import glob
import random
import numpy as np
import cv2
from PIL import Image
import albumentations as A
import json
import shutil
from tqdm import tqdm

from augmentations import *

# Configuration
CROP_SIZE = 640
VAL_COUNT = 42
TEST_COUNT = 21
AUGMENTATION_MULTIPLIER = 1  # 3x training data like Roboflow
MIN_ANNOTATION_VISIBILITY = 0.3


def load_annotations(label_path):
    """Load YOLO annotations from label file"""
    if not label_path or not os.path.exists(label_path):
        return []

    annotations = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        annotations.append([class_id, x_center, y_center, width, height])
    except Exception as e:
        print(f"Error reading annotations from {label_path}: {e}")

    return annotations


def find_all_images(images_folder):
    """Find all images that have corresponding label files"""
    print("Finding images with label files...")

    labels_folder = os.path.join(os.path.dirname(images_folder), "labels")
    if not os.path.exists(labels_folder):
        raise FileNotFoundError(f"Labels folder not found: {labels_folder}")

    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []

    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(images_folder, ext)))
        image_files.extend(glob.glob(os.path.join(images_folder, ext.upper())))

    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))

    # Only keep images that have corresponding label files
    valid_images = []
    for image_path in image_files:
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]
        label_path = os.path.join(labels_folder, f"{base_name}.txt")

        if os.path.exists(label_path):
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    valid_images.append({
                        'path': image_path,
                        'label_path': label_path,
                        'name': image_name,
                        'base_name': base_name,
                        'width': width,
                        'height': height
                    })
            except Exception as e:
                print(f"Error reading {image_path}: {e}")

    print(f"Found {len(valid_images)} images with label files")
    return valid_images


def crop_image_systematically(image_info):
    """Crop image into 640x640 tiles systematically"""
    image = cv2.imread(image_info['path'])
    if image is None:
        return []

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    annotations = load_annotations(image_info['label_path'])

    h, w = image.shape[:2]
    crops = []

    # Calculate how many 640x640 crops fit
    cols = w // CROP_SIZE
    rows = h // CROP_SIZE

    crop_counter = 0
    for row in range(rows):
        for col in range(cols):
            y_start = row * CROP_SIZE
            x_start = col * CROP_SIZE
            y_end = y_start + CROP_SIZE
            x_end = x_start + CROP_SIZE

            # Extract 640x640 crop
            crop = image[y_start:y_end, x_start:x_end]

            # Transform annotations for this crop
            crop_annotations = transform_annotations_for_crop(
                annotations, x_start, y_start, CROP_SIZE, w, h
            )

            # Store crop
            crop_info = {
                'crop_id': f"{image_info['base_name']}_crop_{crop_counter:03d}",
                'image': crop,
                'annotations': crop_annotations,
                'source_image': image_info['name']
            }

            crops.append(crop_info)
            crop_counter += 1

    return crops


def transform_annotations_for_crop(annotations, crop_x, crop_y, crop_size, original_width, original_height):
    """Transform YOLO annotations for crop"""
    if not annotations:
        return []

    transformed = []

    for ann in annotations:
        class_id, x_center, y_center, width, height = ann

        # Convert to pixel coordinates
        x_center_px = x_center * original_width
        y_center_px = y_center * original_height
        width_px = width * original_width
        height_px = height * original_height

        # Calculate bounding box
        x1 = x_center_px - width_px / 2
        y1 = y_center_px - height_px / 2
        x2 = x_center_px + width_px / 2
        y2 = y_center_px + height_px / 2

        # Calculate intersection with crop
        crop_x1, crop_y1 = crop_x, crop_y
        crop_x2, crop_y2 = crop_x + crop_size, crop_y + crop_size

        intersect_x1 = max(x1, crop_x1)
        intersect_y1 = max(y1, crop_y1)
        intersect_x2 = min(x2, crop_x2)
        intersect_y2 = min(y2, crop_y2)

        if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
            # Calculate intersection area
            intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
            original_area = width_px * height_px

            # Keep annotation if >50% visible
            if intersect_area / original_area >= MIN_ANNOTATION_VISIBILITY:
                # Adjust to crop coordinates
                new_x_center = (intersect_x1 + intersect_x2) / 2 - crop_x
                new_y_center = (intersect_y1 + intersect_y2) / 2 - crop_y
                new_width = intersect_x2 - intersect_x1
                new_height = intersect_y2 - intersect_y1

                # Normalize to crop dimensions
                norm_x_center = new_x_center / crop_size
                norm_y_center = new_y_center / crop_size
                norm_width = new_width / crop_size
                norm_height = new_height / crop_size

                transformed.append([class_id, norm_x_center, norm_y_center, norm_width, norm_height])

    return transformed


def apply_random_distortion(image, annotations):
    """Apply one randomly selected distortion transform - provides variety without over-augmentation"""

    # Available distortion functions with their probability weights
    # Exclude ThinPlateSpline until parameters are verified
    distortion_functions = [
        apply_thinplate_spline_deformation, # Thin-plate spline deformation
        apply_grid_elastic_deform,  # Localized elastic deformations
        apply_grid_distortion,  # Non-uniform grid warping
        apply_elastic_transform,  # Tissue-like deformation
        apply_optical_distortion,  # Barrel/pincushion lens effects
    ]

    # Randomly select one distortion function
    selected_distortion = random.choice(distortion_functions)

    # Apply the selected distortion
    return selected_distortion(image, annotations)


def apply_augmentations(image, annotations):
    """Apply geometric and photometric augmentations in sequence:

    1. Mirror (flip) - 50% chance (exact transform, no quality loss)
    2. Rotate 100% - 0-360 degrees (on clean/flipped image)
    3. Random distortion - one of 4 distortion types (localized effects)
    4. Perspective - 100% chance (realistic camera angles)
    5. Crop/Zoom - 0-15% using normal distribution
    """

    # Step 1: Mirror (flip) with 50% chance - exact transform first
    augmented_image, augmented_annotations = image, annotations
    if random.random() < 0.5:
        augmented_image, augmented_annotations = apply_mirror_flip(
            augmented_image, augmented_annotations)

    # Step 2: Rotate 100% - random 0-360 degrees
    augmented_image, augmented_annotations = apply_full_rotation(
        augmented_image, augmented_annotations
    )

    # Step 3: Random distortion - one of 5 distortion types
    augmented_image, augmented_annotations = apply_random_distortion(
        augmented_image, augmented_annotations
    )

    # Step 4: Perspective - 100% chance
    augmented_image, augmented_annotations = apply_perspective_distortion(
        augmented_image, augmented_annotations
    )

    # Step 5: Crop/Zoom - 0-15% using normal distribution
    augmented_image, augmented_annotations = apply_crop_zoom_normal_dist(
        augmented_image, augmented_annotations
    )

    return augmented_image, augmented_annotations


def apply_roboflow_augmentation(crop_info):
    """
    Apply augmentation with geometric transforms from random_crop_augmentation.py
    """

    # Apply geometric augmentations first
    augmented_image, augmented_annotations = apply_augmentations(
        crop_info['image'], crop_info['annotations']
    )
    """
    # Build color/effect augmentation list
    augmentations = []

    # COLOR AUGMENTATIONS: Choose AT MOST ONE per image (mutual exclusivity)
    color_choice = random.random()
    color_applied = False

    if color_choice < 0.22:  # 22% chance (Exact Roboflow)
        augmentations.append(A.ToGray(p=1.0))
        color_applied = True
    elif not color_applied and color_choice < 0.35:  # 13% chance
        hue_shift = random.uniform(-34, 34)  # Exact Roboflow range
        augmentations.append(A.HueSaturationValue(
            hue_shift_limit=(hue_shift-0.1, hue_shift+0.1),
            sat_shift_limit=0,
            val_shift_limit=0,
            p=1.0
        ))
        color_applied = True
    elif not color_applied and color_choice < 0.48:  # 13% chance
        sat_shift = random.uniform(-34, 34)  # Exact Roboflow range
        augmentations.append(A.HueSaturationValue(
            hue_shift_limit=0,
            sat_shift_limit=(sat_shift-0.1, sat_shift+0.1),
            val_shift_limit=0,
            p=1.0
        ))
        color_applied = True
    elif not color_applied and color_choice < 0.60:  # 12% chance
        brightness_shift = random.uniform(-0.10, 0.10)  # Conservative range
        augmentations.append(A.RandomBrightnessContrast(
            brightness_limit=(brightness_shift-0.01, brightness_shift+0.01),
            contrast_limit=0,
            p=1.0
        ))
        color_applied = True
    elif not color_applied and color_choice < 0.68:  # 8% chance
        exposure_shift = random.uniform(-15, 15) / 100.0  # Exact Roboflow range
        augmentations.append(A.RandomBrightnessContrast(
            brightness_limit=(exposure_shift-0.01, exposure_shift+0.01),
            contrast_limit=0,
            p=1.0
        ))
        color_applied = True

    # EFFECTS: Choose AT MOST ONE per image
    effects_choice = random.random()

    if effects_choice < 0.15:  # 15% chance
        blur_limit = random.choice([3])  # Up to 3.1px -> use 3px
        augmentations.append(A.Blur(blur_limit=(blur_limit, blur_limit), p=1.0))
    elif effects_choice < 0.25:  # 10% chance
        # Up to 1.64% of pixels - convert to std_range
        noise_std = random.uniform(0.005, 0.02)  # Conservative interpretation
        augmentations.append(A.GaussNoise(std_range=(noise_std, noise_std), mean_range=(0, 0), p=1.0))

    # Apply color/effect augmentations if any
    if augmentations:
        pipeline = A.Compose(augmentations, bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_area=0,
            min_visibility=0.3
        ))

        if augmented_annotations:
            class_labels = [ann[0] for ann in augmented_annotations]
            bboxes = [[ann[1], ann[2], ann[3], ann[4]] for ann in augmented_annotations]

            try:
                result = pipeline(image=augmented_image, bboxes=bboxes, class_labels=class_labels)

                # Convert back to annotation format
                final_annotations = []
                for bbox, class_id in zip(result['bboxes'], result['class_labels']):
                    final_annotations.append([class_id, bbox[0], bbox[1], bbox[2], bbox[3]])

                return result['image'], final_annotations
            except Exception as e:
                print(f"Color/effect augmentation failed: {e}")
                return augmented_image, augmented_annotations
        else:
            try:
                result = pipeline(image=augmented_image, bboxes=[], class_labels=[])
                return result['image'], []
            except Exception as e:
                print(f"Color/effect augmentation failed: {e}")
                return augmented_image, []
    """
    return augmented_image, augmented_annotations


def save_crop(crop_info, output_dir, split_name, augment_suffix=""):
    """Save a crop to the appropriate directory"""
    images_dir = os.path.join(output_dir, split_name, "images")
    labels_dir = os.path.join(output_dir, split_name, "labels")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Generate filename
    filename = f"{crop_info['crop_id']}{augment_suffix}"

    # Save image
    image_path = os.path.join(images_dir, f"{filename}.jpg")
    image_bgr = cv2.cvtColor(crop_info['image'], cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, image_bgr)

    # Always create label file (empty if no annotations)
    label_path = os.path.join(labels_dir, f"{filename}.txt")
    with open(label_path, 'w') as f:
        if crop_info['annotations']:
            for ann in crop_info['annotations']:
                f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")


def create_dataset_yaml(output_dir):
    """Create dataset.yaml file"""
    yaml_content = f"""path: {os.path.abspath(output_dir)}
train: train/images
val: val/images
test: test/images

nc: 1
names:
  0: vehicle
"""

    with open(os.path.join(output_dir, "dataset.yaml"), 'w') as f:
        f.write(yaml_content)


def main():
    """Main execution function"""
    print("Clean Crop, Split & Random Crop Augmentation Geometric Transforms")
    print("=" * 65)
    print("Methodology:")
    print("1. Crop all images into 640x640 tiles")
    print(f"2. Random split: {VAL_COUNT} val, {TEST_COUNT} test, rest train")
    print(f"3. Augment train data {AUGMENTATION_MULTIPLIER}x with geometric augmentations from random_crop_augmentation.py")

    # Hardcoded paths
    input_folder = r"C:\NTNU\Custom_Obj_Det\datasets\vehicle_detection_improved.v1i.yolov12\train"
    output_folder = r"C:\NTNU\Custom_Obj_Det\datasets\dataset_geometric"

    if not os.path.exists(input_folder):
        print(f"Error: Input folder not found: {input_folder}")
        return

    images_folder = os.path.join(input_folder, "images")
    if not os.path.exists(images_folder):
        print(f"Error: Images folder not found: {images_folder}")
        return

    print(f"\nProcessing images from: {images_folder}")
    print(f"Output will be saved to: {output_folder}")

    try:
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        print("Random seed set to 42 for reproducibility")

        # Step 1: Find all images with label files
        valid_images = find_all_images(images_folder)
        if not valid_images:
            print("No images with label files found!")
            return

        # Step 2: Crop all images systematically
        print(f"\nCropping {len(valid_images)} images into {CROP_SIZE}x{CROP_SIZE} tiles...")
        all_crops = []

        for image_info in tqdm(valid_images, desc="Cropping images"):
            crops = crop_image_systematically(image_info)
            all_crops.extend(crops)

        print(f"Generated {len(all_crops)} total crops")

        if len(all_crops) < VAL_COUNT + TEST_COUNT:
            print(f"Error: Not enough crops ({len(all_crops)}) for val+test ({VAL_COUNT + TEST_COUNT})")
            return

        # Step 3: Random split
        print(f"\nRandomly splitting crops...")
        random.shuffle(all_crops)

        val_crops = all_crops[:VAL_COUNT]
        test_crops = all_crops[VAL_COUNT:VAL_COUNT + TEST_COUNT]
        train_crops = all_crops[VAL_COUNT + TEST_COUNT:]

        print(f"Split: {len(train_crops)} train, {len(val_crops)} val, {len(test_crops)} test")

        # Step 4: Save validation and test sets (no augmentation)
        print("\nSaving validation and test sets...")

        for crop in tqdm(val_crops, desc="Saving validation"):
            save_crop(crop, output_folder, "val")

        for crop in tqdm(test_crops, desc="Saving test"):
            save_crop(crop, output_folder, "test")

        # Step 5: Save training set with augmentation
        print(f"\nSaving training set with {AUGMENTATION_MULTIPLIER}x random crop augmentation geometric transforms...")

        # Save original training crops
        for crop in tqdm(train_crops, desc="Saving original training"):
            save_crop(crop, output_folder, "train")

        # Generate augmented versions (2 additional versions for 3x total)
        for aug_round in range(1, AUGMENTATION_MULTIPLIER):
            print(f"Generating augmentation round {aug_round}...")
            for i, crop in enumerate(tqdm(train_crops, desc=f"Augmenting round {aug_round}")):
                # Apply augmentation
                aug_image, aug_annotations = apply_roboflow_augmentation(crop)

                # Create augmented crop
                aug_crop = {
                    'crop_id': crop['crop_id'],
                    'image': aug_image,
                    'annotations': aug_annotations,
                    'source_image': crop['source_image']
                }

                # Save with augmentation suffix
                save_crop(aug_crop, output_folder, "train", f"_aug{aug_round}")

        # Step 6: Create dataset files
        create_dataset_yaml(output_folder)

        # Copy classes.txt if it exists
        classes_file = os.path.join(input_folder, "classes.txt")
        if os.path.exists(classes_file):
            shutil.copy2(classes_file, os.path.join(output_folder, "classes.txt"))

        # Step 7: Save statistics
        final_train_count = len(train_crops) * AUGMENTATION_MULTIPLIER
        stats = {
            'methodology': 'clean_crop_split_augment_random_crop_geometric',
            'input_images': len(valid_images),
            'total_crops': len(all_crops),
            'split': {
                'train_original': len(train_crops),
                'train_total': final_train_count,
                'val': len(val_crops),
                'test': len(test_crops)
            },
            'augmentation_multiplier': AUGMENTATION_MULTIPLIER,
            'crop_size': CROP_SIZE,
            'geometric_augmentations_applied': [
                'MIRROR/FLIP: 50% chance of either horizontal OR vertical flip (mutually exclusive)',
                'ROTATION: 100% chance, random 0-360 degrees',
                'SHEAR: 100% chance, both directions, normal distribution (μ=0°, σ=8°), clipped to ±15°',
                'CROP/ZOOM: 0-14% using normal distribution',
                'COLOR/EFFECTS: Unchanged from original Roboflow specifications'
            ]
        }

        with open(os.path.join(output_folder, "dataset_stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)

        # Final summary
        print(f"\n" + "=" * 50)
        print("DATASET CREATION COMPLETE!")
        print("=" * 50)
        print(f"Final dataset:")
        print(f"   Training: {final_train_count} images ({len(train_crops)} × {AUGMENTATION_MULTIPLIER})")
        print(f"   Validation: {len(val_crops)} images")
        print(f"   Test: {len(test_crops)} images")
        print(f"   Total: {final_train_count + len(val_crops) + len(test_crops)} images")
        print(f"\nGeometric Augmentations Applied:")
        print(f"  1. Mirror/Flip: 50% chance")
        print(f"  2. Rotation: 100% (0-360°)")
        print(f"  3. Shear: 100% (both directions, normal distribution, ±15°)")
        print(f"  4. Crop/Zoom: 0-14% (normal distribution)")
        print(f"Dataset saved to: {output_folder}")
        print(f"Ready for YOLOv12 training!")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()