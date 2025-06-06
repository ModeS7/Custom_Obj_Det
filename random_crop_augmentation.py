"""
Random Crop Augmentation Script for Satellite Vehicle Detection

Generates 600 diverse 640x640 training crops with optimized augmentation sequence:
1. Mirror (flip) - 50% chance
2. Rotate 100% - random 0-360 degrees
3. Shear 100% - random intensity and direction
4. Photometric augmentations - 100% probability

Features:
- Area-weighted image selection (larger images more likely)
- Normal distribution scaling (μ=1.0, σ=0.2)
- Smart annotation handling (50% minimum visibility)
- Satellite-optimized augmentation parameters
"""

import os
import glob
import random
import numpy as np
import cv2
from PIL import Image
import albumentations as A
import json
from tqdm import tqdm

# Configuration
TARGET_IMAGES = 600
CROP_SIZE = 640
SCALE_MEAN = 1.0  # Normal scale focus
SCALE_STD = 0.2  # Standard deviation for random zoom
MIN_SCALE = 0.7  # Minimum zoom (70%)
MAX_SCALE = 1.4  # Maximum zoom (140%)
MIN_ANNOTATION_VISIBILITY = 0.5  # 50% minimum visibility
KEEP_ZERO_ANNOTATION_CROPS = True  # Keep crops with no annotations
MAX_ZERO_ANNOTATION_RATIO = 0.2  # Max 20% of dataset can be zero-annotation


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


def scan_training_images(training_folder):
    """Scan training folder and calculate area weights"""
    print("Scanning training images...")

    images_folder = os.path.join(training_folder, "images")
    labels_folder = os.path.join(training_folder, "labels")

    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"Training images folder not found: {images_folder}")

    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []

    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(images_folder, ext)))
        image_files.extend(glob.glob(os.path.join(images_folder, ext.upper())))

    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))

    image_info = []
    total_area = 0

    for image_path in image_files:
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                area = width * height

                # Find corresponding label file
                image_name = os.path.basename(image_path)
                base_name = os.path.splitext(image_name)[0]
                label_path = os.path.join(labels_folder, f"{base_name}.txt")

                if not os.path.exists(label_path):
                    label_path = None

                image_info.append({
                    'path': image_path,
                    'label_path': label_path,
                    'name': image_name,
                    'base_name': base_name,
                    'width': width,
                    'height': height,
                    'area': area
                })

                total_area += area

        except Exception as e:
            print(f"Error reading {image_path}: {e}")
            continue

    # Calculate selection weights (area-weighted)
    for img_info in image_info:
        img_info['weight'] = img_info['area'] / total_area

    print(f"Found {len(image_info)} training images")
    print(
        f"Area range: {min(info['area'] for info in image_info):,} - {max(info['area'] for info in image_info):,} pixels")

    return image_info


def weighted_random_selection(image_info_list):
    """Select image based on area weights"""
    weights = [img['weight'] for img in image_info_list]
    return random.choices(image_info_list, weights=weights, k=1)[0]


def random_scale_image(image, annotations):
    """Apply random scaling with normal distribution"""
    # Generate scale factor with normal distribution, clipped to range
    scale = np.random.normal(SCALE_MEAN, SCALE_STD)
    scale = np.clip(scale, MIN_SCALE, MAX_SCALE)

    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize image
    scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Annotations remain the same (normalized coordinates don't change with scaling)
    return scaled_image, annotations, scale


def get_random_crop_position(image_shape, crop_size):
    """Get random crop position ensuring crop fits within image"""
    h, w = image_shape[:2]

    if h < crop_size or w < crop_size:
        return None, None

    max_y = h - crop_size
    max_x = w - crop_size

    crop_y = random.randint(0, max_y)
    crop_x = random.randint(0, max_x)

    return crop_x, crop_y


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


def apply_photometric_augmentations_normal_dist(image):
    """Apply photometric augmentations using normal distribution

    All adjustments use normal distribution centered at neutral values:
    - Most images get little change (around neutral)
    - Some images get moderate changes
    - Few images get extreme changes (at clipping limits)
    """

    # Convert to float for processing
    img_float = image.astype(np.float32) / 255.0

    # ✅ BRIGHTNESS: Normal distribution around 0, most images get little change
    brightness_delta = np.random.normal(0, 0.08)  # Mean=0, std=0.08 (wider spread)
    brightness_delta = np.clip(brightness_delta, -0.15, 0.15)  # Clip extremes
    img_float = np.clip(img_float + brightness_delta, 0, 1)

    # ✅ CONTRAST: Normal distribution around 1.0, most images get little change
    contrast_factor = np.random.normal(1.0, 0.1)  # Mean=1.0 (neutral), std=0.1
    contrast_factor = np.clip(contrast_factor, 0.85, 1.15)  # Clip extremes (1.0 ± 0.15)
    img_float = np.clip(img_float * contrast_factor, 0, 1)

    # Convert back to uint8
    img_uint8 = (img_float * 255).astype(np.uint8)

    # ✅ NOISE: Normal distribution around 0, most images get little noise
    noise_std = abs(np.random.normal(0, 6.0))  # Mean=0, std=6, take absolute value
    noise_std = np.clip(noise_std, 0, 15.0)    # Cap maximum noise

    # Apply noise to image
    noise = np.random.normal(0, noise_std, img_uint8.shape).astype(np.float32)
    img_with_noise = img_uint8.astype(np.float32) + noise
    img_with_noise = np.clip(img_with_noise, 0, 255).astype(np.uint8)

    # ✅ BLUR/SHARPENING: Single normal distribution - negative=blur, positive=sharpen
    blur_sharpen_factor = np.random.normal(0, 2.0)  # Mean=0, std=2
    blur_sharpen_factor = np.clip(blur_sharpen_factor, -4.0, 4.0)  # Clip extremes

    if blur_sharpen_factor < -0.3:  # Negative side: Apply blur
        # Convert factor to blur kernel size (larger negative = more blur)
        blur_intensity = abs(blur_sharpen_factor)
        blur_kernel = int(3 + blur_intensity * 2)  # Scale 3-11 based on intensity

        # Ensure odd kernel size
        if blur_kernel % 2 == 0:
            blur_kernel += 1

        # Clip to reasonable range
        blur_kernel = min(blur_kernel, 11)

        final_img = cv2.GaussianBlur(img_with_noise, (blur_kernel, blur_kernel), 0)

    elif blur_sharpen_factor > 0.3:  # Positive side: Apply sharpening
        # Convert factor to sharpening intensity
        sharpen_alpha = min(blur_sharpen_factor * 0.15, 0.6)  # Scale to reasonable range

        # Create sharpening kernel
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(img_with_noise, -1, kernel)

        # Blend with original based on intensity
        final_img = cv2.addWeighted(img_with_noise, 1 - sharpen_alpha, sharpened, sharpen_alpha, 0)

    else:  # Around 0: No blur or sharpening (most images - natural sharpness)
        final_img = img_with_noise

    return final_img


def apply_augmentations(image, annotations):
    """Apply geometric and photometric augmentations in your specified order:

    1. Mirror (flip) - 50% chance (exact transform, no quality loss)
    2. Rotate 100% - 0-360 degrees (on clean/flipped image)
    3. Shear 100% - random intensity and direction (final interpolation step)
    4. Photometric augmentations
    """

    # ✅ STEP 1: Mirror (flip) with 50% chance - exact transform first
    augmented_image, augmented_annotations = apply_mirror_flip(image, annotations)

    # ✅ STEP 2: Rotate 100% - random 0-360 degrees
    augmented_image, augmented_annotations = apply_full_rotation(
        augmented_image, augmented_annotations
    )

    # ✅ STEP 3: Shear 100% - random intensity and direction
    augmented_image, augmented_annotations = apply_random_shear(
        augmented_image, augmented_annotations
    )

    # ✅ STEP 4: Photometric augmentations (100% probability with normal distribution)
    final_image = apply_photometric_augmentations_normal_dist(augmented_image)

    return final_image, augmented_annotations


def apply_mirror_flip(image, annotations):
    """Apply mirror (flip) with 50% chance - exact transform, no interpolation"""

    # 50% chance of applying either horizontal or vertical flip (not both)
    if random.random() < 0.5:
        # Randomly choose between horizontal or vertical flip
        if random.random() < 0.5:
            flip_transform = A.HorizontalFlip(p=1.0)
        else:
            flip_transform = A.VerticalFlip(p=1.0)

        flip_pipeline = A.Compose([flip_transform], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_area=0,
            min_visibility=0.3
        ))

        if annotations:
            class_labels = [ann[0] for ann in annotations]
            bboxes = [[ann[1], ann[2], ann[3], ann[4]] for ann in annotations]

            result = flip_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)

            # Convert back to annotation format
            final_annotations = []
            for bbox, class_id in zip(result['bboxes'], result['class_labels']):
                final_annotations.append([class_id, bbox[0], bbox[1], bbox[2], bbox[3]])

            return result['image'], final_annotations
        else:
            result = flip_pipeline(image=image, bboxes=[], class_labels=[])
            return result['image'], []

    # No flip applied
    return image, annotations


def apply_full_rotation(image, annotations):
    """Apply rotation 100% - random 0-360 degrees"""

    # Generate random angle from 0-360 degrees
    angle = random.uniform(0, 360)

    # Convert to -180 to +180 range for albumentations
    if angle > 180:
        angle = angle - 360

    rotation_pipeline = A.Compose([
        A.Rotate(limit=(angle - 0.1, angle + 0.1), p=1.0, border_mode=cv2.BORDER_CONSTANT)
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_area=0,
        min_visibility=0.3
    ))

    if annotations:
        class_labels = [ann[0] for ann in annotations]
        bboxes = [[ann[1], ann[2], ann[3], ann[4]] for ann in annotations]

        result = rotation_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)

        # Convert back to annotation format
        final_annotations = []
        for bbox, class_id in zip(result['bboxes'], result['class_labels']):
            final_annotations.append([class_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        return result['image'], final_annotations
    else:
        result = rotation_pipeline(image=image, bboxes=[], class_labels=[])
        return result['image'], []


def apply_random_shear(image, annotations):
    """Apply shear 100% - random intensity and direction using normal distribution

    Shear transforms create parallelogram distortions by moving one edge
    of the image relative to the opposite edge. This simulates perspective
    changes common in satellite imagery due to viewing angle variations.

    Uses normal distribution: most images get light shear, few get extreme shear.
    """

    # ✅ IMPROVED: Normal distribution around 0° (most images get little shear)
    shear_x = np.random.normal(0, 8.0)  # Mean=0°, std=8°
    shear_y = np.random.normal(0, 8.0)  # Mean=0°, std=8°

    # Clip to reasonable extremes
    shear_x = np.clip(shear_x, -30, 30)
    shear_y = np.clip(shear_y, -30, 30)

    # Randomly decide to apply shear in x, y, or both directions
    apply_x = random.random() < 0.7  # 70% chance for x-shear
    apply_y = random.random() < 0.7  # 70% chance for y-shear

    # Ensure at least one direction is applied
    if not apply_x and not apply_y:
        if random.random() < 0.5:
            apply_x = True
        else:
            apply_y = True

    # Set unused directions to 0
    if not apply_x:
        shear_x = 0
    if not apply_y:
        shear_y = 0

    shear_pipeline = A.Compose([
        A.Affine(
            shear={'x': shear_x, 'y': shear_y},  # Apply shear in both directions
            p=1.0,  # 100% probability
            border_mode=cv2.BORDER_CONSTANT,
            fill=0
        ),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_area=0,
        min_visibility=0.3
    ))

    if annotations:
        class_labels = [ann[0] for ann in annotations]
        bboxes = [[ann[1], ann[2], ann[3], ann[4]] for ann in annotations]

        result = shear_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)

        # Convert back to annotation format
        final_annotations = []
        for bbox, class_id in zip(result['bboxes'], result['class_labels']):
            final_annotations.append([class_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        return result['image'], final_annotations
    else:
        result = shear_pipeline(image=image, bboxes=[], class_labels=[])
        return result['image'], []


def generate_random_crop(image_info):
    """Generate a single random crop with your specified augmentation sequence"""

    # Load image
    image = cv2.imread(image_info['path'])
    if image is None:
        return None, None, None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load annotations
    annotations = load_annotations(image_info['label_path'])

    # Apply random scaling
    scaled_image, scaled_annotations, scale_factor = random_scale_image(image, annotations)

    # Get random crop position
    crop_x, crop_y = get_random_crop_position(scaled_image.shape, CROP_SIZE)
    if crop_x is None:
        return None, None, None  # Image too small after scaling

    # Crop image
    cropped_image = scaled_image[crop_y:crop_y + CROP_SIZE, crop_x:crop_x + CROP_SIZE]

    # Transform annotations for crop
    original_height, original_width = scaled_image.shape[:2]
    cropped_annotations = transform_annotations_for_crop(
        scaled_annotations, crop_x, crop_y, CROP_SIZE, original_width, original_height
    )

    # ✅ Apply your specified augmentation sequence
    augmented_image, final_annotations = apply_augmentations(
        cropped_image, cropped_annotations
    )

    # Generate metadata
    crop_metadata = {
        'source_image': image_info['name'],
        'scale_factor': scale_factor,
        'crop_position': (crop_x, crop_y),
        'original_size': (image_info['width'], image_info['height']),
        'scaled_size': (original_width, original_height),
        'annotation_count': len(final_annotations)
    }

    return augmented_image, final_annotations, crop_metadata


def create_output_structure(output_folder):
    """Create output folder structure"""
    print(f"Creating output structure: {output_folder}")

    images_dir = os.path.join(output_folder, "images")
    labels_dir = os.path.join(output_folder, "labels")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    return images_dir, labels_dir


def save_statistics(output_folder, generation_stats, successful_crops):
    """Save generation statistics"""
    stats = {
        'total_generated': len(successful_crops),
        'target_images': TARGET_IMAGES,
        'generation_attempts': generation_stats['attempts'],
        'success_rate': len(successful_crops) / generation_stats['attempts'] if generation_stats['attempts'] > 0 else 0,
        'zero_annotation_crops': sum(1 for crop in successful_crops if crop['metadata']['annotation_count'] == 0),
        'annotation_distribution': {},
        'source_image_usage': {},
        'scale_factor_stats': {
            'mean': np.mean([crop['metadata']['scale_factor'] for crop in successful_crops]),
            'std': np.std([crop['metadata']['scale_factor'] for crop in successful_crops]),
            'min': min(crop['metadata']['scale_factor'] for crop in successful_crops),
            'max': max(crop['metadata']['scale_factor'] for crop in successful_crops)
        }
    }

    # Annotation count distribution
    for crop in successful_crops:
        count = crop['metadata']['annotation_count']
        stats['annotation_distribution'][count] = stats['annotation_distribution'].get(count, 0) + 1

    # Source image usage
    for crop in successful_crops:
        source = crop['metadata']['source_image']
        stats['source_image_usage'][source] = stats['source_image_usage'].get(source, 0) + 1

    # Save statistics
    stats_file = os.path.join(output_folder, "generation_statistics.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Statistics saved to: {stats_file}")
    return stats


def main():
    """Main execution function"""
    print("Random Crop Augmentation Script")
    print("=" * 40)
    print(f"Target: {TARGET_IMAGES} augmented images")
    print(f"Crop size: {CROP_SIZE}x{CROP_SIZE}")
    print(f"Scale range: {MIN_SCALE:.1f}x - {MAX_SCALE:.1f}x (μ={SCALE_MEAN:.1f}, σ={SCALE_STD:.1f})")
    print(f"Min annotation visibility: {MIN_ANNOTATION_VISIBILITY:.0%}")

    # Get input and output folders
    training_folder = input("Enter path to training folder (from dataset_splitter): ").strip()
    if not os.path.exists(training_folder):
        print(f"Error: Training folder not found: {training_folder}")
        return

    output_folder = input("Enter path for augmented output folder: ").strip()

    print(f"\nProcessing training images from: {training_folder}")
    print(f"Output will be saved to: {output_folder}")

    # Confirm before proceeding
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return

    try:
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        print("Random seed set to 42 for reproducibility")

        # Scan training images
        image_info_list = scan_training_images(training_folder)
        if not image_info_list:
            print("No training images found!")
            return

        # Create output structure
        images_dir, labels_dir = create_output_structure(output_folder)

        # No pipeline creation needed - augmentations applied directly in apply_augmentations()

        # Generate random crops
        print(f"\nGenerating {TARGET_IMAGES} random crops with your specified augmentation sequence:")
        print("✅ 1. Mirror (flip) - 50% chance (exact transform)")
        print("✅ 2. Rotate 100% - random 0-360 degrees")
        print("✅ 3. Shear 100% - random intensity and direction")
        print(f"✅ 4. Photometric augmentations - All use normal distribution around neutral:")
        print(f"    • Brightness: Most neutral, some bright/dark")
        print(f"    • Contrast: Most neutral, some high/low")
        print(f"    • Noise: Most clean, some noisy")
        print(f"    • Blur/Sharpen: Most natural, some blurred OR sharpened (never both)")

        successful_crops = []
        zero_annotation_count = 0
        generation_stats = {'attempts': 0}

        with tqdm(total=TARGET_IMAGES, desc="Generating crops") as pbar:
            while len(successful_crops) < TARGET_IMAGES:
                generation_stats['attempts'] += 1

                # Select image based on area weights
                selected_image = weighted_random_selection(image_info_list)

                # Generate random crop with your specified augmentation sequence
                aug_image, aug_annotations, metadata = generate_random_crop(selected_image)

                if aug_image is None:
                    continue  # Failed to generate crop

                # Check zero-annotation policy
                if len(aug_annotations) == 0:
                    if not KEEP_ZERO_ANNOTATION_CROPS:
                        continue
                    if zero_annotation_count / len(successful_crops) >= MAX_ZERO_ANNOTATION_RATIO:
                        continue
                    zero_annotation_count += 1

                # Save augmented image
                crop_id = f"aug_{len(successful_crops):05d}"
                image_filename = f"{crop_id}.jpg"
                image_path = os.path.join(images_dir, image_filename)

                # Convert RGB back to BGR for OpenCV
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(image_path, aug_image_bgr)

                # Save annotations if any
                if aug_annotations:
                    label_filename = f"{crop_id}.txt"
                    label_path = os.path.join(labels_dir, label_filename)

                    with open(label_path, 'w') as f:
                        for ann in aug_annotations:
                            f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")

                # Record successful crop
                successful_crops.append({
                    'filename': image_filename,
                    'metadata': metadata
                })

                pbar.update(1)

        # Save statistics and summary
        stats = save_statistics(output_folder, generation_stats, successful_crops)

        # Copy classes.txt if it exists
        classes_file = os.path.join(training_folder, "classes.txt")
        if os.path.exists(classes_file):
            import shutil
            shutil.copy2(classes_file, os.path.join(output_folder, "classes.txt"))

        # Create data.yaml for training
        yaml_content = f"""path: {os.path.abspath(output_folder)}
train: images
val: # Set validation path separately
test: # Set test path separately

nc: 1
names:
  0: vehicle
"""

        with open(os.path.join(output_folder, "data.yaml"), 'w') as f:
            f.write(yaml_content)

        # Final summary
        print(f"\n" + "=" * 50)
        print("AUGMENTED DATASET GENERATION COMPLETE!")
        print("=" * 50)
        print(f"Generated: {len(successful_crops)} augmented images")
        print(f"Success rate: {stats['success_rate']:.1%}")
        print(f"Augmentation sequence applied to each crop:")
        print(f"  1. Mirror (flip): 50% chance")
        print(f"  2. Rotation: 100% (0-360°)")
        print(f"  3. Shear: 100% (random intensity/direction)")
        print(f"  4. Photometric: 100% (brightness/contrast/noise normal dist + blur OR sharpen)")
        print(
            f"Zero-annotation crops: {stats['zero_annotation_crops']} ({stats['zero_annotation_crops'] / len(successful_crops) * 100:.1f}%)")
        print(
            f"Scale factor range: {stats['scale_factor_stats']['min']:.2f} - {stats['scale_factor_stats']['max']:.2f}")
        print(f"Average scale: {stats['scale_factor_stats']['mean']:.2f} ± {stats['scale_factor_stats']['std']:.2f}")
        print(f"\nDataset saved to: {output_folder}")
        print(f"Check generation_statistics.json for detailed breakdown")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()