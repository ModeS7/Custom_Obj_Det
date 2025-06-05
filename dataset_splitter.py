# Script to make split datasett into test,
# validation with 640x640 and leave training
# images without test or validation data.

import os
import glob
import shutil
import random
from pathlib import Path
from PIL import Image
import json

# Configuration
TARGET_TEST_TILES = 55  # Back to higher targets with aggressive approach
TARGET_VAL_TILES = 35  # Back to higher targets
TILE_SIZE = 640
STRIP_WIDTH = 640
MIN_IMAGES_FOR_STRIPS = 8  # Minimum images to try
MAX_IMAGES_FOR_STRIPS = 29  # Can use all images if needed
MIN_DIMENSION = 1280  # Need at least 2x strip width/height


def scan_images_and_labels(images_folder):
    """Scan all images and their corresponding label files"""
    print("Scanning images and labels...")

    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []

    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(images_folder, ext)))
        image_files.extend(glob.glob(os.path.join(images_folder, ext.upper())))

    # Remove duplicates (case-insensitive filesystems)
    image_files = list(set(image_files))
    image_files.sort()

    # Check for corresponding labels
    labels_folder = os.path.join(os.path.dirname(images_folder), "labels")
    if not os.path.exists(labels_folder):
        print(f"Warning: Labels folder not found at {labels_folder}")
        labels_folder = None

    image_info = []

    for image_path in image_files:
        try:
            with Image.open(image_path) as img:
                width, height = img.size

                # Check for corresponding label file
                image_name = os.path.basename(image_path)
                base_name = os.path.splitext(image_name)[0]
                label_path = None

                if labels_folder:
                    potential_label = os.path.join(labels_folder, f"{base_name}.txt")
                    if os.path.exists(potential_label):
                        label_path = potential_label

                # Determine valid strip directions
                valid_directions = []
                if width >= MIN_DIMENSION:
                    valid_directions.extend(["left", "right"])
                if height >= MIN_DIMENSION:
                    valid_directions.extend(["top", "bottom"])

                image_info.append({
                    'path': image_path,
                    'label_path': label_path,
                    'name': image_name,
                    'base_name': base_name,
                    'width': width,
                    'height': height,
                    'valid_directions': valid_directions,
                    'usable': len(valid_directions) > 0
                })

        except Exception as e:
            print(f"Error reading {image_path}: {e}")
            continue

    print(f"Found {len(image_info)} images")
    usable_count = sum(1 for img in image_info if img['usable'])
    with_labels = sum(1 for img in image_info if img['label_path'])
    print(f"Usable images (min dimension >= {MIN_DIMENSION}px): {usable_count}")
    print(f"Images with labels: {with_labels}")

    return image_info, labels_folder


def select_strip_images_aggressive(image_info):
    """Aggressively select strips to maximize tile yield"""
    print("\nSelecting images for strip cutting (aggressive mode)...")

    # Filter usable images
    usable_images = [img for img in image_info if img['usable']]

    if not usable_images:
        print("No usable images found!")
        return [], []

    # Target total tiles
    target_total_tiles = TARGET_TEST_TILES + TARGET_VAL_TILES

    # Track used strips per image: {image_name: [directions_used]}
    used_strips = {}
    strip_selections = []
    estimated_total_tiles = 0

    print(f"Target: {target_total_tiles} tiles")

    # Phase 1: Use each image once with random direction
    print("Phase 1: Using each image once...")
    random.shuffle(usable_images)

    for img in usable_images:
        if estimated_total_tiles >= target_total_tiles:
            break

        # Choose random direction
        chosen_direction = random.choice(img['valid_directions'])

        # Estimate tiles for this strip
        if chosen_direction in ["left", "right"]:
            estimated_tiles = img['height'] // TILE_SIZE
        else:  # top, bottom
            estimated_tiles = img['width'] // TILE_SIZE

        # Record this selection
        strip_selections.append({
            'image': img,
            'direction': chosen_direction,
            'estimated_tiles': estimated_tiles
        })

        used_strips[img['name']] = [chosen_direction]
        estimated_total_tiles += estimated_tiles

        print(f"  {img['name']} ({chosen_direction}) -> ~{estimated_tiles} tiles")

    # Phase 2: If still need more tiles, reuse images with different directions
    if estimated_total_tiles < target_total_tiles:
        print(f"\nPhase 2: Need {target_total_tiles - estimated_total_tiles} more tiles, reusing images...")

        # Create list of possible additional strips
        additional_options = []

        for img in usable_images:
            if img['name'] in used_strips:
                used_directions = used_strips[img['name']]
                available_directions = [d for d in img['valid_directions'] if d not in used_directions]

                for direction in available_directions:
                    # Check for spatial overlap
                    if not would_overlap(used_directions, direction):
                        if direction in ["left", "right"]:
                            estimated_tiles = img['height'] // TILE_SIZE
                        else:
                            estimated_tiles = img['width'] // TILE_SIZE

                        additional_options.append({
                            'image': img,
                            'direction': direction,
                            'estimated_tiles': estimated_tiles
                        })

        # Sort by tile potential and add until target reached
        additional_options.sort(key=lambda x: x['estimated_tiles'], reverse=True)

        for option in additional_options:
            if estimated_total_tiles >= target_total_tiles:
                break

            strip_selections.append(option)
            used_strips[option['image']['name']].append(option['direction'])
            estimated_total_tiles += option['estimated_tiles']

            print(f"  {option['image']['name']} ({option['direction']}) -> ~{option['estimated_tiles']} tiles")

    # Calculate training images (images not used for strips at all)
    used_image_names = set(used_strips.keys())
    training_images = [img for img in image_info if img['name'] not in used_image_names]

    print(f"\nSelected {len(strip_selections)} strips from {len(used_image_names)} images")
    print(f"Estimated total tiles: {estimated_total_tiles} (target: {target_total_tiles})")
    print(f"Preserved {len(training_images)} images completely for training")

    return strip_selections, training_images


def would_overlap(used_directions, new_direction):
    """Check if a new strip direction would overlap with existing strips"""
    # Only same-side strips overlap (left with left, top with top, etc.)
    # Perpendicular directions (left/right with top/bottom) don't overlap
    return new_direction in used_directions


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


def transform_annotations_for_strip(annotations, strip_direction, image_width, image_height):
    """Transform annotations for strip crop"""
    if not annotations:
        return []

    transformed = []

    for ann in annotations:
        class_id, x_center, y_center, width, height = ann

        # Convert to pixel coordinates
        x_center_px = x_center * image_width
        y_center_px = y_center * image_height
        width_px = width * image_width
        height_px = height * image_height

        # Calculate bounding box
        x1 = x_center_px - width_px / 2
        y1 = y_center_px - height_px / 2
        x2 = x_center_px + width_px / 2
        y2 = y_center_px + height_px / 2

        # Define strip boundaries
        if strip_direction == "left":
            strip_x1, strip_y1 = 0, 0
            strip_x2, strip_y2 = STRIP_WIDTH, image_height
            offset_x, offset_y = 0, 0
        elif strip_direction == "right":
            strip_x1, strip_y1 = image_width - STRIP_WIDTH, 0
            strip_x2, strip_y2 = image_width, image_height
            offset_x, offset_y = image_width - STRIP_WIDTH, 0
        elif strip_direction == "top":
            strip_x1, strip_y1 = 0, 0
            strip_x2, strip_y2 = image_width, STRIP_WIDTH
            offset_x, offset_y = 0, 0
        elif strip_direction == "bottom":
            strip_x1, strip_y1 = 0, image_height - STRIP_WIDTH
            strip_x2, strip_y2 = image_width, image_height
            offset_x, offset_y = 0, image_height - STRIP_WIDTH

        # Check intersection with strip
        intersect_x1 = max(x1, strip_x1)
        intersect_y1 = max(y1, strip_y1)
        intersect_x2 = min(x2, strip_x2)
        intersect_y2 = min(y2, strip_y2)

        if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
            # Calculate intersection area
            intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
            original_area = width_px * height_px

            # Keep annotation if >50% visible
            if intersect_area / original_area > 0.5:
                # Adjust to strip coordinates
                new_x_center = (intersect_x1 + intersect_x2) / 2 - offset_x
                new_y_center = (intersect_y1 + intersect_y2) / 2 - offset_y
                new_width = intersect_x2 - intersect_x1
                new_height = intersect_y2 - intersect_y1

                # Get strip dimensions
                if strip_direction in ["left", "right"]:
                    strip_width, strip_height = STRIP_WIDTH, image_height
                else:  # top, bottom
                    strip_width, strip_height = image_width, STRIP_WIDTH

                # Normalize to strip dimensions
                norm_x_center = new_x_center / strip_width
                norm_y_center = new_y_center / strip_height
                norm_width = new_width / strip_width
                norm_height = new_height / strip_height

                transformed.append([class_id, norm_x_center, norm_y_center, norm_width, norm_height])

    return transformed


def transform_annotations_for_remaining(annotations, strip_direction, image_width, image_height):
    """Transform annotations for remaining training area"""
    if not annotations:
        return []

    transformed = []

    for ann in annotations:
        class_id, x_center, y_center, width, height = ann

        # Convert to pixel coordinates
        x_center_px = x_center * image_width
        y_center_px = y_center * image_height
        width_px = width * image_width
        height_px = height * image_height

        # Calculate bounding box
        x1 = x_center_px - width_px / 2
        y1 = y_center_px - height_px / 2
        x2 = x_center_px + width_px / 2
        y2 = y_center_px + height_px / 2

        # Define remaining area boundaries
        if strip_direction == "left":
            remain_x1, remain_y1 = STRIP_WIDTH, 0
            remain_x2, remain_y2 = image_width, image_height
            offset_x, offset_y = STRIP_WIDTH, 0
            remain_width, remain_height = image_width - STRIP_WIDTH, image_height
        elif strip_direction == "right":
            remain_x1, remain_y1 = 0, 0
            remain_x2, remain_y2 = image_width - STRIP_WIDTH, image_height
            offset_x, offset_y = 0, 0
            remain_width, remain_height = image_width - STRIP_WIDTH, image_height
        elif strip_direction == "top":
            remain_x1, remain_y1 = 0, STRIP_WIDTH
            remain_x2, remain_y2 = image_width, image_height
            offset_x, offset_y = 0, STRIP_WIDTH
            remain_width, remain_height = image_width, image_height - STRIP_WIDTH
        elif strip_direction == "bottom":
            remain_x1, remain_y1 = 0, 0
            remain_x2, remain_y2 = image_width, image_height - STRIP_WIDTH
            offset_x, offset_y = 0, 0
            remain_width, remain_height = image_width, image_height - STRIP_WIDTH

        # Check intersection with remaining area
        intersect_x1 = max(x1, remain_x1)
        intersect_y1 = max(y1, remain_y1)
        intersect_x2 = min(x2, remain_x2)
        intersect_y2 = min(y2, remain_y2)

        if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
            # Calculate intersection area
            intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
            original_area = width_px * height_px

            # Keep annotation if >50% visible
            if intersect_area / original_area > 0.5:
                # Adjust to remaining area coordinates
                new_x_center = (intersect_x1 + intersect_x2) / 2 - offset_x
                new_y_center = (intersect_y1 + intersect_y2) / 2 - offset_y
                new_width = intersect_x2 - intersect_x1
                new_height = intersect_y2 - intersect_y1

                # Normalize to remaining area dimensions
                norm_x_center = new_x_center / remain_width
                norm_y_center = new_y_center / remain_height
                norm_width = new_width / remain_width
                norm_height = new_height / remain_height

                transformed.append([class_id, norm_x_center, norm_y_center, norm_width, norm_height])

    return transformed


def extract_strip_and_tiles(image_info, strip_direction, temp_folder):
    """Extract strip from image and create tiles"""
    try:
        with Image.open(image_info['path']) as img:
            width, height = img.size

            # Extract strip
            if strip_direction == "left":
                strip = img.crop((0, 0, STRIP_WIDTH, height))
            elif strip_direction == "right":
                strip = img.crop((width - STRIP_WIDTH, 0, width, height))
            elif strip_direction == "top":
                strip = img.crop((0, 0, width, STRIP_WIDTH))
            elif strip_direction == "bottom":
                strip = img.crop((0, height - STRIP_WIDTH, width, height))

            # Load and transform annotations for strip
            annotations = load_annotations(image_info['label_path'])
            strip_annotations = transform_annotations_for_strip(
                annotations, strip_direction, width, height
            )

            # Create tiles from strip
            tiles_info = []

            if strip_direction in ["left", "right"]:
                strip_width, strip_height = STRIP_WIDTH, height
                num_tiles = strip_height // TILE_SIZE

                for i in range(num_tiles):
                    y_start = i * TILE_SIZE
                    y_end = y_start + TILE_SIZE

                    tile = strip.crop((0, y_start, STRIP_WIDTH, y_end))

                    # Transform annotations for this tile
                    tile_annotations = []
                    for ann in strip_annotations:
                        class_id, x_center, y_center, width_norm, height_norm = ann

                        # Convert to strip coordinates
                        y_center_px = y_center * strip_height
                        height_px = height_norm * strip_height

                        # Check if annotation intersects with tile
                        ann_y1 = y_center_px - height_px / 2
                        ann_y2 = y_center_px + height_px / 2

                        if ann_y2 > y_start and ann_y1 < y_end:
                            # Adjust for tile coordinates
                            new_y_center = (y_center_px - y_start) / TILE_SIZE
                            new_height = height_norm * strip_height / TILE_SIZE

                            if new_y_center >= 0 and new_y_center <= 1 and new_height > 0:
                                tile_annotations.append([class_id, x_center, new_y_center, width_norm, new_height])

                    # Save tile and annotations
                    tile_id = f"{image_info['base_name']}_{strip_direction}_tile_{i}"
                    tile_path = os.path.join(temp_folder, f"{tile_id}.jpg")
                    tile.save(tile_path, "JPEG", quality=95)

                    tiles_info.append({
                        'tile_id': tile_id,
                        'tile_path': tile_path,
                        'annotations': tile_annotations
                    })

            else:  # top, bottom
                strip_width, strip_height = width, STRIP_WIDTH
                num_tiles = strip_width // TILE_SIZE

                for i in range(num_tiles):
                    x_start = i * TILE_SIZE
                    x_end = x_start + TILE_SIZE

                    tile = strip.crop((x_start, 0, x_end, STRIP_WIDTH))

                    # Transform annotations for this tile
                    tile_annotations = []
                    for ann in strip_annotations:
                        class_id, x_center, y_center, width_norm, height_norm = ann

                        # Convert to strip coordinates
                        x_center_px = x_center * strip_width
                        width_px = width_norm * strip_width

                        # Check if annotation intersects with tile
                        ann_x1 = x_center_px - width_px / 2
                        ann_x2 = x_center_px + width_px / 2

                        if ann_x2 > x_start and ann_x1 < x_end:
                            # Adjust for tile coordinates
                            new_x_center = (x_center_px - x_start) / TILE_SIZE
                            new_width = width_norm * strip_width / TILE_SIZE

                            if new_x_center >= 0 and new_x_center <= 1 and new_width > 0:
                                tile_annotations.append([class_id, new_x_center, y_center, new_width, height_norm])

                    # Save tile and annotations
                    tile_id = f"{image_info['base_name']}_{strip_direction}_tile_{i}"
                    tile_path = os.path.join(temp_folder, f"{tile_id}.jpg")
                    tile.save(tile_path, "JPEG", quality=95)

                    tiles_info.append({
                        'tile_id': tile_id,
                        'tile_path': tile_path,
                        'annotations': tile_annotations
                    })

            return tiles_info

    except Exception as e:
        print(f"Error processing {image_info['path']}: {e}")
        return []


def save_remaining_area_multiple(image_info, directions_used, output_folder):
    """Save the remaining area after multiple strip extractions"""
    try:
        with Image.open(image_info['path']) as img:
            width, height = img.size

            # Calculate remaining area after all strips
            remaining_area = calculate_remaining_area(width, height, directions_used)

            if not remaining_area:
                print(f"  No remaining area for {image_info['name']} after strips: {directions_used}")
                return None

            # Extract remaining area
            x1, y1, x2, y2 = remaining_area
            remaining = img.crop((x1, y1, x2, y2))

            # Save remaining area
            directions_str = "_".join(sorted(directions_used))
            remaining_name = f"{image_info['base_name']}_remaining_{directions_str}.jpg"
            remaining_path = os.path.join(output_folder, "images", remaining_name)
            remaining.save(remaining_path, "JPEG", quality=95)

            # Transform annotations for remaining area
            annotations = load_annotations(image_info['label_path'])
            remaining_annotations = transform_annotations_for_remaining_multiple(
                annotations, directions_used, width, height, remaining_area
            )

            # Save transformed annotations
            if remaining_annotations:
                label_name = f"{image_info['base_name']}_remaining_{directions_str}.txt"
                label_path = os.path.join(output_folder, "labels", label_name)

                with open(label_path, 'w') as f:
                    for ann in remaining_annotations:
                        f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")

            return remaining_name

    except Exception as e:
        print(f"Error saving remaining area for {image_info['path']}: {e}")
        return None


def calculate_remaining_area(width, height, directions_used):
    """Calculate the largest remaining rectangular area after strip removal"""
    # Start with full image
    remaining_x1, remaining_y1 = 0, 0
    remaining_x2, remaining_y2 = width, height

    # Remove each strip
    for direction in directions_used:
        if direction == "left":
            remaining_x1 = max(remaining_x1, STRIP_WIDTH)
        elif direction == "right":
            remaining_x2 = min(remaining_x2, width - STRIP_WIDTH)
        elif direction == "top":
            remaining_y1 = max(remaining_y1, STRIP_WIDTH)
        elif direction == "bottom":
            remaining_y2 = min(remaining_y2, height - STRIP_WIDTH)

    # Check if there's still a valid area
    if remaining_x2 <= remaining_x1 or remaining_y2 <= remaining_y1:
        return None  # No remaining area

    return (remaining_x1, remaining_y1, remaining_x2, remaining_y2)


def transform_annotations_for_remaining_multiple(annotations, directions_used, image_width, image_height,
                                                 remaining_area):
    """Transform annotations for remaining area after multiple strips"""
    if not annotations or not remaining_area:
        return []

    remaining_x1, remaining_y1, remaining_x2, remaining_y2 = remaining_area
    remaining_width = remaining_x2 - remaining_x1
    remaining_height = remaining_y2 - remaining_y1

    transformed = []

    for ann in annotations:
        class_id, x_center, y_center, width, height = ann

        # Convert to pixel coordinates
        x_center_px = x_center * image_width
        y_center_px = y_center * image_height
        width_px = width * image_width
        height_px = height * image_height

        # Calculate bounding box
        x1 = x_center_px - width_px / 2
        y1 = y_center_px - height_px / 2
        x2 = x_center_px + width_px / 2
        y2 = y_center_px + height_px / 2

        # Check intersection with remaining area
        intersect_x1 = max(x1, remaining_x1)
        intersect_y1 = max(y1, remaining_y1)
        intersect_x2 = min(x2, remaining_x2)
        intersect_y2 = min(y2, remaining_y2)

        if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
            # Calculate intersection area
            intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
            original_area = width_px * height_px

            # Keep annotation if >50% visible
            if intersect_area / original_area > 0.5:
                # Adjust to remaining area coordinates
                new_x_center = (intersect_x1 + intersect_x2) / 2 - remaining_x1
                new_y_center = (intersect_y1 + intersect_y2) / 2 - remaining_y1
                new_width = intersect_x2 - intersect_x1
                new_height = intersect_y2 - intersect_y1

                # Normalize to remaining area dimensions
                norm_x_center = new_x_center / remaining_width
                norm_y_center = new_y_center / remaining_height
                norm_width = new_width / remaining_width
                norm_height = new_height / remaining_height

                transformed.append([class_id, norm_x_center, norm_y_center, norm_width, norm_height])

    return transformed


def create_dataset_structure(output_folder):
    """Create the output folder structure"""
    print(f"\nCreating dataset structure in: {output_folder}")

    # Create directories
    test_dir = os.path.join(output_folder, "test")
    val_dir = os.path.join(output_folder, "val")
    training_dir = os.path.join(output_folder, "training")
    temp_dir = os.path.join(output_folder, "temp_tiles")

    for base_dir in [test_dir, val_dir, training_dir]:
        os.makedirs(os.path.join(base_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "labels"), exist_ok=True)

    os.makedirs(temp_dir, exist_ok=True)

    return test_dir, val_dir, training_dir, temp_dir


def save_allocation_log(output_folder, strip_selections, training_images,
                        test_tiles, val_tiles, all_tiles_info):
    """Save detailed allocation log"""
    log_path = os.path.join(output_folder, "allocation_log.txt")

    # Group selections by image for display
    selections_by_image = {}
    for selection in strip_selections:
        img_name = selection['image']['name']
        if img_name not in selections_by_image:
            selections_by_image[img_name] = []
        selections_by_image[img_name].append(selection)

    with open(log_path, 'w') as f:
        f.write("DATASET ALLOCATION LOG (Aggressive Random Strip Approach)\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Configuration:\n")
        f.write(f"- Target test tiles: {TARGET_TEST_TILES}\n")
        f.write(f"- Target val tiles: {TARGET_VAL_TILES}\n")
        f.write(f"- Total target: {TARGET_TEST_TILES + TARGET_VAL_TILES}\n")
        f.write(f"- Tile size: {TILE_SIZE}x{TILE_SIZE}\n")
        f.write(f"- Strip width: {STRIP_WIDTH}\n")
        f.write(f"- Images used for strips: {len(selections_by_image)}\n")
        f.write(f"- Total strip extractions: {len(strip_selections)}\n")
        f.write(f"- Images preserved for training: {len(training_images)}\n\n")

        f.write(f"Results:\n")
        f.write(f"- Test tiles: {len(test_tiles)}\n")
        f.write(f"- Validation tiles: {len(val_tiles)}\n")
        f.write(f"- Total tiles from strips: {len(all_tiles_info)}\n\n")

        # Strip cutting details
        f.write(f"STRIP CUTTING DETAILS:\n")
        f.write("-" * 40 + "\n")
        for img_name, img_selections in selections_by_image.items():
            img_info = img_selections[0]['image']  # Get image info
            directions = [sel['direction'] for sel in img_selections]
            total_tiles = sum(1 for tile in all_tiles_info
                              if tile['tile_id'].startswith(img_info['base_name']))

            f.write(f"{img_info['name']} ({img_info['width']}x{img_info['height']}):\n")
            for selection in img_selections:
                f.write(f"  -> {selection['direction']} strip\n")
            f.write(f"  -> Total tiles: {total_tiles}\n\n")

        # Training images
        f.write(f"\nTRAINING IMAGES (completely preserved):\n")
        f.write("-" * 40 + "\n")
        for img_info in training_images:
            has_labels = "with labels" if img_info['label_path'] else "no labels"
            f.write(f"{img_info['name']} ({img_info['width']}x{img_info['height']}) - {has_labels}\n")

        # Random allocation details
        f.write(f"\nRANDOM TILE ALLOCATION:\n")
        f.write("-" * 40 + "\n")
        f.write("Test tiles:\n")
        for tile in test_tiles:
            f.write(f"  {tile['tile_id']}.jpg\n")

        f.write("\nValidation tiles:\n")
        for tile in val_tiles:
            f.write(f"  {tile['tile_id']}.jpg\n")

    print(f"Allocation log saved to: {log_path}")


def main():
    """Main execution function"""
    print("Dataset Strip Splitter (Random 4-Direction)")
    print("=" * 45)

    # Set random seed for reproducibility
    random.seed(42)
    print("Random seed set to 42 for reproducibility")

    # Get input and output folders
    images_folder = input("Enter path to images folder: ").strip()
    if not os.path.exists(images_folder):
        print(f"Error: Images folder '{images_folder}' not found!")
        return

    output_folder = input("Enter path for output dataset folder: ").strip()

    print(f"\nProcessing images from: {images_folder}")
    print(f"Output will be saved to: {output_folder}")
    print(f"Target: ~{TARGET_TEST_TILES} test tiles, ~{TARGET_VAL_TILES} validation tiles")
    print(f"Strategy: Aggressive strip extraction - reuse images with different directions if needed")

    # Confirm before proceeding
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return

    try:
        # Phase 1: Scan images and labels
        image_info, labels_folder = scan_images_and_labels(images_folder)
        if not image_info:
            print("No valid images found!")
            return

        # Phase 2: Aggressive strip selection
        strip_selections, training_images = select_strip_images_aggressive(image_info)
        if not strip_selections:
            print("No strips available for cutting!")
            return

        # Phase 3: Create output structure
        test_dir, val_dir, training_dir, temp_dir = create_dataset_structure(output_folder)

        # Phase 4: Extract strips based on selections
        print(f"\nExtracting strips from selections...")
        all_tiles_info = []

        for selection in strip_selections:
            img_info = selection['image']
            chosen_direction = selection['direction']

            print(f"Processing {img_info['name']} ({chosen_direction} strip)...")

            # Extract strip and create tiles
            tiles_info = extract_strip_and_tiles(img_info, chosen_direction, temp_dir)
            all_tiles_info.extend(tiles_info)

        print(f"Total tiles created: {len(all_tiles_info)}")

        # Phase 5: Save remaining areas for training (group by image)
        print(f"\nSaving remaining areas for training...")

        # Group selections by image
        selections_by_image = {}
        for selection in strip_selections:
            img_name = selection['image']['name']
            if img_name not in selections_by_image:
                selections_by_image[img_name] = []
            selections_by_image[img_name].append(selection)

        # Save remaining areas for each image
        for img_name, img_selections in selections_by_image.items():
            # Use the first selection's image info (they're all the same image)
            img_info = img_selections[0]['image']

            # Get all strip directions used for this image
            directions_used = [sel['direction'] for sel in img_selections]

            remaining_name = save_remaining_area_multiple(img_info, directions_used, training_dir)
            if remaining_name:
                print(f"  Saved remaining area: {remaining_name}")

        # Phase 6: Random tile allocation
        print(f"\nRandomly allocating {len(all_tiles_info)} tiles to test and validation sets...")
        print(f"\nRandomly allocating {len(all_tiles_info)} tiles to test and validation sets...")

        if len(all_tiles_info) < (TARGET_TEST_TILES + TARGET_VAL_TILES):
            print(
                f"Warning: Only {len(all_tiles_info)} tiles available, less than target {TARGET_TEST_TILES + TARGET_VAL_TILES}")

            # Adjust allocation proportionally
            total_available = len(all_tiles_info)
            test_ratio = TARGET_TEST_TILES / (TARGET_TEST_TILES + TARGET_VAL_TILES)

            actual_test_count = max(1, int(total_available * test_ratio))
            actual_val_count = total_available - actual_test_count

            print(f"Adjusted allocation: {actual_test_count} test, {actual_val_count} validation")
        else:
            actual_test_count = TARGET_TEST_TILES
            actual_val_count = TARGET_VAL_TILES

        random.shuffle(all_tiles_info)

        test_tiles = all_tiles_info[:actual_test_count]
        val_tiles = all_tiles_info[actual_test_count:actual_test_count + actual_val_count]

        # Move tiles to final locations
        for tile_info in test_tiles:
            # Move image
            final_image_path = os.path.join(test_dir, "images", f"{tile_info['tile_id']}.jpg")
            shutil.move(tile_info['tile_path'], final_image_path)

            # Save annotations
            if tile_info['annotations']:
                final_label_path = os.path.join(test_dir, "labels", f"{tile_info['tile_id']}.txt")
                with open(final_label_path, 'w') as f:
                    for ann in tile_info['annotations']:
                        f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")

        for tile_info in val_tiles:
            # Move image
            final_image_path = os.path.join(val_dir, "images", f"{tile_info['tile_id']}.jpg")
            shutil.move(tile_info['tile_path'], final_image_path)

            # Save annotations
            if tile_info['annotations']:
                final_label_path = os.path.join(val_dir, "labels", f"{tile_info['tile_id']}.txt")
                with open(final_label_path, 'w') as f:
                    for ann in tile_info['annotations']:
                        f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")

        # Phase 6: Copy complete training images
        print(f"\nCopying {len(training_images)} complete images to training set...")
        for img_info in training_images:
            # Copy image
            final_image_path = os.path.join(training_dir, "images", img_info['name'])
            shutil.copy2(img_info['path'], final_image_path)

            # Copy labels if they exist
            if img_info['label_path']:
                final_label_path = os.path.join(training_dir, "labels", f"{img_info['base_name']}.txt")
                shutil.copy2(img_info['label_path'], final_label_path)

        # Clean up temp folder
        shutil.rmtree(temp_dir)

        # Phase 7: Save documentation
        save_allocation_log(output_folder, strip_selections, training_images,
                            test_tiles, val_tiles, all_tiles_info)

        # Copy classes.txt if it exists
        classes_file = os.path.join(os.path.dirname(images_folder), "classes.txt")
        if os.path.exists(classes_file):
            for split_dir in [test_dir, val_dir, training_dir]:
                shutil.copy2(classes_file, os.path.join(split_dir, "classes.txt"))

        # Summary
        unique_images_used = len(set(sel['image']['name'] for sel in strip_selections))

        print(f"\n" + "=" * 60)
        print("DATASET CREATION COMPLETE!")
        print("=" * 60)
        print(f"Test tiles: {len(test_tiles)}")
        print(f"Validation tiles: {len(val_tiles)}")
        print(f"Total strip extractions: {len(strip_selections)}")
        print(f"Unique images used for strips: {unique_images_used}")
        print(f"Complete training images: {len(training_images)}")
        print(f"Training area fragments: {unique_images_used} (from strip cutting)")
        print(f"\nDataset saved to: {output_folder}")
        print(f"Check allocation_log.txt for detailed breakdown")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()