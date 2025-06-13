import random
import cv2
import numpy as np
import albumentations as A

def apply_mirror_flip(image, annotations):
    """Apply mirror (flip) with 50% chance - exact transform, no interpolation"""
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
    """Apply shear 100% - random intensity in both directions using normal distribution"""
    # Normal distribution around 0° (most images get little shear)
    shear_x = np.random.normal(0, 2.0)  # Mean=0°, std=2°
    shear_y = np.random.normal(0, 2.0)  # Mean=0°, std=2°

    # Clip to 15 degrees
    shear_x = np.clip(shear_x, -15, 15)
    shear_y = np.clip(shear_y, -15, 15)

    # Apply shear in both directions - creates unique vector every time

    shear_pipeline = A.Compose([
        A.Affine(
            shear={'x': shear_x, 'y': shear_y},
            p=1.0,
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


def apply_perspective_distortion(image, annotations):
    """Apply perspective transform - more realistic than simple shear"""

    perspective_pipeline = A.Compose([
        A.Perspective(
            scale=0.1,  # Scale range for perspective distortion
            p=1.0,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fit_output=True,  # Maintains full image visibility
            keep_size=True    # Keeps original image size
        )
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_area=0,
        min_visibility=0.3
    ))

    if annotations:
        class_labels = [ann[0] for ann in annotations]
        bboxes = [[ann[1], ann[2], ann[3], ann[4]] for ann in annotations]

        result = perspective_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)

        final_annotations = []
        for bbox, class_id in zip(result['bboxes'], result['class_labels']):
            final_annotations.append([class_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        return result['image'], final_annotations
    else:
        result = perspective_pipeline(image=image, bboxes=[], class_labels=[])
        return result['image'], []


def apply_crop_zoom_normal_dist(image, annotations):
    """Apply crop/zoom 0-15% using normal distribution"""
    # Normal distribution around 1.0 (no zoom), std=0.06 to get 0-15% range
    zoom_factor = np.random.normal(1.0, 0.09)
    # Clip to 0-15% zoom range (1.0 to 1.15)
    zoom_factor = np.clip(zoom_factor, 1.0, 1.25)

    zoom_pipeline = A.Compose([
        A.Affine(scale=zoom_factor, p=1.0, border_mode=cv2.BORDER_CONSTANT, fill=0)
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_area=0,
        min_visibility=0.3
    ))

    if annotations:
        class_labels = [ann[0] for ann in annotations]
        bboxes = [[ann[1], ann[2], ann[3], ann[4]] for ann in annotations]

        result = zoom_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)

        # Convert back to annotation format
        final_annotations = []
        for bbox, class_id in zip(result['bboxes'], result['class_labels']):
            final_annotations.append([class_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        return result['image'], final_annotations
    else:
        result = zoom_pipeline(image=image, bboxes=[], class_labels=[])
        return result['image'], []


def apply_thinplate_spline_deformation(image, annotations):
    """Apply ThinPlateSpline smooth deformation - parameters need verification"""
    # Parameters estimated based on typical TPS usage - NEEDS VERIFICATION
    tps_pipeline = A.Compose([
        A.ThinPlateSpline(
            scale_range=[0.01, 0.04],  # Scale range for deformation
            p=1.0,
        )
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_area=0,
        min_visibility=0.3
    ))

    # Implementation follows same pattern as other transforms
    if annotations:
        class_labels = [ann[0] for ann in annotations]
        bboxes = [[ann[1], ann[2], ann[3], ann[4]] for ann in annotations]

        result = tps_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)

        final_annotations = []
        for bbox, class_id in zip(result['bboxes'], result['class_labels']):
            final_annotations.append([class_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        return result['image'], final_annotations
    else:
        result = tps_pipeline(image=image, bboxes=[], class_labels=[])
        return result['image'], []


def apply_grid_elastic_deform(image, annotations):
    """Apply GridElasticDeform - localized elastic deformations with grid control"""

    magnitude = random.randint(8, 16)

    grid_elastic_pipeline = A.Compose([
        A.GridElasticDeform(
            num_grid_xy=(6, 6),
            magnitude=magnitude,
            p=1.0
        )
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_area=0,
        min_visibility=0.3
    ))

    if annotations:
        class_labels = [ann[0] for ann in annotations]
        bboxes = [[ann[1], ann[2], ann[3], ann[4]] for ann in annotations]

        result = grid_elastic_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)

        final_annotations = []
        for bbox, class_id in zip(result['bboxes'], result['class_labels']):
            final_annotations.append([class_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        return result['image'], final_annotations
    else:
        result = grid_elastic_pipeline(image=image, bboxes=[], class_labels=[])
        return result['image'], []


def apply_grid_distortion(image, annotations):
    """Apply GridDistortion - non-uniform grid warping with localized distortions"""

    grid_distortion_pipeline = A.Compose([
        A.GridDistortion(
            num_steps=4,
            distort_limit=[-0.2, 0.2],
            p=1.0
        )
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_area=0,
        min_visibility=0.3
    ))

    if annotations:
        class_labels = [ann[0] for ann in annotations]
        bboxes = [[ann[1], ann[2], ann[3], ann[4]] for ann in annotations]

        result = grid_distortion_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)

        final_annotations = []
        for bbox, class_id in zip(result['bboxes'], result['class_labels']):
            final_annotations.append([class_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        return result['image'], final_annotations
    else:
        result = grid_distortion_pipeline(image=image, bboxes=[], class_labels=[])
        return result['image'], []


def apply_elastic_transform(image, annotations):
    """Apply ElasticTransform - tissue-like elastic deformation for realistic distortions"""

    # Alpha controls deformation strength - higher = more distortion
    # Good range for vehicles: 50-120 (moderate to strong elastic effects)
    alpha = random.uniform(50, 120)

    # Sigma controls smoothness - 10:1 ratio with alpha is recommended
    # Lower sigma = sharper distortions, higher sigma = smoother distortions
    sigma = alpha * random.uniform(0.08, 0.12)  # Maintains ~10:1 ratio

    elastic_pipeline = A.Compose([
        A.ElasticTransform(
            alpha=alpha,
            sigma=sigma,
            p=1.0
        )
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_area=0,
        min_visibility=0.3
    ))

    if annotations:
        class_labels = [ann[0] for ann in annotations]
        bboxes = [[ann[1], ann[2], ann[3], ann[4]] for ann in annotations]

        result = elastic_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)

        final_annotations = []
        for bbox, class_id in zip(result['bboxes'], result['class_labels']):
            final_annotations.append([class_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        return result['image'], final_annotations
    else:
        result = elastic_pipeline(image=image, bboxes=[], class_labels=[])
        return result['image'], []


def apply_optical_distortion(image, annotations):
    """Apply OpticalDistortion - barrel/pincushion lens distortion effects"""

    optical_distortion_pipeline = A.Compose([
        A.OpticalDistortion(
            distort_limit=[-0.2, 0.2],
            p=1.0
        )
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_area=0,
        min_visibility=0.3
    ))

    if annotations:
        class_labels = [ann[0] for ann in annotations]
        bboxes = [[ann[1], ann[2], ann[3], ann[4]] for ann in annotations]

        result = optical_distortion_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)

        final_annotations = []
        for bbox, class_id in zip(result['bboxes'], result['class_labels']):
            final_annotations.append([class_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        return result['image'], final_annotations
    else:
        result = optical_distortion_pipeline(image=image, bboxes=[], class_labels=[])
        return result['image'], []
