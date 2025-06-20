# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1VHg3PDopRQphBSO-Zwh1CLvfGjsAeEGe
"""

"""
YOLOv3 Training Script for Google Colab
This script sets up the environment, downloads your Roboflow dataset, and trains a YOLOv3 model.
"""

# INITIAL SETUP
!pip install ultralytics roboflow

from ultralytics import YOLO
import os
from google.colab import drive
from roboflow import Roboflow

# Mount Google Drive to save models
drive.mount('/content/drive')
save_dir = '/content/drive/MyDrive/yolov3tiny_training'
os.makedirs(save_dir, exist_ok=True)

# DOWNLOAD DATASET FROM ROBOFLOW
from API import *
rf = Roboflow(api_key=api_key)
project = rf.workspace(workspace_id).project(original_project_id)
version = project.version(2)
dataset = version.download("yolov8")

# Print dataset information
print(f"Dataset downloaded to: {dataset.location}")
print(f"Data configuration file: {dataset.location}/data.yaml")

# TRAIN THE MODEL
# YOLOv3-tiny is the lightweight version, similar to YOLOv8n
model = YOLO('yolov3-tiny.pt')  # Load pre-trained YOLOv3-tiny model

# TRAINING CONFIGURATION
# Modified parameters appropriate for YOLOv3
epochs = 100             # For small dataset, 100-150 epochs is often sufficient
batch_size = 16          # Smaller batch size for limited VRAM
image_size = 640         # Standard YOLOv3 input size
patience = 20            # Early stopping patience

# Start training
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=epochs,
    batch=batch_size,
    imgsz=image_size,
    patience=patience,
    project=save_dir,
    name='yolov3tiny_custom',
    exist_ok=True,
    pretrained=True,
    optimizer="AdamW",   # Advanced optimizer
    cos_lr=True,         # Cosine learning rate schedule
    lr0=0.001,           # Initial learning rate
    lrf=0.01,            # Final learning rate ratio
    weight_decay=0.0001, # Weight decay
    warmup_epochs=3,     # Warmup epochs
    seed=0,                 # For reproducibility
    device=0,               # Use GPU 0
)

# VALIDATE THE TRAINED MODEL
metrics = model.val(data=f"{dataset.location}/data.yaml")
print(f"Validation metrics: {metrics}")

# EXPORT THE MODEL
# Get the path of the best weights
weights_dir = os.path.join(save_dir, 'yolov3_custom', 'weights')
best_pt = os.path.join(weights_dir, 'best.pt')

# Export to ONNX format
model.export(format="onnx", imgsz=image_size)  # Creates best.onnx in the same folder

# Print actual paths for both PT and ONNX models
print(f"PyTorch model saved at: {best_pt}")
print(f"ONNX model exported to: {os.path.join(weights_dir, 'best.onnx')}")

# RUN INFERENCE ON A TEST IMAGE
# Find all test images in the directory
import glob

# Get a list of all test images
test_image_dir = f"{dataset.location}/test/images"
test_images = glob.glob(f"{test_image_dir}/*.jpg") + glob.glob(f"{test_image_dir}/*.jpeg") + glob.glob(f"{test_image_dir}/*.png")

if test_images:
    # Use the first test image found
    test_image_path = test_images[9]
    print(f"Running inference on: {test_image_path}")
    # Perform inference
    results = model.predict(test_image_path, save=True, conf=0.25)
    print(f"Inference results saved to: {results[0].save_dir}")

    # If you want to display the results directly in Colab
    from IPython.display import Image, display
    display(Image(os.path.join(results[0].save_dir, os.path.basename(test_image_path))))
else:
    print(f"No test images found in {test_image_dir}")
    print("Try running inference on a specific image with:")
    print("model.predict('/path/to/your/image.jpg', save=True, conf=0.25)")

# If you want to display the results directly in Colab
from IPython.display import Image, display
display(Image(os.path.join(results[0].save_dir, os.path.basename(test_image_path))))

print("Training, validation, and export complete!")