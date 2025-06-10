"""
YOLOv12 Training Script for Custom Vehicle Dataset - Local PC Version
Adapted for RTX 3090 and Windows environment.

INSTALLATION INSTRUCTIONS:
==========================
Before running this script, install the required packages:

1. Install basic requirements:
   pip install PyYAML wheel

2. Install YOLOv12:
   git clone https://github.com/sunsmarterjie/yolov12.git
   cd yolov12
   pip install -e .

3. (Optional) Install Flash Attention for better performance:
   pip install flash-attn --no-build-isolation

Then run this script: python train_yolov12.py
"""

import os
import yaml
import glob
from pathlib import Path
from ultralytics import YOLO

def setup_dataset():
    """Setup dataset paths and create dataset.yaml"""
    # Your dataset path
    LOCAL_DATASET_PATH = r"C:\NTNU\Custom_Obj_Det\datasets\My First Project.v2i.yolov12"

    print(f"Setting up dataset at: {LOCAL_DATASET_PATH}")

    if not os.path.exists(LOCAL_DATASET_PATH):
        raise FileNotFoundError(f"Dataset path not found: {LOCAL_DATASET_PATH}")

    # Verify dataset structure
    train_imgs = os.path.join(LOCAL_DATASET_PATH, "train", "images")
    train_labels = os.path.join(LOCAL_DATASET_PATH, "train", "labels")
    test_imgs = os.path.join(LOCAL_DATASET_PATH, "test", "images")
    test_labels = os.path.join(LOCAL_DATASET_PATH, "test", "labels")

    for path_name, path in [("Train images", train_imgs), ("Train labels", train_labels),
                           ("Test images", test_imgs), ("Test labels", test_labels)]:
        if not os.path.exists(path):
            print(f"Warning: {path_name} folder not found at {path}")
        else:
            file_count = len(glob.glob(os.path.join(path, "*.*")))
            print(f"✓ {path_name}: {file_count} files")

    # Create dataset.yaml file
    data_yaml_path = os.path.join(LOCAL_DATASET_PATH, 'dataset.yaml')

    # Convert Windows path to forward slashes for YAML
    dataset_path_yaml = LOCAL_DATASET_PATH.replace('\\', '/')

    yaml_data = {
        'path': dataset_path_yaml,
        'train': 'train/images',
        'val': 'test/images',    # Using test as validation since no val folder exists
        'test': 'test/images',
        'nc': 1,
        'names': {0: 'vehicle'}
    }

    with open(data_yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    print(f"✓ Dataset YAML created at: {data_yaml_path}")

    return data_yaml_path, LOCAL_DATASET_PATH

def download_pretrained_weights():
    """Download YOLOv12 pre-trained weights"""
    weights_file = "yolov12n.pt"

    if not os.path.exists(weights_file):
        print("Downloading YOLOv12 pre-trained weights...")
        import urllib.request
        url = "https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12n.pt"
        urllib.request.urlretrieve(url, weights_file)
        print(f"✓ Downloaded: {weights_file}")
    else:
        print(f"✓ Using existing weights: {weights_file}")

    return weights_file

def train_model(data_yaml_path, weights_file):
    """Train the YOLOv12 model"""

    # Load pre-trained model
    model = YOLO(weights_file)
    print(f"✓ Loaded model: {weights_file}")

    # Training configuration - optimized for RTX 3090
    epochs = 100
    batch_size = 16
    image_size = 640
    patience = 20

    # Output directory
    save_dir = r"C:\NTNU\Custom_Obj_Det\yolov12_training"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n🚀 Starting training with configuration:")
    print(f"   • Epochs: {epochs}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Image size: {image_size}")
    print(f"   • Patience: {patience}")
    print(f"   • Save directory: {save_dir}")

    # Start training
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        patience=patience,
        scale=0.5,           # Model-specific augmentation parameter
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.1,
        project=save_dir,
        name='yolov12n_vehicle_detection',
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",   # Optimizer for attention-based models
        cos_lr=True,         # Cosine learning rate schedule
        lr0=0.001,           # Initial learning rate
        lrf=0.01,            # Final learning rate ratio
        weight_decay=0.0001, # Weight decay
        warmup_epochs=3,     # Warmup epochs
        close_mosaic=10,     # Disable mosaic augmentation in final epochs
        seed=0,              # For reproducibility
        device=0,            # Use GPU 0 (RTX 3090)
    )

    print("✅ Training completed!")
    return model, save_dir

def validate_model(model, data_yaml_path):
    """Validate the trained model"""
    print("\n📊 Validating model...")
    metrics = model.val(data=data_yaml_path)
    print(f"✓ Validation completed")
    return metrics

def export_model(model, save_dir, image_size):
    """Export model to ONNX format"""
    print("\n📦 Exporting model to ONNX...")

    # Get paths
    weights_dir = os.path.join(save_dir, 'yolov12n_vehicle_detection', 'weights')
    best_pt = os.path.join(weights_dir, 'best.pt')

    # Export to ONNX format
    model.export(format="onnx", imgsz=image_size)

    # Print paths
    best_onnx = os.path.join(weights_dir, 'best.onnx')
    print(f"✓ PyTorch model: {best_pt}")
    print(f"✓ ONNX model: {best_onnx}")

    return best_pt, best_onnx

def test_inference(model, dataset_path):
    """Run inference on a test image"""
    print("\n🔍 Testing inference...")

    # Find test images
    test_image_dir = os.path.join(dataset_path, 'test', 'images')
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend(glob.glob(os.path.join(test_image_dir, ext)))

    if test_images:
        # Use the first test image found
        test_image_path = test_images[0]
        print(f"Running inference on: {os.path.basename(test_image_path)}")

        # Perform inference
        results = model.predict(test_image_path, save=True, conf=0.25)
        print(f"✓ Results saved to: {results[0].save_dir}")

        # Print detection results
        for r in results:
            print(f"  Detected {len(r.boxes)} objects")
            if len(r.boxes) > 0:
                for i, box in enumerate(r.boxes):
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    print(f"    Object {i+1}: Class {cls}, Confidence: {conf:.3f}")

        return results[0].save_dir
    else:
        print(f"⚠ No test images found in {test_image_dir}")
        return None

def main():
    """Main execution function"""
    print("YOLOv12 Training Script for RTX 3090")
    print("=" * 50)

    try:
        # Check if ultralytics is available
        try:
            from ultralytics import YOLO
            print("✓ YOLOv12 detected. Proceeding with training...")
        except ImportError:
            print("❌ Error: ultralytics/YOLOv12 not found!")
            print("\nPlease install YOLOv12 first:")
            print("1. git clone https://github.com/sunsmarterjie/yolov12.git")
            print("2. cd yolov12")
            print("3. pip install -e .")
            print("\nThen run this script again.")
            return

        # Step 1: Setup dataset
        data_yaml_path, dataset_path = setup_dataset()

        # Step 2: Download pre-trained weights
        weights_file = download_pretrained_weights()

        # Step 3: Train model
        model, save_dir = train_model(data_yaml_path, weights_file)

        # Step 4: Validate model
        validate_model(model, data_yaml_path)

        # Step 5: Export model
        best_pt, best_onnx = export_model(model, save_dir, 640)

        # Step 6: Test inference
        inference_dir = test_inference(model, dataset_path)

        # Final summary
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETE!")
        print("=" * 60)
        print(f"📁 Model files saved in: {save_dir}")
        print(f"🏆 Best weights: {best_pt}")
        print(f"📦 ONNX model: {best_onnx}")
        if inference_dir:
            print(f"🔍 Test results: {inference_dir}")

        print("\n✨ Your YOLOv12 model is ready for use!")
        print("Example usage:")
        print("  from ultralytics import YOLO")
        print(f"  model = YOLO('{best_pt}')")
        print("  results = model.predict('image.jpg', save=True, conf=0.25)")

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()