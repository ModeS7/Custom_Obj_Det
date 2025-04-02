import os
import time
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import io
from matplotlib import pyplot as plt

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
CONFIG = {
    'batch_size': 4,  # Adjust based on GPU memory
    'num_workers': 4,  # Number of worker threads for data loading
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'epochs': 50,
    'seed': 42,
    'img_size': 640,  # Common size for object detection
    'data_dir': 'datasets/x3_666',  # Update with your path
    'checkpoint_dir': 'checkpoints',
    'checkpoint_path': None,  # Path to checkpoint if resuming training
    'model_save_path': 'models/best_model.pth',
    'use_mixed_precision': True,  # Use mixed precision training for speed and memory efficiency
    'tensorboard_dir': 'logs/object_detection',  # Directory for TensorBoard logs
    'log_images_interval': 100,  # Log images to TensorBoard every N batches
    'visualize_model_graph': True,  # Visualize model graph in TensorBoard
}

# Set random seeds for reproducibility
pl.seed_everything(CONFIG['seed'])


# Dataset class for YOLO format (Roboflow typically exports in this format)
class RoboflowDataset(Dataset):
    def __init__(self, data_dir, img_size=640, transforms=None, subset='train'):
        self.data_dir = data_dir
        self.img_size = img_size
        self.transforms = transforms
        self.subset = subset

        # Load dataset configuration
        with open(os.path.join(data_dir, 'data.yaml'), 'r') as f:
            self.yaml = yaml.safe_load(f)

        self.class_names = self.yaml['names']
        self.num_classes = len(self.class_names)

        # Set paths based on subset
        if subset == 'train':
            img_dir = os.path.join(data_dir, 'train', 'images')
            label_dir = os.path.join(data_dir, 'train', 'labels')
        elif subset == 'valid':
            img_dir = os.path.join(data_dir, 'valid', 'images')
            label_dir = os.path.join(data_dir, 'valid', 'labels')
        else:
            img_dir = os.path.join(data_dir, 'test', 'images')
            label_dir = os.path.join(data_dir, 'test', 'labels')

        # Get image and label file paths
        self.img_files = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.label_files = [os.path.join(label_dir, os.path.splitext(os.path.basename(img))[0] + '.txt')
                            for img in self.img_files]

        print(f"Loaded {len(self.img_files)} images for {subset} set")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.img_files[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get image dimensions
        height, width = image.shape[:2]

        # Load labels (YOLO format)
        label_path = self.label_files[idx]
        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    # YOLO format: class_id, x_center, y_center, width, height (normalized)
                    class_id = int(data[0])
                    x_center = float(data[1]) * width
                    y_center = float(data[2]) * height
                    box_width = float(data[3]) * width
                    box_height = float(data[4]) * height

                    # Convert to (x_min, y_min, x_max, y_max) format for PyTorch
                    x_min = x_center - box_width / 2
                    y_min = y_center - box_height / 2
                    x_max = x_center + box_width / 2
                    y_max = y_center + box_height / 2

                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)

        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # Apply transformations if any
        if self.transforms and len(boxes) > 0:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.tensor(transformed['labels'], dtype=torch.int64)
        else:
            # Convert to tensors
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # Create target dict
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        # Handle empty boxes
        if len(boxes) == 0:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros(0, dtype=torch.int64)

        return image, target


# Define transforms
def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(height=CONFIG['img_size'], width=CONFIG['img_size']),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(height=CONFIG['img_size'], width=CONFIG['img_size']),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


# Collate function for data loader
def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images), targets


# Create model
def get_model(num_classes):
    # Load pre-trained Faster R-CNN model
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # Replace the classifier with a new one for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# Lightning Module for training
class ObjectDetectionModule(pl.LightningModule):
    def __init__(self, model, config, class_names=None):
        super().__init__()
        self.model = model
        self.config = config
        self.best_map = 0.0
        self.class_names = class_names
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def on_fit_start(self):
        # Log model graph to TensorBoard
        if self.logger:
            dummy_input = torch.zeros(1, 3, self.config['img_size'], self.config['img_size'], device=self.device)
            self.logger.experiment.add_graph(self.model, dummy_input)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        # Convert targets to the format expected by the model
        target_list = []
        for t in targets:
            target_dict = {}
            target_dict['boxes'] = t['boxes']
            target_dict['labels'] = t['labels']
            target_list.append(target_dict)

        # Forward pass
        loss_dict = self.model(images, target_list)

        # Sum all losses
        losses = sum(loss for loss in loss_dict.values())

        # Log losses
        self.log('train_loss', losses, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f'train_{k}', v, on_step=True, on_epoch=True)

        # Log learning rate
        opt = self.optimizers()
        if isinstance(opt, list):
            opt = opt[0]
        self.log('learning_rate', opt.param_groups[0]['lr'], on_step=True, on_epoch=False)

        # Periodically log sample images with detections to TensorBoard (every 100 batches)
        if batch_idx % 100 == 0:
            self._log_images(images, targets, prefix='train', max_images=4)

        return losses

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        # Convert targets to the format expected by the model
        target_list = []
        for t in targets:
            target_dict = {}
            target_dict['boxes'] = t['boxes']
            target_dict['labels'] = t['labels']
            target_list.append(target_dict)

        # Forward pass in eval mode
        self.model.eval()
        with torch.no_grad():
            loss_dict = self.model(images, target_list)

            # Get predictions for visualization
            predictions = self.model(images)

        # Sum all losses
        losses = sum(loss for loss in loss_dict.values())

        # Log losses
        self.log('val_loss', losses, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f'val_{k}', v, on_epoch=True)

        # Store outputs for epoch end processing
        self.validation_step_outputs.append((losses, images, targets, predictions))

        return losses

    def on_validation_epoch_end(self):
        # Log sample validation images with predictions
        if len(self.validation_step_outputs) > 0:
            sample_idx = np.random.randint(0, len(self.validation_step_outputs))
            _, images, targets, predictions = self.validation_step_outputs[sample_idx]
            self._log_images(images, targets, predictions=predictions, prefix='val', max_images=8)

        # Clear outputs
        self.validation_step_outputs.clear()

    def _log_images(self, images, targets, predictions=None, prefix='', max_images=4):
        """Log images with bounding boxes to TensorBoard"""
        if not isinstance(self.logger, TensorBoardLogger):
            return

        # Denormalize images for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(images.device)
        images_denorm = images * std + mean
        images_denorm = torch.clamp(images_denorm, 0, 1)

        num_images = min(max_images, len(images))
        fig, axs = plt.subplots(1, num_images, figsize=(20, 5))
        if num_images == 1:
            axs = [axs]

        for i in range(num_images):
            img = images_denorm[i].permute(1, 2, 0).cpu().numpy()
            ax = axs[i]
            ax.imshow(img)

            # Draw ground truth boxes
            if targets and i < len(targets):
                boxes = targets[i]['boxes'].cpu().numpy()
                labels = targets[i]['labels'].cpu().numpy()

                for box, label in zip(boxes, labels):
                    x1, y1, x2, y2 = box
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='green', linewidth=2)
                    ax.add_patch(rect)

                    # Add label if class names are available
                    if self.class_names and label < len(self.class_names):
                        class_name = self.class_names[label]
                        ax.text(x1, y1, class_name, color='white', backgroundcolor='green', fontsize=8)

            # Draw predicted boxes if provided
            if predictions and i < len(predictions):
                pred_boxes = predictions[i]['boxes'].cpu().numpy()
                pred_scores = predictions[i]['scores'].cpu().numpy()
                pred_labels = predictions[i]['labels'].cpu().numpy()

                # Filter by confidence threshold
                threshold = 0.5
                indices = pred_scores > threshold

                for box, score, label in zip(pred_boxes[indices], pred_scores[indices], pred_labels[indices]):
                    x1, y1, x2, y2 = box
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)

                    # Add label and score
                    label_text = f"Class {label}"
                    if self.class_names and label < len(self.class_names):
                        label_text = self.class_names[label]
                    ax.text(x1, y1 - 10, f"{label_text}: {score:.2f}", color='white', backgroundcolor='red', fontsize=8)

            ax.axis('off')

        # Log to TensorBoard
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = plt.imread(buf)
        plt.close(fig)

        # Convert to tensor and log
        img_tensor = torch.tensor(img).permute(2, 0, 1)
        self.logger.experiment.add_image(f'{prefix}_detections', img_tensor, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }


# Main training function
def train():
    # Create checkpoint directory if it doesn't exist
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG['model_save_path']), exist_ok=True)

    # Create transforms
    train_transforms = get_transforms(train=True)
    valid_transforms = get_transforms(train=False)

    # Create datasets
    train_dataset = RoboflowDataset(
        data_dir=CONFIG['data_dir'],
        transforms=train_transforms,
        subset='train'
    )
    valid_dataset = RoboflowDataset(
        data_dir=CONFIG['data_dir'],
        transforms=valid_transforms,
        subset='valid'
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Create model
    num_classes = train_dataset.num_classes + 1  # +1 for background class
    model = get_model(num_classes)

    # Create Lightning module
    detection_module = ObjectDetectionModule(
        model=model,
        config=CONFIG,
        class_names=train_dataset.class_names
    )

    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=CONFIG['checkpoint_dir'],
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        filename='model-{epoch:02d}-{val_loss:.3f}',
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True
    )

    # Create logger
    logger = TensorBoardLogger('logs', name='object_detection')

    # Create LR Monitor callback
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=CONFIG['epochs'],
        precision=16 if CONFIG['use_mixed_precision'] else 32,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1
    )

    # Add custom TensorBoard summary with dataset statistics
    tb_logger = logger.experiment

    # Log dataset info
    tb_logger.add_text('Dataset Info', f"Original images: 264, Total with augmentation: 666", 0)
    tb_logger.add_text('Classes', f"Number of classes: {train_dataset.num_classes}", 0)
    class_names_str = "\n".join([f"{i}: {name}" for i, name in enumerate(train_dataset.class_names)])
    tb_logger.add_text('Class Names', class_names_str, 0)

    # Log model summary
    model_info = f"Model: Faster R-CNN with ResNet-50 backbone\n"
    model_info += f"Input size: {CONFIG['img_size']}x{CONFIG['img_size']}\n"
    model_info += f"Batch size: {CONFIG['batch_size']}\n"
    model_info += f"Learning rate: {CONFIG['learning_rate']}"
    tb_logger.add_text('Model Info', model_info, 0)

    # Log hardware info
    hardware_info = f"GPU: GTX 1080 Ti\n"
    hardware_info += f"Mixed precision: {CONFIG['use_mixed_precision']}"
    tb_logger.add_text('Hardware Info', hardware_info, 0)

    # Sample a few images from the training set with annotations for visualization
    sample_idx = np.random.randint(0, len(train_dataset), size=4)
    sample_images = []
    for idx in sample_idx:
        image, target = train_dataset[idx]
        # Convert to numpy for visualization
        img_np = image.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)

        # Draw bounding boxes
        img_with_boxes = img_np.copy()
        for box, label in zip(target['boxes'], target['labels']):
            x1, y1, x2, y2 = box.numpy().astype(int)
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_name = train_dataset.class_names[label] if label < len(
                train_dataset.class_names) else f"Class {label}"
            cv2.putText(img_with_boxes, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        sample_images.append(img_with_boxes)

    # Create a grid of images
    sample_grid = torchvision.utils.make_grid([torch.from_numpy(img).permute(2, 0, 1) for img in sample_images])
    tb_logger.add_image('Sample Training Images', sample_grid, 0)

    print("Starting training... TensorBoard logs will be saved to 'logs/object_detection'")
    print("To view TensorBoard, run: tensorboard --logdir=logs/object_detection")

    # Train model
    if CONFIG['checkpoint_path']:
        print(f"Resuming training from checkpoint: {CONFIG['checkpoint_path']}")
        trainer.fit(detection_module, train_loader, valid_loader, ckpt_path=CONFIG['checkpoint_path'])
    else:
        trainer.fit(detection_module, train_loader, valid_loader)

    # Save the best model
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Best model saved to {best_model_path}")
        # You can also save just the model weights if you prefer
        best_model = ObjectDetectionModule.load_from_checkpoint(
            best_model_path,
            model=model,
            config=CONFIG,
            class_names=train_dataset.class_names
        )
        torch.save(best_model.model.state_dict(), CONFIG['model_save_path'])
        print(f"Model weights saved to {CONFIG['model_save_path']}")

    print("\nTraining completed! To analyze results, run:")
    print("tensorboard --logdir=logs/object_detection")


if __name__ == "__main__":
    train()