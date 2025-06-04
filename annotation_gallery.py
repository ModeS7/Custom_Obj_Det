import cv2
import numpy as np
import onnxruntime as ort
import time
import sys
import os
import glob
from pathlib import Path
import yaml
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QSlider, QFileDialog, QComboBox, QTextEdit,
                             QProgressBar, QFrame, QSizePolicy, QScrollArea, QListWidget,
                             QListWidgetItem, QSplitter, QGroupBox, QGridLayout, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSettings
from PIL import Image

# Constants
AVAILABLE_MODELS = ["yolo12.onnx", "yolo8.onnx", "yolo3.onnx"]
DEFAULT_MODEL = AVAILABLE_MODELS[0]
DEFAULT_CONF_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_TILE_OVERLAP = 10
INPUT_SIZE = 640

# Vehicle class mapping (extend as needed)
CLASS_NAMES = {
    0: "vehicle"  # Generic vehicle class - modify based on your specific needs
}


class DetectionThread(QThread):
    """Thread for running tiled object detection on images"""
    detection_complete = pyqtSignal(list, float, object)
    model_loaded = pyqtSignal(str)

    def __init__(self, model_path):
        super().__init__()
        self.running = True
        self.frame = None
        self.new_frame = False
        self.new_model = False
        self.conf_threshold = DEFAULT_CONF_THRESHOLD
        self.iou_threshold = DEFAULT_IOU_THRESHOLD
        self.tile_overlap = DEFAULT_TILE_OVERLAP
        self.model_path = model_path
        self.new_model_path = None
        self.current_image_info = None
        self.setup_model()

    def setup_model(self):
        """Initialize the ONNX model"""
        try:
            self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            print(f"Model loaded: {self.model_path}")
            self.model_loaded.emit(os.path.basename(self.model_path))
            return True
        except Exception as e:
            print(f"Error loading model {self.model_path}: {e}")
            return False

    def set_frame(self, frame, image_info=None):
        """Set a new frame for processing"""
        self.frame = frame
        self.current_image_info = image_info
        self.new_frame = True

    def set_conf_threshold(self, value):
        self.conf_threshold = value

    def set_iou_threshold(self, value):
        self.iou_threshold = value

    def set_tile_overlap(self, value):
        self.tile_overlap = value

    def change_model(self, model_path):
        self.new_model_path = model_path
        self.new_model = True

    def stop(self):
        self.running = False
        self.wait()

    def run(self):
        while self.running:
            # Check for model changes
            if self.new_model:
                self.new_model = False
                if self.new_model_path and self.new_model_path != self.model_path:
                    self.model_path = self.new_model_path
                    self.setup_model()

            # Process frames when available
            if self.new_frame and self.frame is not None:
                self.new_frame = False
                start_time = time.time()

                # Process with tiling
                detections = self.process_tiled_frame()
                fps = 1 / (time.time() - start_time)
                self.detection_complete.emit(detections, fps, self.current_image_info)

            # Sleep to avoid high CPU usage
            time.sleep(0.01)

    def create_tiles(self, frame):
        """Split the frame into multiple tiles with overlap"""
        h, w = frame.shape[:2]

        # Calculate tile dimensions
        tile_width = INPUT_SIZE - self.tile_overlap
        tile_height = INPUT_SIZE - self.tile_overlap

        # Calculate tile counts
        n_tiles_w = int(np.ceil(w / tile_width))
        n_tiles_h = int(np.ceil(h / tile_height))

        tiles = []
        tile_info = []

        # Create tiles
        for row in range(n_tiles_h):
            for col in range(n_tiles_w):
                # Calculate tile coordinates
                x1 = col * tile_width
                y1 = row * tile_height
                x2 = min(x1 + INPUT_SIZE, w)
                y2 = min(y1 + INPUT_SIZE, h)

                # Adjust coordinates to avoid going out of bounds
                if x2 == w:
                    x1 = max(0, w - INPUT_SIZE)
                if y2 == h:
                    y1 = max(0, h - INPUT_SIZE)

                # Extract tile
                tile = frame[y1:y2, x1:x2]

                # Ensure the tile is the right size
                th, tw = tile.shape[:2]
                if th < INPUT_SIZE or tw < INPUT_SIZE:
                    pad_tile = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
                    pad_tile[0:th, 0:tw] = tile
                    tile = pad_tile

                tiles.append(tile)
                tile_info.append({'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1})

        return tiles, tile_info

    def process_tiled_frame(self):
        """Process multiple tiles and combine results"""
        tiles, tile_info = self.create_tiles(self.frame)
        all_detections = []

        # Process each tile
        for i, (tile, info) in enumerate(zip(tiles, tile_info)):
            tile_detections = self.process_frame(tile)

            # Adjust coordinates based on tile position
            for j in range(len(tile_detections)):
                box, score, class_id = tile_detections[j]
                adjusted_box = [
                    box[0] + info['x'],
                    box[1] + info['y'],
                    box[2] + info['x'],
                    box[3] + info['y']
                ]
                tile_detections[j] = (adjusted_box, score, class_id)

            all_detections.extend(tile_detections)

        # Apply global NMS if needed
        if len(all_detections) > 0:
            boxes = np.array([det[0] for det in all_detections])
            scores = np.array([det[1] for det in all_detections])
            class_ids = np.array([det[2] for det in all_detections])

            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
            if len(indices) > 0:
                return [(boxes[i], scores[i], class_ids[i]) for i in indices]

        return []

    def process_frame(self, frame):
        """Process a single frame through the YOLO model"""
        try:
            # Preprocess the frame
            input_tensor, orig_size = self.preprocess(frame)

            # Run inference
            outputs = self.session.run(None, {self.input_name: input_tensor})

            # Postprocess the outputs
            detections = self.postprocess(outputs, orig_size)

            return detections
        except Exception as e:
            print(f"Error processing frame: {e}")
            return []

    def preprocess(self, frame):
        """Preprocess image for YOLO"""
        original_size = frame.shape[:2]
        h, w = frame.shape[:2]

        # Resize while maintaining aspect ratio
        scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        # Create square canvas with border
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        offset_x, offset_y = (INPUT_SIZE - new_w) // 2, (INPUT_SIZE - new_h) // 2
        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

        # Normalize and transpose
        img = canvas.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, 0)  # Add batch dimension

        return img, (h, w, scale, offset_x, offset_y)

    def postprocess(self, outputs, orig_size):
        """Process YOLO outputs to get detections"""
        h, w, scale, offset_x, offset_y = orig_size

        # Extract predictions
        predictions = np.squeeze(outputs[0]).T

        # Filter by confidence
        scores = np.max(predictions[:, 4:], axis=1)
        mask = scores > self.conf_threshold
        predictions = predictions[mask]

        if len(predictions) == 0:
            return []

        # Extract boxes and classes
        boxes = predictions[:, :4]
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        scores = np.max(predictions[:, 4:], axis=1)

        # Convert center format to corner format
        x_center, y_center = boxes[:, 0], boxes[:, 1]
        width, height = boxes[:, 2], boxes[:, 3]

        # Convert to original image coordinates
        x1 = (x_center - width / 2 - offset_x) / scale
        y1 = (y_center - height / 2 - offset_y) / scale
        x2 = (x_center + width / 2 - offset_x) / scale
        y2 = (y_center + height / 2 - offset_y) / scale

        # Clip to image boundaries
        x1 = np.clip(x1, 0, w)
        y1 = np.clip(y1, 0, h)
        x2 = np.clip(x2, 0, w)
        y2 = np.clip(y2, 0, h)

        # Create final boxes
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)

        if len(indices) == 0:
            return []

        # Return detection results
        return [(boxes[i], scores[i], class_ids[i]) for i in indices]


class AnnotationGalleryWindow(QMainWindow):
    """Main window for interactive gallery annotation"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Annotation Gallery")
        self.setMinimumSize(1200, 800)

        # Initialize settings
        self.settings = QSettings("AnnotationGallery", "Settings")

        # App state
        self.current_model = DEFAULT_MODEL
        self.conf_threshold = DEFAULT_CONF_THRESHOLD
        self.iou_threshold = DEFAULT_IOU_THRESHOLD
        self.tile_overlap = DEFAULT_TILE_OVERLAP

        # Image management
        self.image_folder = None
        self.image_files = []
        self.current_image_index = 0
        self.current_image = None
        self.current_detections = []
        self.fps = 0

        # Annotation management
        self.annotations_saved = {}  # Track which images have saved annotations
        self.stored_annotations = {}  # Store actual annotation data per image
        self.current_annotations = []  # Current image annotations in YOLO format

        # Setup UI
        self.setup_ui()

        # Start detection thread
        self.detection_thread = DetectionThread(self.current_model)
        self.detection_thread.detection_complete.connect(self.update_detections)
        self.detection_thread.model_loaded.connect(self.update_model_info)
        self.detection_thread.start()

        # Load settings
        self.load_settings()

        # Center window
        self.center_window()

    def setup_ui(self):
        """Setup the main window UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        central_widget.setLayout(QHBoxLayout())
        central_widget.layout().addWidget(main_splitter)

        # Left panel - Image list and controls
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)

        # Right panel - Image display and detection info
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)

        # Set splitter proportions
        main_splitter.setSizes([300, 900])

    def create_left_panel(self):
        """Create left control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Folder selection
        folder_group = QGroupBox("Dataset Folder")
        folder_layout = QVBoxLayout(folder_group)

        self.folder_label = QLabel("No folder selected")
        self.folder_label.setWordWrap(True)
        folder_layout.addWidget(self.folder_label)

        self.select_folder_btn = QPushButton("Select Image Folder")
        self.select_folder_btn.clicked.connect(self.select_folder)
        folder_layout.addWidget(self.select_folder_btn)

        layout.addWidget(folder_group)

        # Image list
        list_group = QGroupBox("Images")
        list_layout = QVBoxLayout(list_group)

        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.image_selected)
        list_layout.addWidget(self.image_list)

        # Progress info
        self.progress_label = QLabel("0 / 0 images")
        list_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        list_layout.addWidget(self.progress_bar)

        layout.addWidget(list_group)

        # Model controls
        model_group = QGroupBox("Model Settings")
        model_layout = QVBoxLayout(model_group)

        self.model_combo = QComboBox()
        self.model_combo.addItems(AVAILABLE_MODELS)
        self.model_combo.currentTextChanged.connect(self.model_changed)
        model_layout.addWidget(QLabel("Model:"))
        model_layout.addWidget(self.model_combo)

        layout.addWidget(model_group)

        # Parameter controls
        param_group = QGroupBox("Detection Parameters")
        param_layout = QGridLayout(param_group)

        # Confidence threshold
        param_layout.addWidget(QLabel("Confidence:"), 0, 0)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(5)
        self.conf_slider.setMaximum(95)
        self.conf_slider.setValue(int(DEFAULT_CONF_THRESHOLD * 100))
        self.conf_slider.valueChanged.connect(self.conf_changed)
        param_layout.addWidget(self.conf_slider, 0, 1)

        self.conf_label = QLabel(f"{DEFAULT_CONF_THRESHOLD:.2f}")
        param_layout.addWidget(self.conf_label, 0, 2)

        # IoU threshold
        param_layout.addWidget(QLabel("IoU:"), 1, 0)
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setMinimum(5)
        self.iou_slider.setMaximum(95)
        self.iou_slider.setValue(int(DEFAULT_IOU_THRESHOLD * 100))
        self.iou_slider.valueChanged.connect(self.iou_changed)
        param_layout.addWidget(self.iou_slider, 1, 1)

        self.iou_label = QLabel(f"{DEFAULT_IOU_THRESHOLD:.2f}")
        param_layout.addWidget(self.iou_label, 1, 2)

        # Tile overlap
        param_layout.addWidget(QLabel("Tile Overlap:"), 2, 0)
        self.overlap_slider = QSlider(Qt.Horizontal)
        self.overlap_slider.setMinimum(0)
        self.overlap_slider.setMaximum(50)
        self.overlap_slider.setValue(DEFAULT_TILE_OVERLAP)
        self.overlap_slider.valueChanged.connect(self.overlap_changed)
        param_layout.addWidget(self.overlap_slider, 2, 1)

        self.overlap_label = QLabel(f"{DEFAULT_TILE_OVERLAP}px")
        param_layout.addWidget(self.overlap_label, 2, 2)

        layout.addWidget(param_group)

        # Export controls
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)

        self.save_current_btn = QPushButton("Save Current Annotations")
        self.save_current_btn.clicked.connect(self.save_current_annotations)
        self.save_current_btn.setEnabled(False)
        export_layout.addWidget(self.save_current_btn)

        self.export_all_btn = QPushButton("Export All for Roboflow")
        self.export_all_btn.clicked.connect(self.export_all_annotations)
        self.export_all_btn.setEnabled(False)
        export_layout.addWidget(self.export_all_btn)

        layout.addWidget(export_group)

        layout.addStretch()
        return panel

    def create_right_panel(self):
        """Create right display panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Navigation controls
        nav_layout = QHBoxLayout()

        self.prev_btn = QPushButton("◀ Previous")
        self.prev_btn.clicked.connect(self.previous_image)
        self.prev_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)

        self.image_info_label = QLabel("No image loaded")
        self.image_info_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.image_info_label)

        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self.next_image)
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.next_btn)

        layout.addLayout(nav_layout)

        # Image display
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_scroll.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_scroll.setWidget(self.image_label)

        layout.addWidget(self.image_scroll, 1)

        # Detection info
        info_group = QGroupBox("Detection Info")
        info_layout = QVBoxLayout(info_group)

        self.detection_info = QTextEdit()
        self.detection_info.setMaximumHeight(100)
        self.detection_info.setReadOnly(True)
        info_layout.addWidget(self.detection_info)

        layout.addWidget(info_group)

        return panel

    def select_folder(self):
        """Select folder containing images"""
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.load_image_folder(folder)

    def load_image_folder(self, folder_path):
        """Load images from selected folder"""
        self.image_folder = folder_path
        self.folder_label.setText(f"Folder: {os.path.basename(folder_path)}")

        # Find all image files using pathlib for better handling
        folder = Path(folder_path)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

        # Use set to avoid duplicates on case-insensitive filesystems
        image_files_set = set()

        for file_path in folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files_set.add(str(file_path))

        self.image_files = sorted(list(image_files_set))

        # Populate image list
        self.image_list.clear()
        self.annotations_saved = {}
        self.stored_annotations = {}  # Clear stored annotations for new folder

        for i, image_path in enumerate(self.image_files):
            item = QListWidgetItem(os.path.basename(image_path))
            item.setData(Qt.UserRole, image_path)
            self.image_list.addItem(item)
            self.annotations_saved[i] = False

        # Update UI
        self.update_progress()
        self.current_image_index = 0

        if self.image_files:
            self.load_current_image()
            self.export_all_btn.setEnabled(True)

        print(f"Loaded {len(self.image_files)} images from {folder_path}")

    def update_progress(self):
        """Update progress display"""
        total = len(self.image_files)
        saved = sum(self.annotations_saved.values())

        self.progress_label.setText(f"{saved} / {total} annotated")
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(saved)

        # Update list item colors
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if self.annotations_saved.get(i, False):
                item.setBackground(QColor(200, 255, 200))  # Light green for saved
            else:
                item.setBackground(QColor(255, 255, 255))  # White for unsaved

    def image_selected(self, item):
        """Handle image selection from list"""
        image_path = item.data(Qt.UserRole)
        if image_path in self.image_files:
            self.current_image_index = self.image_files.index(image_path)
            self.load_current_image()

    def load_current_image(self):
        """Load and display current image"""
        if not self.image_files or self.current_image_index >= len(self.image_files):
            return

        image_path = self.image_files[self.current_image_index]

        try:
            # Load image
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                print(f"Failed to load image: {image_path}")
                return

            # Convert BGR to RGB for display
            display_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)

            # Update UI
            self.image_info_label.setText(
                f"Image {self.current_image_index + 1} of {len(self.image_files)}: "
                f"{os.path.basename(image_path)} ({self.current_image.shape[1]}x{self.current_image.shape[0]})"
            )

            # Update navigation buttons
            self.prev_btn.setEnabled(self.current_image_index > 0)
            self.next_btn.setEnabled(self.current_image_index < len(self.image_files) - 1)

            # Update list selection
            self.image_list.setCurrentRow(self.current_image_index)

            # Run detection
            image_info = {
                'path': image_path,
                'name': os.path.basename(image_path),
                'width': self.current_image.shape[1],
                'height': self.current_image.shape[0]
            }

            self.detection_thread.set_frame(self.current_image.copy(), image_info)

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    def previous_image(self):
        """Load previous image"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()

    def next_image(self):
        """Load next image"""
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_current_image()

    def update_detections(self, detections, fps, image_info):
        """Update detections and display"""
        self.current_detections = detections
        self.fps = fps

        # Convert detections to YOLO format
        if image_info:
            self.current_annotations = self.convert_to_yolo_format(
                detections, image_info['width'], image_info['height']
            )

        # Update display
        self.update_image_display()

        # Update detection info
        info_text = f"FPS: {fps:.1f}\n"
        info_text += f"Detections: {len(detections)}\n"
        info_text += f"Model: {self.current_model}\n"
        info_text += f"Conf: {self.conf_threshold:.2f}, IoU: {self.iou_threshold:.2f}, Overlap: {self.tile_overlap}px"

        if detections:
            info_text += "\n\nDetected objects:"
            for i, (box, score, class_id) in enumerate(detections):
                class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
                info_text += f"\n{i + 1}. {class_name}: {score:.3f}"

        self.detection_info.setText(info_text)

        # Enable save button if we have detections
        self.save_current_btn.setEnabled(len(detections) > 0)

    def convert_to_yolo_format(self, detections, img_width, img_height):
        """Convert detections to YOLO format (normalized coordinates)"""
        yolo_annotations = []

        for box, score, class_id in detections:
            x1, y1, x2, y2 = box

            # Calculate center and dimensions
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            width = x2 - x1
            height = y2 - y1

            # Normalize coordinates
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            width_norm = width / img_width
            height_norm = height / img_height

            # YOLO format: class_id x_center y_center width height
            yolo_annotations.append(
                f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")

        return yolo_annotations

    def update_image_display(self):
        """Update image display with detection overlays"""
        if self.current_image is None:
            return

        # Create display image with detections
        display_image = self.current_image.copy()
        display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)

        # Draw detections
        for box, score, class_id in self.current_detections:
            x1, y1, x2, y2 = map(int, box)

            # Draw rectangle
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
            label = f"{class_name}: {score:.2f}"

            # Get text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw background rectangle
            cv2.rectangle(display_image, (x1, y1 - text_height - 10),
                          (x1 + text_width, y1), (0, 255, 0), -1)

            # Draw text
            cv2.putText(display_image, label, (x1, y1 - 5),
                        font, font_scale, (0, 0, 0), thickness)

        # Convert to QPixmap and display
        h, w, ch = display_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(display_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Scale image to fit display while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        label_size = self.image_label.size()

        if label_size.width() > 0 and label_size.height() > 0:
            scaled_pixmap = pixmap.scaled(
                label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.setPixmap(pixmap)

    def save_current_annotations(self):
        """Save current image annotations"""
        if not self.current_annotations or not self.image_files:
            return

        try:
            # Store the actual annotation data
            self.stored_annotations[self.current_image_index] = self.current_annotations.copy()

            # Mark as saved
            self.annotations_saved[self.current_image_index] = True
            self.update_progress()

            # Show confirmation
            image_name = os.path.basename(self.image_files[self.current_image_index])
            QMessageBox.information(self, "Saved",
                                    f"Annotations saved for {image_name}\n"
                                    f"({len(self.current_annotations)} detections)")

            print(
                f"Saved annotations for image {self.current_image_index + 1}: {len(self.current_annotations)} detections")

        except Exception as e:
            print(f"Error saving annotations: {e}")
            QMessageBox.warning(self, "Error", f"Failed to save annotations: {e}")

    def export_all_annotations(self):
        """Export all annotations for Roboflow"""
        if not self.image_files:
            QMessageBox.warning(self, "Warning", "No images loaded")
            return

        # Check if we have any saved annotations
        saved_count = sum(self.annotations_saved.values())
        if saved_count == 0:
            QMessageBox.warning(self, "Warning", "No annotations have been saved")
            return

        # Select output folder
        output_folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not output_folder:
            return

        try:
            exported_count = self.create_roboflow_dataset(output_folder)
            QMessageBox.information(self, "Success",
                                    f"Dataset exported successfully!\n"
                                    f"Location: {output_folder}\n"
                                    f"Images with annotations: {exported_count}")

        except Exception as e:
            print(f"Error exporting dataset: {e}")
            QMessageBox.warning(self, "Error", f"Failed to export dataset: {e}")

    def create_roboflow_dataset(self, output_folder):
        """Create dataset structure for Roboflow"""
        # Create folder structure
        images_folder = os.path.join(output_folder, "images")
        labels_folder = os.path.join(output_folder, "labels")

        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(labels_folder, exist_ok=True)

        exported_count = 0

        # Copy images and create annotation files for saved annotations
        for i, image_path in enumerate(self.image_files):
            if not self.annotations_saved.get(i, False):
                continue

            # Copy image
            image_name = os.path.basename(image_path)
            output_image_path = os.path.join(images_folder, image_name)

            # Use PIL to ensure compatibility
            with Image.open(image_path) as img:
                img.save(output_image_path)

            # Create annotation file
            base_name = os.path.splitext(image_name)[0]
            annotation_file = os.path.join(labels_folder, f"{base_name}.txt")

            # Write stored annotations to file
            stored_annotations = self.stored_annotations.get(i, [])
            with open(annotation_file, 'w') as f:
                for annotation_line in stored_annotations:
                    f.write(annotation_line + '\n')

            exported_count += 1
            print(f"Exported {image_name} with {len(stored_annotations)} annotations")

        # Create classes.txt file
        classes_file = os.path.join(output_folder, "classes.txt")
        with open(classes_file, 'w') as f:
            for class_id in sorted(CLASS_NAMES.keys()):
                f.write(f"{CLASS_NAMES[class_id]}\n")

        # Create data.yaml file
        yaml_content = {
            'path': output_folder,
            'train': 'images',
            'val': 'images',  # For now, using same folder
            'names': CLASS_NAMES
        }

        yaml_file = os.path.join(output_folder, "data.yaml")
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        print(f"Dataset exported to {output_folder}")
        print(f"Exported {exported_count} images with annotations")
        return exported_count

    def model_changed(self, model_name):
        """Handle model change"""
        self.current_model = model_name
        self.detection_thread.change_model(model_name)

        # Re-run detection on current image if available
        if self.current_image is not None:
            self.load_current_image()

    def conf_changed(self, value):
        """Handle confidence threshold change"""
        self.conf_threshold = value / 100.0
        self.conf_label.setText(f"{self.conf_threshold:.2f}")
        self.detection_thread.set_conf_threshold(self.conf_threshold)

        # Re-run detection
        if self.current_image is not None:
            self.load_current_image()

    def iou_changed(self, value):
        """Handle IoU threshold change"""
        self.iou_threshold = value / 100.0
        self.iou_label.setText(f"{self.iou_threshold:.2f}")
        self.detection_thread.set_iou_threshold(self.iou_threshold)

        # Re-run detection
        if self.current_image is not None:
            self.load_current_image()

    def overlap_changed(self, value):
        """Handle tile overlap change"""
        self.tile_overlap = value
        self.overlap_label.setText(f"{self.tile_overlap}px")
        self.detection_thread.set_tile_overlap(self.tile_overlap)

        # Re-run detection
        if self.current_image is not None:
            self.load_current_image()

    def update_model_info(self, model_name):
        """Update UI after model is loaded"""
        self.current_model = model_name
        self.setWindowTitle(f"Interactive Annotation Gallery - Model: {model_name}")

    def center_window(self):
        """Center window on screen"""
        screen = QApplication.desktop().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2,
                  (screen.height() - size.height()) // 2)

    def load_settings(self):
        """Load settings from previous session"""
        # Load last used folder
        last_folder = self.settings.value("last_folder", "")
        if last_folder and os.path.exists(last_folder):
            self.load_image_folder(last_folder)

        # Load parameter values
        self.conf_threshold = float(self.settings.value("conf_threshold", DEFAULT_CONF_THRESHOLD))
        self.iou_threshold = float(self.settings.value("iou_threshold", DEFAULT_IOU_THRESHOLD))
        self.tile_overlap = int(self.settings.value("tile_overlap", DEFAULT_TILE_OVERLAP))

        # Update UI
        self.conf_slider.setValue(int(self.conf_threshold * 100))
        self.iou_slider.setValue(int(self.iou_threshold * 100))
        self.overlap_slider.setValue(self.tile_overlap)

    def save_settings(self):
        """Save settings for next session"""
        if self.image_folder:
            self.settings.setValue("last_folder", self.image_folder)

        self.settings.setValue("conf_threshold", self.conf_threshold)
        self.settings.setValue("iou_threshold", self.iou_threshold)
        self.settings.setValue("tile_overlap", self.tile_overlap)

    def closeEvent(self, event):
        """Handle window close"""
        self.save_settings()

        if hasattr(self, 'detection_thread'):
            self.detection_thread.stop()

        super().closeEvent(event)


def main():
    """Main function"""
    app = QApplication(sys.argv)
    app.setApplicationName("Interactive Annotation Gallery")
    app.setOrganizationName("AnnotationTools")

    window = AnnotationGalleryWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()