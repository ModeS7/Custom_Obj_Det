import cv2
import numpy as np
import onnxruntime as ort
import time
import sys
import argparse
import dxcam
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QComboBox, QPushButton,
                             QVBoxLayout, QLabel, QDialog, QHBoxLayout, QRadioButton,
                             QButtonGroup, QListWidget, QListWidgetItem, QFrame)
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QImage, QPixmap, QCursor
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QRect, QSize, QPoint
import threading
import pygetwindow as gw  # For window management
import os

# Available models
AVAILABLE_MODELS = ["yolo12.onnx", "yolo8.onnx", "yolo3.onnx"]
DEFAULT_MODEL = AVAILABLE_MODELS[0]

# Configuration
DEFAULT_CONF_THRESHOLD = 0.8
DEFAULT_IOU_THRESHOLD = 0.8
INPUT_SIZE = 640  # YOLO input size
TILE_OVERLAP = 50  # Overlap between tiles in pixels

# Capture modes
CAPTURE_MODE_DISPLAY = 0  # Capture entire display
CAPTURE_MODE_WINDOW = 1  # Capture specific window

# Detection modes
MODE_SINGLE_FRAME = 0  # Original mode - resize entire frame
MODE_TILED = 1  # New mode - split into tiles


class WindowOutlineWidget(QWidget):
    """Widget to draw an outline around the selected window"""

    def __init__(self, rect, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.rect = rect
        self.setGeometry(self.rect)

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(QColor(0, 255, 0), 3)  # Green, 3px wide
        painter.setPen(pen)
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)


class WindowSelectionDialog(QDialog):
    """Dialog for selecting a specific window for capture"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Window")
        self.selected_window = None
        self.outline_widget = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Instructions
        instructions = QLabel("Select a window to capture:")
        layout.addWidget(instructions)

        # Window list
        self.window_list = QListWidget()
        self.populate_window_list()
        self.window_list.currentRowChanged.connect(self.preview_window)
        layout.addWidget(self.window_list)

        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("Select")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.resize(400, 500)

    def populate_window_list(self):
        """Populate the list with visible windows"""
        self.windows = []

        # Get all visible windows
        for window in gw.getAllWindows():
            if window.visible and window.title:
                self.windows.append(window)
                item = QListWidgetItem(f"{window.title} ({window.width}x{window.height})")
                self.window_list.addItem(item)

    def preview_window(self, index):
        """Show preview outline around the selected window"""
        if index < 0 or index >= len(self.windows):
            return

        # Remove previous outline if exists
        if self.outline_widget:
            self.outline_widget.close()
            self.outline_widget = None

        # Get window info
        window = self.windows[index]
        rect = QRect(window.left, window.top, window.width, window.height)

        # Create outline widget
        self.outline_widget = WindowOutlineWidget(rect)
        self.outline_widget.show()

    def get_selected_window(self):
        """Return the selected window object"""
        index = self.window_list.currentRow()
        if index >= 0 and index < len(self.windows):
            return self.windows[index]
        return None

    def closeEvent(self, event):
        """Clean up when dialog is closed"""
        if self.outline_widget:
            self.outline_widget.close()
        super().closeEvent(event)


class TargetSelectionDialog(QDialog):
    """Initial dialog for selecting capture target (display or window)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Capture Target")
        self.selected_display = None
        self.selected_window = None
        self.capture_mode = CAPTURE_MODE_DISPLAY

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Mode selection
        mode_group = QButtonGroup(self)
        mode_box = QFrame()
        mode_layout = QVBoxLayout(mode_box)

        self.display_radio = QRadioButton("Capture Display")
        self.display_radio.setChecked(True)
        self.display_radio.toggled.connect(self.update_mode)
        mode_group.addButton(self.display_radio)
        mode_layout.addWidget(self.display_radio)

        self.window_radio = QRadioButton("Capture Browser Window")
        self.window_radio.toggled.connect(self.update_mode)
        mode_group.addButton(self.window_radio)
        mode_layout.addWidget(self.window_radio)

        layout.addWidget(mode_box)

        # Display selection (initially visible)
        self.display_frame = QFrame()
        display_layout = QVBoxLayout(self.display_frame)

        display_label = QLabel("Select display:")
        display_layout.addWidget(display_label)

        self.display_combo = QComboBox()
        self.populate_displays()
        display_layout.addWidget(self.display_combo)

        layout.addWidget(self.display_frame)

        # Window selection button (initially hidden)
        self.window_frame = QFrame()
        window_layout = QVBoxLayout(self.window_frame)

        self.window_info = QLabel("No window selected")
        window_layout.addWidget(self.window_info)

        self.select_window_button = QPushButton("Select Window")
        self.select_window_button.clicked.connect(self.open_window_selection)
        window_layout.addWidget(self.select_window_button)

        layout.addWidget(self.window_frame)
        self.window_frame.hide()

        # OK/Cancel buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.resize(400, 300)

    def update_mode(self):
        """Update UI based on selected mode"""
        if self.display_radio.isChecked():
            self.capture_mode = CAPTURE_MODE_DISPLAY
            self.display_frame.show()
            self.window_frame.hide()
        else:
            self.capture_mode = CAPTURE_MODE_WINDOW
            self.display_frame.hide()
            self.window_frame.show()

    def populate_displays(self):
        """Populate the dropdown with available displays"""
        app = QApplication.instance()
        screens = app.screens()

        for i, screen in enumerate(screens):
            display_name = f"Display {i + 1}: {screen.name()} ({screen.geometry().width()}x{screen.geometry().height()})"
            self.display_combo.addItem(display_name, i)

    def open_window_selection(self):
        """Open the window selection dialog"""
        dialog = WindowSelectionDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            window = dialog.get_selected_window()
            if window:
                self.selected_window = window
                self.window_info.setText(f"Selected: {window.title} ({window.width}x{window.height})")

    def get_target_info(self):
        """Return selected target information"""
        if self.capture_mode == CAPTURE_MODE_DISPLAY:
            return {
                'mode': CAPTURE_MODE_DISPLAY,
                'display_idx': self.display_combo.currentData()
            }
        else:
            if not self.selected_window:
                # If no window selected, default to the first display
                return {
                    'mode': CAPTURE_MODE_DISPLAY,
                    'display_idx': 0
                }
            return {
                'mode': CAPTURE_MODE_WINDOW,
                'window': self.selected_window
            }


class ModelSelectionDialog(QDialog):
    """Dialog for selecting model to use"""

    def __init__(self, current_model, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Model")
        self.current_model = current_model
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Instructions
        instructions = QLabel("Select YOLO model to use:")
        layout.addWidget(instructions)

        # Model list
        self.model_list = QListWidget()
        self.populate_model_list()
        layout.addWidget(self.model_list)

        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("Select")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.resize(300, 200)

    def populate_model_list(self):
        """Populate the list with available models"""
        for model_name in AVAILABLE_MODELS:
            item = QListWidgetItem(model_name)
            self.model_list.addItem(item)

            # Select current model in the list
            if model_name == self.current_model:
                self.model_list.setCurrentItem(item)

    def get_selected_model(self):
        """Return the selected model name"""
        current_row = self.model_list.currentRow()
        if current_row >= 0 and current_row < len(AVAILABLE_MODELS):
            return AVAILABLE_MODELS[current_row]
        return DEFAULT_MODEL


class DetectionThread(QThread):
    """Thread for running object detection"""
    detection_complete = pyqtSignal(list, float)
    model_loaded = pyqtSignal(str)

    def __init__(self, model_path, use_cuda=False):
        super().__init__()
        self.running = True
        self.frame_ready = threading.Event()
        self.model_change = threading.Event()
        self.frame = None
        self.conf_threshold = DEFAULT_CONF_THRESHOLD
        self.iou_threshold = DEFAULT_IOU_THRESHOLD
        self.use_cuda = use_cuda
        self.model_path = model_path
        self.new_model_path = None
        self.setup_model()
        self.tile_info = None  # Store tile coordinates
        self.detection_mode = MODE_SINGLE_FRAME  # Default to single frame mode

    def setup_model(self):
        """Initialize the ONNX model"""
        providers = ['CPUExecutionProvider']

        if self.use_cuda and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print(f"Using CUDA with model: {self.model_path}")
        else:
            print(f"Using CPU with model: {self.model_path}")

        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            print(f"Model loaded: {self.model_path}")
            self.model_loaded.emit(os.path.basename(self.model_path))
            return True
        except Exception as e:
            print(f"Error loading model {self.model_path}: {e}")
            return False

    def set_frame(self, frame, tile_info=None):
        """Set a new frame for processing"""
        self.frame = frame
        self.tile_info = tile_info
        self.frame_ready.set()

    def set_conf_threshold(self, value):
        """Set confidence threshold"""
        self.conf_threshold = value

    def set_iou_threshold(self, value):
        """Set IoU threshold"""
        self.iou_threshold = value

    def change_model(self, model_path):
        """Change the model being used"""
        self.new_model_path = model_path
        self.model_change.set()

    def set_detection_mode(self, mode):
        """Set detection mode"""
        self.detection_mode = mode

    def stop(self):
        """Stop the thread"""
        self.running = False
        self.frame_ready.set()
        self.model_change.set()
        self.wait()

    def run(self):
        """Main thread loop"""
        while self.running:
            # Check for model change requests
            if self.model_change.is_set():
                self.model_change.clear()
                if self.new_model_path and self.new_model_path != self.model_path:
                    print(f"Changing model from {self.model_path} to {self.new_model_path}")
                    self.model_path = self.new_model_path
                    self.setup_model()

            # Wait for a frame to process
            ready = threading.Event.wait(self.frame_ready, timeout=0.1)
            if not ready:
                continue  # No frame ready, check for model changes

            self.frame_ready.clear()

            if not self.running or self.frame is None:
                continue

            start_time = time.time()

            # Process frame(s) based on mode
            if self.detection_mode == MODE_TILED and isinstance(self.frame, list):
                # Tiled mode - process each tile and combine results
                all_detections = []

                for i, (tile, tile_info) in enumerate(zip(self.frame, self.tile_info)):
                    tile_detections = self.process_frame(tile)

                    # Adjust detection coordinates to the original image space
                    for j in range(len(tile_detections)):
                        box, score, class_id = tile_detections[j]
                        # Adjust coordinates based on tile position
                        adjusted_box = [
                            box[0] + tile_info['x'],
                            box[1] + tile_info['y'],
                            box[2] + tile_info['x'],
                            box[3] + tile_info['y']
                        ]
                        tile_detections[j] = (adjusted_box, score, class_id)

                    all_detections.extend(tile_detections)

                # Apply global NMS to remove duplicate detections across tiles
                if len(all_detections) > 0:
                    boxes = np.array([det[0] for det in all_detections])
                    scores = np.array([det[1] for det in all_detections])
                    class_ids = np.array([det[2] for det in all_detections])

                    # Apply NMS with current IOU threshold
                    indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)

                    if len(indices) > 0:
                        all_detections = [(boxes[i], scores[i], class_ids[i]) for i in indices]
                    else:
                        all_detections = []
            else:
                # Single frame mode - process the entire frame
                all_detections = self.process_frame(self.frame)

            fps = 1 / (time.time() - start_time)
            self.detection_complete.emit(all_detections, fps)

    def process_frame(self, frame):
        """Process a frame through the model"""
        try:
            # Preprocessing
            input_tensor, orig_size = self.preprocess(frame)

            # Run inference
            outputs = self.session.run(None, {self.input_name: input_tensor})

            # Postprocessing
            detections = self.postprocess(outputs, orig_size)

            return detections
        except Exception as e:
            print(f"Error processing frame: {e}")
            return []

    def preprocess(self, frame):
        """Preprocessing for YOLO model"""
        original_size = frame.shape[:2]
        h, w = frame.shape[:2]

        # Calculate scale to fit INPUT_SIZE
        scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize frame
        resized = cv2.resize(frame, (new_w, new_h))

        # Create square canvas
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        offset_x, offset_y = (INPUT_SIZE - new_w) // 2, (INPUT_SIZE - new_h) // 2
        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

        # Normalize and convert to correct format
        img = canvas.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, 0)  # Add batch dimension

        return img, (h, w, scale, offset_x, offset_y)

    def postprocess(self, outputs, orig_size):
        """Postprocessing for YOLO outputs"""
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

        # Apply NMS with current IOU threshold
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)

        if len(indices) == 0:
            return []

        # Return detection results
        return [(boxes[i], scores[i], class_ids[i]) for i in indices]


class DetectionWindow(QMainWindow):
    """Separate window for displaying detections"""

    def __init__(self, use_cuda=False):
        super().__init__()
        self.use_cuda = use_cuda

        # Set window title and properties
        self.setWindowTitle("Vehicle Detection")
        self.setMinimumSize(800, 600)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create image display label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.image_label)

        # Create info panel
        info_panel = QWidget()
        info_layout = QHBoxLayout(info_panel)

        # Info display
        self.info_label = QLabel()
        self.info_label.setFont(QFont("Arial", 10))
        info_layout.addWidget(self.info_label)

        # Controls panel
        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)

        # Add confidence threshold slider
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Confidence:"))
        self.conf_threshold = DEFAULT_CONF_THRESHOLD
        controls_layout.addLayout(confidence_layout)

        # Add IoU threshold slider
        iou_layout = QHBoxLayout()
        iou_layout.addWidget(QLabel("IoU:"))
        self.iou_threshold = DEFAULT_IOU_THRESHOLD
        controls_layout.addLayout(iou_layout)

        # Add controls description
        controls_help = QLabel(
            "Controls:\n"
            "  ↑/↓: Adjust confidence\n"
            "  ←/→: Adjust IoU threshold\n"
            "  T: Toggle tiled mode\n"
            "  L: Toggle labels\n"
            "  M: Change model\n"
            "  C: Change capture target\n"
            "  Q: Quit"
        )
        controls_layout.addWidget(controls_help)

        info_layout.addWidget(controls_panel)
        main_layout.addWidget(info_panel)

        # Current model
        self.current_model = DEFAULT_MODEL

        # Current detection mode
        self.detection_mode = MODE_SINGLE_FRAME

        # Show confidence labels flag
        self.show_labels = False

        # Initialize state
        self.detections = []
        self.current_frame = None
        self.fps = 0
        self.tile_count = 0

        # Capture state
        self.capture_mode = None
        self.display_idx = None
        self.target_window = None
        self.camera = None
        self.window_rect = None
        self.monitor_info = None

        # Window monitor timer (for browser window mode)
        self.window_monitor_timer = QTimer(self)
        self.window_monitor_timer.timeout.connect(self.check_window_changes)

        # Get monitor information
        self.get_monitor_info()

        # Initial target selection
        if not self.select_initial_target():
            print("No target selected. Exiting.")
            sys.exit(0)

        # Start detection thread
        self.detection_thread = DetectionThread(self.current_model, use_cuda)
        self.detection_thread.detection_complete.connect(self.update_detections)
        self.detection_thread.model_loaded.connect(self.update_model_info)
        self.detection_thread.start()

        # Start capture timer
        self.capture_timer = QTimer(self)
        self.capture_timer.timeout.connect(self.capture_frame)
        self.capture_timer.start(33)  # ~30 FPS

        # Position the window strategically
        self.position_window_strategically()

    def update_model_info(self, model_name):
        """Update UI after model is loaded"""
        self.current_model = model_name
        # Update window title to include model name
        self.setWindowTitle(f"Vehicle Detection - Model: {model_name}")

    def select_model(self):
        """Open dialog to select a new model"""
        # Pause capture
        self.capture_timer.stop()

        dialog = ModelSelectionDialog(self.current_model, self)
        if dialog.exec_() == QDialog.Accepted:
            selected_model = dialog.get_selected_model()
            if selected_model != self.current_model:
                print(f"Changing model to {selected_model}")
                # Send model change request to detection thread
                self.detection_thread.change_model(selected_model)

        # Resume capture
        self.capture_timer.start()

    def cycle_model(self):
        """Cycle to the next available model"""
        # Find current model index
        current_index = AVAILABLE_MODELS.index(self.current_model)

        # Get next model (with wraparound)
        next_index = (current_index + 1) % len(AVAILABLE_MODELS)
        next_model = AVAILABLE_MODELS[next_index]

        print(f"Cycling model from {self.current_model} to {next_model}")
        self.detection_thread.change_model(next_model)

    def get_monitor_info(self):
        """Get information about all monitors"""
        app = QApplication.instance()
        screens = app.screens()

        self.monitor_info = []

        for screen in screens:
            geometry = screen.geometry()
            self.monitor_info.append({
                'width': geometry.width(),
                'height': geometry.height(),
                'left': geometry.left(),
                'top': geometry.top(),
                'right': geometry.right(),
                'bottom': geometry.bottom()
            })

    def select_initial_target(self):
        """Show initial target selection dialog"""
        dialog = TargetSelectionDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            target_info = dialog.get_target_info()
            return self.setup_capture_target(target_info)
        return False

    def change_capture_target(self):
        """Change the capture target during runtime"""
        # Pause capture
        self.capture_timer.stop()

        dialog = TargetSelectionDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            target_info = dialog.get_target_info()
            result = self.setup_capture_target(target_info)

            # Resume capture
            self.capture_timer.start()
            return result

        # Resume capture with existing settings
        self.capture_timer.start()
        return True

    def setup_capture_target(self, target_info):
        """Set up the capture target based on selection"""
        # Clean up existing capture if any
        if self.camera:
            try:
                self.camera.release()
            except:
                pass
            self.camera = None

        if self.window_monitor_timer.isActive():
            self.window_monitor_timer.stop()

        self.capture_mode = target_info['mode']

        if self.capture_mode == CAPTURE_MODE_DISPLAY:
            # Display mode
            self.display_idx = target_info['display_idx']
            self.target_window = None
            app = QApplication.instance()
            screens = app.screens()

            if self.display_idx >= len(screens):
                self.display_idx = 0  # Default to first display if invalid

            self.screen = screens[self.display_idx]
            self.screen_rect = self.screen.geometry()
            self.screen_width = self.screen_rect.width()
            self.screen_height = self.screen_rect.height()

            try:
                self.camera = dxcam.create(output_idx=self.display_idx)
                print(f"Screen capture initialized for display {self.display_idx}")
                return True
            except Exception as e:
                print(f"Error initializing screen capture: {e}")
                return False

        elif self.capture_mode == CAPTURE_MODE_WINDOW:
            # Window mode
            self.target_window = target_info['window']
            self.update_window_info()

            try:
                # For window capture, first try to use display capture and crop the frames
                # This is more reliable than using dxcam's region parameter
                display_index = self.get_display_containing_window()
                if display_index is None:
                    display_index = 0  # Default to first display

                print(f"Creating camera for display {display_index} to capture window")
                self.camera = dxcam.create(output_idx=display_index)

                print(f"Window capture initialized for: {self.target_window.title}")

                # Start window monitor timer
                self.window_monitor_timer.start(500)  # Check every half second
                return True
            except Exception as e:
                print(f"Error initializing window capture: {e}")

                # Fallback to display capture
                try:
                    print("Falling back to display capture")
                    self.capture_mode = CAPTURE_MODE_DISPLAY
                    self.display_idx = 0
                    self.camera = dxcam.create(output_idx=self.display_idx)
                    return True
                except Exception as e2:
                    print(f"Fallback failed: {e2}")
                    return False

        return False

    def get_display_containing_window(self):
        """Find which display contains the target window"""
        if not self.target_window:
            return 0

        # Get window position
        win_left = self.target_window.left
        win_top = self.target_window.top
        win_right = win_left + self.target_window.width
        win_bottom = win_top + self.target_window.height

        # Find which monitor contains most of the window
        best_match = 0
        best_overlap = 0

        for idx, monitor in enumerate(self.monitor_info):
            # Calculate overlap area
            overlap_left = max(win_left, monitor['left'])
            overlap_top = max(win_top, monitor['top'])
            overlap_right = min(win_right, monitor['right'])
            overlap_bottom = min(win_bottom, monitor['bottom'])

            if overlap_right > overlap_left and overlap_bottom > overlap_top:
                overlap_area = (overlap_right - overlap_left) * (overlap_bottom - overlap_top)
                if overlap_area > best_overlap:
                    best_overlap = overlap_area
                    best_match = idx

        return best_match

    def update_window_info(self):
        """Update window position and size information"""
        if not self.target_window:
            return

        try:
            # Update window with current state
            window_list = gw.getWindowsWithTitle(self.target_window.title)
            if not window_list:
                print(f"Window '{self.target_window.title}' not found")
                return

            self.target_window = window_list[0]

            # Store the window rect
            self.window_rect = QRect(
                self.target_window.left, self.target_window.top,
                self.target_window.width, self.target_window.height
            )

            self.screen_width = self.target_window.width
            self.screen_height = self.target_window.height
        except Exception as e:
            print(f"Error updating window info: {e}")

    def check_window_changes(self):
        """Check for window position/size changes"""
        if not self.target_window:
            return

        try:
            # Find the window by title
            windows = gw.getWindowsWithTitle(self.target_window.title)
            if not windows:
                print(f"Window '{self.target_window.title}' not found.")
                return

            current_window = windows[0]

            # Check if window position or size has changed
            current_rect = QRect(
                current_window.left, current_window.top,
                current_window.width, current_window.height
            )

            if (current_rect != self.window_rect):
                print("Window position or size changed. Updating capture.")

                # Update the target window reference
                self.target_window = current_window

                # Store the new rect
                self.window_rect = current_rect

                # Update size info
                self.screen_width = current_window.width
                self.screen_height = current_window.height

        except Exception as e:
            print(f"Error checking window changes: {e}")

    def toggle_detection_mode(self):
        """Toggle between single frame and tiled detection modes"""
        self.detection_mode = MODE_TILED if self.detection_mode == MODE_SINGLE_FRAME else MODE_SINGLE_FRAME

        # Update detection thread mode
        self.detection_thread.set_detection_mode(self.detection_mode)

        print(f"Switched to {'Tiled' if self.detection_mode == MODE_TILED else 'Single Frame'} mode")

    def toggle_labels(self):
        """Toggle confidence labels on/off"""
        self.show_labels = not self.show_labels
        print(f"Labels {'on' if self.show_labels else 'off'}")
        self.update_display()  # Update immediately to reflect change

    def position_window_strategically(self):
        """Position window strategically based on capture mode"""
        app = QApplication.instance()
        screens = app.screens()

        if self.capture_mode == CAPTURE_MODE_DISPLAY:
            # If capturing a display, try to position on another display if available
            if len(screens) > 1:
                for i, screen in enumerate(screens):
                    if i != self.display_idx:
                        # Position window on this screen
                        screen_geo = screen.geometry()
                        window_size = self.size()
                        x = screen_geo.x() + (screen_geo.width() - window_size.width()) // 2
                        y = screen_geo.y() + (screen_geo.height() - window_size.height()) // 2
                        self.move(x, y)
                        return

        # Default positioning
        self.resize(800, 600)
        center_point = QApplication.desktop().screenGeometry().center()
        self.move(center_point - QPoint(400, 300))  # Center the window

    def create_tiles(self, frame):
        """Split the frame into multiple tiles with overlap"""
        h, w = frame.shape[:2]

        # Calculate number of tiles in each dimension
        # We subtract TILE_OVERLAP to account for the overlap between tiles
        tile_width = INPUT_SIZE - TILE_OVERLAP
        tile_height = INPUT_SIZE - TILE_OVERLAP

        # Calculate how many tiles we need in each dimension
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

                # Adjust coordinates to ensure we don't go out of bounds
                if x2 == w:
                    x1 = max(0, w - INPUT_SIZE)
                if y2 == h:
                    y1 = max(0, h - INPUT_SIZE)

                # Extract tile
                tile = frame[y1:y2, x1:x2]

                # Ensure the tile is the right size (should be 640x640)
                # If it's smaller, we pad it
                th, tw = tile.shape[:2]
                if th < INPUT_SIZE or tw < INPUT_SIZE:
                    pad_tile = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
                    pad_tile[0:th, 0:tw] = tile
                    tile = pad_tile

                tiles.append(tile)
                tile_info.append({'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1})

        return tiles, tile_info

    def capture_frame(self):
        """Capture screen/window and process based on detection mode"""
        if self.camera:
            # dxcam returns the frame in RGB format
            try:
                frame = self.camera.grab()
                if frame is not None:
                    # If in window mode, crop the frame to the window
                    if self.capture_mode == CAPTURE_MODE_WINDOW and self.window_rect:
                        # Determine the display being captured
                        display_idx = self.get_display_containing_window()
                        if display_idx is not None:
                            # Calculate window position relative to the display
                            display_info = self.monitor_info[display_idx]

                            # Calculate window coordinates relative to captured display
                            rel_left = self.window_rect.left() - display_info['left']
                            rel_top = self.window_rect.top() - display_info['top']
                            rel_right = rel_left + self.window_rect.width()
                            rel_bottom = rel_top + self.window_rect.height()

                            # Ensure we don't go out of bounds
                            rel_left = max(0, rel_left)
                            rel_top = max(0, rel_top)
                            rel_right = min(display_info['width'], rel_right)
                            rel_bottom = min(display_info['height'], rel_bottom)

                            # Crop the frame to the window area
                            if (rel_right > rel_left and rel_bottom > rel_top and
                                    rel_right <= frame.shape[1] and rel_bottom <= frame.shape[0]):
                                frame = frame[rel_top:rel_bottom, rel_left:rel_right]
                            else:
                                print(
                                    f"Invalid window region: {rel_left},{rel_top},{rel_right},{rel_bottom} in {frame.shape}")

                    # Store the frame for display
                    self.current_frame = frame

                    # Set thresholds
                    self.detection_thread.set_conf_threshold(self.conf_threshold)
                    self.detection_thread.set_iou_threshold(self.iou_threshold)

                    # Process according to detection mode
                    if self.detection_mode == MODE_TILED:
                        # Create tiles from the frame
                        tiles, tile_info = self.create_tiles(frame)
                        self.tile_count = len(tiles)

                        # Send tiles to the detection thread
                        self.detection_thread.set_frame(tiles, tile_info)
                    else:
                        # Single frame mode - send the whole frame
                        self.tile_count = 0
                        self.detection_thread.set_frame(frame.copy())
            except Exception as e:
                print(f"Error capturing frame: {e}")

    def update_detections(self, detections, fps):
        """Update detections and FPS, then update display"""
        self.detections = detections
        self.fps = fps
        self.update_display()

    def update_display(self):
        """Update the display with current frame and detections"""
        if self.current_frame is None:
            return

        # Make a copy of current frame to draw on - frame is in RGB format from dxcam
        display_frame = self.current_frame.copy()

        # Convert to BGR for OpenCV operations
        display_frame_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

        # Draw detection boxes on the frame - OpenCV expects BGR format
        for box, score, class_id in self.detections:
            x1, y1, x2, y2 = map(int, box)

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            # Draw green rectangle using BGR color order (0,255,0 is green in BGR)
            cv2.rectangle(display_frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Draw confidence labels if enabled
            if self.show_labels:
                # Just show the confidence score with a transparent background
                label = f"{score:.2f}"

                # Get text size for proper positioning
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

                # Create a transparent overlay for text background
                overlay = display_frame_bgr.copy()
                cv2.rectangle(overlay, (x1, y1 - 25), (x1 + text_size[0] + 10, y1), (0, 0, 0), -1)

                # Apply the overlay with transparency
                alpha = 0.6  # 60% opacity
                cv2.addWeighted(overlay, alpha, display_frame_bgr, 1 - alpha, 0, display_frame_bgr)

                # Draw the confidence score in white
                cv2.putText(display_frame_bgr, label, (x1 + 5, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Convert back to RGB for Qt
        display_frame_rgb = cv2.cvtColor(display_frame_bgr, cv2.COLOR_BGR2RGB)

        # Convert to QImage for display
        h, w = display_frame_rgb.shape[:2]
        bytes_per_line = 3 * w
        qimg = QImage(display_frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Scale to fit the label while preserving aspect ratio
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

        # Update info text
        mode_text = "Tiled" if self.detection_mode == MODE_TILED else "Single Frame"
        label_status = "On" if self.show_labels else "Off"
        tile_info = f" | Tiles: {self.tile_count}" if self.detection_mode == MODE_TILED else ""

        target_text = ""
        if self.capture_mode == CAPTURE_MODE_DISPLAY:
            target_text = f"Display {self.display_idx}"
        else:
            target_text = f"Window: {self.target_window.title if self.target_window else 'Unknown'}"

        info_text = (f"Model: {self.current_model} | Target: {target_text} | Mode: {mode_text} | "
                     f"Labels: {label_status} | FPS: {self.fps:.1f} | "
                     f"Detections: {len(self.detections)}{tile_info} | "
                     f"Conf: {self.conf_threshold:.2f} | IoU: {self.iou_threshold:.2f}")
        self.info_label.setText(info_text)

    def resizeEvent(self, event):
        """Handle window resize"""
        super().resizeEvent(event)
        if self.current_frame is not None:
            self.update_display()

    def keyPressEvent(self, event):
        """Handle keyboard input"""
        if event.key() == Qt.Key_Q:
            self.close()  # Quit
        elif event.key() == Qt.Key_T:
            # Toggle tiled mode
            self.toggle_detection_mode()
        elif event.key() == Qt.Key_L:
            # Toggle labels
            self.toggle_labels()
        elif event.key() == Qt.Key_M:
            # Cycle through models
            self.cycle_model()
        elif event.key() == Qt.Key_C:
            # Change capture target
            self.change_capture_target()
        elif event.key() == Qt.Key_Up:
            # Increase confidence threshold
            self.conf_threshold = min(0.95, self.conf_threshold + 0.05)
            print(f"Confidence threshold: {self.conf_threshold:.2f}")
        elif event.key() == Qt.Key_Down:
            # Decrease confidence threshold
            self.conf_threshold = max(0.05, self.conf_threshold - 0.05)
            print(f"Confidence threshold: {self.conf_threshold:.2f}")
        elif event.key() == Qt.Key_Right:
            # Increase IoU threshold
            self.iou_threshold = min(0.95, self.iou_threshold + 0.05)
            print(f"IoU threshold: {self.iou_threshold:.2f}")
        elif event.key() == Qt.Key_Left:
            # Decrease IoU threshold
            self.iou_threshold = max(0.05, self.iou_threshold - 0.05)
            print(f"IoU threshold: {self.iou_threshold:.2f}")

    def closeEvent(self, event):
        """Clean up on close"""
        if hasattr(self, 'detection_thread'):
            self.detection_thread.stop()

        if hasattr(self, 'camera') and self.camera:
            try:
                self.camera.release()
            except:
                pass

        if hasattr(self, 'window_monitor_timer') and self.window_monitor_timer.isActive():
            self.window_monitor_timer.stop()

        super().closeEvent(event)


def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Vehicle Detection with Model Selection")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA acceleration")
    args = parser.parse_args()

    # Start application
    app = QApplication(sys.argv)
    window = DetectionWindow(use_cuda=args.cuda)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()