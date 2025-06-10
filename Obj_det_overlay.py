import cv2
import numpy as np
import onnxruntime as ort
import time
import sys
import dxcam
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QComboBox, QPushButton,
                             QVBoxLayout, QLabel, QDialog, QHBoxLayout, QRadioButton,
                             QButtonGroup, QListWidget, QListWidgetItem, QFrame, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap, QCursor
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QRect
import pygetwindow as gw
import os

# Constants
AVAILABLE_MODELS = ["yolo12.onnx", "yolo8.onnx", "yolo3.onnx", "best.onnx"]
DEFAULT_MODEL = AVAILABLE_MODELS[0]
DEFAULT_CONF_THRESHOLD = 0.8
DEFAULT_IOU_THRESHOLD = 0.95
INPUT_SIZE = 640
TILE_OVERLAP = 10

# Capture and detection modes
CAPTURE_MODE_DISPLAY = 0
CAPTURE_MODE_WINDOW = 1
MODE_SINGLE_FRAME = 0
MODE_TILED = 1


class TargetSelectionDialog(QDialog):
    """Dialog for selecting capture target (display or window)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Capture Target")
        self.selected_window = None
        self.capture_mode = CAPTURE_MODE_DISPLAY
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Mode selection
        mode_group = QButtonGroup(self)
        self.display_radio = QRadioButton("Capture Display")
        self.display_radio.setChecked(True)
        self.display_radio.toggled.connect(self.update_mode)
        mode_group.addButton(self.display_radio)

        self.window_radio = QRadioButton("Capture Browser Window")
        self.window_radio.toggled.connect(self.update_mode)
        mode_group.addButton(self.window_radio)

        mode_box = QFrame()
        mode_layout = QVBoxLayout(mode_box)
        mode_layout.addWidget(self.display_radio)
        mode_layout.addWidget(self.window_radio)
        layout.addWidget(mode_box)

        # Display selection
        self.display_frame = QFrame()
        display_layout = QVBoxLayout(self.display_frame)
        display_layout.addWidget(QLabel("Select display:"))

        self.display_combo = QComboBox()
        self.populate_displays()
        display_layout.addWidget(self.display_combo)
        layout.addWidget(self.display_frame)

        # Window selection
        self.window_frame = QFrame()
        window_layout = QVBoxLayout(self.window_frame)
        self.window_info = QLabel("No window selected")
        window_layout.addWidget(self.window_info)

        self.select_window_button = QPushButton("Select Window")
        self.select_window_button.clicked.connect(self.select_window)
        window_layout.addWidget(self.select_window_button)
        layout.addWidget(self.window_frame)
        self.window_frame.hide()

        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.resize(400, 300)

    def update_mode(self):
        self.capture_mode = CAPTURE_MODE_WINDOW if self.window_radio.isChecked() else CAPTURE_MODE_DISPLAY
        self.display_frame.setVisible(self.capture_mode == CAPTURE_MODE_DISPLAY)
        self.window_frame.setVisible(self.capture_mode == CAPTURE_MODE_WINDOW)

    def populate_displays(self):
        screens = QApplication.instance().screens()
        for i, screen in enumerate(screens):
            display_name = f"Display {i + 1}: {screen.name()} ({screen.geometry().width()}x{screen.geometry().height()})"
            self.display_combo.addItem(display_name, i)

    def select_window(self):
        windows = []
        window_list = QListWidget()

        # Populate window list
        for window in gw.getAllWindows():
            if window.visible and window.title:
                windows.append(window)
                window_list.addItem(f"{window.title} ({window.width}x{window.height})")

        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Window")
        dialog_layout = QVBoxLayout(dialog)
        dialog_layout.addWidget(QLabel("Select a window to capture:"))
        dialog_layout.addWidget(window_list)

        button_layout = QHBoxLayout()
        ok_button = QPushButton("Select")
        ok_button.clicked.connect(dialog.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        dialog_layout.addLayout(button_layout)

        dialog.resize(400, 500)

        # Show dialog and process result
        if dialog.exec_() == QDialog.Accepted:
            index = window_list.currentRow()
            if index >= 0 and index < len(windows):
                self.selected_window = windows[index]
                self.window_info.setText(
                    f"Selected: {self.selected_window.title} ({self.selected_window.width}x{self.selected_window.height})")

    def get_target_info(self):
        if self.capture_mode == CAPTURE_MODE_DISPLAY:
            return {
                'mode': CAPTURE_MODE_DISPLAY,
                'display_idx': self.display_combo.currentData()
            }
        else:
            if not self.selected_window:
                return {
                    'mode': CAPTURE_MODE_DISPLAY,
                    'display_idx': 0
                }
            return {
                'mode': CAPTURE_MODE_WINDOW,
                'window': self.selected_window
            }


class DetectionThread(QThread):
    """Thread for running object detection"""
    detection_complete = pyqtSignal(list, float)
    model_loaded = pyqtSignal(str)

    def __init__(self, model_path):
        super().__init__()
        self.running = True
        self.frame = None
        self.new_frame = False
        self.new_model = False
        self.conf_threshold = DEFAULT_CONF_THRESHOLD
        self.iou_threshold = DEFAULT_IOU_THRESHOLD
        self.model_path = model_path
        self.new_model_path = None
        self.tile_info = None
        self.detection_mode = MODE_SINGLE_FRAME
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

    def set_frame(self, frame, tile_info=None):
        """Set a new frame for processing"""
        self.frame = frame
        self.tile_info = tile_info
        self.new_frame = True

    def set_conf_threshold(self, value):
        self.conf_threshold = value

    def set_iou_threshold(self, value):
        self.iou_threshold = value

    def change_model(self, model_path):
        self.new_model_path = model_path
        self.new_model = True

    def set_detection_mode(self, mode):
        self.detection_mode = mode

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

                # Process based on detection mode
                if self.detection_mode == MODE_TILED and isinstance(self.frame, list):
                    detections = self.process_tiled_frame()
                else:
                    detections = self.process_frame(self.frame)

                fps = 1 / (time.time() - start_time)
                self.detection_complete.emit(detections, fps)

            # Sleep to avoid high CPU usage
            time.sleep(0.01)

    def process_tiled_frame(self):
        """Process multiple tiles and combine results"""
        all_detections = []

        # Process each tile
        for i, (tile, tile_info) in enumerate(zip(self.frame, self.tile_info)):
            tile_detections = self.process_frame(tile)

            # Adjust coordinates based on tile position
            for j in range(len(tile_detections)):
                box, score, class_id = tile_detections[j]
                adjusted_box = [
                    box[0] + tile_info['x'],
                    box[1] + tile_info['y'],
                    box[2] + tile_info['x'],
                    box[3] + tile_info['y']
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


class DetectionWindow(QMainWindow):
    """Main window for displaying detections"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vehicle Detection")
        self.setMinimumSize(800, 600)

        # App state
        self.current_model = DEFAULT_MODEL
        self.detection_mode = MODE_SINGLE_FRAME
        self.show_labels = False
        self.detections = []
        self.current_frame = None
        self.fps = 0
        self.conf_threshold = DEFAULT_CONF_THRESHOLD
        self.iou_threshold = DEFAULT_IOU_THRESHOLD
        self.capture_mode = None
        self.display_idx = None
        self.target_window = None
        self.camera = None

        # Setup UI
        self.setup_ui()

        # Get monitor info
        self.monitor_info = self.get_monitor_info()

        # Initial target selection
        if not self.select_initial_target():
            print("No target selected. Exiting.")
            sys.exit(0)

        # Start detection thread
        self.detection_thread = DetectionThread(self.current_model)
        self.detection_thread.detection_complete.connect(self.update_detections)
        self.detection_thread.model_loaded.connect(self.update_model_info)
        self.detection_thread.start()

        # Start capture timer
        self.capture_timer = QTimer(self)
        self.capture_timer.timeout.connect(self.capture_frame)
        self.capture_timer.start(33)  # ~30 FPS

        # Center window on screen
        self.center_window()

    def setup_ui(self):
        """Setup the main window UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Image display - Set to expand to fill available space
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(False)  # Don't stretch the image, keep aspect ratio
        main_layout.addWidget(self.image_label, 1)  # 1 is the stretch factor

        # Info panel - Fixed height but adequate for controls
        info_panel = QWidget()
        info_panel.setMinimumHeight(120)  # Increased minimum height
        info_layout = QHBoxLayout(info_panel)

        # Info display
        self.info_label = QLabel()
        self.info_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.info_label.setWordWrap(True)  # Allow wrapping if needed
        info_layout.addWidget(self.info_label, 2)  # Give more space to info

        # Controls help - Fixed width to prevent it from being squeezed
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
        controls_help.setMinimumWidth(200)  # Ensure minimum width for controls text
        controls_help.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        info_layout.addWidget(controls_help, 1)  # Give less space but still enough

        main_layout.addWidget(info_panel, 0)  # 0 means no stretch

    def get_monitor_info(self):
        """Get information about all monitors"""
        monitor_info = []
        screens = QApplication.instance().screens()

        for screen in screens:
            geometry = screen.geometry()
            monitor_info.append({
                'width': geometry.width(),
                'height': geometry.height(),
                'left': geometry.left(),
                'top': geometry.top(),
                'right': geometry.right(),
                'bottom': geometry.bottom()
            })

        print(f"Detected {len(monitor_info)} monitors")
        for i, monitor in enumerate(monitor_info):
            print(f"Monitor {i}: {monitor['width']}x{monitor['height']} at ({monitor['left']},{monitor['top']})")

        return monitor_info

    def update_model_info(self, model_name):
        """Update UI after model is loaded"""
        self.current_model = model_name
        self.setWindowTitle(f"Vehicle Detection - Model: {model_name}")

    def cycle_model(self):
        """Cycle to next model"""
        try:
            current_index = AVAILABLE_MODELS.index(self.current_model)
        except ValueError:
            current_index = 0

        next_index = (current_index + 1) % len(AVAILABLE_MODELS)
        next_model = AVAILABLE_MODELS[next_index]

        print(f"Cycling model from {self.current_model} to {next_model}")
        self.detection_thread.change_model(next_model)

    def select_initial_target(self):
        """Show initial target selection dialog"""
        dialog = TargetSelectionDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            target_info = dialog.get_target_info()
            return self.setup_capture_target(target_info)
        return False

    def change_capture_target(self):
        """Change the capture target during runtime"""
        self.capture_timer.stop()

        dialog = TargetSelectionDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            target_info = dialog.get_target_info()
            result = self.setup_capture_target(target_info)
            self.capture_timer.start()
            return result

        self.capture_timer.start()
        return True

    def setup_capture_target(self, target_info):
        """Setup capture target based on selection"""
        # Clean up existing capture
        if self.camera:
            try:
                self.camera.release()
            except:
                pass
            self.camera = None

        self.capture_mode = target_info['mode']

        if self.capture_mode == CAPTURE_MODE_DISPLAY:
            # Display mode
            self.display_idx = target_info['display_idx']
            self.target_window = None

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

            # Get display that contains this window
            self.window_display_idx = self.get_display_containing_window(self.target_window)
            if self.window_display_idx is None:
                self.window_display_idx = 0

            try:
                # Create camera for the specific display
                self.camera = dxcam.create(output_idx=self.window_display_idx)
                print(f"Window capture initialized for: {self.target_window.title}")
                return True
            except Exception as e:
                print(f"Error initializing window capture: {e}")
                return False

        return False

    def get_display_containing_window(self, window):
        """Find which display contains the target window"""
        if not window:
            return 0

        # Get window position
        win_left = window.left
        win_top = window.top
        win_right = win_left + window.width
        win_bottom = win_top + window.height

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

    def center_window(self):
        """Center window on screen"""
        screen = QApplication.desktop().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2,
                  (screen.height() - size.height()) // 2)

    def create_tiles(self, frame):
        """Split the frame into multiple tiles with overlap"""
        h, w = frame.shape[:2]

        # Calculate tile dimensions
        tile_width = INPUT_SIZE - TILE_OVERLAP
        tile_height = INPUT_SIZE - TILE_OVERLAP

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

    def capture_frame(self):
        """Capture screen/window and process"""
        if not self.camera:
            return

        try:
            frame = self.camera.grab()
            if frame is None:
                return

            # If window mode, crop to window
            if self.capture_mode == CAPTURE_MODE_WINDOW and self.target_window:
                try:
                    # Get updated window position
                    windows = gw.getWindowsWithTitle(self.target_window.title)
                    if windows:
                        window = windows[0]

                        # Get monitor info
                        monitor = self.monitor_info[self.window_display_idx]

                        # Calculate position relative to monitor
                        rel_x = max(0, window.left - monitor['left'])
                        rel_y = max(0, window.top - monitor['top'])
                        rel_right = min(monitor['width'], rel_x + window.width)
                        rel_bottom = min(monitor['height'], rel_y + window.height)

                        # Crop if coordinates are valid
                        if (rel_right > rel_x and rel_bottom > rel_y and
                                rel_x < frame.shape[1] and rel_y < frame.shape[0] and
                                rel_right <= frame.shape[1] and rel_bottom <= frame.shape[0]):
                            frame = frame[rel_y:rel_bottom, rel_x:rel_right]
                except Exception as e:
                    print(f"Error cropping to window: {e}")

            # Store frame for display
            self.current_frame = frame

            # Process according to detection mode
            if self.detection_mode == MODE_TILED:
                # Create tiles
                tiles, tile_info = self.create_tiles(frame)
                self.detection_thread.set_frame(tiles, tile_info)
            else:
                # Single frame mode
                self.detection_thread.set_frame(frame.copy())
        except Exception as e:
            print(f"Error capturing frame: {e}")

    def update_detections(self, detections, fps):
        """Update detections and FPS"""
        self.detections = detections
        self.fps = fps
        self.update_display()

    def update_display(self):
        """Update the display with current frame and detections"""
        if self.current_frame is None:
            return

        # Make a copy of current frame
        display_frame = self.current_frame.copy()

        # Convert to BGR for OpenCV operations
        display_frame_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

        # Draw detection boxes
        for box, score, class_id in self.detections:
            x1, y1, x2, y2 = map(int, box)

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            # Draw rectangle
            cv2.rectangle(display_frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Draw confidence labels if enabled
            if self.show_labels:
                label = f"{score:.2f}"

                # Get text size
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

                # Create transparent overlay
                overlay = display_frame_bgr.copy()
                cv2.rectangle(overlay, (x1, y1 - 25), (x1 + text_size[0] + 10, y1), (0, 0, 0), -1)

                # Apply overlay with transparency
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, display_frame_bgr, 1 - alpha, 0, display_frame_bgr)

                # Draw text
                cv2.putText(display_frame_bgr, label, (x1 + 5, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Convert back to RGB for Qt
        display_frame_rgb = cv2.cvtColor(display_frame_bgr, cv2.COLOR_BGR2RGB)

        # Convert to QImage for display
        h, w = display_frame_rgb.shape[:2]
        bytes_per_line = 3 * w
        qimg = QImage(display_frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Get current label size for proper scaling
        label_size = self.image_label.size()

        # Only scale if we have a valid size
        if label_size.width() > 0 and label_size.height() > 0:
            # Create a pixmap and scale it to fit the label's current size
            pixmap = QPixmap.fromImage(qimg)
            scaled_pixmap = pixmap.scaled(
                label_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

        # Update info text
        mode_label = "T" if self.detection_mode == MODE_TILED else "S"
        labels_status = "L" if self.show_labels else "-"

        info_text = (f"Model: {self.current_model} | "
                     f"FPS: {self.fps:.1f} | "
                     f"Det: {len(self.detections)} | "
                     f"C: {self.conf_threshold:.2f} | "
                     f"IoU: {self.iou_threshold:.2f} | "
                     f"Mode: {mode_label}{labels_status} | "
                     f"Size: {display_frame.shape[1]}x{display_frame.shape[0]}")
        self.info_label.setText(info_text)

    def resizeEvent(self, event):
        """Handle window resize"""
        super().resizeEvent(event)
        # Ensure we update the display whenever the window is resized
        # This is crucial for making the image area expand with the window
        QTimer.singleShot(0, self.update_display)

    def keyPressEvent(self, event):
        """Handle keyboard input"""
        key = event.key()

        if key == Qt.Key_Q:
            self.close()  # Quit
        elif key == Qt.Key_T:
            # Toggle tiled mode
            self.detection_mode = MODE_TILED if self.detection_mode == MODE_SINGLE_FRAME else MODE_SINGLE_FRAME
            self.detection_thread.set_detection_mode(self.detection_mode)
        elif key == Qt.Key_L:
            # Toggle labels
            self.show_labels = not self.show_labels
            self.update_display()
        elif key == Qt.Key_M:
            # Cycle through models
            self.cycle_model()
        elif key == Qt.Key_C:
            # Change capture target
            self.change_capture_target()
        elif key == Qt.Key_Up:
            # Increase confidence threshold
            self.conf_threshold = min(0.95, self.conf_threshold + 0.05)
            self.detection_thread.set_conf_threshold(self.conf_threshold)
        elif key == Qt.Key_Down:
            # Decrease confidence threshold
            self.conf_threshold = max(0.05, self.conf_threshold - 0.05)
            self.detection_thread.set_conf_threshold(self.conf_threshold)
        elif key == Qt.Key_Right:
            # Increase IoU threshold
            self.iou_threshold = min(0.95, self.iou_threshold + 0.05)
            self.detection_thread.set_iou_threshold(self.iou_threshold)
        elif key == Qt.Key_Left:
            # Decrease IoU threshold
            self.iou_threshold = max(0.05, self.iou_threshold - 0.05)
            self.detection_thread.set_iou_threshold(self.iou_threshold)

    def closeEvent(self, event):
        """Clean up on close"""
        if hasattr(self, 'detection_thread'):
            self.detection_thread.stop()

        if hasattr(self, 'camera') and self.camera:
            try:
                self.camera.release()
            except:
                pass

        super().closeEvent(event)


def main():
    """Main function"""
    app = QApplication(sys.argv)
    window = DetectionWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()