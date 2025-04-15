import cv2
import numpy as np
import onnxruntime as ort
import time
import sys
import argparse
import dxcam
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QComboBox, QPushButton, QVBoxLayout, QLabel, QDialog, \
    QHBoxLayout
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QRect, QSize
import threading

# Configuration
MODEL_PATH = "yolo12.onnx"
CLASS_NAMES = ["vehicle"]
DEFAULT_CONF_THRESHOLD = 0.8
INPUT_SIZE = 640
IOU_THRESHOLD = 0.8

# Display modes (properly defined like in original code)
DISPLAY_MODE_TRANSPARENT = 0
DISPLAY_MODE_SEMI_TRANSPARENT = 1
DISPLAY_MODE_FULL_IMAGE = 2


class DisplaySelectionDialog(QDialog):
    """Dialog for selecting which display to use"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Display")
        self.selected_display = 0

        layout = QVBoxLayout()

        # Create display selection dropdown
        self.label = QLabel("Select which display to use:")
        layout.addWidget(self.label)

        self.display_combo = QComboBox()
        self.populate_displays()
        layout.addWidget(self.display_combo)

        # Add OK button
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

    def populate_displays(self):
        """Populate the dropdown with available displays"""
        app = QApplication.instance()
        screens = app.screens()

        for i, screen in enumerate(screens):
            display_name = f"Display {i + 1}: {screen.name()} ({screen.geometry().width()}x{screen.geometry().height()})"
            self.display_combo.addItem(display_name, i)

    def get_selected_display(self):
        """Return the selected display index"""
        return self.display_combo.currentData()


class DetectionThread(QThread):
    """Thread for running object detection"""
    detection_complete = pyqtSignal(list, float)

    def __init__(self, model_path, use_cuda=False):
        super().__init__()
        self.running = True
        self.frame_ready = threading.Event()
        self.frame = None
        self.conf_threshold = DEFAULT_CONF_THRESHOLD
        self.use_cuda = use_cuda
        self.model_path = model_path
        self.setup_model()

    def setup_model(self):
        """Initialize the ONNX model"""
        providers = ['CPUExecutionProvider']

        if self.use_cuda and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("Using CUDA")
        else:
            print("Using CPU")

        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"Model loaded: {self.model_path}")

    def set_frame(self, frame):
        """Set a new frame for processing"""
        self.frame = frame
        self.frame_ready.set()

    def set_conf_threshold(self, value):
        """Set confidence threshold"""
        self.conf_threshold = value

    def stop(self):
        """Stop the thread"""
        self.running = False
        self.frame_ready.set()
        self.wait()

    def run(self):
        """Main thread loop"""
        while self.running:
            self.frame_ready.wait()
            self.frame_ready.clear()

            if not self.running or self.frame is None:
                continue

            start_time = time.time()
            detections = self.process_frame(self.frame)
            fps = 1 / (time.time() - start_time)

            self.detection_complete.emit(detections, fps)

    def process_frame(self, frame):
        """Process a frame through the model"""
        try:
            # Basic preprocessing
            input_tensor, orig_size = self.preprocess(frame)

            # Run inference
            outputs = self.session.run(None, {self.input_name: input_tensor})

            # Basic postprocessing
            detections = self.postprocess(outputs, orig_size)

            return detections
        except Exception as e:
            print(f"Error processing frame: {e}")
            return []

    def preprocess(self, frame):
        """Minimal preprocessing for YOLO model"""
        original_size = frame.shape[:2]

        # Use frame as is - dxcam returns RGB format
        # Note: We're not converting to BGR here to maintain model accuracy

        # Resize to square input with padding
        h, w = frame.shape[:2]
        scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
        new_w, new_h = int(w * scale), int(h * scale)

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
        """Basic postprocessing for YOLO outputs"""
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
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, IOU_THRESHOLD)

        if len(indices) == 0:
            return []

        # Return detection results
        return [(boxes[i], scores[i], class_ids[i]) for i in indices]


class DetectionWindow(QMainWindow):
    """Separate window for displaying detections"""

    def __init__(self, display_idx=None, use_cuda=False):
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

        # Add controls description
        controls_help = QLabel(
            "Controls:\n"
            "  ↑/↓: Adjust confidence\n"
            "  Q: Quit"
        )
        controls_layout.addWidget(controls_help)

        info_layout.addWidget(controls_panel)
        main_layout.addWidget(info_panel)

        # Get screen info based on display_idx or let user select
        self.setup_display(display_idx)

        # Initialize state
        self.detections = []
        self.current_frame = None
        self.fps = 0

        # Initialize screen capture for selected display
        try:
            # Use selected display for capture
            self.camera = dxcam.create(output_idx=self.display_idx)
            print(f"Screen capture initialized for display {self.display_idx}")
        except Exception as e:
            print(f"Error initializing screen capture: {e}")
            self.camera = None

        # Start detection thread
        self.detection_thread = DetectionThread(MODEL_PATH, use_cuda)
        self.detection_thread.detection_complete.connect(self.update_detections)
        self.detection_thread.start()

        # Start capture timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.capture_frame)
        self.timer.start(33)  # ~30 FPS

        # Position the window on a different screen than the one being captured
        self.position_window_strategically()

    def setup_display(self, display_idx=None):
        """Set up the display to use for capture"""
        app = QApplication.instance()
        screens = app.screens()

        # If no display index specified or invalid, prompt user
        if display_idx is None or display_idx < 0 or display_idx >= len(screens):
            dialog = DisplaySelectionDialog()
            if dialog.exec_() == QDialog.Accepted:
                self.display_idx = dialog.get_selected_display()
            else:
                # Default to primary display if dialog canceled
                self.display_idx = 0
        else:
            self.display_idx = display_idx

        # Set up screen info for the selected display
        self.screen = screens[self.display_idx]
        self.screen_rect = self.screen.geometry()
        self.screen_width = self.screen_rect.width()
        self.screen_height = self.screen_rect.height()

        print(f"Using display {self.display_idx}: {self.screen.name()} ({self.screen_width}x{self.screen_height})")

    def position_window_strategically(self):
        """Position window on different screen than capture if possible"""
        app = QApplication.instance()
        screens = app.screens()

        # If we have multiple screens, put window on a different screen than capture
        if len(screens) > 1:
            for i, screen in enumerate(screens):
                if i != self.display_idx:
                    # Position window on this screen
                    screen_geo = screen.geometry()
                    window_size = self.size()
                    x = screen_geo.x() + (screen_geo.width() - window_size.width()) // 2
                    y = screen_geo.y() + (screen_geo.height() - window_size.height()) // 2
                    self.move(x, y)
                    print(f"Positioned window on screen {i}")
                    return

        # If only one screen or couldn't position elsewhere
        # Move to a corner of the screen we're capturing, but sized appropriately
        self.resize(self.screen_width // 2, self.screen_height // 2)
        self.move(0, 0)

    def capture_frame(self):
        """Capture screen and send to detection thread"""
        if self.camera:
            # dxcam returns the frame in RGB format
            frame = self.camera.grab()
            if frame is not None:
                # Store the frame for display
                self.current_frame = frame
                # Set the confidence threshold
                self.detection_thread.set_conf_threshold(self.conf_threshold)
                # Send a copy to the detection thread
                self.detection_thread.set_frame(frame.copy())

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

            # Add label
            label = f"{CLASS_NAMES[class_id]}: {score:.2f}"
            # Draw black background for text
            cv2.rectangle(display_frame_bgr, (x1, y1 - 30), (x1 + len(label) * 10, y1), (0, 0, 0), -1)
            # Draw white text
            cv2.putText(display_frame_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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
        info_text = (f"FPS: {self.fps:.1f} | "
                     f"Detections: {len(self.detections)} | "
                     f"Confidence: {self.conf_threshold:.2f} | "
                     f"Display: {self.display_idx} ({self.screen_width}x{self.screen_height})")
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
        elif event.key() == Qt.Key_Up:
            self.conf_threshold = min(0.9, self.conf_threshold + 0.05)  # Increase confidence
            print(f"Confidence threshold: {self.conf_threshold:.2f}")
        elif event.key() == Qt.Key_Down:
            self.conf_threshold = max(0.1, self.conf_threshold - 0.05)  # Decrease confidence
            print(f"Confidence threshold: {self.conf_threshold:.2f}")

    def closeEvent(self, event):
        """Clean up on close"""
        if hasattr(self, 'detection_thread'):
            self.detection_thread.stop()

        if hasattr(self, 'camera') and self.camera:
            self.camera.release()

        super().closeEvent(event)


def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Simple Object Detection Viewer")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA acceleration")
    parser.add_argument("--display", type=int, default=None, help="Display index to use (defaults to selection dialog)")
    args = parser.parse_args()

    # Start application
    app = QApplication(sys.argv)
    window = DetectionWindow(display_idx=args.display, use_cuda=args.cuda)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()