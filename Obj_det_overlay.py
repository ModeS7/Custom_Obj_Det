import cv2
import numpy as np
import onnxruntime as ort
import time
import sys
import argparse
import dxcam
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QImage
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
import threading

# Configuration
MODEL_PATH = "yolo12.onnx"
CLASS_NAMES = ["vehicle"]
DEFAULT_CONF_THRESHOLD = 0.25
INPUT_SIZE = 640
IOU_THRESHOLD = 0.45

# Display modes (properly defined like in original code)
DISPLAY_MODE_TRANSPARENT = 0
DISPLAY_MODE_SEMI_TRANSPARENT = 1
DISPLAY_MODE_FULL_IMAGE = 2


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


class DetectionOverlay(QMainWindow):
    """Main window for displaying detection overlay"""

    def __init__(self, use_cuda=False):
        super().__init__()
        self.use_cuda = use_cuda

        # Get screen info
        self.screen = QApplication.primaryScreen()
        self.screen_width = self.screen.geometry().width()
        self.screen_height = self.screen.geometry().height()

        # Set up the window
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(0, 0, self.screen_width, self.screen_height)
        self.setCentralWidget(QWidget())

        # Initialize state
        self.detections = []
        self.current_frame = None
        self.fps = 0
        self.conf_threshold = DEFAULT_CONF_THRESHOLD
        self.display_mode = DISPLAY_MODE_TRANSPARENT

        # Initialize screen capture
        try:
            self.camera = dxcam.create(output_idx=0)
            print("Screen capture initialized")
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

    def capture_frame(self):
        """Capture screen and send to detection thread"""
        if self.camera:
            frame = self.camera.grab()
            if frame is not None:
                self.current_frame = frame
                self.detection_thread.set_conf_threshold(self.conf_threshold)
                self.detection_thread.set_frame(frame.copy())

    def update_detections(self, detections, fps):
        """Update detections and FPS"""
        self.detections = detections
        self.fps = fps
        self.update()  # Trigger repaint

    def paintEvent(self, event):
        """Draw overlay"""
        painter = QPainter(self)

        # Handle different display modes correctly
        if self.display_mode == DISPLAY_MODE_TRANSPARENT:
            # Fully transparent background
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.fillRect(self.rect(), Qt.transparent)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        elif self.display_mode == DISPLAY_MODE_SEMI_TRANSPARENT:
            # Semi-transparent dark overlay
            painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
        elif self.display_mode == DISPLAY_MODE_FULL_IMAGE and self.current_frame is not None:
            # Show the captured frame as background
            h, w = self.current_frame.shape[:2]
            qimg = QImage(self.current_frame.data, w, h,
                          self.current_frame.strides[0], QImage.Format_RGB888).rgbSwapped()
            painter.drawImage(0, 0, qimg)

        # Draw info panel
        painter.setFont(QFont("Arial", 12, QFont.Bold))
        painter.fillRect(10, 10, 320, 150, QColor(0, 0, 0, 180))

        painter.setPen(QColor(255, 255, 255))
        info_lines = [
            f"FPS: {self.fps:.1f}",
            f"Detections: {len(self.detections)}",
            f"Confidence: {self.conf_threshold:.2f}",
            f"Display Mode: {self.display_mode}",
            f"Controls:",
            f"  D: Cycle display modes",
            f"  ↑/↓: Adjust confidence",
            f"  Q: Quit"
        ]

        for i, line in enumerate(info_lines):
            painter.drawText(20, 30 + i * 16, line)

        # Draw detection boxes with thicker borders and clearer labels
        pen = QPen(QColor(0, 255, 0))
        pen.setWidth(3)  # Thicker lines for better visibility
        painter.setPen(pen)

        for box, score, class_id in self.detections:
            x1, y1, x2, y2 = map(int, box)

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            # Draw filled box with different appearance based on display mode
            if self.display_mode == DISPLAY_MODE_TRANSPARENT:
                # For transparent mode, use a more visible fill
                painter.fillRect(x1, y1, x2 - x1, y2 - y1, QColor(0, 255, 0, 40))
            elif self.display_mode == DISPLAY_MODE_SEMI_TRANSPARENT:
                # For semi-transparent, use a different color
                painter.fillRect(x1, y1, x2 - x1, y2 - y1, QColor(0, 255, 0, 60))

            # Draw box outline
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

            # Draw label with better visibility
            label = f"{CLASS_NAMES[class_id]}: {score:.2f}"
            text_width = painter.fontMetrics().width(label) + 10

            # Background for text
            painter.fillRect(x1, y1 - 25, text_width, 20, QColor(0, 0, 0, 200))

            # Text
            painter.setPen(QColor(0, 255, 0))
            painter.drawText(x1 + 5, y1 - 10, label)

            # Reset pen for next box
            pen = QPen(QColor(0, 255, 0))
            pen.setWidth(3)
            painter.setPen(pen)

    def keyPressEvent(self, event):
        """Handle keyboard input"""
        if event.key() == Qt.Key_Q:
            self.close()  # Quit
        elif event.key() == Qt.Key_D:
            # Cycle through display modes properly
            self.display_mode = (self.display_mode + 1) % 3
            print(f"Display mode: {self.display_mode}")
        elif event.key() == Qt.Key_Up:
            self.conf_threshold = min(0.9, self.conf_threshold + 0.05)  # Increase confidence
        elif event.key() == Qt.Key_Down:
            self.conf_threshold = max(0.1, self.conf_threshold - 0.05)  # Decrease confidence

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
    parser = argparse.ArgumentParser(description="Simple Object Detection Overlay")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA acceleration")
    args = parser.parse_args()

    # Start application
    app = QApplication(sys.argv)
    overlay = DetectionOverlay(use_cuda=args.cuda)
    overlay.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()