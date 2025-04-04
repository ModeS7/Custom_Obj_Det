import cv2
import numpy as np
import onnxruntime as ort
from mss import mss
import time
import sys
import argparse
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QImage
from PyQt5.QtCore import Qt, QTimer

#############################################################
# Configuration Parameters
#############################################################

# Model Configuration
MODEL_PATH = "best.onnx"
CLASS_NAMES = ["vehicle"]  # Update with your classes
DEFAULT_CONF_THRESHOLD = 0.25
MIN_CONF_THRESHOLD = 0.1
MAX_CONF_THRESHOLD = 0.9
CONF_STEP = 0.05
IOU_THRESHOLD = 0.45
INPUT_SIZE = 640

# Display Configuration
DISPLAY_MODE_TRANSPARENT = 0
DISPLAY_MODE_SEMI_TRANSPARENT = 1
DISPLAY_MODE_FULL_IMAGE = 2


#############################################################
# Main Overlay Class
#############################################################

class DetectionOverlay(QMainWindow):
    """Main class for transparent vehicle detection overlay"""

    def __init__(self, use_cuda=False):
        super().__init__()

        # Store CUDA preference
        self.use_cuda = use_cuda

        # Initialize screen info
        self.screen = QApplication.primaryScreen()
        self.screen_geometry = self.screen.geometry()
        self.screen_width = self.screen_geometry.width()
        self.screen_height = self.screen_geometry.height()

        print(f"Screen resolution: {self.screen_width}x{self.screen_height}")
        print(f"CUDA acceleration: {'Enabled' if self.use_cuda else 'Disabled'}")

        # Set window properties
        self.setup_window()

        # Initialize state
        self.init_state()

        # Initialize model and screen capture
        self.init_model()

        # Start processing timer
        self.start_processing()

    def setup_window(self):
        """Configure the overlay window"""
        # Set window flags for transparency and always-on-top
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(0, 0, self.screen_width, self.screen_height)

        # Central transparent widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

    def init_state(self):
        """Initialize state variables"""
        # Detection and display state
        self.detections = []
        self.display_mode = DISPLAY_MODE_TRANSPARENT
        self.current_frame = None
        self.fps = 0
        self.conf_threshold = DEFAULT_CONF_THRESHOLD

    def init_model(self):
        """Initialize the ONNX model and screen capture"""
        # Set execution providers based on CUDA preference
        if self.use_cuda:
            try:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.session = ort.InferenceSession(MODEL_PATH, providers=providers)
                self.cuda_available = True
                print("CUDA initialization successful")
            except Exception as e:
                print(f"CUDA initialization failed: {e}")
                print("Falling back to CPU")
                self.session = ort.InferenceSession(MODEL_PATH)
                self.cuda_available = False
        else:
            # CPU only
            self.session = ort.InferenceSession(MODEL_PATH)
            self.cuda_available = False

        # Get input details
        self.input_name = self.session.get_inputs()[0].name

        # Screen capture setup - use actual screen dimensions
        self.sct = mss()
        self.monitor = {
            "top": 0,
            "left": 0,
            "width": self.screen_width,
            "height": self.screen_height
        }

    def start_processing(self):
        """Start the frame processing timer"""
        self.processing_timer = QTimer(self)
        self.processing_timer.timeout.connect(self.process_frame)
        self.processing_timer.start(1)  # Run as fast as possible

    #############################################################
    # Frame Processing Methods
    #############################################################

    def process_frame(self):
        """Main processing function for each frame"""
        start_time = time.time()

        # Capture and prepare frame
        self.capture_frame()

        # Run detection
        self.detect_objects()

        # Calculate FPS
        self.fps = 1 / (time.time() - start_time)

        # Trigger repaint of the overlay
        self.update()

    def capture_frame(self):
        """Capture the current screen contents"""
        frame = np.array(self.sct.grab(self.monitor))
        self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def detect_objects(self):
        """Run object detection on the current frame"""
        # Preprocess
        input_tensor, orig_size, padding_info = self.preprocess(self.current_frame)

        # Inference
        outputs = self.session.run(None, {self.input_name: input_tensor})

        # Postprocess
        self.detections = self.postprocess(outputs, orig_size, padding_info)

    #############################################################
    # Model Processing Methods
    #############################################################

    def preprocess(self, frame):
        """Preprocess the frame for model input"""
        # Original dimensions
        h, w = frame.shape[:2]

        # Calculate scaling factor (longest side)
        scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize with aspect ratio
        resized = cv2.resize(frame, (new_w, new_h))

        # Create canvas with center placement
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        x_offset = (INPUT_SIZE - new_w) // 2
        y_offset = (INPUT_SIZE - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        # Normalize and transpose
        image = canvas.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)  # HWC to CHW
        input_tensor = np.expand_dims(image, axis=0)

        return input_tensor, (w, h), (x_offset, y_offset, scale)

    def postprocess(self, outputs, orig_size, padding_info):
        """Postprocess model outputs to get detections"""
        orig_w, orig_h = orig_size
        x_offset, y_offset, scale = padding_info

        predictions = np.squeeze(outputs[0]).T
        scores = np.max(predictions[:, 4:], axis=1)
        valid = scores > self.conf_threshold
        predictions = predictions[valid]
        scores = scores[valid]

        if len(scores) == 0:
            return []

        # Extract boxes in xywh format
        boxes = predictions[:, :4]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Convert xywh (center) to xyxy (corners)
        x_center = boxes[:, 0]
        y_center = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]

        x1 = (x_center - width / 2 - x_offset) / scale
        y1 = (y_center - height / 2 - y_offset) / scale
        x2 = (x_center + width / 2 - x_offset) / scale
        y2 = (y_center + height / 2 - y_offset) / scale

        # Stack corrected coordinates
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Clip boxes to screen boundaries
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, IOU_THRESHOLD)

        return [(boxes[i], scores[i], class_ids[i]) for i in indices]

    #############################################################
    # Drawing Methods
    #############################################################

    def paintEvent(self, event):
        """Main drawing method for the overlay"""
        painter = QPainter(self)

        # Draw background based on display mode
        self.draw_background(painter)

        # Draw information overlay
        self.draw_info_panel(painter)

        # Draw all detections
        self.draw_detections(painter)

    def draw_background(self, painter):
        """Draw the background based on current display mode"""
        if self.display_mode == DISPLAY_MODE_TRANSPARENT:
            # Transparent background
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.fillRect(self.rect(), Qt.transparent)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

        elif self.display_mode == DISPLAY_MODE_SEMI_TRANSPARENT:
            # Semi-transparent dark background
            painter.fillRect(self.rect(), QColor(0, 0, 0, 100))

        elif self.display_mode == DISPLAY_MODE_FULL_IMAGE and self.current_frame is not None:
            # Draw the full captured image
            height, width = self.current_frame.shape[:2]
            qimg = QImage(self.current_frame.data, width, height,
                          self.current_frame.strides[0], QImage.Format_RGB888).rgbSwapped()
            painter.drawImage(0, 0, qimg)

    def draw_info_panel(self, painter):
        """Draw the information panel with stats"""
        # Set font for text
        font = QFont("Arial", 12, QFont.Bold)
        painter.setFont(font)

        # Draw info panel background
        painter.fillRect(5, 5, 350, 130, QColor(0, 0, 0, 180))

        # Draw stats text
        painter.setPen(QColor(255, 50, 50))
        painter.drawText(10, 30, f"FPS: {self.fps:.1f}")
        painter.drawText(10, 50, f"Detections: {len(self.detections)}")
        painter.drawText(10, 70, f"Display Mode: {self.display_mode} (D to change)")
        painter.drawText(10, 90, f"Screen: {self.screen_width}x{self.screen_height}")
        painter.drawText(10, 110, f"Confidence: {self.conf_threshold:.2f} (↑/↓ to adjust)")

        # Display CUDA status with appropriate color
        if self.cuda_available:
            painter.setPen(QColor(50, 255, 50))  # Green for CUDA enabled
            painter.drawText(10, 130, "CUDA: Enabled")
        else:
            painter.setPen(QColor(255, 255, 50))  # Yellow for CPU only
            painter.drawText(10, 130, "CUDA: Disabled (CPU mode)")

    def draw_detections(self, painter):
        """Draw all detected objects"""
        # Default color for all detections
        box_color = QColor(0, 255, 0)
        pen = QPen(box_color)
        pen.setWidth(3)

        for box, score, class_id in self.detections:
            x1, y1, x2, y2 = map(int, box)

            # Ensure valid box dimensions
            if x2 <= x1 or y2 <= y1:
                continue

            # Set pen for drawing
            painter.setPen(pen)

            # Draw rectangle with semi-transparent fill in transparent mode
            if self.display_mode == DISPLAY_MODE_TRANSPARENT:
                painter.fillRect(x1, y1, x2 - x1, y2 - y1, QColor(0, 255, 0, 40))

            # Draw rectangle outline
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

            # Prepare label text
            class_name = CLASS_NAMES[class_id]
            label = f"{class_name}: {score:.2f}"

            # Draw label background
            text_width = painter.fontMetrics().width(label)
            painter.fillRect(x1, y1 - 25, text_width + 10, 20, QColor(0, 0, 0, 180))

            # Draw label text
            painter.setPen(box_color)
            painter.drawText(x1 + 5, y1 - 10, label)

    #############################################################
    # Event Handlers
    #############################################################

    def keyPressEvent(self, event):
        """Handle keyboard input"""
        if event.key() == Qt.Key_Q:
            # Exit application
            self.close()

        elif event.key() == Qt.Key_D:
            # Toggle display mode
            self.display_mode = (self.display_mode + 1) % 3
            print(f"Display mode changed to: {self.display_mode}")

        elif event.key() == Qt.Key_Up:
            # Increase confidence threshold
            self.conf_threshold = min(MAX_CONF_THRESHOLD, self.conf_threshold + CONF_STEP)
            print(f"Confidence threshold increased to: {self.conf_threshold:.2f}")

        elif event.key() == Qt.Key_Down:
            # Decrease confidence threshold
            self.conf_threshold = max(MIN_CONF_THRESHOLD, self.conf_threshold - CONF_STEP)
            print(f"Confidence threshold decreased to: {self.conf_threshold:.2f}")


#############################################################
# Main Application
#############################################################

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Vehicle Detection Overlay')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA acceleration if available')
    args = parser.parse_args()

    # Create application
    app = QApplication(sys.argv)

    # Create and show overlay with CUDA preference
    overlay = DetectionOverlay(use_cuda=args.cuda)
    overlay.show()

    # Run event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()