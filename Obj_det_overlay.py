import cv2
import numpy as np
import onnxruntime as ort
from mss import mss
import time
import sys
import argparse
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QImage
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from collections import deque
import threading

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

# Performance Configuration
FRAME_INTERVAL = 20  # ms between frames (50 FPS target instead of 1000)
TRACKING_HISTORY = 5  # Number of frames to maintain tracking


#############################################################
# Object Tracker Class
#############################################################

class SimpleTracker:
    """Simple object tracker to reduce detection flicker"""

    def __init__(self, max_disappeared=5, iou_threshold=0.3):
        self.next_object_id = 0
        self.objects = {}  # Format: {ID: (box, score, class_id, disappeared_count)}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold

    def _calculate_iou(self, boxA, boxB):
        """Calculate IoU between two boxes"""
        # Determine intersection coordinates
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute intersection area
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # Compute area of both bounding boxes
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # Compute IoU
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def update(self, detections):
        """Update tracker with new detections"""
        # If no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.objects.keys()):
                box, score, class_id, disappeared_count = self.objects[object_id]
                self.objects[object_id] = (box, score, class_id, disappeared_count + 1)

                # Remove objects that have disappeared for too long
                if self.objects[object_id][3] > self.max_disappeared:
                    del self.objects[object_id]

            return self.objects

        # Initialize array for current frame's matched indices
        matched_indices = set()
        unmatched_detections = []

        # Try to match new detections with existing objects
        for i, (box, score, class_id) in enumerate(detections):
            max_iou = self.iou_threshold
            match_id = None

            for object_id, (existing_box, existing_score, existing_class, _) in self.objects.items():
                # Only match objects of the same class
                if class_id != existing_class:
                    continue

                iou = self._calculate_iou(box, existing_box)

                if iou > max_iou:
                    max_iou = iou
                    match_id = object_id

            if match_id is not None:
                # Update the matched object with new detection
                self.objects[match_id] = (box, score, class_id, 0)  # Reset disappeared counter
                matched_indices.add(match_id)
            else:
                unmatched_detections.append((box, score, class_id))

        # Check for objects that have disappeared
        for object_id in list(self.objects.keys()):
            if object_id not in matched_indices:
                box, score, class_id, disappeared_count = self.objects[object_id]
                self.objects[object_id] = (box, score, class_id, disappeared_count + 1)

                # Remove objects that have disappeared for too long
                if self.objects[object_id][3] > self.max_disappeared:
                    del self.objects[object_id]

        # Register new objects
        for box, score, class_id in unmatched_detections:
            self.objects[self.next_object_id] = (box, score, class_id, 0)
            self.next_object_id += 1

        return self.objects

    def get_active_objects(self):
        """Return only the active objects (not disappeared)"""
        return [(self.objects[object_id][0], self.objects[object_id][1], self.objects[object_id][2])
                for object_id in self.objects if self.objects[object_id][3] <= self.max_disappeared // 2]


#############################################################
# Threading Classes for Parallel Processing
#############################################################

class DetectionThread(QThread):
    """Thread for running object detection processing"""

    # Signal to emit when detection is complete
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
        # Set execution providers based on CUDA preference
        if self.use_cuda:
            try:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.session = ort.InferenceSession(self.model_path, providers=providers)
                self.cuda_available = True
                print("CUDA initialization successful")
            except Exception as e:
                print(f"CUDA initialization failed: {e}")
                print("Falling back to CPU")
                self.session = ort.InferenceSession(self.model_path)
                self.cuda_available = False
        else:
            # CPU only
            self.session = ort.InferenceSession(self.model_path)
            self.cuda_available = False

        # Get input details
        self.input_name = self.session.get_inputs()[0].name

    def set_frame(self, frame):
        """Set frame to be processed"""
        self.frame = frame
        self.frame_ready.set()

    def set_conf_threshold(self, value):
        """Update confidence threshold"""
        self.conf_threshold = value

    def stop(self):
        """Stop the detection thread"""
        self.running = False
        self.frame_ready.set()  # Release any waiting frame_ready.wait()
        self.wait()

    def run(self):
        """Main thread loop"""
        while self.running:
            # Wait for a new frame to be ready
            self.frame_ready.wait()
            self.frame_ready.clear()

            if not self.running:
                break

            # Skip if no frame is available
            if self.frame is None:
                continue

            # Process frame
            start_time = time.time()
            detections = self.process_frame(self.frame)
            fps = 1 / (time.time() - start_time)

            # Emit results
            self.detection_complete.emit(detections, fps)

    def process_frame(self, frame):
        """Process a frame to detect objects"""
        try:
            # Preprocess
            input_tensor, orig_size, padding_info = self.preprocess(frame)

            # Inference
            outputs = self.session.run(None, {self.input_name: input_tensor})

            # Postprocess
            detections = self.postprocess(outputs, orig_size, padding_info)
            return detections
        except Exception as e:
            print(f"Error processing frame: {e}")
            return []

    def preprocess(self, frame):
        """Preprocess the frame for model input"""
        # Original dimensions
        h, w = frame.shape[:2]

        # Calculate scaling factor (longest side)
        scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize with aspect ratio (use INTER_LINEAR for faster resize)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

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

        # Initialize model and screen capture in separate thread
        self.init_processing_thread()

        # Initialize tracker
        self.tracker = SimpleTracker(max_disappeared=5, iou_threshold=0.3)

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
        self.detection_fps = 0

        # Initialize screen capture
        self.sct = mss()
        self.monitor = {
            "top": 0,
            "left": 0,
            "width": self.screen_width,
            "height": self.screen_height
        }

        # Frame stats queue for FPS averaging
        self.fps_queue = deque(maxlen=10)

    def init_processing_thread(self):
        """Initialize the detection processing thread"""
        self.detection_thread = DetectionThread(MODEL_PATH, self.use_cuda)
        self.detection_thread.detection_complete.connect(self.on_detection_complete)
        self.detection_thread.start()

    def on_detection_complete(self, detections, fps):
        """Handle detection results from thread"""
        # Update FPS stats
        self.detection_fps = fps
        self.fps_queue.append(fps)
        self.fps = sum(self.fps_queue) / len(self.fps_queue) if self.fps_queue else 0

        # Update tracker with new detections
        self.tracker.update(detections)
        self.detections = self.tracker.get_active_objects()

        # Request repaint of the window
        self.update()

    def start_processing(self):
        """Start the frame processing timer"""
        self.processing_timer = QTimer(self)
        self.processing_timer.timeout.connect(self.capture_frame)
        self.processing_timer.start(FRAME_INTERVAL)  # 50 FPS target

    def capture_frame(self):
        """Capture the current screen contents and submit for processing"""
        try:
            # Capture frame
            frame = np.array(self.sct.grab(self.monitor))

            # Convert colorspace (only once, before submitting to thread)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Store current frame for display if needed
            self.current_frame = frame_rgb

            # Submit frame to detection thread
            self.detection_thread.set_conf_threshold(self.conf_threshold)
            self.detection_thread.set_frame(frame_rgb.copy())

        except Exception as e:
            print(f"Error capturing frame: {e}")

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
        painter.fillRect(5, 5, 350, 150, QColor(0, 0, 0, 180))

        # Draw stats text
        painter.setPen(QColor(255, 50, 50))
        painter.drawText(10, 30, f"FPS: {self.fps:.1f}")
        painter.drawText(10, 50, f"Processing FPS: {self.detection_fps:.1f}")
        painter.drawText(10, 70, f"Detections: {len(self.detections)}")
        painter.drawText(10, 90, f"Display Mode: {self.display_mode} (D to change)")
        painter.drawText(10, 110, f"Screen: {self.screen_width}x{self.screen_height}")
        painter.drawText(10, 130, f"Confidence: {self.conf_threshold:.2f} (↑/↓ to adjust)")

        # Display CUDA status with appropriate color
        if self.detection_thread.cuda_available:
            painter.setPen(QColor(50, 255, 50))  # Green for CUDA enabled
            painter.drawText(10, 150, "CUDA: Enabled")
        else:
            painter.setPen(QColor(255, 255, 50))  # Yellow for CPU only
            painter.drawText(10, 150, "CUDA: Disabled (CPU mode)")

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

    def closeEvent(self, event):
        """Clean up resources when closing"""
        # Stop detection thread
        if hasattr(self, 'detection_thread'):
            self.detection_thread.stop()

        super().closeEvent(event)


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