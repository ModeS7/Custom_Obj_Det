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
from collections import deque
import threading

# Configuration Parameters
MODEL_PATH = "best1.onnx"  # Path to the ONNX model file
CLASS_NAMES = ["vehicle"]  # List of class names the model can detect
DEFAULT_CONF_THRESHOLD, MIN_CONF_THRESHOLD, MAX_CONF_THRESHOLD = 0.25, 0.1, 0.9  # Confidence thresholds for detection
CONF_STEP = 0.05  # Step size for adjusting confidence threshold
IOU_THRESHOLD = 0.45  # Intersection over Union threshold for Non-Maximum Suppression
INPUT_SIZE = 640  # Input size for the YOLO model
DISPLAY_MODE_TRANSPARENT, DISPLAY_MODE_SEMI_TRANSPARENT, DISPLAY_MODE_FULL_IMAGE = 0, 1, 2  # Display mode constants
FRAME_INTERVAL = 20  # Interval between frame captures in milliseconds
TRACKING_HISTORY = 5  # Number of past positions to track for trajectory
DXCAM_TARGET_FPS = 60  # Target frame rate for the DXcam capture


class EnhancedTracker:
    """
    Object tracker that maintains object identities across frames.
    Uses IoU matching with velocity prediction to reduce flickering.
    """

    def __init__(self, max_disappeared=7, iou_threshold=0.3, maintain_threshold=0.15, alpha=0.7):
        """
        Initialize the tracker with configurable parameters.

        Args:
            max_disappeared: Maximum number of frames an object can disappear before being removed
            iou_threshold: Minimum IoU required to match detections with existing objects
            maintain_threshold: Minimum confidence to keep tracking an object
            alpha: Smoothing factor for box coordinates (higher = more weight to new detection)
        """
        self.next_object_id = 0
        self.objects = {}  # Format: {ID: (box, score, class_id, disappeared_count, velocity)}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.maintain_threshold = maintain_threshold
        self.alpha = alpha

    def _calculate_iou(self, boxA, boxB):
        """Calculate Intersection over Union between two bounding boxes"""
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def _smooth_box(self, old_box, new_box):
        """Apply exponential smoothing to box coordinates to reduce jitter"""
        return [old_box[i] * (1 - self.alpha) + new_box[i] * self.alpha for i in range(4)]

    def _calculate_velocity(self, old_box, new_box):
        """Calculate velocity vector (dx, dy) based on centers of old and new boxes"""
        old_center = [(old_box[0] + old_box[2]) / 2, (old_box[1] + old_box[3]) / 2]
        new_center = [(new_box[0] + new_box[2]) / 2, (new_box[1] + new_box[3]) / 2]
        return [new_center[0] - old_center[0], new_center[1] - old_center[1]]

    def _predict_box(self, box, velocity):
        """Predict new box position based on previous velocity"""
        if velocity is None:
            return box
        dx, dy = velocity
        return [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]

    def update(self, detections):
        """
        Update tracker state with new detections.

        This method:
        1. Handles case with no detections by updating disappeared counts
        2. Matches detections with existing objects using IoU
        3. Updates matched objects with new positions and resets disappeared count
        4. Registers new objects for unmatched detections
        5. Updates and removes objects that have disappeared for too long

        Args:
            detections: List of (box, score, class_id) tuples from current frame

        Returns:
            Dictionary of tracked objects with their IDs
        """
        # Handle case with no detections
        if len(detections) == 0:
            for object_id in list(self.objects.keys()):
                box, score, class_id, disappeared_count, velocity = self.objects[object_id]

                # Predict new position based on velocity
                if velocity is not None and disappeared_count < self.max_disappeared // 2:
                    predicted_box = self._predict_box(box, velocity)
                    self.objects[object_id] = (predicted_box, score * 0.9, class_id, disappeared_count + 1, velocity)
                else:
                    self.objects[object_id] = (box, score * 0.9, class_id, disappeared_count + 1, velocity)

                # Remove objects that have disappeared for too long
                if self.objects[object_id][3] > self.max_disappeared:
                    del self.objects[object_id]
            return self.objects

        # Initialize matching variables
        matched_indices = set()
        unmatched_detections = []

        # Create predicted objects for matching
        predicted_objects = {}
        for object_id, (box, score, class_id, disappeared, velocity) in self.objects.items():
            if velocity is not None and disappeared < self.max_disappeared // 2:
                predicted_box = self._predict_box(box, velocity)
                predicted_objects[object_id] = (predicted_box, score, class_id, disappeared, velocity)
            else:
                predicted_objects[object_id] = (box, score, class_id, disappeared, velocity)

        # Match detections with existing objects
        for i, (box, score, class_id) in enumerate(detections):
            max_iou = self.iou_threshold
            match_id = None

            for object_id, (existing_box, existing_score, existing_class, _, _) in predicted_objects.items():
                if class_id != existing_class:
                    continue

                iou = self._calculate_iou(box, existing_box)
                if iou > max_iou:
                    max_iou = iou
                    match_id = object_id

            if match_id is not None:
                # Update matched object
                orig_box, orig_score, orig_class, disappeared, orig_velocity = self.objects[match_id]
                velocity = self._calculate_velocity(orig_box, box)
                smoothed_box = self._smooth_box(orig_box, box)
                updated_score = max(score, orig_score * 0.95)
                self.objects[match_id] = (smoothed_box, updated_score, class_id, 0, velocity)
                matched_indices.add(match_id)
            elif score >= self.maintain_threshold:
                unmatched_detections.append((box, score, class_id))

        # Update disappeared objects
        for object_id in list(self.objects.keys()):
            if object_id not in matched_indices:
                box, score, class_id, disappeared_count, velocity = self.objects[object_id]

                if velocity is not None and disappeared_count < self.max_disappeared // 2:
                    predicted_box = self._predict_box(box, velocity)
                    self.objects[object_id] = (predicted_box, score * 0.9, class_id, disappeared_count + 1, velocity)
                else:
                    self.objects[object_id] = (box, score * 0.9, class_id, disappeared_count + 1, velocity)

                if (self.objects[object_id][3] > self.max_disappeared or
                        self.objects[object_id][1] < self.maintain_threshold * 0.5):
                    del self.objects[object_id]

        # Register new objects
        for box, score, class_id in unmatched_detections:
            self.objects[self.next_object_id] = (box, score, class_id, 0, None)
            self.next_object_id += 1

        return self.objects

    def get_active_objects(self):
        """
        Return list of active objects for display.
        Only includes objects that haven't disappeared for too long.
        """
        return [(self.objects[object_id][0], self.objects[object_id][1], self.objects[object_id][2])
                for object_id in self.objects if self.objects[object_id][3] <= self.max_disappeared // 2]


class DetectionThread(QThread):
    """
    Worker thread that handles model inference to detect vehicles.
    Runs the ONNX model in a separate thread to prevent UI freezing.
    """
    detection_complete = pyqtSignal(list, float)  # Signal emitted when detection is complete

    def __init__(self, model_path, use_cuda=False):
        """
        Initialize the detection thread with model path and CUDA option.

        Args:
            model_path: Path to the ONNX model file
            use_cuda: Whether to use CUDA acceleration if available
        """
        super().__init__()
        self.running = True
        self.frame_ready = threading.Event()  # Event to signal when a new frame is ready for processing
        self.frame = None
        self.conf_threshold = DEFAULT_CONF_THRESHOLD
        self.use_cuda = use_cuda
        self.model_path = model_path
        self.setup_model()

    def setup_model(self):
        """Initialize the ONNX model with CUDA support if available and requested"""
        self.cuda_available = False

        if self.use_cuda:
            try:
                # First check if CUDA provider is actually available in providers list
                if 'CUDAExecutionProvider' not in ort.get_available_providers():
                    print("CUDA requested but not available in ONNX Runtime providers")
                    print("Available providers:", ort.get_available_providers())
                    print("Falling back to CPU execution")
                    self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
                    return

                # Try with CUDA provider
                providers = [
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB GPU memory limit
                    }),
                    'CPUExecutionProvider'
                ]

                # Try creating session with CUDA
                try:
                    self.session = ort.InferenceSession(self.model_path, providers=providers)
                    active_provider = self.session.get_providers()[0]

                    if 'CUDA' in active_provider:
                        self.cuda_available = True
                        print(f"CUDA initialization successful - Using provider: {active_provider}")
                    else:
                        print(f"Warning: CUDA requested but not used. Active provider: {active_provider}")
                        print("Check if CUDA and cuDNN requirements are met")
                except Exception as e:
                    print(f"Error creating session with CUDA: {str(e)}")
                    print("Falling back to CPU execution")
                    self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            except Exception as e:
                print(f"CUDA initialization failed: {str(e)}")
                print("Falling back to CPU execution")
                self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        else:
            # CPU only
            print("Using CPU execution (CUDA not requested)")
            self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])

        # Get input details
        self.input_name = self.session.get_inputs()[0].name
        print(f"Model loaded successfully with input name: {self.input_name}")
        print(f"Active providers: {self.session.get_providers()}")

    def set_frame(self, frame):
        """Set a new frame for processing and signal the worker thread"""
        self.frame = frame
        self.frame_ready.set()

    def set_conf_threshold(self, value):
        """Update the confidence threshold for detection"""
        self.conf_threshold = value

    def stop(self):
        """Stop the thread safely"""
        self.running = False
        self.frame_ready.set()  # Wake up the thread if it's waiting
        self.wait()

    def run(self):
        """Main thread loop that processes frames as they become available"""
        while self.running:
            self.frame_ready.wait()  # Wait for a new frame
            self.frame_ready.clear()

            if not self.running or self.frame is None:
                continue

            start_time = time.time()
            detections = self.process_frame(self.frame)
            fps = 1 / (time.time() - start_time)

            self.detection_complete.emit(detections, fps)

    def process_frame(self, frame):
        """Process a frame through the ONNX model pipeline"""
        try:
            input_tensor, orig_size, padding_info = self.preprocess(frame)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            detections = self.postprocess(outputs, orig_size, padding_info)
            return detections
        except Exception as e:
            print(f"Error processing frame: {e}")
            return []

    def preprocess(self, frame):
        """
        Preprocess the frame for the YOLO model.

        Steps:
        1. Resize the image maintaining aspect ratio
        2. Pad to square input size
        3. Normalize pixel values
        4. Transpose from HWC to CHW format (PyTorch/ONNX convention)
        5. Add batch dimension

        Returns:
            input_tensor: Preprocessed tensor ready for the model
            orig_size: Original image dimensions (width, height)
            padding_info: Information about padding and scaling for postprocessing
        """
        h, w = frame.shape[:2]
        scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create a square canvas and place the resized image in the center
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)  # Fill with gray (114)
        x_offset, y_offset = (INPUT_SIZE - new_w) // 2, (INPUT_SIZE - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        # Normalize and convert to correct format
        image = canvas.astype(np.float32) / 255.0  # Normalize to [0,1]
        input_tensor = np.expand_dims(image.transpose(2, 0, 1), axis=0)  # HWC to CHW and add batch dim

        return input_tensor, (w, h), (x_offset, y_offset, scale)

    def postprocess(self, outputs, orig_size, padding_info):
        """
        Process model outputs to return detections.

        Steps:
        1. Extract predictions and filter by confidence threshold
        2. Convert normalized coordinates to pixel coordinates
        3. Convert from center coordinates to corner coordinates
        4. Scale back to original image size
        5. Apply Non-Maximum Suppression

        Returns:
            List of (box, score, class_id) tuples for valid detections
        """
        orig_w, orig_h = orig_size
        x_offset, y_offset, scale = padding_info

        predictions = np.squeeze(outputs[0]).T
        scores = np.max(predictions[:, 4:], axis=1)
        valid = scores > self.conf_threshold
        predictions = predictions[valid]
        scores = scores[valid]

        if len(scores) == 0:
            return []

        boxes = predictions[:, :4]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Convert xywh to xyxy (center coordinates to corner coordinates)
        x_center, y_center = boxes[:, 0], boxes[:, 1]
        width, height = boxes[:, 2], boxes[:, 3]

        # Remove padding and rescale to original image size
        x1 = (x_center - width / 2 - x_offset) / scale
        y1 = (y_center - height / 2 - y_offset) / scale
        x2 = (x_center + width / 2 - x_offset) / scale
        y2 = (y_center + height / 2 - y_offset) / scale

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Clip boxes to screen boundaries
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, IOU_THRESHOLD)

        return [(boxes[i], scores[i], class_ids[i]) for i in indices]


class DetectionOverlay(QMainWindow):
    """
    Main application window that displays the detection overlay.
    Renders detection boxes on top of the screen in real-time.
    """

    def __init__(self, use_cuda=False):
        """
        Initialize the overlay window with all components.

        Args:
            use_cuda: Whether to use CUDA acceleration if available
        """
        super().__init__()
        self.use_cuda = use_cuda

        # Initialize screen info
        self.screen = QApplication.primaryScreen()
        self.screen_geometry = self.screen.geometry()
        self.screen_width = self.screen_geometry.width()
        self.screen_height = self.screen_geometry.height()

        # Setup window and initialize
        self.setup_window()
        self.init_state()
        self.init_processing_thread()
        self.tracker = EnhancedTracker(max_disappeared=7, iou_threshold=0.3,
                                       maintain_threshold=0.15, alpha=0.7)
        self.start_processing()

    def setup_window(self):
        """Configure the window properties for overlay display"""
        # Create a frameless, always-on-top, transparent window
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(0, 0, self.screen_width, self.screen_height)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

    def init_state(self):
        """Initialize application state and variables"""
        self.detections = []
        self.display_mode = DISPLAY_MODE_TRANSPARENT
        self.current_frame = None
        self.fps = 0
        self.conf_threshold = DEFAULT_CONF_THRESHOLD
        self.detection_fps = 0
        self.capture_fps = 0
        self.last_capture_time = time.time()

        # Initialize screen capture using DXcam (faster than traditional methods)
        try:
            self.camera = dxcam.create(output_idx=0, output_color="BGR")
        except Exception as e:
            print(f"Error initializing DXcam: {e}")
            self.camera = None

        # FPS tracking queues for smoothing
        self.fps_queue = deque(maxlen=30)
        self.capture_fps_queue = deque(maxlen=30)

    def init_processing_thread(self):
        """Initialize and start the detection thread"""
        self.detection_thread = DetectionThread(MODEL_PATH, self.use_cuda)
        self.detection_thread.detection_complete.connect(self.on_detection_complete)
        self.detection_thread.start()

    def on_detection_complete(self, detections, fps):
        """
        Called when detection processing completes.
        Updates tracking and triggers UI refresh.
        """
        self.detection_fps = fps
        self.fps_queue.append(fps)
        self.fps = sum(self.fps_queue) / len(self.fps_queue) if self.fps_queue else 0

        self.tracker.update(detections)
        self.detections = self.tracker.get_active_objects()
        self.update()  # Trigger a repaint of the UI

    def start_processing(self):
        """Start the timer for regular frame capture"""
        self.processing_timer = QTimer(self)
        self.processing_timer.timeout.connect(self.capture_frame)
        self.processing_timer.start(FRAME_INTERVAL)

    def capture_frame(self):
        """Capture a new frame from the screen and send it for processing"""
        try:
            # Calculate and update capture FPS
            current_time = time.time()
            elapsed = current_time - self.last_capture_time
            self.last_capture_time = current_time

            if elapsed > 0:
                self.capture_fps_queue.append(1.0 / elapsed)
                self.capture_fps = sum(self.capture_fps_queue) / len(self.capture_fps_queue)

            if self.camera is not None:
                frame = self.camera.grab()
                if frame is None:
                    return

                self.current_frame = frame
                self.detection_thread.set_conf_threshold(self.conf_threshold)
                self.detection_thread.set_frame(frame.copy())

        except Exception as e:
            print(f"Error capturing frame: {e}")

    def paintEvent(self, event):
        """
        Qt event handler for painting the window.
        Draws the background, information panel, and detection boxes.
        """
        painter = QPainter(self)

        # Draw background based on display mode
        if self.display_mode == DISPLAY_MODE_TRANSPARENT:
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.fillRect(self.rect(), Qt.transparent)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        elif self.display_mode == DISPLAY_MODE_SEMI_TRANSPARENT:
            painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
        elif self.display_mode == DISPLAY_MODE_FULL_IMAGE and self.current_frame is not None:
            height, width = self.current_frame.shape[:2]
            qimg = QImage(self.current_frame.data, width, height,
                          self.current_frame.strides[0], QImage.Format_RGB888).rgbSwapped()
            painter.drawImage(0, 0, qimg)

        # Draw info panel with performance metrics and settings
        font = QFont("Arial", 12, QFont.Bold)
        painter.setFont(font)
        painter.fillRect(5, 5, 350, 190, QColor(0, 0, 0, 180))

        painter.setPen(QColor(255, 50, 50))
        info_lines = [
            f"FPS: {self.fps:.1f}",
            f"Processing FPS: {self.detection_fps:.1f}",
            f"Capture FPS: {self.capture_fps:.1f}",
            f"Detections: {len(self.detections)}",
            f"Display Mode: {self.display_mode} (D to change)",
            f"Screen: {self.screen_width}x{self.screen_height}",
            f"Confidence: {self.conf_threshold:.2f} (↑/↓ to adjust)",
            f"Tracking: Enhanced (reduced flickering)"
        ]

        for i, line in enumerate(info_lines):
            painter.drawText(10, 30 + i * 20, line)

        # Display CUDA status with appropriate color
        if self.detection_thread.cuda_available:
            painter.setPen(QColor(50, 255, 50))
            painter.drawText(10, 190, "CUDA: Enabled")
        else:
            painter.setPen(QColor(255, 255, 50))
            painter.drawText(10, 190, "CUDA: Disabled (CPU mode)")

        # Draw detection boxes and labels
        box_color = QColor(0, 255, 0)
        pen = QPen(box_color)
        pen.setWidth(3)

        for box, score, class_id in self.detections:
            x1, y1, x2, y2 = map(int, box)

            if x2 <= x1 or y2 <= y1:
                continue

            painter.setPen(pen)

            if self.display_mode == DISPLAY_MODE_TRANSPARENT:
                painter.fillRect(x1, y1, x2 - x1, y2 - y1, QColor(0, 255, 0, 40))

            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

            class_name = CLASS_NAMES[class_id]
            label = f"{class_name}: {score:.2f}"

            text_width = painter.fontMetrics().width(label)
            painter.fillRect(x1, y1 - 25, text_width + 10, 20, QColor(0, 0, 0, 180))

            painter.setPen(box_color)
            painter.drawText(x1 + 5, y1 - 10, label)

    def keyPressEvent(self, event):
        """Handle keyboard events for controlling the application"""
        if event.key() == Qt.Key_Q:
            self.close()  # Q to quit
        elif event.key() == Qt.Key_D:
            self.display_mode = (self.display_mode + 1) % 3  # D to cycle display modes
        elif event.key() == Qt.Key_Up:
            self.conf_threshold = min(MAX_CONF_THRESHOLD,
                                      self.conf_threshold + CONF_STEP)  # Up arrow to increase confidence
        elif event.key() == Qt.Key_Down:
            self.conf_threshold = max(MIN_CONF_THRESHOLD,
                                      self.conf_threshold - CONF_STEP)  # Down arrow to decrease confidence

    def closeEvent(self, event):
        """Clean up resources when the window is closed"""
        if hasattr(self, 'detection_thread'):
            self.detection_thread.stop()

        if hasattr(self, 'camera') and self.camera is not None:
            self.camera.release()

        super().closeEvent(event)


def main():
    """Main function to start the application with command line arguments"""
    # Diagnostic check for CUDA availability
    print("===== ONNX Runtime GPU Diagnostic =====")
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")

    if 'CUDAExecutionProvider' in ort.get_available_providers():
        print("CUDA is available for ONNX Runtime")
        # Try to get CUDA device properties if torch is available
        try:
            import torch
            if torch.cuda.is_available():
                print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
                print(f"CUDA device count: {torch.cuda.device_count()}")
        except ImportError:
            print("PyTorch not available for additional CUDA diagnostics")
    else:
        print("CUDA provider not found in ONNX Runtime")
        print("CUDA requirements: cuDNN 9.* and CUDA 12.*, and the latest MSVC runtime.")
        print("See: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements")
    print("=======================================")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Vehicle Detection Overlay')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA acceleration if available')
    parser.add_argument('--region', type=str, help='Region to capture in format left,top,right,bottom')
    args = parser.parse_args()

    # Start the application
    app = QApplication(sys.argv)
    overlay = DetectionOverlay(use_cuda=args.cuda)
    overlay.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()