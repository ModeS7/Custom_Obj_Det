import cv2
import numpy as np
import onnxruntime as ort
from mss import mss
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtGui import QPainter, QColor, QPen, QFont
from PyQt5.QtCore import Qt, QTimer

# Configuration
MODEL_PATH = "best.onnx"
CLASS_NAMES = ["vehicle"]  # Update with your classes
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
INPUT_SIZE = 640


class DetectionOverlay(QMainWindow):
    def __init__(self, screen_width, screen_height):
        super().__init__()

        # Set window flags for transparency and always on top
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(0, 0, screen_width, screen_height)

        # Central transparent widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Store detections
        self.detections = []
        self.fps = 0

        # Debug modes: 0=normal overlay, 1=debug with background, 2=debug with full image
        self.debug_mode = 0
        self.current_frame = None  # Store current frame for debug visualization

        # Setup ONNX model
        self.session = ort.InferenceSession(MODEL_PATH)
        self.input_name = self.session.get_inputs()[0].name

        # Screen capture - use the exact same monitor settings as original code
        self.sct = mss()
        self.monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}  # Match original resolution

        # Store original dimensions for scaling if needed
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Create a processing timer - use 1ms for maximum speed
        self.processing_timer = QTimer(self)
        self.processing_timer.timeout.connect(self.process_frame)
        self.processing_timer.start(1)  # Run as fast as possible

    def process_frame(self):
        start_time = time.time()

        # Capture screen
        frame = np.array(self.sct.grab(self.monitor))
        orig_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Store the frame for debug view
        self.current_frame = orig_frame.copy()

        # Preprocess
        input_tensor, orig_size, padding_info = self.preprocess(orig_frame)

        # Inference
        outputs = self.session.run(None, {self.input_name: input_tensor})

        # Postprocess
        self.detections = self.postprocess(outputs, orig_size, padding_info)

        # Calculate FPS
        self.fps = 1 / (time.time() - start_time)

        # Trigger repaint
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)

        # Different debug modes
        if self.debug_mode == 0:
            # Normal transparent overlay
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.fillRect(self.rect(), Qt.transparent)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        elif self.debug_mode == 1:
            # Semi-transparent background
            painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
        elif self.debug_mode == 2 and self.current_frame is not None:
            # Draw the actual image
            height, width = self.current_frame.shape[:2]
            qimg = QImage(self.current_frame.data, width, height,
                          self.current_frame.strides[0], QImage.Format_RGB888).rgbSwapped()
            painter.drawImage(0, 0, qimg)

        # Set font for text
        font = QFont("Arial", 12, QFont.Bold)
        painter.setFont(font)

        # Draw FPS and detection count with background for better visibility
        painter.fillRect(5, 5, 200, 70, QColor(0, 0, 0, 180))  # Semi-transparent background
        painter.setPen(QColor(255, 50, 50))
        painter.drawText(10, 30, f"FPS: {self.fps:.1f}")
        painter.drawText(10, 50, f"Detections: {len(self.detections)}")
        painter.drawText(10, 70, f"Mode: {self.debug_mode} (Press D to change)")

        # Draw detections
        for box, score, class_id in self.detections:
            x1, y1, x2, y2 = map(int, box)

            # Ensure valid box dimensions
            if x2 > x1 and y2 > y1:
                # Set color and pen width for the bounding box - make it more visible
                pen = QPen(QColor(0, 255, 0))
                pen.setWidth(3)  # Thicker line
                painter.setPen(pen)

                # Draw rectangle with fill for better visibility
                if self.debug_mode == 0:
                    # In transparent mode, add background to rectangle
                    painter.fillRect(x1, y1, x2 - x1, y2 - y1, QColor(0, 255, 0, 40))

                painter.drawRect(x1, y1, x2 - x1, y2 - y1)

                # Make label more visible with background
                label = f"{CLASS_NAMES[class_id]}: {score:.2f}"
                text_width = painter.fontMetrics().width(label)
                painter.fillRect(x1, y1 - 25, text_width + 10, 20, QColor(0, 0, 0, 180))
                painter.setPen(QColor(0, 255, 0))
                painter.drawText(x1 + 5, y1 - 10, label)
                painter.setPen(QColor(0, 255, 0))

    def preprocess(self, frame):
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
        orig_w, orig_h = orig_size
        x_offset, y_offset, scale = padding_info

        predictions = np.squeeze(outputs[0]).T
        scores = np.max(predictions[:, 4:], axis=1)
        valid = scores > CONF_THRESHOLD
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
        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)

        return [(boxes[i], scores[i], class_ids[i]) for i in indices]

    def keyPressEvent(self, event):
        # Exit on 'q' press
        if event.key() == Qt.Key_Q:
            self.close()
        # Toggle debug mode with 'd' key
        elif event.key() == Qt.Key_D:
            self.debug_mode = (self.debug_mode + 1) % 3
            print(f"Debug mode changed to: {self.debug_mode}")


def main():
    app = QApplication([])

    # Get screen resolution
    screen = app.primaryScreen()
    screen_size = screen.size()
    screen_width, screen_height = screen_size.width(), screen_size.height()

    # Create overlay
    overlay = DetectionOverlay(screen_width, screen_height)
    overlay.show()

    app.exec_()


if __name__ == "__main__":
    main()