"""
YOLOv12 Screen Capture Detection - With fixed NMS implementation
"""

import cv2
import numpy as np
import mss
import time
import argparse
from PIL import Image, ImageTk
import tkinter as tk
import os
import sys
from onnxruntime import InferenceSession

class YOLOv12ScreenDetector:
    def __init__(self, model_path, class_names_path, confidence=0.01, iou_threshold=0.45):
        """Initialize the YOLOv12 screen detector."""
        # Load the class names
        try:
            with open(class_names_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"Error: Class names file not found at {class_names_path}")
            sys.exit(1)

        # Generate random colors for each class
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)

        # Load the ONNX model
        try:
            self.session = InferenceSession(model_path)
            print(f"Model loaded successfully")
        except Exception as e:
            print(f"Error loading the ONNX model: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # Get model info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]

        # Model parameters
        self.img_size = self.input_shape[2:4] if len(self.input_shape) == 4 else (640, 640)
        self.confidence_threshold = confidence
        self.iou_threshold = iou_threshold

        print(f"Model loaded successfully. Input shape: {self.img_size}")
        print(f"Detecting {len(self.class_names)} classes: {', '.join(self.class_names)}")

    def preprocess(self, img):
        """Preprocess the image for the model."""
        # Get original dimensions
        orig_h, orig_w = img.shape[:2]

        # Resize maintaining aspect ratio
        scale = min(self.img_size[0] / orig_w, self.img_size[1] / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        # Resize the image
        resized = cv2.resize(img, (new_w, new_h))

        # Create a canvas with the expected dimensions
        canvas = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)

        # Paste the resized image into the center of the canvas
        offset_x, offset_y = (self.img_size[0] - new_w) // 2, (self.img_size[1] - new_h) // 2
        canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized

        # Convert to RGB (OpenCV uses BGR by default)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        # Normalize and convert to the expected format
        canvas = canvas.astype(np.float32) / 255.0
        canvas = np.transpose(canvas, (2, 0, 1))  # HWC to CHW (channels first)
        canvas = np.expand_dims(canvas, axis=0)  # Add batch dimension

        return canvas, (scale, offset_x, offset_y, orig_h, orig_w)

    def process_yolov12_output(self, pred, img_height, img_width):
        """
        Process YOLOv12 model output tensor.

        Args:
            pred: Raw model output with shape (bs, 5, 8400)
            img_height, img_width: Original image dimensions

        Returns:
            Processed detections
        """
        print(f"Output shape: {pred.shape}")

        # Show the range of values for each channel
        for i in range(pred.shape[1]):
            min_val = np.min(pred[0, i, :])
            max_val = np.max(pred[0, i, :])
            mean_val = np.mean(pred[0, i, :])
            print(f"  Channel {i}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}")

        # Extract box data from prediction
        # In YOLOv12's output format, the first dimension is batch, second is parameters, third is boxes
        cx = pred[0, 0, :]  # center x (normalized 0-1 or absolute pixels)
        cy = pred[0, 1, :]  # center y (normalized 0-1 or absolute pixels)
        w = pred[0, 2, :]   # width (normalized 0-1 or absolute pixels)
        h = pred[0, 3, :]   # height (normalized 0-1 or absolute pixels)
        conf = pred[0, 4, :] # confidence

        # Check if values are normalized (0-1) or absolute
        # If max values are significantly greater than 1, they're likely absolute pixel values
        cx_max = np.max(cx)
        cy_max = np.max(cy)

        # Determine if coordinates are normalized or absolute based on their ranges
        is_normalized = cx_max <= 1.0 and cy_max <= 1.0

        # Find high confidence boxes
        mask = conf > self.confidence_threshold
        indices = np.nonzero(mask)[0]

        print(f"Found {len(indices)} boxes with confidence > {self.confidence_threshold}")

        if len(indices) == 0:
            return []

        # Create a list to store filtered boxes
        boxes = []

        # Extract filtered boxes
        for i in indices:
            # Convert coordinates to pixel coordinates
            if is_normalized:
                # Normalized coordinates (0-1)
                x1 = int((cx[i] - w[i]/2) * img_width)
                y1 = int((cy[i] - h[i]/2) * img_height)
                x2 = int((cx[i] + w[i]/2) * img_width)
                y2 = int((cy[i] + h[i]/2) * img_height)
            else:
                # Absolute pixel coordinates
                x1 = int(cx[i] - w[i]/2)
                y1 = int(cy[i] - h[i]/2)
                x2 = int(cx[i] + w[i]/2)
                y2 = int(cy[i] + h[i]/2)

            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, img_width-1))
            y1 = max(0, min(y1, img_height-1))
            x2 = max(0, min(x2, img_width-1))
            y2 = max(0, min(y2, img_height-1))

            # Skip invalid boxes (zero width or height)
            if x2 <= x1 or y2 <= y1:
                continue

            # Add to boxes list with confidence and class ID (0 for vehicle)
            boxes.append([x1, y1, x2, y2, conf[i], 0])

        if not boxes:
            return []

        # Convert to numpy array
        boxes = np.array(boxes)

        # Apply non-max suppression
        final_boxes = self.apply_nms(boxes)

        return final_boxes

    def apply_nms(self, boxes, iou_threshold=None):
        """Apply non-maximum suppression to boxes."""
        if len(boxes) == 0:
            return []

        if iou_threshold is None:
            iou_threshold = self.iou_threshold

        # Sort by confidence (highest first)
        indices = np.argsort(boxes[:, 4])[::-1]
        boxes_sorted = boxes[indices]

        keep = []
        while len(boxes_sorted) > 0:
            # Keep the box with highest confidence
            keep.append(boxes_sorted[0])

            if len(boxes_sorted) == 1:
                break

            # Calculate IoU between first box and all other boxes
            ious = self.calculate_iou(boxes_sorted[0], boxes_sorted[1:])

            # Find boxes with IoU below threshold
            inds = np.where(ious <= iou_threshold)[0]

            # Update boxes_sorted to only include boxes to keep
            boxes_sorted = boxes_sorted[1:][inds] if len(inds) > 0 else np.empty((0, 6))

        return np.array(keep)

    def calculate_iou(self, box, boxes):
        """Calculate IoU between a box and multiple boxes."""
        # Extract coordinates of first box
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

        # Calculate area of first box
        area1 = max(0, (x2 - x1)) * max(0, (y2 - y1))

        # Extract coordinates of other boxes
        xx1 = boxes[:, 0]
        yy1 = boxes[:, 1]
        xx2 = boxes[:, 2]
        yy2 = boxes[:, 3]

        # Calculate areas of other boxes
        areas = np.maximum(0, (xx2 - xx1)) * np.maximum(0, (yy2 - yy1))

        # Calculate intersection
        inter_x1 = np.maximum(x1, xx1)
        inter_y1 = np.maximum(y1, yy1)
        inter_x2 = np.minimum(x2, xx2)
        inter_y2 = np.minimum(y2, yy2)

        # Calculate intersection area
        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        # Calculate IoU with protection against division by zero
        union_area = area1 + areas - inter_area
        iou = np.zeros_like(union_area, dtype=np.float32)
        valid = union_area > 0
        iou[valid] = inter_area[valid] / union_area[valid]

        return iou

    def detect(self, img):
        """Perform detection on an image."""
        # Preprocess the image
        input_tensor, preprocess_info = self.preprocess(img)
        scale, offset_x, offset_y, orig_h, orig_w = preprocess_info

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # Process the YOLOv12 output
        predictions = outputs[0]  # Shape (1, 5, 8400)

        # We need to further process the output to convert YOLOv12's format to usable boxes
        boxes = self.process_yolov12_output(predictions, orig_h, orig_w)

        # Create a copy of the original image for drawing
        result_img = img.copy()

        # Draw boxes on the image
        for box in boxes:
            x1, y1, x2, y2, conf, class_id = box

            # Draw bounding box
            color = tuple(map(int, self.colors[int(class_id)]))
            cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Draw label
            label = f"{self.class_names[int(class_id)]}: {conf:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y1_label = max(int(y1), label_height + 10)
            cv2.rectangle(result_img, (int(x1), y1_label - label_height - 10),
                         (int(x1) + label_width, y1_label), color, -1)
            cv2.putText(result_img, label, (int(x1), y1_label - 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return result_img

    def capture_screen(self, monitor=None, capture_rate=1, output_dir=None):
        """Capture the screen and perform detection."""
        # Initialize screen capture
        with mss.mss() as sct:
            # Get the monitor parameters
            if monitor is None:
                monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
            else:
                monitor = sct.monitors[monitor]

            # Create output directory if needed
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                print(f"Saving detection frames to {output_dir}")

            # Use PIL for display if OpenCV GUI is not available
            try:
                cv2.namedWindow("YOLOv12 Screen Detection", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("YOLOv12 Screen Detection", 1280, 720)
                opencv_gui_available = True
            except:
                from PIL import Image, ImageTk
                import tkinter as tk

                opencv_gui_available = False
                # Create a tkinter window
                root = tk.Tk()
                root.title("YOLOv12 Screen Detection")
                root.geometry("1280x720")

                # Create a label to display the image
                label = tk.Label(root)
                label.pack(fill=tk.BOTH, expand=True)

                # Handle window close
                def on_closing():
                    root.quit()
                    root.destroy()

                root.protocol("WM_DELETE_WINDOW", on_closing)
                print("Using Tkinter for display (OpenCV GUI not available)")

            print("Press 'q' to quit.")

            last_time = time.time()
            frame_count = 0
            running = True

            while running:
                # Capture screen
                screenshot = sct.grab(monitor)
                img = np.array(screenshot)

                # Convert from BGRA to BGR (remove alpha channel)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                # Only perform detection at the specified rate
                current_time = time.time()
                if current_time - last_time >= capture_rate:
                    # Perform detection
                    try:
                        result_img = self.detect(img)
                    except Exception as e:
                        print(f"Error in detection: {e}")
                        import traceback
                        traceback.print_exc()
                        result_img = img  # Fall back to original image

                    last_time = current_time

                    # Calculate FPS
                    fps = 1 / (time.time() - current_time + 0.001)  # Avoid division by zero
                    cv2.putText(result_img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Save frame if output_dir is specified
                    if output_dir is not None:
                        frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
                        cv2.imwrite(frame_path, result_img)
                        frame_count += 1
                else:
                    result_img = img

                # Display the result
                if opencv_gui_available:
                    # Display with OpenCV
                    cv2.imshow("YOLOv12 Screen Detection", result_img)

                    # Exit if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        running = False
                else:
                    # Display with PIL and tkinter
                    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(result_img_rgb)

                    # Resize to fit window
                    pil_img = pil_img.resize((1280, 720), Image.LANCZOS)

                    # Convert to PhotoImage and update label
                    try:
                        tk_img = ImageTk.PhotoImage(image=pil_img)
                        label.configure(image=tk_img)
                        label.image = tk_img

                        # Update tkinter window
                        root.update()
                    except Exception as e:
                        print(f"Error updating display: {e}")
                        running = False

                    # Check if window was closed
                    if not root.winfo_exists():
                        running = False

            # Clean up
            if opencv_gui_available:
                cv2.destroyAllWindows()
            elif root.winfo_exists():
                root.quit()
                root.destroy()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="YOLOv12 Screen Capture Detection")
    parser.add_argument("--model", type=str, required=True, help="Path to the ONNX model")
    parser.add_argument("--classes", type=str, required=True, help="Path to the class names file")
    parser.add_argument("--confidence", type=float, default=0.01, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--monitor", type=int, default=None, help="Monitor number to capture (default: entire screen)")
    parser.add_argument("--rate", type=float, default=0.1, help="Capture rate in seconds (default: 0.1)")
    parser.add_argument("--output", type=str, default=None, help="Directory to save output frames (optional)")
    args = parser.parse_args()

    # Initialize the detector
    detector = YOLOv12ScreenDetector(
        model_path=args.model,
        class_names_path=args.classes,
        confidence=args.confidence,
        iou_threshold=args.iou
    )

    # Start screen capture and detection
    detector.capture_screen(monitor=args.monitor, capture_rate=args.rate, output_dir=args.output)

if __name__ == "__main__":
    main()