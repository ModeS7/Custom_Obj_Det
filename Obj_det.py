import cv2
import numpy as np
import onnxruntime as ort
from mss import mss
import time

# Configuration
MODEL_PATH = "best.onnx"
CLASS_NAMES = ["vehicle"]  # Update with your classes
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
INPUT_SIZE = 640

# Initialize ONNX Runtime
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# Screen capture setup
sct = mss()
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}  # Match your actual resolution


def preprocess(frame):
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


def postprocess(outputs, orig_size, padding_info):
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


def main():
    while True:
        start_time = time.time()

        # Capture screen
        frame = np.array(sct.grab(monitor))
        orig_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        orig_h, orig_w = orig_frame.shape[:2]

        # Preprocess
        input_tensor, orig_size, padding_info = preprocess(orig_frame)

        # Inference
        outputs = session.run(None, {input_name: input_tensor})

        # Postprocess
        detections = postprocess(outputs, orig_size, padding_info)

        # Draw results
        for box, score, class_id in detections:
            x1, y1, x2, y2 = map(int, box)
            # Ensure valid box dimensions
            if x2 > x1 and y2 > y1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{CLASS_NAMES[class_id]}: {score:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show FPS
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display
        cv2.imshow("Screen Detection", cv2.resize(frame, (1280, 720)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()