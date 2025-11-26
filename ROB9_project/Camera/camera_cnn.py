# Camera/camera_cnn.py

import pyrealsense2 as rs
import numpy as np
import cv2
from pathlib import Path

import torch
from ultralytics import YOLO


# -------------------------------------------------------------------------
# YOLO model config
# -------------------------------------------------------------------------

# Class names must match your ssda.yaml
CLASS_NAMES = {
    0: "scratch",
    1: "pest",
}

# Minimum red ratio inside a YOLO box for it to be considered "on an apple"
APPLE_MIN_RED_FOR_DEFECT = 0.45  # 45% of pixels in box must be red (tune as needed)


def get_model_path() -> Path:
    """
    Returns the path to your best trained YOLO model.

    Assumes this file is at:
        ROB9_project/Camera/camera_cnn.py

    and the weights are at:
        ROB9_project/runs_ssDA/yolov8s_ssda_phase2/weights/best.pt
    """
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "runs_ssDA" / "yolov8s_ssda_phase2" / "weights" / "best.pt"


def load_yolo_model() -> YOLO:
    """
    Load the YOLO model on GPU only.
    Raises an error if CUDA is not available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "[YOLO] CUDA is not available, but GPU-only mode was requested. "
            "Make sure you have an NVIDIA GPU, drivers, and the correct PyTorch build installed."
        )

    device_str = "cuda:0"
    model_path = get_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLO weights not found at {model_path}. "
            f"Make sure the file exists, or adjust get_model_path()."
        )

    print(f"[YOLO] Loading model from: {model_path}")
    model = YOLO(str(model_path))

    # Move model to GPU explicitly
    model.to(device_str)
    print(f"[YOLO] Using device: {device_str}")

    return model


# -------------------------------------------------------------------------
# Redness estimation using OpenCV
# -------------------------------------------------------------------------

def compute_red_ratio_bgr(image_bgr: np.ndarray) -> float:
    """
    Estimate how much of the image is "red" using HSV thresholding.

    Returns:
        red_ratio in [0, 1] = red_pixels / total_pixels

    Note:
        image_bgr should be a non-empty BGR image (np.ndarray).
    """
    if image_bgr is None or image_bgr.size == 0:
        return 0.0

    # Convert BGR to HSV
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Tight red thresholds to avoid white/gray objects being called "red".
    # Hue around red (0–10 and 160–180), high saturation and reasonable value.
    lower_red1 = np.array([0, 140, 80], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)

    lower_red2 = np.array([160, 140, 80], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    total_pixels = image_bgr.shape[0] * image_bgr.shape[1]
    if total_pixels == 0:
        return 0.0

    red_pixels = int(np.count_nonzero(red_mask))
    red_ratio = red_pixels / float(total_pixels)
    return red_ratio


# -------------------------------------------------------------------------
# YOLO inference on a frame
# -------------------------------------------------------------------------

def run_yolo_on_frame(
    model: YOLO,
    frame_bgr: np.ndarray,
    conf_th: float = 0.5,
    red_threshold: float = 0.5,
):
    """
    Run YOLO on a single BGR frame and return:
      - annotated_frame (BGR with boxes/labels and overlays)
      - list of valid defects [(class_name, confidence), ...]
      - decision: "accept", "reject_defect", or "reject_color"
      - red_ratio: fraction of red pixels in [0, 1] over the whole frame

    Inference is forced onto GPU device 0.

    A detection is only considered a valid defect if the red ratio inside
    its bounding box is >= APPLE_MIN_RED_FOR_DEFECT. This filters out
    detections on non-apple objects like white toilet paper.
    """
    # --- YOLO inference (GPU only) ---
    results = model(
        frame_bgr,
        conf=conf_th,
        verbose=False,
        device=0,  # GPU index 0 (cuda:0)
    )
    r = results[0]

    # Start from YOLO's own annotated frame (boxes + labels)
    annotated = r.plot()

    valid_defects = []

    # Filter detections by "apple-like" color inside each box
    if r.boxes is not None and len(r.boxes) > 0:
        h, w, _ = frame_bgr.shape
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = CLASS_NAMES.get(cls_id, str(cls_id))

            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))

            if x2 <= x1 or y2 <= y1:
                continue

            roi = frame_bgr[y1:y2, x1:x2]
            roi_red = compute_red_ratio_bgr(roi)

            # Debug print (optional)
            # print(f"Box {cls_name} conf={conf:.2f}, roi_red={roi_red:.2f}")

            if roi_red >= APPLE_MIN_RED_FOR_DEFECT:
                valid_defects.append((cls_name, conf))
            else:
                # Uncomment if you want to see what's discarded:
                # print(f"Discard {cls_name} (conf={conf:.2f}), roi_red={roi_red:.2f}")
                pass

    # --- Redness estimation on the whole frame (for grading) ---
    red_ratio = compute_red_ratio_bgr(frame_bgr)

    # --- Decision logic ---
    # 1) If any valid defect -> reject_defect
    # 2) Else if red_ratio < threshold -> reject_color
    # 3) Else -> accept
    if valid_defects:
        decision = "reject_defect"
    elif red_ratio < red_threshold:
        decision = "reject_color"
    else:
        decision = "accept"

    # Console logging for debugging / logging
    if decision == "reject_defect":
        print("[DECISION] REJECT_DEFECT due to:")
        for name, conf in valid_defects:
            print(f"  - {name} (conf={conf:.2f})")
    elif decision == "reject_color":
        print(f"[DECISION] REJECT_COLOR: red_ratio={red_ratio:.2f}")
    else:
        print(f"[DECISION] ACCEPT: red_ratio={red_ratio:.2f}, defects=0")

    # --- Overlay decision and redness info on annotated frame ---
    red_percent = int(round(red_ratio * 100))
    num_defects = len(valid_defects)

    if decision == "reject_defect":
        text_decision = f"Decision: REJECT_DEFECT (defects={num_defects})"
    elif decision == "reject_color":
        text_decision = f"Decision: REJECT_COLOR (defects={num_defects})"
    else:
        text_decision = f"Decision: ACCEPT (defects={num_defects})"

    text_red = f"Redness: {red_percent}% (target >= {int(red_threshold * 100)}%)"

    # Color for decision text
    if decision == "accept":
        color_decision = (0, 255, 0)  # green
    else:
        color_decision = (0, 0, 255)  # red

    cv2.putText(
        annotated,
        text_decision,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color_decision,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        annotated,
        text_red,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Extra line with defect list if any
    if num_defects > 0:
        defect_str = ", ".join(f"{name}({conf:.2f})" for name, conf in valid_defects)
        cv2.putText(
            annotated,
            f"Defects: {defect_str}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated, valid_defects, decision, red_ratio


# -------------------------------------------------------------------------
# RealSense camera + YOLO
# -------------------------------------------------------------------------


def preview_rgb_ir_depth_with_yolo():
    """
    Main preview:
      - RGB stream + YOLOv8s apple defect detection (pest/scratch) on GPU,
        filtered so defects are only counted on "red-enough" regions.
      - Redness estimation (%) on RGB (whole frame)
      - IR stream for visualisation
      - Depth stream (colormap) for visualisation and future UR5 3D work

    Press 'q' or ESC to quit.
    """
    # Load YOLO model (GPU only)
    model = load_yolo_model()

    # RealSense pipeline and config
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable depth stream (for future UR5 work)
    config.enable_stream(
        rs.stream.depth,
        640, 480,
        rs.format.z16,
        30,
    )

    # Enable RGB stream
    config.enable_stream(
        rs.stream.color,
        640, 480,
        rs.format.bgr8,
        30,
    )

    # Enable one of the infrared imagers (index 1 or 2)
    config.enable_stream(
        rs.stream.infrared,
        1,               # 1 = left IR imager
        640, 480,
        rs.format.y8,
        30,
    )

    # Align depth to the color stream for easier use later
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Start streaming
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()

            # Align depth to color
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            ir_frame = frames.get_infrared_frame(1)  # IR not aligned, just visual

            if not depth_frame or not color_frame or not ir_frame:
                continue

            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            ir_image = np.asanyarray(ir_frame.get_data())

            # Depth colormap for visualisation
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_scaled = cv2.convertScaleAbs(depth_image, alpha=0.03)
            depth_colormap = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)

            # Run YOLO + redness on the RGB frame (GPU)
            annotated, defects, decision, red_ratio = run_yolo_on_frame(
                model,
                color_image,
                conf_th=0.5,
                red_threshold=0.20,  # 50% red requirement
            )

            # Show windows
            cv2.imshow("RGB + YOLO + Redness", annotated)
            cv2.imshow("IR (NIR)", ir_image)
            cv2.imshow("Depth (colormap)", depth_colormap)

            # Quit on 'q' or ESC
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    preview_rgb_ir_depth_with_yolo()
