from pathlib import Path
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

# Class names consistent with ssda.yaml
CLASS_NAMES = {
    0: "pest",
    1: "scratch",
}


def get_model_path() -> Path:
    """
    Returns the path to your best trained model.
    Adjust this if you prefer a different location.
    """
    project_root = Path(__file__).resolve().parents[1]
    # Using the AAU-trained yolov8s weights you tested
    return project_root / "runs_ssDA" / "yolov8s_ssda_phase2" / "weights" / "best.pt"


def load_model(device: str = "cuda") -> YOLO:
    """
    Load the YOLO model once and reuse it.
    device: "cuda" or "cpu"
    """
    model_path = get_model_path()
    print(f"[YOLO] Loading model from: {model_path}")
    model = YOLO(str(model_path))
    # Ultralytics will auto-pick device, but you can hint:
    model.to(device)
    return model


def detect_defects(
    model: YOLO,
    frame_bgr: np.ndarray,
    conf_th: float = 0.5,
) -> Tuple[np.ndarray, List[Tuple[str, float]], str]:
    """
    Run YOLO on a single BGR frame (from OpenCV / RealSense).

    Returns:
      annotated_frame_bgr: frame with boxes/labels drawn
      defects: list of (class_name, confidence)
      decision: "accept" if no defects, "reject" if any defect is found
    """
    # YOLO can handle BGR np.ndarray directly
    results = model(
        frame_bgr,
        conf=conf_th,
        verbose=False,
    )

    r = results[0]

    # YOLO's built-in plotting (returns BGR image with overlays)
    annotated = r.plot()

    defects = []
    if r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
            defects.append((cls_name, conf))

    decision = "reject" if defects else "accept"
    return annotated, defects, decision
