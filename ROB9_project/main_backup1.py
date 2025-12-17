#!/usr/bin/env python3
"""
ROB9 MAIN
- Checks calibration files
- Runs optional calibration steps
- Loads YOLO + RealSense
- Loads UV->XY homography
- Runs a simple pick/place decision loop
"""

from __future__ import annotations

import json
import os
import sys
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import cv2
import pyrealsense2 as rs

import torch
from ultralytics import YOLO


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
UR_DIR = PROJECT_ROOT / "UR5"
CAM_DIR = PROJECT_ROOT / "Camera"
CNN_DIR = PROJECT_ROOT / "CNN"

SIMPLE_CAL = UR_DIR / "simple_calibration.json"
UV_TCP = UR_DIR / "uv_tcp_poses.json"
HOMO_JSON = UR_DIR / "homography_uv_to_base_xy.json"


# -----------------------------
# Configuration
# -----------------------------
# YOLO weights (AppleRF trained model)
YOLO_WEIGHTS = PROJECT_ROOT / "runs_AppleRF" / "yolov8s_applerf_phase2" / "weights" / "best.pt"

# Decision thresholds
CONF_APPLE = 0.90
REDNESS_ACCEPT_THRESHOLD = 0.00     # >= 90% redness => pick (your requirement)
PEST_REJECT_THRESHOLD = 0.50        # >= 50% pest confidence => reject

# RealSense streams
W, H, FPS = 640, 480, 30


# -----------------------------
# Helpers
# -----------------------------
def ask_yn(prompt: str, default_yes: bool = True) -> bool:
    suffix = " [Y/n]: " if default_yes else " [y/N]: "
    ans = input(prompt + suffix).strip().lower()
    if ans == "":
        return default_yes
    return ans.startswith("y")


def run_cmd(cmd: List[str]) -> None:
    print("\n[CMD] " + " ".join(cmd) + "\n")
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def pose6_from_list(x: Any) -> np.ndarray:
    arr = np.array(x, dtype=float).reshape(6)
    return arr


def _get_pose_any(cal: Dict[str, Any], keys: List[str]) -> np.ndarray:
    """
    Try multiple key names (backwards-compatible).
    """
    for k in keys:
        if k in cal:
            return pose6_from_list(cal[k])
    raise KeyError(f"None of these keys exist in calibration JSON: {keys}. Found keys: {list(cal.keys())}")


@dataclass
class Calibration:
    workspace_pose: np.ndarray
    accept_pose: np.ndarray
    reject_pose: np.ndarray


def load_simple_calibration(path: Path) -> Calibration:
    cal = load_json(path)

    # Backwards-compatible mapping:
    # - workspace_pose can be stored as "workspace" OR "workspace_pose" OR "center"
    workspace_pose = _get_pose_any(cal, ["workspace_pose", "workspace", "center"])

    # - accept pose can be stored as "accept_pose" OR "accept"
    accept_pose = _get_pose_any(cal, ["accept_pose", "accept"])

    # - reject pose can be stored as "reject_pose" OR "reject"
    reject_pose = _get_pose_any(cal, ["reject_pose", "reject"])

    return Calibration(
        workspace_pose=workspace_pose,
        accept_pose=accept_pose,
        reject_pose=reject_pose,
    )


def load_homography(path: Path) -> np.ndarray:
    obj = load_json(path)
    # Support a few common key names
    for k in ["H", "homography", "H_uv_to_xy"]:
        if k in obj:
            Hm = np.array(obj[k], dtype=float).reshape(3, 3)
            return Hm
    # Or sometimes stored flat
    if "data" in obj:
        Hm = np.array(obj["data"], dtype=float).reshape(3, 3)
        return Hm
    raise KeyError(f"Homography JSON missing expected key. Found keys: {list(obj.keys())}")


def uv_to_xy(Hm: np.ndarray, u: float, v: float) -> Tuple[float, float]:
    pt = np.array([u, v, 1.0], dtype=float).reshape(3, 1)
    out = Hm @ pt
    out /= (out[2, 0] + 1e-12)
    x = float(out[0, 0])
    y = float(out[1, 0])
    return x, y


# -----------------------------
# Redness (simple HSV)
# -----------------------------
def red_ratio_bgr(img_bgr: np.ndarray) -> float:
    if img_bgr is None or img_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # tighter reds (reduce false positives)
    lower1 = np.array([0, 140, 80], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([160, 140, 80], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)
    return float(np.count_nonzero(mask)) / float(img_bgr.shape[0] * img_bgr.shape[1] + 1e-12)


# -----------------------------
# YOLO
# -----------------------------
def load_yolo(weights: Path) -> YOLO:
    if not weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")

    model = YOLO(str(weights))

    # force GPU if available
    if torch.cuda.is_available():
        model.to("cuda:0")
        print("[YOLO] Using GPU cuda:0")
    else:
        print("[YOLO] CUDA not available -> CPU (will be slower)")

    return model


def pick_best_apple_detection(results, class_name="apple", conf_min=0.5):
    """
    Returns best box (xyxy, conf, cls_name) or None.
    """
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None

    names = r.names  # dict id->name
    best = None
    best_conf = -1.0

    for b in r.boxes:
        cls_id = int(b.cls[0])
        conf = float(b.conf[0])
        name = names.get(cls_id, str(cls_id))
        if name != class_name:
            continue
        if conf < conf_min:
            continue
        if conf > best_conf:
            xyxy = b.xyxy[0].cpu().numpy().astype(float)
            best = (xyxy, conf, name)
            best_conf = conf

    return best


def extract_defects(results, defect_names=("pest", "scratch", "bruise", "rotten"), conf_min=0.25):
    r = results[0]
    out = []
    if r.boxes is None or len(r.boxes) == 0:
        return out
    names = r.names
    for b in r.boxes:
        cls_id = int(b.cls[0])
        conf = float(b.conf[0])
        name = names.get(cls_id, str(cls_id))
        if name in defect_names and conf >= conf_min:
            out.append((name, conf, b.xyxy[0].cpu().numpy().astype(float)))
    return out


# -----------------------------
# RealSense
# -----------------------------
def start_realsense() -> Tuple[rs.pipeline, rs.align]:
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    return pipeline, align


def get_frames(pipeline: rs.pipeline, align: rs.align) -> Tuple[np.ndarray, np.ndarray]:
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    color = frames.get_color_frame()
    depth = frames.get_depth_frame()
    if not color or not depth:
        raise RuntimeError("Missing frames from RealSense")
    color_img = np.asanyarray(color.get_data())
    depth_img = np.asanyarray(depth.get_data())
    return color_img, depth_img


# -----------------------------
# Main logic
# -----------------------------
def main():
    print("\n=== ROB9 MAIN (camera + YOLO + UV->XY + UR pick/place) ===\n")
    print(f"[PATH] Project root: {PROJECT_ROOT}")

    # 1) Check simple calibration (work area / accept / reject)
    if SIMPLE_CAL.exists():
        print(f"[OK] Found: {SIMPLE_CAL}")
    else:
        print(f"[MISS] Missing: {SIMPLE_CAL}")
        print("[INFO] Run UR5/ur_controller.py first to record workspace/accept/reject.")
        sys.exit(1)

    # Camera calibration prompt (runs your camera calibration script if needed)
    cam_done = ask_yn("Have you done camera calibration (intrinsics/extrinsics) for this setup?", default_yes=True)
    if not cam_done:
        run_cmd(["python3", str(CAM_DIR / "camera_calibration.py")])

    # 2) UV->TCP points
    if UV_TCP.exists():
        print(f"[OK] Found: {UV_TCP}")
    else:
        print(f"[MISS] Missing: {UV_TCP}")
        print("[INFO] Running UV corner recorder...")
        run_cmd(["python3", str(UR_DIR / "uv_corner_recorder.py")])

    # 3) Homography
    if HOMO_JSON.exists():
        print(f"[OK] Found: {HOMO_JSON}")
    else:
        print(f"[MISS] Missing: {HOMO_JSON}")
        print("[INFO] Computing homography...")
        run_cmd(["python3", str(UR_DIR / "compute_homography.py")])

    # Load calibration (robust keys)
    cal = load_simple_calibration(SIMPLE_CAL)

    # Load homography matrix
    Hm = load_homography(HOMO_JSON)

    # Load YOLO
    print(f"[YOLO] Weights: {YOLO_WEIGHTS}")
    model = load_yolo(YOLO_WEIGHTS)

    # Start camera
    pipeline, align = start_realsense()

    print("\n[RUN] Starting live loop. Press 'q' to quit.\n")

    try:
        while True:
            color_img, depth_img = get_frames(pipeline, align)

            # YOLO inference (GPU if available)
            results = model(color_img, conf=0.25, verbose=False, device=0 if torch.cuda.is_available() else "cpu")

            # Find best apple
            best_apple = pick_best_apple_detection(results, class_name="apple", conf_min=CONF_APPLE)
            defects = extract_defects(results)

            annotated = results[0].plot()

            # If no apple found, just show feed
            if best_apple is None:
                cv2.putText(annotated, "No apple detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("ROB9 MAIN", annotated)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q") or k == 27:
                    break
                continue

            xyxy, apple_conf, _ = best_apple
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Redness on the apple ROI
            roi = color_img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            rr = red_ratio_bgr(roi)

            # Pest reject logic: if any pest >= threshold -> reject
            reject_defect = False
            for name, conf, _box in defects:
                if name == "pest" and conf >= PEST_REJECT_THRESHOLD:
                    reject_defect = True

            # “Pick only if redness >= 90%”
            should_pick = rr >= REDNESS_ACCEPT_THRESHOLD

            if not should_pick:
                decision = "IDLE (not red enough)"
            elif reject_defect:
                decision = "PICK -> REJECT"
            else:
                decision = "PICK -> ACCEPT"

            # Map pixel to base XY using homography
            bx, by = uv_to_xy(Hm, cx, cy)

            # Overlay
            cv2.circle(annotated, (cx, cy), 6, (255, 255, 0), -1)
            cv2.putText(annotated, f"apple_conf={apple_conf:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"red={rr*100:.0f}%", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"uv=({cx},{cy}) -> base_xy=({bx:.3f},{by:.3f})", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, decision, (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0) if "ACCEPT" in decision else (0, 0, 255), 2, cv2.LINE_AA)

            # NOTE: Robot/gripper action hooks:
            # Here you would:
            #   - move robot HOME
            #   - move to (bx, by) approach
            #   - vacuum on
            #   - lift
            #   - move to accept_pose or reject_pose
            #   - vacuum off
            #
            # For now we ONLY display decision + mapping.

            cv2.imshow("ROB9 MAIN", annotated)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q") or k == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
