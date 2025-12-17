#!/usr/bin/env python3
"""
ROB9 main orchestrator
- Checks/creates:
  - UR bin/workspace calibration: UR5/simple_calibration.json (from your existing UR controller script)
  - UV+TCP dataset:              UR5/uv_tcp_poses.json      (uv_corner_recorder.py)
  - Homography UV->base XY:      UR5/homography_uv_to_base_xy.json (compute_homography.py)
- Runs runtime loop:
  - Move HOME pose
  - Capture RGB frame
  - YOLO detect apples + defects
  - Choose best apple in CENTER workspace and not in drop zones
  - (u,v) -> (x,y) using homography
  - Approach -> pick (vacuum on) -> lift
  - Place to accept/reject -> vacuum off
  - Return HOME, repeat
"""

from __future__ import annotations

import json
import os
import sys
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import cv2
import pyrealsense2 as rs

import torch
from ultralytics import YOLO

# --- Robot control (your existing ROS2 URScript publisher controller)
# ur_controller.py contains SimpleURDemo + Pose6
try:
    from UR5.ur_controller import SimpleURDemo, Pose6
except Exception as e:
    SimpleURDemo = None
    Pose6 = None
    _UR_IMPORT_ERR = e


# ---------------------------------------------------------------------
# Paths (assumes this file is ROB9_project/main.py)
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
UR_DIR = PROJECT_ROOT / "UR5"
CAM_DIR = PROJECT_ROOT / "Camera"
CNN_DIR = PROJECT_ROOT / "CNN"

SIMPLE_CALIB_JSON = UR_DIR / "simple_calibration.json"
UV_TCP_JSON = UR_DIR / "uv_tcp_poses.json"
HOMOGRAPHY_JSON = UR_DIR / "homography_uv_to_base_xy.json"

# If you deploy AppleRF model:
YOLO_WEIGHTS_DEFAULT = PROJECT_ROOT / "runs_AppleRF" / "yolov8s_applerf_phase2" / "weights" / "best.pt"
# If you deploy SSDA model instead, swap to:
# YOLO_WEIGHTS_DEFAULT = PROJECT_ROOT / "runs_ssDA" / "yolov8s_ssda_phase2" / "weights" / "best.pt"


# ---------------------------------------------------------------------
# Tunables (match your project rules)
# ---------------------------------------------------------------------
APPLE_CLASS = "apple"
DEFECT_CLASSES = {"pest", "scratch"}  # ignore bruise/rotten for now if you want

# pick apple only if detector is confident enough
APPLE_MIN_CONF_TO_PICK = 0.90

# decision threshold: if defect confidence >= this -> reject
DEFECT_MIN_CONF_REJECT = 0.50

# accept color threshold (redness on ROI of apple)
MIN_REDNESS_ACCEPT = 0.50  # 50% red requirement (change to 0.90 if that’s truly the spec)

# ignore apples too close to accept/reject drop locations (meters)
DROP_ZONE_RADIUS_M = 0.18

# approach & pick heights
APPROACH_Z_OFFSET = 0.10   # approach 10cm above pick Z
LIFT_Z_OFFSET = 0.12       # lift 12cm after vacuum on

# center workspace ROI (in image coordinates)
CENTER_ROI = (0.20, 0.20, 0.80, 0.80)  # xmin,ymin,xmax,ymax as fractions of width/height

# YOLO inference
YOLO_IMG_SIZE = 640
YOLO_CONF = 0.25


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def prompt_yn(msg: str, default_yes: bool = False) -> bool:
    d = "Y/n" if default_yes else "y/N"
    while True:
        s = input(f"{msg} [{d}]: ").strip().lower()
        if not s:
            return default_yes
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False


def run_cmd(cmd: str, cwd: Optional[Path] = None) -> None:
    """Run a bash command (useful for ROS sourcing)."""
    print(f"\n[CMD] {cmd}\n")
    subprocess.run(["bash", "-lc", cmd], cwd=str(cwd) if cwd else None, check=True)


def file_exists_or_explain(path: Path, hint: str) -> bool:
    if path.is_file():
        print(f"[OK] Found: {path}")
        return True
    print(f"[MISSING] {path}\n  -> {hint}")
    return False


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_homography(path: Path) -> np.ndarray:
    d = load_json(path)
    H = np.array(d["H_uv_to_xy"], dtype=np.float64)
    assert H.shape == (3, 3)
    return H


def uv_to_xy(H: np.ndarray, u: float, v: float) -> Tuple[float, float]:
    p = np.array([u, v, 1.0], dtype=np.float64)
    q = H @ p
    if abs(q[2]) < 1e-9:
        raise RuntimeError("Homography produced invalid scale (w≈0).")
    x = q[0] / q[2]
    y = q[1] / q[2]
    return float(x), float(y)


def compute_red_ratio_bgr(image_bgr: np.ndarray) -> float:
    """HSV red thresholding (robust-ish)."""
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # tighten these if you get "toilet paper is red" style false positives
    lower1 = np.array([0, 140, 80], np.uint8)
    upper1 = np.array([10, 255, 255], np.uint8)
    lower2 = np.array([160, 140, 80], np.uint8)
    upper2 = np.array([180, 255, 255], np.uint8)

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)

    total = image_bgr.shape[0] * image_bgr.shape[1]
    red = int(np.count_nonzero(mask))
    return red / float(total + 1e-9)


# ---------------------------------------------------------------------
# Gripper helpers (VG10)
# ---------------------------------------------------------------------
def vg10_grip(channel_a: int = 60, channel_b: int = 60) -> None:
    """
    Calls /vg10_grip service. Assumes vg10 node is already running.
    """
    cmd = f"""
    source /opt/ros/humble/setup.bash
    ros2 service call /vg10_grip vg_control_interfaces/srv/VacuumSet "{{channel_a: {channel_a}, channel_b: {channel_b}}}"
    """
    run_cmd(cmd)


def vg10_release() -> None:
    cmd = """
    source /opt/ros/humble/setup.bash
    ros2 service call /vg10_release std_srvs/srv/Trigger "{}"
    """
    run_cmd(cmd)


# ---------------------------------------------------------------------
# RealSense
# ---------------------------------------------------------------------
@dataclass
class RSStreams:
    pipeline: rs.pipeline
    align: rs.align


def start_realsense() -> RSStreams:
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)
    align = rs.align(rs.stream.color)
    return RSStreams(pipeline=pipeline, align=align)


def get_color_frame(streams: RSStreams) -> np.ndarray:
    frames = streams.pipeline.wait_for_frames()
    aligned = streams.align.process(frames)
    color = aligned.get_color_frame()
    if not color:
        raise RuntimeError("No color frame.")
    return np.asanyarray(color.get_data())


# ---------------------------------------------------------------------
# YOLO selection logic
# ---------------------------------------------------------------------
@dataclass
class Detection:
    cls_name: str
    conf: float
    xyxy: Tuple[int, int, int, int]
    center_uv: Tuple[int, int]


def yolo_detect(model: YOLO, frame_bgr: np.ndarray) -> List[Detection]:
    r = model(frame_bgr, imgsz=YOLO_IMG_SIZE, conf=YOLO_CONF, verbose=False, device=0)[0]
    dets: List[Detection] = []

    names = r.names  # id -> name
    if r.boxes is None:
        return dets

    h, w = frame_bgr.shape[:2]
    for b in r.boxes:
        cls_id = int(b.cls[0])
        name = str(names.get(cls_id, cls_id))
        conf = float(b.conf[0])

        x1, y1, x2, y2 = b.xyxy[0].tolist()
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(w - 1, int(x2)); y2 = min(h - 1, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue

        cu = int((x1 + x2) / 2)
        cv = int((y1 + y2) / 2)
        dets.append(Detection(name, conf, (x1, y1, x2, y2), (cu, cv)))

    return dets


def in_center_roi(center_uv: Tuple[int, int], frame_shape: Tuple[int, int, int]) -> bool:
    h, w = frame_shape[:2]
    xmin, ymin, xmax, ymax = CENTER_ROI
    u, v = center_uv
    return (xmin * w) <= u <= (xmax * w) and (ymin * h) <= v <= (ymax * h)


def decide_accept_reject(
    frame_bgr: np.ndarray,
    apple_det: Detection,
    defect_dets: List[Detection],
) -> Tuple[str, float]:
    """
    Returns (decision, apple_red_ratio)
    decision in {"accept", "reject_defect", "reject_color"}
    """
    x1, y1, x2, y2 = apple_det.xyxy
    roi = frame_bgr[y1:y2, x1:x2]
    red_ratio = compute_red_ratio_bgr(roi)

    reject_defect = any(d.cls_name in DEFECT_CLASSES and d.conf >= DEFECT_MIN_CONF_REJECT for d in defect_dets)
    if reject_defect:
        return "reject_defect", red_ratio

    if red_ratio < MIN_REDNESS_ACCEPT:
        return "reject_color", red_ratio

    return "accept", red_ratio


# ---------------------------------------------------------------------
# Robot runtime (ROS2 URScript interface)
# ---------------------------------------------------------------------
def start_robot_controller() -> SimpleURDemo:
    if SimpleURDemo is None:
        raise RuntimeError(f"Could not import UR controller from UR5/ur_controller.py: {_UR_IMPORT_ERR}")

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    import threading

    rclpy.init(args=None)
    node = SimpleURDemo()

    exec_ = MultiThreadedExecutor()
    exec_.add_node(node)

    t = threading.Thread(target=exec_.spin, daemon=True)
    t.start()

    print("[UR] Controller node started + spinning.")
    return node


def pose6_from_list(lst: list) -> Pose6:
    return Pose6(lst[0], lst[1], lst[2], lst[3], lst[4], lst[5])


# ---------------------------------------------------------------------
# Calibration “runner” hooks (calls your existing scripts if you want)
# ---------------------------------------------------------------------
def maybe_run_teach_bins_and_workspace():
    """
    If you want main.py to launch your teaching script automatically, point this
    to the correct file. You said you run: python3 ur_controller.py, but that one
    is also used as demo. Your "teaching" one may be ur5_controller.py.
    """
    # Prefer ur5_controller.py if it exists
    candidate = UR_DIR / "ur5_controller.py"
    if candidate.is_file():
        run_cmd(f"python3 {candidate}")
        return

    # fallback: tell user what to run
    print("[INFO] Could not auto-run teaching script (UR5/ur5_controller.py missing).")
    print("       Run your teaching step manually to produce simple_calibration.json.")


def maybe_run_camera_calibration():
    candidate = CAM_DIR / "camera_calibration.py"
    if candidate.is_file():
        run_cmd(f"python3 {candidate}")
        return
    print("[INFO] camera_calibration.py not found. Skipping auto camera calibration.")


def maybe_run_uv_corner_recorder():
    candidate = UR_DIR / "uv_corner_recorder.py"
    if candidate.is_file():
        run_cmd(f"python3 {candidate}")
        return
    raise FileNotFoundError("UR5/uv_corner_recorder.py not found.")


def maybe_run_compute_homography():
    candidate = UR_DIR / "compute_homography.py"
    if candidate.is_file():
        run_cmd(f"python3 {candidate}")
        return
    raise FileNotFoundError("UR5/compute_homography.py not found.")


# ---------------------------------------------------------------------
# Main runtime loop
# ---------------------------------------------------------------------
def main():
    print("\n=== ROB9 MAIN (camera + YOLO + UV->XY + UR pick/place) ===\n")
    print(f"[PATH] Project root: {PROJECT_ROOT}")

    # -----------------------------
    # Step A) Ensure UR bin/workspace calibration exists
    # -----------------------------
    if not file_exists_or_explain(
        SIMPLE_CALIB_JSON,
        "Run your UR teaching script so it creates UR5/simple_calibration.json (workspace + accept + reject).",
    ):
        if prompt_yn("Run the UR teaching script now?", default_yes=True):
            maybe_run_teach_bins_and_workspace()

    if not SIMPLE_CALIB_JSON.is_file():
        print("[FATAL] Missing simple_calibration.json. Cannot continue.")
        sys.exit(1)

    # -----------------------------
    # Step B) Ensure camera calibration (optional)
    # -----------------------------
    if prompt_yn("Have you done camera calibration (intrinsics/extrinsics) for this setup?", default_yes=True) is False:
        maybe_run_camera_calibration()

    # -----------------------------
    # Step C) Ensure UV TCP poses exist
    # -----------------------------
    if not file_exists_or_explain(
        UV_TCP_JSON,
        "Run UR5/uv_corner_recorder.py to create UV+TCP pairs including HOME pose.",
    ):
        if prompt_yn("Run uv_corner_recorder.py now?", default_yes=True):
            maybe_run_uv_corner_recorder()

    if not UV_TCP_JSON.is_file():
        print("[FATAL] Missing uv_tcp_poses.json. Cannot continue.")
        sys.exit(1)

    # -----------------------------
    # Step D) Ensure homography exists
    # -----------------------------
    if not file_exists_or_explain(
        HOMOGRAPHY_JSON,
        "Run UR5/compute_homography.py to create homography_uv_to_base_xy.json",
    ):
        if prompt_yn("Compute homography now?", default_yes=True):
            maybe_run_compute_homography()

    if not HOMOGRAPHY_JSON.is_file():
        print("[FATAL] Missing homography file. Cannot continue.")
        sys.exit(1)

    # -----------------------------
    # Load calibration data
    # -----------------------------
    cal = load_json(SIMPLE_CALIB_JSON)
    home_data = load_json(UV_TCP_JSON)["home_pose"]

    H = load_homography(HOMOGRAPHY_JSON)

    workspace_pose = pose6_from_list(cal.get("workspace") or cal.get("center"))
    accept_pose = pose6_from_list(cal["accept_pose"])
    reject_pose = pose6_from_list(cal["reject_pose"])
    home_pose = pose6_from_list(home_data["tcp_pose"])

    print("\n[CALIB] Loaded:")
    print(f"  workspace: {workspace_pose.to_list()}")
    print(f"  accept   : {accept_pose.to_list()}")
    print(f"  reject   : {reject_pose.to_list()}")
    print(f"  home     : {home_pose.to_list()}")

    # -----------------------------
    # Load YOLO
    # -----------------------------
    weights = Path(os.getenv("YOLO_WEIGHTS", str(YOLO_WEIGHTS_DEFAULT)))
    if not weights.is_file():
        print(f"[FATAL] YOLO weights not found: {weights}")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available. You asked for GPU inference. Fix PyTorch/CUDA.")
        sys.exit(1)

    print(f"\n[YOLO] Loading: {weights}")
    model = YOLO(str(weights))
    model.to("cuda:0")
    print("[YOLO] Using cuda:0")

    # -----------------------------
    # Start robot controller
    # -----------------------------
    if not prompt_yn("Start autonomous run now?", default_yes=True):
        print("[OK] Exiting (idle).")
        return

    ur = start_robot_controller()

    # -----------------------------
    # Start camera
    # -----------------------------
    streams = start_realsense()
    print("[CAM] RealSense streaming started.")

    # Go HOME once
    print("[UR] Going HOME...")
    ur.move_linear(home_pose, speed=0.25, accel=0.5)
    time.sleep(0.2)

    print("\n[RUN] Press CTRL+C to stop.\n")

    try:
        while True:
            # Ensure we're at HOME before grabbing a frame
            ur.move_linear(home_pose, speed=0.25, accel=0.5)

            frame = get_color_frame(streams)

            dets = yolo_detect(model, frame)
            apples = [d for d in dets if d.cls_name == APPLE_CLASS and d.conf >= APPLE_MIN_CONF_TO_PICK]

            # stop condition: no apples in center ROI
            apples_center = [a for a in apples if in_center_roi(a.center_uv, frame.shape)]
            if not apples_center:
                print("[RUN] No apples in center ROI. Waiting...")
                time.sleep(0.5)
                continue

            # Choose best apple (highest confidence)
            apple = max(apples_center, key=lambda d: d.conf)

            # Collect defects that overlap with this apple box (simple overlap check)
            ax1, ay1, ax2, ay2 = apple.xyxy
            defect_dets = []
            for d in dets:
                if d.cls_name in DEFECT_CLASSES:
                    dx1, dy1, dx2, dy2 = d.xyxy
                    # IoU-ish overlap (quick)
                    ix1, iy1 = max(ax1, dx1), max(ay1, dy1)
                    ix2, iy2 = min(ax2, dx2), min(ay2, dy2)
                    if ix2 > ix1 and iy2 > iy1:
                        defect_dets.append(d)

            decision, apple_red = decide_accept_reject(frame, apple, defect_dets)

            u, v = apple.center_uv
            x, y = uv_to_xy(H, u, v)

            # ignore near drop zones (robot-space exclusion)
            def near(p: Pose6) -> bool:
                return (x - p.x) ** 2 + (y - p.y) ** 2 <= (DROP_ZONE_RADIUS_M ** 2)

            if near(accept_pose) or near(reject_pose):
                print("[SKIP] Apple maps into drop-zone region. Skipping.")
                time.sleep(0.2)
                continue

            print(f"\n[TARGET] apple conf={apple.conf:.2f} uv=({u},{v}) -> xy=({x:.3f},{y:.3f}) red={apple_red:.2f} decision={decision}")

            # Build pick poses using workspace orientation + Z
            pick_z = workspace_pose.z
            pick_pose = Pose6(x, y, pick_z, workspace_pose.rx, workspace_pose.ry, workspace_pose.rz)
            approach_pose = Pose6(x, y, pick_z + APPROACH_Z_OFFSET, workspace_pose.rx, workspace_pose.ry, workspace_pose.rz)
            lift_pose = Pose6(x, y, pick_z + LIFT_Z_OFFSET, workspace_pose.rx, workspace_pose.ry, workspace_pose.rz)

            # Approach -> descend -> vacuum -> lift
            ur.move_linear(approach_pose, speed=0.20, accel=0.4)
            ur.move_linear(pick_pose, speed=0.12, accel=0.25)

            # vacuum ON (tune channels)
            vg10_grip(channel_a=60, channel_b=60)
            time.sleep(0.25)

            ur.move_linear(lift_pose, speed=0.20, accel=0.4)

            # Place
            drop_pose = accept_pose if decision == "accept" else reject_pose
            drop_approach = Pose6(drop_pose.x, drop_pose.y, drop_pose.z + 0.12, drop_pose.rx, drop_pose.ry, drop_pose.rz)

            ur.move_linear(drop_approach, speed=0.25, accel=0.5)
            ur.move_linear(drop_pose, speed=0.15, accel=0.3)

            # vacuum OFF
            vg10_release()
            time.sleep(0.15)

            # back up and go home
            ur.move_linear(drop_approach, speed=0.25, accel=0.5)
            ur.move_linear(home_pose, speed=0.25, accel=0.5)

    except KeyboardInterrupt:
        print("\n[STOP] Stopping...")
    finally:
        try:
            streams.pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[DONE]")


if __name__ == "__main__":
    main()
