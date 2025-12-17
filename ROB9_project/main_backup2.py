#!/usr/bin/env python3
"""
ROB9 MAIN
- Checks calibration files
- Runs optional calibration steps
- Loads YOLO + RealSense
- Loads UV->XY homography
- Runs a simple pick/place decision loop

This file is designed to NOT crash if your JSON keys are named:
  - center/accept/reject   (your current simple_calibration.json)
or:
  - workspace/accept_pose/reject_pose (older code style)
"""

from __future__ import annotations

import json
import sys
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional

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

SIMPLE_CAL = UR_DIR / "simple_calibration.json"
UV_TCP = UR_DIR / "uv_tcp_poses.json"
HOMO_JSON = UR_DIR / "homography_uv_to_base_xy.json"


# -----------------------------
# Configuration
# -----------------------------
YOLO_WEIGHTS = PROJECT_ROOT / "runs_AppleRF" / "yolov8s_applerf_phase2" / "weights" / "best.pt"

# Decision thresholds (tune as needed)
CONF_APPLE = 0.90
REDNESS_PICK_THRESHOLD = 0.00      # Only pick apples >= 90% red (your requirement)
PEST_REJECT_THRESHOLD = 0.50       # If pest >= 50% -> reject
SCRATCH_REJECT_THRESHOLD = 0.50    # If scratch >= 50% -> reject (optional but useful)

# RealSense streams
W, H, FPS = 640, 480, 30

# Robot + pick safety/tuning
URSCRIPT_TOPIC = "/urscript_interface/script_command"

APPROACH_Z_OFFSET = 0.00   # relative to workspace_pose.z (0 = same)
PICK_Z_DROP = 0.06         # go DOWN this much from workspace_pose.z to touch apple (tune!)
LIFT_Z_UP = 0.10           # lift up after picking (meters)
BIN_APPROACH_Z_UP = 0.10   # approach bin from above

# "Don't pick" exclusion radius around accept/reject zones (meters)
BIN_EXCLUSION_RADIUS = 0.18

# Vacuum settings (VG10)
VG10_GRIP_SRV = "/vg10_grip"
VG10_RELEASE_SRV = "/vg10_release"
VG10_GRIP_A = 20
VG10_GRIP_B = 20


# -----------------------------
# Helpers
# -----------------------------
def ask_yn(prompt: str, default_yes: bool = True) -> bool:
    suffix = " [Y/n]: " if default_yes else " [y/N]: "
    ans = input(prompt + suffix).strip().lower()
    if ans == "":
        return default_yes
    return ans.startswith("y")


def run_cmd(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    print("\n[CMD] " + " ".join(cmd) + "\n")
    return subprocess.run(cmd, check=check, capture_output=False)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def pose6_from_list(x: Any) -> np.ndarray:
    return np.array(x, dtype=float).reshape(6)


def _get_pose_any(cal: Dict[str, Any], keys: List[str]) -> np.ndarray:
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
    workspace_pose = _get_pose_any(cal, ["workspace_pose", "workspace", "center"])
    accept_pose = _get_pose_any(cal, ["accept_pose", "accept"])
    reject_pose = _get_pose_any(cal, ["reject_pose", "reject"])
    return Calibration(workspace_pose=workspace_pose, accept_pose=accept_pose, reject_pose=reject_pose)


def load_homography(path: Path) -> np.ndarray:
    obj = load_json(path)
    for k in ["H", "homography", "H_uv_to_xy"]:
        if k in obj:
            return np.array(obj[k], dtype=float).reshape(3, 3)
    if "data" in obj:
        return np.array(obj["data"], dtype=float).reshape(3, 3)
    raise KeyError(f"Homography JSON missing expected key. Found keys: {list(obj.keys())}")


def uv_to_xy(Hm: np.ndarray, u: float, v: float) -> Tuple[float, float]:
    pt = np.array([u, v, 1.0], dtype=float).reshape(3, 1)
    out = Hm @ pt
    out /= (out[2, 0] + 1e-12)
    return float(out[0, 0]), float(out[1, 0])


def red_ratio_bgr(img_bgr: np.ndarray) -> float:
    if img_bgr is None or img_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 140, 80], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([160, 140, 80], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)
    return float(np.count_nonzero(mask)) / float(img_bgr.shape[0] * img_bgr.shape[1] + 1e-12)


# -----------------------------
# ROS2 URScript + Vacuum
# -----------------------------
def _ros2_ok() -> bool:
    """Return True if ros2 can resolve std_msgs/msg/String."""
    try:
        r = subprocess.run(
            ["ros2", "interface", "show", "std_msgs/msg/String"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return r.returncode == 0
    except FileNotFoundError:
        return False


def publish_urscript(script: str) -> None:
    """
    Publish a URScript program to the UR driver topic.
    IMPORTANT: You must have sourced ROS2 + ur_ws in this terminal.
    """
    safe = script.replace("\\", "\\\\").replace('"', '\\"')
    msg = f'{{data: "{safe}"}}'
    cmd = ["ros2", "topic", "pub", "--once", URSCRIPT_TOPIC, "std_msgs/msg/String", msg]
    run_cmd(cmd, check=True)


def urscript_movel(pose6: np.ndarray, a: float = 1.2, v: float = 0.25) -> str:
    x, y, z, rx, ry, rz = [float(v) for v in pose6.tolist()]
    return (
        "def prog():\n"
        f"  movel(p[{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}], a={a}, v={v})\n"
        "end\n"
        "prog()\n"
    )


def vacuum_on() -> None:
    cmd = [
        "ros2", "service", "call",
        VG10_GRIP_SRV,
        "vg_control_interfaces/srv/VacuumSet",
        f"{{channel_a: {VG10_GRIP_A}, channel_b: {VG10_GRIP_B}}}",
    ]
    run_cmd(cmd, check=True)


def vacuum_off() -> None:
    cmd = [
        "ros2", "service", "call",
        VG10_RELEASE_SRV,
        "vg_control_interfaces/srv/VacuumSet",
        "{channel_a: 0, channel_b: 0}",
    ]
    run_cmd(cmd, check=True)


# -----------------------------
# YOLO
# -----------------------------
def load_yolo(weights: Path) -> YOLO:
    if not weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")
    model = YOLO(str(weights))
    if torch.cuda.is_available():
        model.to("cuda:0")
        print("[YOLO] Using GPU cuda:0")
    else:
        print("[YOLO] CUDA not available -> CPU (will be slower)")
    return model


def pick_best_apple_detection(results, class_name="apple", conf_min=0.5):
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    names = r.names
    best, best_conf = None, -1.0
    for b in r.boxes:
        cls_id = int(b.cls[0])
        conf = float(b.conf[0])
        name = names.get(cls_id, str(cls_id))
        if name != class_name or conf < conf_min:
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
    return np.asanyarray(color.get_data()), np.asanyarray(depth.get_data())


# -----------------------------
# Pick/place action
# -----------------------------
def dist_xy(p1: np.ndarray, x: float, y: float) -> float:
    return float(np.hypot(p1[0] - x, p1[1] - y))


def make_pose_from_xy(base_pose: np.ndarray, x: float, y: float, z: float) -> np.ndarray:
    p = base_pose.copy()
    p[0] = float(x)
    p[1] = float(y)
    p[2] = float(z)
    return p


def do_pick_and_place(
    cal: Calibration,
    pick_x: float,
    pick_y: float,
    is_reject: bool,
    enable_vacuum: bool,
):
    """
    Simple sequence:
      HOME -> above apple -> down -> vacuum on -> lift -> above bin -> down -> release -> lift -> HOME
    """
    home = cal.workspace_pose
    bin_pose = cal.reject_pose if is_reject else cal.accept_pose

    safe_z = float(home[2] + APPROACH_Z_OFFSET)
    pick_z = float(home[2] - PICK_Z_DROP)

    # Build poses
    above_pick = make_pose_from_xy(home, pick_x, pick_y, safe_z)
    down_pick = make_pose_from_xy(home, pick_x, pick_y, pick_z)

    above_bin = bin_pose.copy()
    above_bin[2] = float(bin_pose[2] + BIN_APPROACH_Z_UP)

    # 1) HOME
    publish_urscript(urscript_movel(home))
    time.sleep(0.5)

    # 2) above apple
    publish_urscript(urscript_movel(above_pick))
    time.sleep(0.5)

    # 3) down to apple
    publish_urscript(urscript_movel(down_pick, a=0.6, v=0.10))
    time.sleep(0.3)

    # 4) vacuum ON
    if enable_vacuum:
        vacuum_on()
    time.sleep(0.4)

    # 5) lift
    lift_pose = above_pick.copy()
    lift_pose[2] = float(safe_z + LIFT_Z_UP)
    publish_urscript(urscript_movel(lift_pose, a=1.0, v=0.20))
    time.sleep(0.5)

    # 6) above bin
    publish_urscript(urscript_movel(above_bin, a=1.2, v=0.25))
    time.sleep(0.5)

    # 7) down to bin
    publish_urscript(urscript_movel(bin_pose, a=0.6, v=0.12))
    time.sleep(0.3)

    # 8) vacuum OFF
    if enable_vacuum:
        vacuum_off()
    time.sleep(0.3)

    # 9) back above bin then HOME
    publish_urscript(urscript_movel(above_bin, a=1.2, v=0.25))
    time.sleep(0.4)
    publish_urscript(urscript_movel(home, a=1.2, v=0.25))
    time.sleep(0.5)


# -----------------------------
# Main
# -----------------------------
def main():
    print("\n=== ROB9 MAIN (camera + YOLO + UV->XY + UR pick/place) ===\n")
    print(f"[PATH] Project root: {PROJECT_ROOT}")

    # 1) Check calibration files
    if not SIMPLE_CAL.exists():
        print(f"[MISS] Missing: {SIMPLE_CAL}")
        print("[INFO] Run UR5/ur_controller.py first to record workspace/accept/reject.")
        sys.exit(1)
    print(f"[OK] Found: {SIMPLE_CAL}")

    if not UV_TCP.exists():
        print(f"[MISS] Missing: {UV_TCP}")
        print("[INFO] Running UV corner recorder...")
        run_cmd(["python3", str(UR_DIR / "uv_corner_recorder.py")], check=True)
    print(f"[OK] Found: {UV_TCP}")

    if not HOMO_JSON.exists():
        print(f"[MISS] Missing: {HOMO_JSON}")
        print("[INFO] Computing homography...")
        run_cmd(["python3", str(UR_DIR / "compute_homography.py")], check=True)
    print(f"[OK] Found: {HOMO_JSON}")

    # Camera calibration prompt
    cam_done = ask_yn("Have you done camera calibration (intrinsics/extrinsics) for this setup?", default_yes=True)
    if not cam_done:
        run_cmd(["python3", str(CAM_DIR / "camera_calibration.py")], check=False)

    cal = load_simple_calibration(SIMPLE_CAL)
    Hm = load_homography(HOMO_JSON)

    # Load YOLO
    print(f"[YOLO] Weights: {YOLO_WEIGHTS}")
    model = load_yolo(YOLO_WEIGHTS)

    # Robot enable
    enable_motion = ask_yn("Enable REAL robot motion + vacuum now?", default_yes=False)
    enable_vacuum = enable_motion  # keep simple: vacuum only if robot enabled

    if enable_motion:
        if not _ros2_ok():
            print("\n[ERROR] ROS2 not sourced in this terminal. Do this before running main:")
            print("  source /opt/ros/humble/setup.bash")
            print("  source ~/ur_ws/install/setup.bash")
            print("  (then activate your venv) and rerun.\n")
            enable_motion = False
            enable_vacuum = False
        else:
            print("[RUN] Robot motion ENABLED.")
            print(f"[RUN] URScript topic: {URSCRIPT_TOPIC}")

    # Start camera
    pipeline, align = start_realsense()
    print("\n[RUN] Starting live loop. Press 'q' to quit.\n")

    last_pick_time = 0.0
    PICK_COOLDOWN_S = 2.0

    try:
        while True:
            color_img, _depth_img = get_frames(pipeline, align)

            results = model(color_img, conf=0.25, verbose=False, device=0 if torch.cuda.is_available() else "cpu")
            best_apple = pick_best_apple_detection(results, class_name="apple", conf_min=CONF_APPLE)
            defects = extract_defects(results)

            annotated = results[0].plot()

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

            roi = color_img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            rr = red_ratio_bgr(roi)

            # defect logic
            reject_defect = False
            for name, conf, _box in defects:
                if name == "pest" and conf >= PEST_REJECT_THRESHOLD:
                    reject_defect = True
                if name == "scratch" and conf >= SCRATCH_REJECT_THRESHOLD:
                    reject_defect = True

            should_pick = rr >= REDNESS_PICK_THRESHOLD

            if not should_pick:
                decision = "IDLE (not red enough)"
            elif reject_defect:
                decision = "PICK -> REJECT"
            else:
                decision = "PICK -> ACCEPT"

            bx, by = uv_to_xy(Hm, cx, cy)

            # Exclusion: don't pick inside bin zones
            too_close_accept = dist_xy(cal.accept_pose, bx, by) < BIN_EXCLUSION_RADIUS
            too_close_reject = dist_xy(cal.reject_pose, bx, by) < BIN_EXCLUSION_RADIUS
            in_bin_zone = too_close_accept or too_close_reject

            # Overlay
            cv2.circle(annotated, (cx, cy), 6, (255, 255, 0), -1)
            cv2.putText(annotated, f"apple_conf={apple_conf:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"red={rr*100:.0f}% (pick>= {int(REDNESS_PICK_THRESHOLD*100)}%)", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"uv=({cx},{cy}) -> base_xy=({bx:.3f},{by:.3f})", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            color_dec = (0, 255, 0) if "ACCEPT" in decision else (0, 0, 255)
            cv2.putText(annotated, decision, (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_dec, 2, cv2.LINE_AA)

            if in_bin_zone:
                cv2.putText(annotated, "IGNORED: in bin zone", (20, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)

            # --- ACTION: pick and place ---
            now = time.time()
            if enable_motion and (now - last_pick_time) > PICK_COOLDOWN_S:
                if should_pick and (not in_bin_zone):
                    is_reject = reject_defect
                    print(f"\n[ACTION] Picking at base_xy=({bx:.3f},{by:.3f}) -> {'REJECT' if is_reject else 'ACCEPT'}")
                    try:
                        do_pick_and_place(cal, bx, by, is_reject=is_reject, enable_vacuum=enable_vacuum)
                        last_pick_time = time.time()
                    except subprocess.CalledProcessError as e:
                        print(f"[ERROR] ROS2 command failed: {e}")
                        print("[HINT] Ensure you sourced ROS2 + ur_ws in THIS terminal, and robot driver is running.")
                        enable_motion = False
                        enable_vacuum = False
                # else idle

            cv2.imshow("ROB9 MAIN", annotated)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q") or k == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
