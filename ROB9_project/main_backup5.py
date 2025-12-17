#!/usr/bin/env python3
"""
ROB9 MAIN (camera + YOLO + UV->XY + UR pick/place)

Behavior:
1) Ask user if these are done (NO auto-run):
   - accept/reject/workspace poses (simple_calibration.json)
   - camera calibration (optional, just a question gate)
   - uv_tcp_poses.json (HOME pose + uv points)
   - homography_uv_to_base_xy.json

2) Start camera + YOLO
3) If motion enabled: move robot to HOME pose (from uv_tcp_poses.json)
4) Loop:
   - detect apple
   - map pixel center (u,v) -> base (x,y) using homography
   - decide ACCEPT/REJECT based on redness + pest/scratch confidence
   - pick -> place -> return HOME
"""

from __future__ import annotations

import json
import sys
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, List

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

SIMPLE_CAL = UR_DIR / "simple_calibration.json"
UV_TCP = UR_DIR / "uv_tcp_poses.json"
HOMO_JSON = UR_DIR / "homography_uv_to_base_xy.json"
ZONES_CAL = UR_DIR / "zones_calibration.json"


# -----------------------------
# Configuration
# -----------------------------
YOLO_WEIGHTS = PROJECT_ROOT / "runs_AppleRF" / "yolov8s_applerf_phase2" / "weights" / "best.pt"

# Apple selection
CONF_APPLE = 0.70  # YOLO confidence for "apple" box to be considered

# Redness threshold (your requirement “>= 50% red”)
REDNESS_PICK_THRESHOLD = 0.00

# Defect reject logic (confidence used as proxy)
PEST_REJECT_THRESHOLD = 0.50
SCRATCH_REJECT_THRESHOLD = 0.50

# RealSense streams
W, H, FPS = 640, 480, 30

# ROS topics/services
URSCRIPT_TOPIC = "/urscript_interface/script_command"

VG10_GRIP_SRV = "/vg10_grip"
VG10_RELEASE_SRV = "/vg10_release"

# You said you want the 4 middle cups; that mapping is in your VG10 node setup.
# These numbers are just “strength” inputs.
VG10_GRIP_A = 20
VG10_GRIP_B = 20

# -----------------------------
# Z / motion tuning (meters)
# -----------------------------
# These are the ONLY numbers you should need to tweak for “how high/low” things are.

# Workspace/apple Z:
# - WORKSPACE_Z_OFFSET shifts the whole working plane up/down (base frame meters).
#   Positive = higher (less negative), Negative = lower (more negative).
WORKSPACE_Z_OFFSET = -0.03

# - PICK_Z_DROP is how far DOWN from the workspace plane you go to “touch” the apple.
#   Increase if you are not reaching the apple; decrease if you are crashing into the table.
PICK_Z_DROP = 0.06

# - APPLE_Z_OFFSET is a fine trim applied on top of PICK_Z_DROP.
#   Positive = pick a bit higher, Negative = pick a bit lower.
APPLE_Z_OFFSET = 0.05

# Bin (accept/reject) Z:
# - These offsets shift the recorded accept/reject poses up/down.
ACCEPT_Z_OFFSET = 0.1
REJECT_Z_OFFSET = 0.1

# -----------------------------
# UR10 motion speed tuning
# -----------------------------
# These map directly to URScript movel() parameters:
#   - a: acceleration (m/s^2)
#   - v: TCP speed (m/s)
#
# Travel moves: HOME <-> above apple <-> above bin
MOVE_A_TRAVEL = 1.2
MOVE_V_TRAVEL = 0.25

# Approach moves: down to apple / down to bin (slow & safe)
MOVE_A_APPROACH_PICK = 0.6
MOVE_V_APPROACH_PICK = 0.10
MOVE_A_APPROACH_BIN = 0.6
MOVE_V_APPROACH_BIN = 0.12

# Lift / retreat speed (after picking)
MOVE_A_LIFT = 1.0
MOVE_V_LIFT = 0.20

# How long to wait after commanding HOME before starting camera/detections
HOME_SETTLE_S = 1.0

# Approach / lift heights:
LIFT_Z_UP = 0.10         # lift after picking (relative to workspace plane)
BIN_APPROACH_Z_UP = 0.10 # approach height above bin pose

# Don't pick inside bins (meters)
BIN_EXCLUSION_RADIUS = 0.18

# Simple cooldown so it doesn't spam pick commands
PICK_COOLDOWN_S = 5.0


# -----------------------------
# Helpers
# -----------------------------
def ask_yn(prompt: str, default_yes: bool = True) -> bool:
    suffix = " [Y/n]: " if default_yes else " [y/N]: "
    ans = input(prompt + suffix).strip().lower()
    if ans == "":
        return default_yes
    return ans.startswith("y")


def run_cmd(cmd: List[str], check: bool = True, timeout_s: float | None = 20.0) -> subprocess.CompletedProcess:
    print("\n[CMD] " + " ".join(cmd) + "\n")
    return subprocess.run(cmd, check=check, capture_output=False, timeout=timeout_s)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def pose6_from_list(x: Any) -> np.ndarray:
    return np.array(x, dtype=float).reshape(6)


def _get_pose_any(cal: Dict[str, Any], keys: List[str]) -> np.ndarray:
    for k in keys:
        if k in cal:
            return pose6_from_list(cal[k])
    raise KeyError(f"Missing keys {keys}. Found: {list(cal.keys())}")


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


@dataclass
class Zones:
    """Bin zones as polygons in base XY."""
    accept_xy: np.ndarray
    reject_xy: np.ndarray


def load_zones_calibration(path: Path) -> Zones:
    """
    Expects zones_calibration.json created by zone_recorder.py:
      {
        "accept_zone": {"corners": [[x,y,z,rx,ry,rz], ...]},
        "reject_zone": {"corners": ...}
      }
    Only (x,y) are used for zone inclusion checks.
    """
    obj = load_json(path)

    def _zone_xy(key: str) -> np.ndarray:
        if key not in obj or "corners" not in obj[key]:
            raise KeyError(f"zones_calibration.json missing '{key}.corners'. Keys: {list(obj.keys())}")
        corners = obj[key]["corners"]
        if not isinstance(corners, list) or len(corners) < 3:
            raise ValueError(f"Zone '{key}' must have at least 3 corners.")
        return np.array([[float(c[0]), float(c[1])] for c in corners], dtype=float)

    return Zones(
        accept_xy=_zone_xy("accept_zone"),
        reject_xy=_zone_xy("reject_zone"),
    )


def point_in_poly_xy(x: float, y: float, poly_xy: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test (treats edge as inside)."""
    if poly_xy is None or len(poly_xy) < 3:
        return False

    inside = False
    n = len(poly_xy)
    j = n - 1

    for i in range(n):
        xi, yi = float(poly_xy[i][0]), float(poly_xy[i][1])
        xj, yj = float(poly_xy[j][0]), float(poly_xy[j][1])

        # Check if point is on the segment (xi,yi)-(xj,yj)
        dx, dy = xj - xi, yj - yi
        if abs(dx) + abs(dy) > 1e-12:
            t = ((x - xi) * dx + (y - yi) * dy) / (dx * dx + dy * dy)
            if 0.0 <= t <= 1.0:
                px, py = xi + t * dx, yi + t * dy
                if (x - px) ** 2 + (y - py) ** 2 < 1e-10:
                    return True

        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        )
        if intersect:
            inside = not inside
        j = i

    return inside



def load_home_pose_from_uv(path: Path) -> np.ndarray:
    """
    Expects uv_tcp_poses.json to contain something like:
      { "home_pose": [x,y,z,rx,ry,rz], "points": {...} }
    """
    obj = load_json(path)
    for k in ["home_pose", "home", "homePose", "homepose"]:
        if k in obj:
            return pose6_from_list(obj[k])
    raise KeyError(f"uv_tcp_poses.json missing HOME pose key. Found keys: {list(obj.keys())}")


def load_homography(path: Path) -> np.ndarray:
    obj = load_json(path)
    for k in ["H", "homography", "H_uv_to_xy"]:
        if k in obj:
            return np.array(obj[k], dtype=float).reshape(3, 3)
    if "data" in obj:
        return np.array(obj["data"], dtype=float).reshape(3, 3)
    raise KeyError(f"Homography JSON missing expected key. Found: {list(obj.keys())}")


def uv_to_xy(Hm: np.ndarray, u: float, v: float) -> Tuple[float, float]:
    pt = np.array([u, v, 1.0], dtype=float).reshape(3, 1)
    out = Hm @ pt
    out /= (out[2, 0] + 1e-12)
    return float(out[0, 0]), float(out[1, 0])


def dist_xy(pose6: np.ndarray, x: float, y: float) -> float:
    return float(np.hypot(pose6[0] - x, pose6[1] - y))


def make_pose_from_xy(base_pose: np.ndarray, x: float, y: float, z: float) -> np.ndarray:
    p = base_pose.copy()
    p[0] = float(x)
    p[1] = float(y)
    p[2] = float(z)
    return p


# -----------------------------
# Redness (HSV)
# -----------------------------
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
# ROS2 helpers
# -----------------------------
def _ros2_ok() -> bool:
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
    Publishes std_msgs/String on /urscript_interface/script_command.
    Uses YAML dict: {data: "..."}.
    """
    safe = script.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    msg = f'{{data: "{safe}"}}'
    cmd = ["ros2", "topic", "pub", "--once", URSCRIPT_TOPIC, "std_msgs/msg/String", msg]
    run_cmd(cmd, check=True)


def urscript_movel(pose6: np.ndarray, a: float = 0.2, v: float = 0.2) -> str:
    x, y, z, rx, ry, rz = [float(vv) for vv in pose6.tolist()]
    return (
        "def prog():\n"
        f"  movel(p[{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}], a={a}, v={v})\n"
        "end\n"
        "prog()\n"
    )


def _service_exists(name: str) -> bool:
    try:
        r = subprocess.run(["ros2", "service", "list"], capture_output=True, text=True, check=True)
        return any(line.strip() == name for line in r.stdout.splitlines())
    except Exception:
        return False


def vacuum_on() -> None:
    cmd = [
        "ros2", "service", "call",
        VG10_GRIP_SRV,
        "vg_control_interfaces/srv/VacuumSet",
        f"{{channel_a: {VG10_GRIP_A}, channel_b: {VG10_GRIP_B}}}",
    ]
    run_cmd(cmd, check=True, timeout_s=10.0)


def vacuum_off() -> None:
    # /vg10_release is VacuumRelease (not VacuumSet)
    cmd = [
        "ros2", "service", "call",
        VG10_RELEASE_SRV,
        "vg_control_interfaces/srv/VacuumRelease",
        "{release_vacuum: 1}",
    ]
    run_cmd(cmd, check=True)



# -----------------------------
# YOLO helpers
# -----------------------------
def load_yolo(weights: Path) -> YOLO:
    if not weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")
    model = YOLO(str(weights))
    if torch.cuda.is_available():
        model.to("cuda:0")
        print("[YOLO] Using GPU cuda:0")
    else:
        print("[YOLO] CUDA not available -> CPU")
    return model


def pick_best_apple_detection(results, conf_min=0.5):
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    names = r.names
    best, best_conf = None, -1.0
    for b in r.boxes:
        cls_id = int(b.cls[0])
        conf = float(b.conf[0])
        name = names.get(cls_id, str(cls_id))
        if name != "apple" or conf < conf_min:
            continue
        if conf > best_conf:
            xyxy = b.xyxy[0].cpu().numpy().astype(float)
            best = (xyxy, conf)
            best_conf = conf
    return best


def extract_defects(results, defect_names=("pest", "scratch"), conf_min=0.25):
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
# Pick/Place sequence
# -----------------------------
def do_pick_and_place(
    cal: Calibration,
    home_pose: np.ndarray,
    pick_x: float,
    pick_y: float,
    is_reject: bool,
    enable_vacuum: bool,
):
    """
    HOME -> above apple -> down -> vacuum -> lift -> above bin -> down -> release -> back HOME
    Uses:
      - home_pose from uv_tcp_poses.json (camera sees full workspace)
      - workspace_pose from simple_calibration.json (orientation + Z reference)
    """
    workspace = cal.workspace_pose

    # --- Z offsets (tweak at top of file) ---
    safe_z = float(workspace[2] + WORKSPACE_Z_OFFSET)
    pick_z = float(safe_z - PICK_Z_DROP + APPLE_Z_OFFSET)

    # pick poses use workspace orientation
    above_pick = make_pose_from_xy(workspace, pick_x, pick_y, safe_z)
    down_pick = make_pose_from_xy(workspace, pick_x, pick_y, pick_z)

    # --- Bin Z offsets (tweak at top of file) ---
    bin_pose_raw = cal.reject_pose if is_reject else cal.accept_pose
    bin_pose = bin_pose_raw.copy()
    bin_pose[2] = float(bin_pose[2] + (REJECT_Z_OFFSET if is_reject else ACCEPT_Z_OFFSET))

    above_bin = bin_pose.copy()
    above_bin[2] = float(bin_pose[2] + BIN_APPROACH_Z_UP)

    # HOME
    publish_urscript(urscript_movel(home_pose, a=MOVE_A_TRAVEL, v=MOVE_V_TRAVEL))
    time.sleep(0.6)

    # above apple
    publish_urscript(urscript_movel(above_pick, a=MOVE_A_TRAVEL, v=MOVE_V_TRAVEL))
    time.sleep(0.6)

    # down
    publish_urscript(urscript_movel(down_pick, a=MOVE_A_APPROACH_PICK, v=MOVE_V_APPROACH_PICK))
    time.sleep(0.4)

    # vacuum on
    if enable_vacuum:
        vacuum_on()
    time.sleep(0.5)

    # lift
    lift_pose = above_pick.copy()
    lift_pose[2] = float(safe_z + LIFT_Z_UP)
    publish_urscript(urscript_movel(lift_pose, a=MOVE_A_LIFT, v=MOVE_V_LIFT))
    time.sleep(0.6)

    # above bin
    publish_urscript(urscript_movel(above_bin, a=MOVE_A_TRAVEL, v=MOVE_V_TRAVEL))
    time.sleep(0.6)

    # down to bin
    publish_urscript(urscript_movel(bin_pose, a=MOVE_A_APPROACH_BIN, v=MOVE_V_APPROACH_BIN))
    time.sleep(0.5)

    # vacuum off
    if enable_vacuum:
        vacuum_off()
    time.sleep(0.4)

    # back above bin then HOME
    publish_urscript(urscript_movel(above_bin, a=MOVE_A_TRAVEL, v=MOVE_V_TRAVEL))
    time.sleep(0.5)
    publish_urscript(urscript_movel(home_pose, a=MOVE_A_TRAVEL, v=MOVE_V_TRAVEL))
    time.sleep(0.6)


# -----------------------------
# Main
# -----------------------------
def main():
    print("\n=== ROB9 MAIN (camera + YOLO + UV->XY + UR pick/place) ===\n")
    print(f"[PATH] Project root: {PROJECT_ROOT}")

    # ---- Step gates (NO auto-run) ----
    if not ask_yn("Have you recorded accept/reject/workspace poses (simple_calibration.json)?", default_yes=True):
        print("\n[STOP] Please run:")
        print("  cd ~/Desktop/ROB9/ROB9_project/UR5")
        print("  python3 ur_controller.py")
        print("and make sure UR5/simple_calibration.json exists.\n")
        return

    if not SIMPLE_CAL.exists():
        print(f"\n[STOP] Missing file: {SIMPLE_CAL}\n")
        return
    print(f"[OK] Found: {SIMPLE_CAL}")

    if not ask_yn("Have you recorded ACCEPT/REJECT box zones (zones_calibration.json)?", default_yes=True):
        print("\n[STOP] Please run:")
        print(f"  python3 {PROJECT_ROOT / 'zone_recorder.py'}")
        print("and make sure zones_calibration.json exists next to main.py.\n")
        return

    if not ZONES_CAL.exists():
        print(f"\n[STOP] Missing file: {ZONES_CAL}\n")
        return
    print(f"[OK] Found: {ZONES_CAL}")

    cam_done = ask_yn("Have you done camera calibration (intrinsics/extrinsics) for this setup?", default_yes=True)
    if not cam_done:
        print("\n[NOTE] Do your camera calibration step manually (you chose 'no').\n")

    if not ask_yn("Have you recorded HOME + (u,v) points (uv_tcp_poses.json)?", default_yes=True):
        print("\n[STOP] Please run:")
        print("  cd ~/Desktop/ROB9/ROB9_project/UR5")
        print("  python3 uv_corner_recorder.py")
        print("and make sure UR5/uv_tcp_poses.json exists.\n")
        return

    if not UV_TCP.exists():
        print(f"\n[STOP] Missing file: {UV_TCP}\n")
        return
    print(f"[OK] Found: {UV_TCP}")

    if not ask_yn("Have you computed homography (homography_uv_to_base_xy.json)?", default_yes=True):
        print("\n[STOP] Please run:")
        print("  cd ~/Desktop/ROB9/ROB9_project/UR5")
        print("  python3 compute_homography.py")
        print("and make sure UR5/homography_uv_to_base_xy.json exists.\n")
        return

    if not HOMO_JSON.exists():
        print(f"\n[STOP] Missing file: {HOMO_JSON}\n")
        return
    print(f"[OK] Found: {HOMO_JSON}")

    # ---- Load calibrations ----
    cal = load_simple_calibration(SIMPLE_CAL)
    home_pose = load_home_pose_from_uv(UV_TCP)
    Hm = load_homography(HOMO_JSON)
    zones = load_zones_calibration(ZONES_CAL)

    # ---- Load YOLO ----
    print(f"[YOLO] Weights: {YOLO_WEIGHTS}")
    model = load_yolo(YOLO_WEIGHTS)

    # ---- Robot enable ----
    enable_motion = ask_yn("Enable REAL robot motion + vacuum now?", default_yes=False)
    enable_vacuum = enable_motion

    if enable_motion:
        if not _ros2_ok():
            print("\n[ERROR] ROS2 is not sourced in THIS terminal.\nDo this first, in the same terminal:")
            print("  source /opt/ros/humble/setup.bash")
            print("  source ~/ur_ws/install/setup.bash")
            print("Then activate venv and rerun main.\n")
            return

        print("[RUN] Robot motion ENABLED.")
        print(f"[RUN] URScript topic: {URSCRIPT_TOPIC}")

    # ---- Move HOME FIRST (camera starts only after robot is at HOME) ----
    if enable_motion:
        print("[RUN] Moving robot to HOME pose (from uv_tcp_poses.json)...")
        publish_urscript(urscript_movel(home_pose, a=MOVE_A_TRAVEL, v=MOVE_V_TRAVEL))
        time.sleep(HOME_SETTLE_S)

    # ---- Start camera AFTER HOME ----
    pipeline, align = start_realsense()
    camera_msg = "\n[RUN] Camera started. Press 'q' to quit.\n"
    if enable_motion:
        camera_msg = "\n[RUN] Camera started (robot is at HOME). Press 'q' to quit.\n"
    print(camera_msg)

    last_pick_time = 0.0
    last_home_time = 0.0

    try:
        while True:
            color_img, _depth_img = get_frames(pipeline, align)

            results = model(
                color_img,
                conf=0.25,
                verbose=False,
                device=0 if torch.cuda.is_available() else "cpu",
            )

            best_apple = pick_best_apple_detection(results, conf_min=CONF_APPLE)
            defects = extract_defects(results, defect_names=("pest", "scratch"))

            annotated = results[0].plot()

            if best_apple is None:
                cv2.putText(annotated, "No apple detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("ROB9 MAIN", annotated)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q") or k == 27:
                    break
                continue

            xyxy, apple_conf = best_apple
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            roi = color_img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            rr = red_ratio_bgr(roi)

            # Reject if strong pest/scratch detection
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

            # Exclusion: avoid picking inside ACCEPT/REJECT box zones (4-corner polygons)
            in_accept_zone = point_in_poly_xy(bx, by, zones.accept_xy)
            in_reject_zone = point_in_poly_xy(bx, by, zones.reject_xy)
            in_bin_zone = in_accept_zone or in_reject_zone

            # Overlay info
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

            # ACTION
            now = time.time()

            # If the apple is already inside a bin zone, ignore and ensure robot is at HOME
            if enable_motion and should_pick and in_bin_zone and (now - last_home_time) > 1.0:
                publish_urscript(urscript_movel(home_pose, a=MOVE_A_TRAVEL, v=MOVE_V_TRAVEL))
                last_home_time = now

            if enable_motion and (now - last_pick_time) > PICK_COOLDOWN_S:
                if should_pick and (not in_bin_zone):
                    is_reject = reject_defect
                    print(f"\n[ACTION] Picking at base_xy=({bx:.3f},{by:.3f}) -> {'REJECT' if is_reject else 'ACCEPT'}")
                    try:
                        do_pick_and_place(
                            cal=cal,
                            home_pose=home_pose,
                            pick_x=bx,
                            pick_y=by,
                            is_reject=is_reject,
                            enable_vacuum=enable_vacuum,
                        )
                        last_pick_time = time.time()
                    except subprocess.CalledProcessError as e:
                        print(f"[ERROR] ROS2 command failed: {e}")
                        print("[HINT] Check you sourced ROS2 + ur_ws in THIS terminal and driver is running.")
                        enable_motion = False
                        enable_vacuum = False

            cv2.imshow("ROB9 MAIN", annotated)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q") or k == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()