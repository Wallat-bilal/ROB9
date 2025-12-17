#!/usr/bin/env python3
"""
uv_corner_recorder.py

Records:
- HOME TCP pose (camera sees full table)
- Multiple labeled (u,v) pixel points + matching TCP poses

NEW:
- Auto-move to saved HOME pose before each point using URScript movel()

Output JSON:
{
  "created_at_unix": ...,
  "robot_model": "ur10",
  "topics": {...},
  "home_pose": [x,y,z,rx,ry,rz],
  "points": [{"label":..., "uv":[u,v], "tcp_pose":[...]}]
}
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String as StringMsg


# ---------------------------- TOPICS ----------------------------------------

TCP_POSE_TOPIC = "/tcp_pose_broadcaster/pose"
URSCRIPT_TOPIC = "/urscript_interface/script_command"

THIS_DIR = Path(__file__).resolve().parent
OUT_JSON = THIS_DIR / "uv_tcp_poses.json"


# ---------------------------- DATA TYPES ------------------------------------

@dataclass
class Pose6:
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz]


# ---------------------- quaternion -> axis-angle ----------------------------

def quat_to_axis_angle(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    """Quaternion -> UR axis-angle vector [rx, ry, rz]."""
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-9:
        return 0.0, 0.0, 0.0
    angle = 2.0 * math.atan2(norm, w)
    ax = x / norm
    ay = y / norm
    az = z / norm
    return ax * angle, ay * angle, az * angle


# --------------------------- ROS2 NODE --------------------------------------

class URPoseRecorder(Node):
    def __init__(self):
        super().__init__("uv_tcp_pose_recorder")

        self._latest_pose: Optional[PoseStamped] = None
        self._pose_event = threading.Event()

        self.create_subscription(PoseStamped, TCP_POSE_TOPIC, self._pose_cb, 10)
        self.script_pub = self.create_publisher(StringMsg, URSCRIPT_TOPIC, 10)

        self.get_logger().info(f"Subscribed: {TCP_POSE_TOPIC}")
        self.get_logger().info(f"Publishing URScript: {URSCRIPT_TOPIC}")

    def _pose_cb(self, msg: PoseStamped):
        self._latest_pose = msg
        self._pose_event.set()

    def get_tcp_pose6(self, timeout_s: float = 2.0) -> Pose6:
        """Get latest TCP pose from ROS and convert to Pose6."""
        if not self._pose_event.wait(timeout=timeout_s):
            raise RuntimeError(
                f"No TCP pose received from {TCP_POSE_TOPIC} in {timeout_s}s. "
                "Is tcp_pose_broadcaster running?"
            )
        if self._latest_pose is None:
            raise RuntimeError("Pose event set but pose is None (unexpected). Try again.")

        p = self._latest_pose.pose
        rx, ry, rz = quat_to_axis_angle(
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w
        )
        return Pose6(
            x=float(p.position.x),
            y=float(p.position.y),
            z=float(p.position.z),
            rx=float(rx),
            ry=float(ry),
            rz=float(rz),
        )

    def send_urscript(self, script: str):
        msg = StringMsg()
        msg.data = script
        self.script_pub.publish(msg)


# --------------------------- URSCRIPT HELPERS -------------------------------

def urscript_movel(p: Pose6, a: float = 0.25, v: float = 0.12) -> str:
    """
    Generate a safe-ish linear move to a TCP pose using URScript.

    a: acceleration (m/s^2)  (keep small for safety)
    v: speed (m/s)          (keep small for safety)
    """
    # UR expects: movel(p[x,y,z,rx,ry,rz], a=..., v=...)
    return (
        f"def __go_home__():\n"
        f"  movel(p[{p.x:.6f},{p.y:.6f},{p.z:.6f},{p.rx:.6f},{p.ry:.6f},{p.rz:.6f}], a={a}, v={v})\n"
        f"end\n"
        f"__go_home__()\n"
    )


# --------------------------- CLI HELPERS ------------------------------------

def yes_no(prompt: str) -> bool:
    while True:
        ans = input(prompt + " [y/n]: ").strip().lower()
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please answer y or n.")


def ask_label(prompt: str) -> str:
    s = input(prompt).strip()
    return s if s else f"pt_{int(time.time())}"


def ask_uv(prompt: str = "Enter pixel") -> Tuple[int, int]:
    while True:
        s = input(prompt + " (format: u v): ").strip()
        parts = s.split()
        if len(parts) != 2:
            print("Type two integers like: 554 210")
            continue
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            print("Both must be integers.")


def record_pose_interactive(node: URPoseRecorder, label: str) -> Pose6:
    input(
        f"\n[UR] Move robot to: {label}\n"
        f"    Then press ENTER here to record TCP pose..."
    )
    pose = node.get_tcp_pose6()
    print(f"[UR] Recorded pose: {pose.to_list()}")
    return pose


def save_json(payload: Dict[str, Any], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[SAVED] {out_path}")


# --------------------------- MAIN -------------------------------------------

def main():
    rclpy.init()
    node = URPoseRecorder()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        print("\n=== UV + TCP POSE RECORDER (with HOME + auto-return) ===")
        robot_model = input("Robot model label (e.g. ur10): ").strip() or "ur10"

        # Give ROS graph a moment to connect publishers/subscribers
        time.sleep(0.5)

        # 1) Record HOME pose once
        print("\nStep 1: Record HOME pose (camera sees full table).")
        home_pose = record_pose_interactive(node, "HOME pose (camera sees full table)")

        payload: Dict[str, Any] = {
            "created_at_unix": time.time(),
            "robot_model": robot_model,
            "topics": {
                "tcp_pose_topic": TCP_POSE_TOPIC,
                "urscript_topic": URSCRIPT_TOPIC,
            },
            "home_pose": home_pose.to_list(),
            "points": [],
        }
        save_json(payload, OUT_JSON)

        # Motion settings (tune if you want)
        home_accel = 0.25
        home_speed = 0.12

        print("\nStep 2: Record points.")
        print("For EACH point:")
        print("  - Script will send robot to HOME automatically")
        print("  - You confirm it arrived (press ENTER)")
        print("  - You enter label + pixel (u v)")
        print("  - You touch the physical point and press ENTER to record pose\n")

        while True:
            if not yes_no("Add a new (u,v) point?"):
                break

            # Auto-return to HOME before each point
            input("\n[STEP] Press ENTER to send robot to HOME automatically...")
            script = urscript_movel(home_pose, a=home_accel, v=home_speed)
            node.send_urscript(script)

            # No feedback channel here, so user confirms arrival
            input("[STEP] Wait until robot reaches HOME, then press ENTER to continue...")

            label = ask_label("Point label (e.g. bottom_left corner1): ")
            u, v = ask_uv("Enter pixel")

            pose = record_pose_interactive(
                node,
                f"Touch physical point for {label} (matching pixel {u},{v})"
            )

            payload["points"].append({
                "label": label,
                "uv": [u, v],
                "tcp_pose": pose.to_list(),
            })

            save_json(payload, OUT_JSON)
            print(f"[OK] Added {label}: uv=({u},{v})")

        print("\nDone.")
        print(f"Output file: {OUT_JSON}")

    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
