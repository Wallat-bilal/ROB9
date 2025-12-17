#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String as StringMsg

# ---------------------------- CONFIG ----------------------------------------
TCP_POSE_TOPIC = "/tcp_pose_broadcaster/pose"
URSCRIPT_TOPIC = "/urscript_interface/script_command"

SAFE_ACC = 0.2
SAFE_VEL = 0.2

PROJECT_ROOT = Path(__file__).resolve().parent
CALIB_PATH = PROJECT_ROOT / "zones_calibration.json"


# ---------------------------- DATA TYPES ------------------------------------
@dataclass
class Pose6:
    """UR-style pose: x, y, z, rx, ry, rz in base frame."""
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz]

    @classmethod
    def from_list(cls, lst):
        return cls(*map(float, lst))


def quat_to_axis_angle(x: float, y: float, z: float, w: float):
    """Quaternion -> UR axis-angle (rx, ry, rz)."""
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-9:
        return 0.0, 0.0, 0.0
    angle = 2.0 * math.atan2(norm, w)
    ax, ay, az = x / norm, y / norm, z / norm
    return ax * angle, ay * angle, az * angle


# --------------------------- ROS2 NODE --------------------------------------
class ZoneRecorder(Node):
    def __init__(self):
        super().__init__("zone_recorder")

        self.current_tcp_pose: Optional[PoseStamped] = None
        self.pose_event = threading.Event()

        self.create_subscription(PoseStamped, TCP_POSE_TOPIC, self._tcp_pose_cb, 10)
        self.get_logger().info(f"Subscribed to TCP pose on {TCP_POSE_TOPIC}")

        self.script_pub = self.create_publisher(StringMsg, URSCRIPT_TOPIC, 10)
        self.get_logger().info(f"Will publish URScript to {URSCRIPT_TOPIC}")

    def _tcp_pose_cb(self, msg: PoseStamped):
        self.current_tcp_pose = msg
        self.pose_event.set()

    def get_tcp_pose(self) -> Pose6:
        if not self.pose_event.wait(timeout=2.0):
            raise RuntimeError(
                f"No TCP pose received from {TCP_POSE_TOPIC}. "
                "Is tcp_pose_broadcaster running?"
            )
        p = self.current_tcp_pose.pose
        rx, ry, rz = quat_to_axis_angle(
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w
        )
        return Pose6(
            x=p.position.x, y=p.position.y, z=p.position.z,
            rx=rx, ry=ry, rz=rz
        )

    def record_pose_interactive(self, label: str) -> Pose6:
        input(
            f"\n[UR] Move robot to: {label}\n"
            f"    Then press ENTER here to record TCP pose..."
        )
        pose = self.get_tcp_pose()
        print(f"[UR] Recorded {label}: {pose.to_list()}")
        return pose


def centroid_pose(corners: List[Pose6]) -> Pose6:
    """Make a drop pose at centroid of the 4 corners. Orientation from first corner."""
    x = sum(p.x for p in corners) / len(corners)
    y = sum(p.y for p in corners) / len(corners)
    z = sum(p.z for p in corners) / len(corners)
    ref = corners[0]
    return Pose6(x=x, y=y, z=z, rx=ref.rx, ry=ref.ry, rz=ref.rz)


def main():
    print("\n=== ZONE RECORDER (accept/reject as 4-corner polygons) ===\n")
    print("You will record:")
    print("  1) workspace_pose (touch table / pick plane reference)")
    print("  2) accept zone corners (4 points)")
    print("  3) reject zone corners (4 points)\n")
    print("IMPORTANT: Record corners in THIS order:")
    print("  bottom_left -> bottom_right -> top_right -> top_left\n")

    rclpy.init()
    node = ZoneRecorder()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        workspace = node.record_pose_interactive("workspace_pose (touch table / pick plane reference)")

        print("\n--- ACCEPT ZONE ---")
        a_bl = node.record_pose_interactive("ACCEPT bottom_left corner")
        a_br = node.record_pose_interactive("ACCEPT bottom_right corner")
        a_tr = node.record_pose_interactive("ACCEPT top_right corner")
        a_tl = node.record_pose_interactive("ACCEPT top_left corner")
        accept_corners = [a_bl, a_br, a_tr, a_tl]
        accept_pose = centroid_pose(accept_corners)

        print("\n--- REJECT ZONE ---")
        r_bl = node.record_pose_interactive("REJECT bottom_left corner")
        r_br = node.record_pose_interactive("REJECT bottom_right corner")
        r_tr = node.record_pose_interactive("REJECT top_right corner")
        r_tl = node.record_pose_interactive("REJECT top_left corner")
        reject_corners = [r_bl, r_br, r_tr, r_tl]
        reject_pose = centroid_pose(reject_corners)

        data: Dict[str, Any] = {
            "workspace_pose": workspace.to_list(),
            "accept_zone": {
                "corner_order": ["bottom_left", "bottom_right", "top_right", "top_left"],
                "corners": [p.to_list() for p in accept_corners],
            },
            "reject_zone": {
                "corner_order": ["bottom_left", "bottom_right", "top_right", "top_left"],
                "corners": [p.to_list() for p in reject_corners],
            },
            # Keep these so existing code can still “drop into a pose”
            "accept_pose": accept_pose.to_list(),
            "reject_pose": reject_pose.to_list(),
        }

        with open(CALIB_PATH, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n[SAVED] {CALIB_PATH}")
        print("[OK] accept_pose/reject_pose were computed as zone centroids (for dropping).")

    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
