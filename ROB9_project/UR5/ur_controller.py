"""
Simple UR demo:

- Connects to UR ROS 2 driver (real robot).
- Saves 3 poses:
    1) Center of the table (used as "home")
    2) Accepted box
    3) Rejected box
- Stores them in simple_calibration.json
- Then optionally runs a demo:
    * You manually pick an apple with the robot (wherever).
    * Script asks if it's accepted or rejected.
    * Robot moves to the corresponding saved box pose:
        - Move above box
        - Move down to box
        - Wait 3 seconds (to drop apple)
        - Move back above box
    * Then robot moves back "home" (center of table, above it).

No recalibration loops, no TCP calibration questions.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String as StringMsg

#from vg10_gripper import VG10Gripper

# ---------------------------- CONFIG ----------------------------------------

TCP_POSE_TOPIC = "/tcp_pose_broadcaster/pose"
URSCRIPT_TOPIC = "/urscript_interface/script_command"

SAFE_ACC = 0.2
SAFE_VEL = 0.2

PROJECT_ROOT = Path(__file__).resolve().parent
CALIB_PATH = PROJECT_ROOT / "simple_calibration.json"


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

    def to_list(self):
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz]

    @classmethod
    def from_list(cls, lst):
        return cls(*map(float, lst))


# ---------------------- quaternion -> axis-angle ----------------------------

def quat_to_axis_angle(x: float, y: float, z: float, w: float):
    """
    Convert quaternion to UR-style axis-angle vector [rx, ry, rz],
    where direction is the rotation axis and magnitude is the angle.
    """
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-9:
        return 0.0, 0.0, 0.0

    angle = 2.0 * math.atan2(norm, w)
    axis_x = x / norm
    axis_y = y / norm
    axis_z = z / norm

    rx = axis_x * angle
    ry = axis_y * angle
    rz = axis_z * angle
    return rx, ry, rz


# --------------------------- ROS2 NODE --------------------------------------

class SimpleURDemo(Node):
    def __init__(self):
        super().__init__("simple_ur_demo")

        # TCP pose subscription
        self.current_tcp_pose: Optional[PoseStamped] = None
        self.pose_event = threading.Event()

        self.create_subscription(
            PoseStamped,
            TCP_POSE_TOPIC,
            self._tcp_pose_cb,
            10,
        )
        self.get_logger().info(f"Subscribed to TCP pose on {TCP_POSE_TOPIC}")

        # URScript publisher
        self.script_pub = self.create_publisher(StringMsg, URSCRIPT_TOPIC, 10)
        self.get_logger().info(f"Will publish URScript to {URSCRIPT_TOPIC}")

        # Storage for calibration poses (in memory)
        self.center_pose: Optional[Pose6] = None
        self.accept_pose: Optional[Pose6] = None
        self.reject_pose: Optional[Pose6] = None

    # --------------- ROS callbacks & pose access -----------------------------

    def _tcp_pose_cb(self, msg: PoseStamped):
        self.current_tcp_pose = msg
        self.pose_event.set()

    def get_tcp_pose(self) -> Pose6:
        """
        Get current TCP pose from ROS and convert to UR-style (x,y,z,rx,ry,rz).
        """
        if not self.pose_event.wait(timeout=2.0):
            raise RuntimeError(
                f"No TCP pose received from {TCP_POSE_TOPIC} within timeout. "
                "Is tcp_pose_broadcaster running?"
            )

        if self.current_tcp_pose is None:
            raise RuntimeError("TCP pose callback didn't set pose yet, try again.")

        p = self.current_tcp_pose.pose
        rx, ry, rz = quat_to_axis_angle(
            p.orientation.x,
            p.orientation.y,
            p.orientation.z,
            p.orientation.w,
        )

        return Pose6(
            x=p.position.x,
            y=p.position.y,
            z=p.position.z,
            rx=rx,
            ry=ry,
            rz=rz,
        )

    # ------------------------ URScript helper --------------------------------

    def _publish_script(self, script: str):
        msg = StringMsg()
        msg.data = script
        self.script_pub.publish(msg)
        self.get_logger().info(f"Sent URScript: {script.strip()}")

    def move_linear(self, pose: Pose6, acc: float = SAFE_ACC, vel: float = SAFE_VEL):
        cmd = (
            f"movel(p[{pose.x:.6f},{pose.y:.6f},{pose.z:.6f},"
            f"{pose.rx:.6f},{pose.ry:.6f},{pose.rz:.6f}], a={acc}, v={vel})\n"
        )
        self._publish_script(cmd)

    # --------------------- Calibration (3 poses) -----------------------------

    def record_pose_interactive(self, label: str) -> Pose6:
        input(
            f"\n[UR] Move the robot so the TCP touches the '{label}' point,\n"
            f"    then press ENTER here..."
        )
        pose = self.get_tcp_pose()
        print(f"[UR] Recorded {label}: {pose.to_list()}")
        return pose

    def calibrate_three_points(self):
        """
        1) Center of table (home)
        2) Accepted box
        3) Rejected box
        """
        print("\n=== SIMPLE WORKSPACE SETUP ===")

        self.center_pose = self.record_pose_interactive("CENTER OF TABLE (HOME)")
        self.accept_pose = self.record_pose_interactive("ACCEPTED BOX")
        self.reject_pose = self.record_pose_interactive("REJECTED BOX")

        data = {
            "center": self.center_pose.to_list(),
            "accept": self.accept_pose.to_list(),
            "reject": self.reject_pose.to_list(),
        }

        with open(CALIB_PATH, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n[UR] Saved calibration poses to {CALIB_PATH}")

    # ---------------------------- Motion logic -------------------------------

    def move_home(self, lift_above_center: float = 0.15):
        """
        Move to 'home' – center of the table with some height above the table.

        Assumes:
        - self.center_pose was recorded with the TCP touching the table center
          (or a few mm above).
        """
        if self.center_pose is None:
            print("[UR] No center pose calibrated.")
            return

        home = Pose6(
            x=self.center_pose.x,
            y=self.center_pose.y,
            z=self.center_pose.z + lift_above_center,
            rx=self.center_pose.rx,
            ry=self.center_pose.ry,
            rz=self.center_pose.rz,
        )
        print("[UR] Moving to HOME (center of table, above)...")
        self.move_linear(home)

    def move_to_box_and_back_home(
            self,
            accepted: bool,
            approach_lift: float = 0.15,
            hover_above_box: float = 0.02,
            dwell_s: float = 3.0,
    ):
        """
        Move from current pose to the chosen box (accept/reject),
        wait a bit at the drop position, then go back 'home'.

        Assumes:
        - self.accept_pose / self.reject_pose were recorded with the TCP
          touching the *top surface* of the box or just above it.
        - We will:
            1) go to a point ABOVE that surface (approach_lift),
            2) descend to a hover pose (hover_above_box),
            3) wait dwell_s seconds,
            4) go back up to the approach pose,
            5) return to home.
        """
        if self.center_pose is None or self.accept_pose is None or self.reject_pose is None:
            raise RuntimeError("Poses not calibrated yet.")

        target_base = self.accept_pose if accepted else self.reject_pose

        # 1) Approach pose above the box (safe height)
        approach_pose = Pose6(
            x=target_base.x,
            y=target_base.y,
            z=target_base.z + approach_lift,
            rx=target_base.rx,
            ry=target_base.ry,
            rz=target_base.rz,
        )

        # 2) Hover pose close to the recorded box surface
        hover_pose = Pose6(
            x=target_base.x,
            y=target_base.y,
            z=target_base.z + hover_above_box,
            rx=target_base.rx,
            ry=target_base.ry,
            rz=target_base.rz,
        )

        label = "ACCEPTED" if accepted else "REJECTED"
        print(f"[UR] Moving apple to {label} BOX...")

        # Move above box, then down, wait, and back up
        self.move_linear(approach_pose)
        self.move_linear(hover_pose)
        print(f"[UR] Waiting {dwell_s:.1f} seconds to drop apple...")
        time.sleep(dwell_s)
        self.move_linear(approach_pose)

        # Then back home (center + height)
        self.move_home(lift_above_center=approach_lift)


# --------------------------- CLI HELPERS ------------------------------------

def yes_no(prompt: str) -> bool:
    while True:
        ans = input(prompt + " [y/n]: ").strip().lower()
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please answer y or n.")


# ------------------------------- MAIN ---------------------------------------

def main():
    rclpy.init()
    node = SimpleURDemo()

    # Spin in background so ROS callbacks run while we use input()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # 1) Calibrate the 3 points
    node.calibrate_three_points()

    # 2) Ask if we should run demo
    if not yes_no("\nRun demo now?"):
        print("[UR] Done. You can use the saved poses later for automation.")
        rclpy.shutdown()
        return

    print(
        "\n=== DEMO MODE ===\n"
        "Instructions:\n"
        "  - Manually move the robot to pick an apple (wherever it is) and close the gripper.\n"
        "  - Then answer whether that apple is ACCEPTED or REJECTED.\n"
        "  - The robot will carry it to the saved box position, wait 3s, then go HOME.\n"
    )

    try:
        while True:
            if not yes_no("Move current apple to a box?"):
                print("[UR] Demo finished.")
                break

            q = input("Is this apple [a]ccepted or [r]ejected? ").strip().lower()
            accepted = q.startswith("a")
            node.move_to_box_and_back_home(accepted=accepted)

    except KeyboardInterrupt:
        print("\n[UR] Demo interrupted by user.")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
