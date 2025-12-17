from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional: quiet down very chatty 'urx' logging
# ---------------------------------------------------------------------------
logging.getLogger("urx").setLevel(logging.WARNING)

try:
    import urx
    from urx import ursecmon
except ImportError:
    urx = None
    ursecmon = None  # type: ignore
    print(
        "[WARN] 'urx' library not found. Install with:\n"
        "       pip install urx\n"
        "This module will not work with a REAL robot until 'urx' is installed."
    )


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# CHANGE THIS to your robot's IP on the network
DEFAULT_ROBOT_IP = "192.168.1.10"

# Default speeds/accelerations – VERY conservative
SAFE_ACC = 0.2      # m/s^2 for movel, rad/s^2 for movej
SAFE_VEL = 0.2      # m/s   for movel, rad/s   for movej

# Path to calibration file
PROJECT_ROOT = Path(__file__).resolve().parent
CALIB_PATH = PROJECT_ROOT / "calibration.json"


# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class Pose:
    """Simple wrapper for a UR pose (x, y, z, rx, ry, rz) in base frame."""
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float

    @classmethod
    def from_list(cls, lst):
        return cls(*map(float, lst))

    def to_list(self):
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz]


@dataclass
class URCalibration:
    """
    Everything we need from calibration for this project.
    All poses are expressed in the robot base frame.

    tcp_pose:       Optional – where the TCP is relative to flange.
                    (If you configure TCP on the pendant you can ignore this.)
    pickup_pose:    Pose at the APPLE PICKUP center (on the table).
    accept_pose:    Pose where apples should be dropped in the ACCEPT box.
    reject_pose:    Pose where apples should be dropped in the REJECT box.
    table_height:   z value of the table plane (optional helper).
    """
    tcp_pose: Optional[Pose] = None
    pickup_pose: Optional[Pose] = None
    accept_pose: Optional[Pose] = None
    reject_pose: Optional[Pose] = None
    table_height: Optional[float] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        for key in ["tcp_pose", "pickup_pose", "accept_pose", "reject_pose"]:
            if d[key] is not None:
                p = d[key]
                d[key] = (
                    p["x"],
                    p["y"],
                    p["z"],
                    p["rx"],
                    p["ry"],
                    p["rz"],
                )
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "URCalibration":
        def pose_or_none(v):
            if v is None:
                return None
            return Pose.from_list(v)

        return cls(
            tcp_pose=pose_or_none(d.get("tcp_pose")),
            pickup_pose=pose_or_none(d.get("pickup_pose")),
            accept_pose=pose_or_none(d.get("accept_pose")),
            reject_pose=pose_or_none(d.get("reject_pose")),
            table_height=d.get("table_height"),
        )


# ---------------------------------------------------------------------------
# MAIN CONTROLLER CLASS
# ---------------------------------------------------------------------------

class URRobotController:
    def __init__(self, ip: str = DEFAULT_ROBOT_IP, use_sim: bool = False):
        """
        Connect to a UR robot using urx.

        If use_sim=True, all motion commands are printed but not sent
        (handy for debugging on a laptop without the robot).
        """
        self.ip = ip
        self.use_sim = use_sim
        self.robot = None

        if not use_sim:
            if urx is None:
                raise RuntimeError(
                    "[UR] 'urx' is not installed. Install with 'pip install urx' "
                    "or run in simulation mode."
                )

            print(f"[UR] Connecting to UR robot at {ip} ...")
            last_error = None
            for attempt in range(5):
                try:
                    # IMPORTANT:
                    #   use_rt=False -> use the older secondary interface instead of RTDE
                    #   This avoids the PoseVector/tolist bug in some urx + firmware combos.
                    self.robot = urx.Robot(ip, use_rt=False)
                    print("[UR] Connected.")
                    break
                except Exception as e:
                    last_error = e
                    print(
                        f"[UR] Error talking to robot (attempt {attempt + 1}/5): {e}\n"
                        "     Is the robot fully booted and not in fault? Retrying..."
                    )
                    time.sleep(1.0)
            else:
                raise RuntimeError(
                    f"[UR] Could not establish a stable connection to robot at {ip} "
                    f"after 5 retries: {last_error}"
                )

        # Load calibration if available
        self.calib: URCalibration = self._load_calibration()

    # ------------------ utility IO ------------------

    def _load_calibration(self) -> URCalibration:
        if CALIB_PATH.is_file():
            with open(CALIB_PATH, "r") as f:
                data = json.load(f)
            print(f"[UR] Loaded calibration from {CALIB_PATH}")
            return URCalibration.from_dict(data)
        else:
            print("[UR] No calibration.json found – starting with empty calibration.")
            return URCalibration()

    def _save_calibration(self):
        data = self.calib.to_dict()
        with open(CALIB_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[UR] Calibration saved to {CALIB_PATH}")

    # ------------------ low-level motion ------------------

    def get_pose(self) -> Pose:
        """
        Get current TCP pose in base frame.

        In simulation mode, returns a dummy pose at origin (all zeros) so that
        the CLI can still run without crashing.
        """
        if self.use_sim:
            print("[SIM] get_pose() -> returning dummy pose [0,0,0,0,0,0]")
            return Pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        p = self.robot.getl()  # [x, y, z, rx, ry, rz]
        return Pose.from_list(p)

    def move_linear(self, pose: Pose, acc: float = SAFE_ACC, vel: float = SAFE_VEL):
        """Linear move to a target pose."""
        if self.use_sim:
            print(f"[SIM] movel to {pose.to_list()}, acc={acc}, vel={vel}")
            return
        self.robot.movel(pose.to_list(), acc=acc, vel=vel, wait=True)

    def move_joints(self, joints, acc: float = SAFE_ACC, vel: float = SAFE_VEL):
        """Joint-space move (if you need it)."""
        if self.use_sim:
            print(f"[SIM] movej to {joints}, acc={acc}, vel={vel}")
            return
        self.robot.movej(joints, acc=acc, vel=vel, wait=True)

    def set_tcp(self, tcp_pose: Pose):
        """Set TCP on the robot (optional; you can also do it via teach pendant)."""
        if self.use_sim:
            print(f"[SIM] set_tcp {tcp_pose.to_list()}")
        else:
            self.robot.set_tcp(tcp_pose.to_list())
        self.calib.tcp_pose = tcp_pose
        self._save_calibration()

    # ------------------ gripper stubs ------------------

    def open_gripper(self):
        """
        TODO: replace this with your actual gripper IO.
        Example for a digital output on pin 0:
            self.robot.set_digital_out(0, False)
        """
        if self.use_sim:
            print("[SIM] open_gripper()")
            return
        print("[UR] Opening gripper (stub). Implement your IO here.")
        # self.robot.set_digital_out(0, False)

    def close_gripper(self):
        """
        TODO: replace this with your actual gripper IO.
        Example for a digital output on pin 0:
            self.robot.set_digital_out(0, True)
        """
        if self.use_sim:
            print("[SIM] close_gripper()")
            return
        print("[UR] Closing gripper (stub). Implement your IO here.")
        # self.robot.set_digital_out(0, True)

    # -----------------------------------------------------------------------
    # CALIBRATION ROUTINES
    # -----------------------------------------------------------------------

    def _record_pose_interactive(self, label: str) -> Pose:
        """
        Helper: tell user to move the robot with the teach pendant & press Enter.
        Then we read and return the current TCP pose.
        """
        input(
            f"\n[UR] Move the robot so the TCP touches the '{label}' point,\n"
            f"    then press ENTER here..."
        )
        pose = self.get_pose()
        print(f"[UR] Recorded {label} pose: {pose.to_list()}")
        return pose

    def calibrate_tcp(self):
        """
        Very simple TCP calibration helper:
        - Ask user to put the gripper TCP at some known reference point.
        - Save that as tcp_pose.
        In practice you may prefer using the built-in TCP wizard on the pendant.
        """
        print("\n[UR] --- TCP calibration ---")
        p = self._record_pose_interactive("TCP reference")
        self.set_tcp(p)
        print("[UR] TCP pose stored. You can also refine it on the teach pendant.")

    def calibrate_workspace(self):
        """
        Calibrate:
            - pickup center
            - accept box drop pose
            - reject box drop pose
            - table height (z of pickup)
        """
        print("\n[UR] --- Workspace calibration ---")

        pickup = self._record_pose_interactive("PICKUP AREA CENTER")
        self.calib.pickup_pose = pickup
        self.calib.table_height = pickup.z

        accept = self._record_pose_interactive("ACCEPT BOX DROP POINT")
        self.calib.accept_pose = accept

        reject = self._record_pose_interactive("REJECT BOX DROP POINT")
        self.calib.reject_pose = reject

        self._save_calibration()
        print("[UR] Workspace poses stored.")

    # -----------------------------------------------------------------------
    # PICK & PLACE PRIMITIVES
    # -----------------------------------------------------------------------

    def move_to_home(self, home_joints=None):
        """
        Move to a safe home pose.
        If home_joints is None, we just print a reminder.
        """
        if home_joints is None:
            print(
                "[UR] No home_joints provided. "
                "Use the pendant to move to a safe pose manually."
            )
            return
        print("[UR] Moving to home pose...")
        self.move_joints(home_joints)

    def pick_apple(
        self,
        pick_pose: Pose,
        approach_height: float = 0.10,
        lift_height: float = 0.20,
    ):
        """
        Simple pick motion:
            1) move above apple at approach_height
            2) descend to pick_pose.z
            3) close gripper
            4) lift to lift_height above pickup plane
        """
        z_table = self.calib.table_height if self.calib.table_height is not None else pick_pose.z

        pre_grasp = Pose(
            x=pick_pose.x,
            y=pick_pose.y,
            z=z_table + approach_height,
            rx=pick_pose.rx,
            ry=pick_pose.ry,
            rz=pick_pose.rz,
        )
        grasp = Pose(
            x=pick_pose.x,
            y=pick_pose.y,
            z=z_table,
            rx=pick_pose.rx,
            ry=pick_pose.ry,
            rz=pick_pose.rz,
        )
        lift = Pose(
            x=pick_pose.x,
            y=pick_pose.y,
            z=z_table + lift_height,
            rx=pick_pose.rx,
            ry=pick_pose.ry,
            rz=pick_pose.rz,
        )

        print("[UR] Executing pick sequence...")
        self.move_linear(pre_grasp)
        self.move_linear(grasp)
        self.close_gripper()
        time.sleep(0.5)
        self.move_linear(lift)

    def place_apple(self, target_pose: Pose):
        """
        Place sequence:
            1) move over box (z + some offset)
            2) descend
            3) open gripper
            4) move back up
        """
        z_table = self.calib.table_height if self.calib.table_height is not None else target_pose.z
        above = Pose(
            x=target_pose.x,
            y=target_pose.y,
            z=z_table + 0.15,
            rx=target_pose.rx,
            ry=target_pose.ry,
            rz=target_pose.rz,
        )
        drop = Pose(
            x=target_pose.x,
            y=target_pose.y,
            z=z_table,
            rx=target_pose.rx,
            ry=target_pose.ry,
            rz=target_pose.rz,
        )

        print("[UR] Executing place sequence...")
        self.move_linear(above)
        self.move_linear(drop)
        self.open_gripper()
        time.sleep(0.3)
        self.move_linear(above)

    def pick_and_place(self, pick_pose: Pose, quality: str):
        """
        quality: "accept" or "reject".
        Uses calibrated accept/reject poses.
        """
        if quality not in {"accept", "reject"}:
            raise ValueError("quality must be 'accept' or 'reject'.")

        if quality == "accept":
            if self.calib.accept_pose is None:
                raise RuntimeError("accept_pose not calibrated.")
            target = self.calib.accept_pose
        else:
            if self.calib.reject_pose is None:
                raise RuntimeError("reject_pose not calibrated.")
            target = self.calib.reject_pose

        self.pick_apple(pick_pose)
        self.place_apple(target)


# ---------------------------------------------------------------------------
# SIMPLE CLI / STATE MACHINE
# ---------------------------------------------------------------------------

def _yes_no(prompt: str) -> bool:
    while True:
        ans = input(prompt + " [y/n]: ").strip().lower()
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please answer y or n.")


# Global controller handle used by the stub
_GLOBAL_CONTROLLER: Optional[URRobotController] = None


def get_next_apple_stub() -> Optional[Tuple[Pose, str]]:
    """
    Placeholder for integration with your camera+CNN.

    Returns:
        (pick_pose_in_base_frame, quality) where quality in {'accept', 'reject'}
        or None if no apple is available.

    For now, we just ask the user to jog to an apple and type 'a' or 'r'.
    """
    print("\n[DEMO] get_next_apple_stub() called.")
    if not _yes_no("Place an apple and move TCP above it. Pick one?"):
        return None

    quality_input = input("Type quality for this apple: [a]ccept / [r]eject: ").strip().lower()
    if quality_input.startswith("r"):
        quality = "reject"
    else:
        quality = "accept"

    controller = _GLOBAL_CONTROLLER
    if controller is None:
        raise RuntimeError("internal error: controller not set")

    pose = controller.get_pose()
    # In a real system, pose would come from vision.
    pose.z = controller.calib.table_height or pose.z
    print(f"[DEMO] Using current pose as pick pose: {pose.to_list()}, quality={quality}")
    return pose, quality


def main_cli():
    """
    Text-based wizard you can run with:

        python ur5_controller.py

    You can later call the same functions from your own main script
    instead of going through this CLI.
    """
    global _GLOBAL_CONTROLLER
    use_sim = _yes_no("Run in SIMULATION mode (no real robot commands)?")
    controller = URRobotController(ip=DEFAULT_ROBOT_IP, use_sim=use_sim)
    _GLOBAL_CONTROLLER = controller

    # 1) Calibration
    print("\n=== CALIBRATION CHECK ===")
    if (
        controller.calib.pickup_pose
        and controller.calib.accept_pose
        and controller.calib.reject_pose
    ):
        print("[UR] Existing calibration found.")
        if _yes_no("Have you already done full calibration and want to KEEP it?"):
            pass
        else:
            controller.calibrate_workspace()
    else:
        print("[UR] Calibration incomplete – starting workspace calibration.")
        controller.calibrate_workspace()

    # Optional TCP step
    if _yes_no(
        "Do you want to (re)calibrate TCP via this script? "
        "(If you already set TCP on pendant, you can say no)"
    ):
        controller.calibrate_tcp()

    # 2) Autonomous loop
    print("\n=== AUTONOMOUS LOOP ===")
    if not _yes_no("Start autonomous pick-and-place loop now?"):
        print("Exiting without running loop.")
        return

    print("\n[UR] Starting demo loop. Ctrl+C or 'n' in stub to stop.")
    try:
        while True:
            result = get_next_apple_stub()
            if result is None:
                print("[DEMO] No more apples, exiting loop.")
                break
            pick_pose, quality = result
            controller.pick_and_place(pick_pose, quality)
    except KeyboardInterrupt:
        print("\n[UR] Interrupted by user. Stopping loop.")

    print("[UR] Demo finished.")


# Backwards-compat: if someone imports UR5Controller by old name
UR5Controller = URRobotController  # alias

if __name__ == "__main__":
    main_cli()
