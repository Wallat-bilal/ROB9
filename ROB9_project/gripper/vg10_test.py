#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String as StringMsg

# Import VG10 helper from the file in the same folder
from vg10_gripper import VG10Gripper, VG10Config

# Topic used by ur_robot_driver to receive URScript
URSCRIPT_TOPIC = "/urscript_interface/script_command"


class VG10TestNode(Node):
    def __init__(self):
        super().__init__("vg10_test_node")

        # Publisher to the URScript interface
        self.script_pub = self.create_publisher(StringMsg, URSCRIPT_TOPIC, 10)

        # VG10 helper
        self.gripper = VG10Gripper(self, self.script_pub, VG10Config())

        self.get_logger().info(
            f"[VG10 TEST] Publishing URScript to {URSCRIPT_TOPIC}"
        )


def main():
    rclpy.init()
    node = VG10TestNode()

    try:
        print(
            """
=== VG10 TEST (gripper/vg10_test.py) ===
Make sure:
  - ur_robot_driver is running
  - External Control program is PLAYING on the robot
  - VG10 URCap is enabled in Installation

Commands:
  g  -> grip (A+B, 60%)
  r  -> release (A+B)
  i  -> idle  (A+B)
  q  -> quit
"""
        )
        while rclpy.ok():
            cmd = input("Enter command [g/r/i/q]: ").strip().lower()
            if cmd == "q":
                break
            elif cmd == "g":
                print("[TEST] Gripping...")
                node.gripper.grip()
            elif cmd == "r":
                print("[TEST] Releasing...")
                node.gripper.release()
            elif cmd == "i":
                print("[TEST] Idle...")
                node.gripper.idle()
            else:
                print("Unknown command.")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
