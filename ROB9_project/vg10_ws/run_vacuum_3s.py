#!/usr/bin/env python3
import time

import rclpy
from rclpy.node import Node

from vg_control_interfaces.srv import VacuumSet, VacuumRelease


class Vacuum3s(Node):
    def __init__(self):
        super().__init__("vacuum_3s")
        self.grip = self.create_client(VacuumSet, "/vg10_grip")
        self.release = self.create_client(VacuumRelease, "/vg10_release")

    def run(self, a=60, b=60, seconds=3.0):
        if not self.grip.wait_for_service(timeout_sec=2.0):
            raise RuntimeError("Service /vg10_grip not available")
        if not self.release.wait_for_service(timeout_sec=2.0):
            raise RuntimeError("Service /vg10_release not available")

        g = VacuumSet.Request()
        g.channel_a = int(a)
        g.channel_b = int(b)

        fut = self.grip.call_async(g)
        rclpy.spin_until_future_complete(self, fut)
        resp = fut.result()
        self.get_logger().info(f"Grip: {resp.success} - {resp.message}")

        time.sleep(float(seconds))

        r = VacuumRelease.Request()
        r.release_vacuum = 1
        fut2 = self.release.call_async(r)
        rclpy.spin_until_future_complete(self, fut2)
        resp2 = fut2.result()
        self.get_logger().info(f"Release: {resp2.success} - {resp2.message}")


def main():
    rclpy.init()
    node = Vacuum3s()
    try:
        node.run(a=20, b=20, seconds=1.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
