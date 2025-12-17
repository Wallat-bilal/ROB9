from __future__ import annotations

import rclpy
from rclpy.node import Node

from vg_control_interfaces.srv import VacuumSet, VacuumRelease


class VG10GripperClient:
    """
    Simple client for vg10_control VGGripperNode.

    Expects services:
      /grip_adjust       (VacuumSet)
      /release_vacuum    (VacuumRelease)
    """

    def __init__(self, node: Node):
        self._node = node
        self._set_cli = node.create_client(VacuumSet, "grip_adjust")
        self._rel_cli = node.create_client(VacuumRelease, "release_vacuum")

        node.get_logger().info("Waiting for VG10 services ...")
        self._set_cli.wait_for_service()
        self._rel_cli.wait_for_service()
        node.get_logger().info("VG10 services are available.")

    def grip(self, a: int = 150, b: int = 0):
        """Turn vacuum ON. Default: channel A 150, B 0."""
        req = VacuumSet.Request()
        req.channel_a = a
        req.channel_b = b

        future = self._set_cli.call_async(req)
        rclpy.spin_until_future_complete(self._node, future)
        res = future.result()
        if res:
            self._node.get_logger().info(
                f"[VG10] grip -> success={res.success}, msg='{res.message}'"
            )

    def release(self):
        """Turn vacuum OFF on both channels."""
        req = VacuumRelease.Request()
        req.release_vacuum = 1

        future = self._rel_cli.call_async(req)
        rclpy.spin_until_future_complete(self._node, future)
        res = future.result()
        if res:
            self._node.get_logger().info(
                f"[VG10] release -> success={res.success}, msg='{res.message}'"
            )
