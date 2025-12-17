from __future__ import annotations

from typing import Optional

from rclpy.node import Node
from rclpy.task import Future
from std_srvs.srv import Trigger

from vg_control_interfaces.srv import VacuumSet


class VG10Gripper:
    """Small helper to call the VG10 services from another ROS2 Node."""

    def __init__(
        self,
        node: Node,
        connect_srv: str = "/vg10_connect",
        grip_srv: str = "/vg10_grip",
        release_srv: str = "/vg10_release",
    ):
        self.node = node
        self._connect_cli = node.create_client(Trigger, connect_srv)
        self._grip_cli = node.create_client(VacuumSet, grip_srv)
        self._release_cli = node.create_client(VacuumSet, release_srv)

    def _wait(self, timeout_sec: float = 2.0) -> bool:
        ok = True
        ok &= self._connect_cli.wait_for_service(timeout_sec=timeout_sec)
        ok &= self._grip_cli.wait_for_service(timeout_sec=timeout_sec)
        ok &= self._release_cli.wait_for_service(timeout_sec=timeout_sec)
        return ok

    def connect(self, timeout_sec: float = 2.0) -> bool:
        if not self._wait(timeout_sec=timeout_sec):
            self.node.get_logger().error("VG10 services not available.")
            return False
        fut: Future = self._connect_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self.node, fut, timeout_sec=timeout_sec)
        resp = fut.result()
        return bool(resp and resp.success)

    def vacuum(self, a: int = 60, b: int = 60, timeout_sec: float = 2.0) -> bool:
        if not self._wait(timeout_sec=timeout_sec):
            self.node.get_logger().error("VG10 services not available.")
            return False
        req = VacuumSet.Request()
        req.channel_a = int(a)
        req.channel_b = int(b)
        fut: Future = self._grip_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, fut, timeout_sec=timeout_sec)
        resp = fut.result()
        return bool(resp and resp.success)

    def release(self, timeout_sec: float = 2.0) -> bool:
        if not self._wait(timeout_sec=timeout_sec):
            self.node.get_logger().error("VG10 services not available.")
            return False
        req = VacuumSet.Request()
        req.channel_a = 0
        req.channel_b = 0
        fut: Future = self._release_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, fut, timeout_sec=timeout_sec)
        resp = fut.result()
        return bool(resp and resp.success)
