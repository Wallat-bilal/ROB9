#!/usr/bin/env python3
from __future__ import annotations

import inspect

import rclpy
from rclpy.node import Node

from pymodbus.client import ModbusTcpClient

from std_msgs.msg import Int32
from std_srvs.srv import Trigger

from vg_control_interfaces.srv import VacuumSet, VacuumRelease


class VGGripperNode(Node):
    """
    VG10 Modbus TCP node.

    Services:
      - /vg10_connect (std_srvs/Trigger)
      - /vg10_grip    (vg_control_interfaces/VacuumSet)
      - /vg10_release (vg_control_interfaces/VacuumRelease)

    Backward compatible service names:
      - /connect_modbus
      - /grip_adjust
      - /release_vacuum
    """

    def __init__(self):
        super().__init__("vg10_gripper_node")

        # Parameters
        self.declare_parameter("ip", "192.168.1.1")
        self.declare_parameter("port", 502)
        self.declare_parameter("changer_addr", 65)

        self.ip = str(self.get_parameter("ip").value)
        self.port = int(self.get_parameter("port").value)
        self.changer_addr = int(self.get_parameter("changer_addr").value)

        self.client: ModbusTcpClient | None = None

        # Last commanded vacuum levels (0-255)
        self._last_a: int = 0
        self._last_b: int = 0

        # Services (new names)
        self.create_service(Trigger, "/vg10_connect", self.connect_srv_cb)
        self.create_service(VacuumSet, "/vg10_grip", self.grip_callback)
        self.create_service(VacuumRelease, "/vg10_release", self.release_callback)

        # Services (old names kept)
        self.create_service(Trigger, "/connect_modbus", self.connect_srv_cb)
        self.create_service(VacuumSet, "/grip_adjust", self.grip_callback)
        self.create_service(VacuumRelease, "/release_vacuum", self.release_callback)

        # Optional topics (kept)
        self.create_subscription(
            Int32, "set_vacuum_a", lambda msg: self.set_vacuum_topic_cb(msg, channel=0), 10
        )
        self.create_subscription(
            Int32, "set_vacuum_b", lambda msg: self.set_vacuum_topic_cb(msg, channel=1), 10
        )

        self.pub_vacuum_a = self.create_publisher(Int32, "vacuum_level_a", 20)
        self.pub_vacuum_b = self.create_publisher(Int32, "vacuum_level_b", 20)

        self.create_timer(0.1, self.read_vacuum_levels)

        ok = self.connect_modbus()
        self.get_logger().info(
            f"VG10 node started. Modbus {'CONNECTED' if ok else 'NOT connected'} "
            f"to {self.ip}:{self.port}, addr={self.changer_addr}"
        )

    # -------------------------
    # pymodbus API compatibility
    # -------------------------
    def _device_kw(self, method) -> dict:
        """
        Return the correct kwarg for addressing the device, depending on pymodbus version.

        pymodbus 4.x: device_id=
        pymodbus 3.x: slave=
        pymodbus 2.x: unit=
        """
        params = inspect.signature(method).parameters
        if "device_id" in params:
            return {"device_id": self.changer_addr}
        if "slave" in params:
            return {"slave": self.changer_addr}
        if "unit" in params:
            return {"unit": self.changer_addr}
        if "slave_id" in params:
            return {"slave_id": self.changer_addr}
        return {}

    def _write_registers(self, address: int, values):
        if self.client is None:
            raise RuntimeError("Modbus client not initialized")
        kw = self._device_kw(self.client.write_registers)
        if kw:
            return self.client.write_registers(address=address, values=values, **kw)
        # Last resort: try positional addressing (very old/odd variants)
        return self.client.write_registers(address, values, self.changer_addr)

    def _read_holding_registers(self, address: int, count: int):
        if self.client is None:
            raise RuntimeError("Modbus client not initialized")
        kw = self._device_kw(self.client.read_holding_registers)
        if kw:
            return self.client.read_holding_registers(address=address, count=count, **kw)
        return self.client.read_holding_registers(address, count, self.changer_addr)

    # -------------------------
    # Connection helpers
    # -------------------------
    def connect_modbus(self) -> bool:
        try:
            if self.client is not None:
                try:
                    self.client.close()
                except Exception:
                    pass

            self.client = ModbusTcpClient(self.ip, port=self.port, timeout=1)
            ok = bool(self.client.connect())
            if ok:
                self.get_logger().info(f"Connected to VG10 Modbus at {self.ip}:{self.port}")
            else:
                self.get_logger().error(f"Failed to connect to VG10 Modbus at {self.ip}:{self.port}")
            return ok
        except Exception as e:
            self.get_logger().error(f"Connection error: {e}")
            return False

    def ensure_connected(self) -> bool:
        if self.client is None:
            return self.connect_modbus()
        try:
            return bool(self.client.connect())
        except Exception:
            return self.connect_modbus()

    # -------------------------
    # Service callbacks
    # -------------------------
    def connect_srv_cb(self, request, response):
        ok = self.connect_modbus()
        response.success = bool(ok)
        response.message = "Connected" if ok else "Failed to connect"
        return response

    def grip_callback(self, request: VacuumSet.Request, response: VacuumSet.Response):
        try:
            if not self.ensure_connected():
                response.success = False
                response.message = "Not connected (call /vg10_connect)"
                return response

            vacuum_a = max(0, min(255, int(request.channel_a)))
            vacuum_b = max(0, min(255, int(request.channel_b)))

            # 0x0100 = grip mode, add level (your ROB10-style approach)
            command = [0x0100 + vacuum_a, 0x0100 + vacuum_b]
            self._write_registers(address=0, values=command)

            self._last_a, self._last_b = vacuum_a, vacuum_b

            response.success = True
            response.message = f"Gripping A={vacuum_a}, B={vacuum_b}"
            return response

        except Exception as e:
            self.get_logger().error(f"Grip error: {e}")
            response.success = False
            response.message = str(e)
            return response

    def release_callback(self, request: VacuumRelease.Request, response: VacuumRelease.Response):
        try:
            if not self.ensure_connected():
                response.success = False
                response.message = "Not connected (call /vg10_connect)"
                return response

            if int(request.release_vacuum) != 1:
                response.success = False
                response.message = "Use release_vacuum: 1"
                return response

            self._write_registers(address=0, values=[0x0000, 0x0000])
            self._last_a, self._last_b = 0, 0

            response.success = True
            response.message = "Vacuum released"
            return response

        except Exception as e:
            self.get_logger().error(f"Release error: {e}")
            response.success = False
            response.message = str(e)
            return response

    # -------------------------
    # Topic callbacks (optional)
    # -------------------------
    def set_vacuum_topic_cb(self, msg: Int32, channel: int):
        try:
            if not self.ensure_connected():
                return
            val = max(0, min(255, int(msg.data)))
            if channel == 0:
                self._last_a = val
            else:
                self._last_b = val
            command = [0x0100 + self._last_a, 0x0100 + self._last_b]
            self._write_registers(address=0, values=command)
        except Exception as e:
            self.get_logger().error(f"Topic set error: {e}")

    def read_vacuum_levels(self):
        # Best-effort read (some devices don’t provide meaningful feedback here)
        try:
            if not self.ensure_connected():
                return
            rr = self._read_holding_registers(address=0, count=2)
            if not rr or getattr(rr, "isError", lambda: True)():
                return
            a = Int32()
            b = Int32()
            a.data = int(rr.registers[0])
            b.data = int(rr.registers[1])
            self.pub_vacuum_a.publish(a)
            self.pub_vacuum_b.publish(b)
        except Exception:
            pass

    def destroy_node(self):
        try:
            if self.client is not None:
                try:
                    self._write_registers(address=0, values=[0x0000, 0x0000])
                except Exception:
                    pass
                try:
                    self.client.close()
                except Exception:
                    pass
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VGGripperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
