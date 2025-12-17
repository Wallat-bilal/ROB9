#!/usr/bin/env python3
import time

import rclpy
from rclpy.node import Node

from pymodbus.client import ModbusTcpClient

from std_msgs.msg import Int32
from std_srvs.srv import Trigger

from vg_control_interfaces.srv import VacuumSet, VacuumRelease


class VGGripperNode(Node):
    """
    VG10 Modbus TCP node.

    Services (new, easy names):
      - /vg10_connect (std_srvs/Trigger)
      - /vg10_grip    (vg_control_interfaces/VacuumSet)
      - /vg10_release (vg_control_interfaces/VacuumRelease)

    Services (backward compatible with your original file):
      - /connect_modbus (std_srvs/Trigger)
      - /grip_adjust    (vg_control_interfaces/VacuumSet)
      - /release_vacuum (vg_control_interfaces/VacuumRelease)

    Topics (kept from your original file):
      - set_vacuum_a (std_msgs/Int32)
      - set_vacuum_b (std_msgs/Int32)
      - vacuum_level_a (std_msgs/Int32)
      - vacuum_level_b (std_msgs/Int32)
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

        # --- Services (NEW names) ---
        self.srv_vg10_connect = self.create_service(Trigger, "/vg10_connect", self.connect_srv_cb)
        self.srv_vg10_grip = self.create_service(VacuumSet, "/vg10_grip", self.grip_adjust_callback)
        self.srv_vg10_release = self.create_service(VacuumRelease, "/vg10_release", self.release_callback)

        # --- Services (OLD names, kept for compatibility) ---
        self.srv_connect_modbus = self.create_service(Trigger, "/connect_modbus", self.connect_srv_cb)
        self.srv_grip_adjust = self.create_service(VacuumSet, "/grip_adjust", self.grip_adjust_callback)
        self.srv_release = self.create_service(VacuumRelease, "/release_vacuum", self.release_callback)

        # Subscribers (kept)
        self.sub_channel_a = self.create_subscription(
            Int32,
            "set_vacuum_a",
            lambda msg: self.set_vacuum_callback(msg, channel=0),
            10,
        )
        self.sub_channel_b = self.create_subscription(
            Int32,
            "set_vacuum_b",
            lambda msg: self.set_vacuum_callback(msg, channel=1),
            10,
        )

        # Publishers (kept)
        self.pub_vacuum_a = self.create_publisher(Int32, "vacuum_level_a", 20)
        self.pub_vacuum_b = self.create_publisher(Int32, "vacuum_level_b", 20)

        # Timer for reading vacuum levels periodically (kept)
        self.create_timer(0.05, self.read_vacuum_levels)

        # Try connect once at startup
        ok = self.connect_modbus()
        self.get_logger().info(
            f"VG10 node started. Modbus {'CONNECTED' if ok else 'NOT connected'} to {self.ip}:{self.port}, slave={self.changer_addr}"
        )

    # -------------------------
    # Modbus connection helpers
    # -------------------------
    def connect_modbus(self) -> bool:
        """Connect to the Modbus server (tool changer)."""
        try:
            if self.client is not None:
                try:
                    self.client.close()
                except Exception:
                    pass

            self.client = ModbusTcpClient(self.ip, port=self.port, timeout=1)
            success = bool(self.client.connect())
            if success:
                self.get_logger().info(f"Connected to VG10 Modbus at {self.ip}:{self.port}")
            else:
                self.get_logger().error(f"Failed to connect to VG10 Modbus at {self.ip}:{self.port}")
            return success
        except Exception as e:
            self.get_logger().error(f"Connection error: {e}")
            return False

    def ensure_connected(self) -> bool:
        """Ensure we have a client and it is connected. Try reconnect if needed."""
        if self.client is None:
            return self.connect_modbus()
        try:
            # pymodbus connect() is idempotent; returns True if connected
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

    def grip_adjust_callback(self, request, response):
        """Grip with both channels at specified vacuum levels."""
        try:
            if not self.ensure_connected():
                response.success = False
                response.message = "Not connected (call /vg10_connect or /connect_modbus)"
                return response

            vacuum_a = max(0, min(255, int(request.channel_a)))
            vacuum_b = max(0, min(255, int(request.channel_b)))

            # Mode: grip (0x0100) + Vacuum level
            command = [0x0100 + vacuum_a, 0x0100 + vacuum_b]
            self.client.write_registers(address=0, values=command, slave=self.changer_addr)

            self.get_logger().info(f"Gripping with channels A:{vacuum_a}, B:{vacuum_b}")
            response.success = True
            response.message = f"Gripping with vacuum levels A:{vacuum_a}, B:{vacuum_b}"
            return response

        except Exception as e:
            self.get_logger().error(f"Grip error: {e}")
            response.success = False
            response.message = str(e)
            return response

    def release_callback(self, request, response):
        """Release vacuum on both channels."""
        try:
            if not self.ensure_connected():
                response.success = False
                response.message = "Not connected (call /vg10_connect or /connect_modbus)"
                return response

            if int(request.release_vacuum) != 1:
                response.success = False
                response.message = "Invalid release_vacuum value. Use 1 to release."
                return response

            command = [0x0000, 0x0000]
            self.client.write_registers(address=0, values=command, slave=self.changer_addr)

            self.get_logger().info("Released vacuum on both channels")
            response.success = True
            response.message = "Vacuum released successfully"
            return response

        except Exception as e:
            self.get_logger().error(f"Release error: {e}")
            response.success = False
            response.message = str(e)
            return response

    # -------------------------
    # Topic callbacks (kept)
    # -------------------------
    def set_vacuum_callback(self, msg, channel: int):
        """Set vacuum level for a specific channel via topic."""
        try:
            if not self.ensure_connected():
                self.get_logger().warn("Not connected; ignoring set_vacuum topic command")
                return

            vacuum_level = max(0, min(255, int(msg.data)))
            # Read current "levels" by using last commanded values is tricky;
            # for simplicity, set both with one changed and one kept at 0 if unknown.
            # Better: use the service calls instead of topics.
            if channel == 0:
                command = [0x0100 + vacuum_level, 0x0100 + 0]
            else:
                command = [0x0100 + 0, 0x0100 + vacuum_level]

            self.client.write_registers(address=0, values=command, slave=self.changer_addr)
        except Exception as e:
            self.get_logger().error(f"Set vacuum topic error: {e}")

    def read_vacuum_levels(self):
        """Read vacuum levels (best-effort)."""
        try:
            if not self.ensure_connected():
                return

            # Some tool-changers expose feedback registers; if yours doesn't,
            # this may fail silently. Keep it best-effort.
            result = self.client.read_holding_registers(address=0, count=2, slave=self.changer_addr)
            if not result or result.isError():
                return

            a = Int32()
            b = Int32()
            a.data = int(result.registers[0])
            b.data = int(result.registers[1])
            self.pub_vacuum_a.publish(a)
            self.pub_vacuum_b.publish(b)

        except Exception:
            pass

    def destroy_node(self):
        # Attempt a safe release on shutdown
        try:
            if self.client is not None:
                try:
                    self.client.write_registers(address=0, values=[0x0000, 0x0000], slave=self.changer_addr)
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
