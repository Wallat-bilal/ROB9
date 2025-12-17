#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String as StringMsg

# Your custom services from vg_control_interfaces
from vg_control_interfaces.srv import VacuumSet, VacuumRelease


URSCRIPT_TOPIC = "/urscript_interface/script_command"


class VG10UrScriptNode(Node):
    def __init__(self):
        super().__init__("vg10_urscript_node")

        # Publisher to URScript interface of ur_robot_driver
        self.script_pub = self.create_publisher(StringMsg, URSCRIPT_TOPIC, 10)

        # Services
        self.srv_grip = self.create_service(
            VacuumSet, "vg10_grip", self.handle_grip
        )
        self.srv_release = self.create_service(
            VacuumRelease, "vg10_release", self.handle_release
        )

        self.get_logger().info(
            f"[VG10] URScript node started. Publishing to {URSCRIPT_TOPIC}"
        )
        self.get_logger().info(
            "Make sure:\n"
            "  - ur_robot_driver is running\n"
            "  - External Control program is PLAYING on the robot\n"
            "  - VG10 URCap is enabled in Installation"
        )

    # ------------- helpers -------------

    def _send_script(self, script: str):
        msg = StringMsg()
        msg.data = script
        self.script_pub.publish(msg)
        self.get_logger().info(f"[VG10] Sent URScript:\n{script.strip()}")

    # ------------- service callbacks -------------

    def handle_grip(self, request: VacuumSet, response: VacuumSet.Response):
        """
        Suck: turn VG10 vacuum ON based on requested levels.

        We'll grip with A+B and use the maximum of channel_a and channel_b
        as the vacuum level, clamped to [0, 80].
        """
        try:
            # Simple policy: use both channels (2) and a single vacuum level
            vac_a = max(0, min(80, int(request.channel_a)))
            vac_b = max(0, min(80, int(request.channel_b)))
            vac = max(vac_a, vac_b)

            channel = 2  # 0=A, 1=B, 2=A+B
            timeout = 2.0
            alert = True

            alert_str = "True" if alert else "False"

            script = f"""
def vg10_prog():
  textmsg("VG10 GRIP via service, ch={channel}, vac={vac}")
  VG10_grip({channel}, {vac}, {timeout}, {alert_str})
end
vg10_prog()
"""
            self._send_script(script)

            response.success = True
            response.message = f"Gripping with A+B at {vac}% vacuum"
        except Exception as e:
            self.get_logger().error(f"handle_grip error: {e}")
            response.success = False
            response.message = str(e)

        return response

    def handle_release(
        self, request: VacuumRelease, response: VacuumRelease.Response
    ):
        """
        Release: turn VG10 vacuum OFF.

        If request.release_vacuum == 1, we release on A+B.
        """
        try:
            if int(request.release_vacuum) != 1:
                response.success = False
                response.message = "release_vacuum must be 1 to release."
                return response

            channel = 2  # A+B
            timeout = 2.0
            autoidle = True
            autoidle_str = "True" if autoidle else "False"

            script = f"""
def vg10_prog():
  textmsg("VG10 RELEASE via service, ch={channel}")
  VG10_release({channel}, {timeout}, {autoidle_str})
end
vg10_prog()
"""
            self._send_script(script)

            response.success = True
            response.message = "Released vacuum on A+B"
        except Exception as e:
            self.get_logger().error(f"handle_release error: {e}")
            response.success = False
            response.message = str(e)

        return response


def main(args=None):
    rclpy.init(args=args)
    node = VG10UrScriptNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
