self.gripper = VG10Gripper(self, self.script_pub)

# URScript publisher
self.script_pub = self.create_publisher(StringMsg, URSCRIPT_TOPIC, 10)
self.get_logger().info(f"Will publish URScript to {URSCRIPT_TOPIC}")

# VG10 gripper helper (new!)
self.gripper = VG10Gripper(self, self.script_pub)

cmd = input("Command [g=grip, r=release, q=quit]: ").strip().lower()
if cmd == "g":
    self.gripper.grip()
elif cmd == "r":
    self.gripper.release()
