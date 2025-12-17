To run main.py

terminal A

source /opt/ros/humble/setup.bash
source ~/ur_ws/install/setup.bash

ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur10 \
  robot_ip:=192.168.1.10 \
  launch_rviz:=false


terminal B

cd ~/Desktop/ROB9/ROB9_project/vg10_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run vg10_control vg10_gripper_node





terminal c



source /opt/ros/humble/setup.bash
source ~/ur_ws/install/setup.bash
source ~/Desktop/ROB9/ROB9_project/vg10_ws/install/setup.bash

source ~/Desktop/ROB9/.venv1/bin/activate
cd ~/Desktop/ROB9/ROB9_project
python3 main.py







Before running the main code make sure to have ros2 service and list of gripper present



