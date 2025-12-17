import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/wallat/Desktop/ROB9/ROB9_project/vg10_ws/install/vg10_control'
