# Camera/camera.py

import pyrealsense2 as rs
import numpy as np
import cv2


def preview_rgb_and_ir():
    # Create pipeline + config
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable color (RGB) stream
    config.enable_stream(
        rs.stream.color,    # RGB sensor
        640, 480,           # resolution
        rs.format.bgr8,     # OpenCV expects BGR
        30                  # fps
    )

    # Enable one of the infrared imagers (index 1 or 2)
    config.enable_stream(
        rs.stream.infrared, # IR / NIR sensor
        1,                  # camera index: 1 = left, 2 = right
        640, 480,
        rs.format.y8,       # 8-bit grayscale
        30
    )

    # Start streaming
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()

            # Get color + infrared frames
            color_frame = frames.get_color_frame()
            ir_frame = frames.get_infrared_frame(1)  # same index as above

            if not color_frame or not ir_frame:
                continue

            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            ir_image = np.asanyarray(ir_frame.get_data())

            # Show windows
            cv2.imshow("D435 RGB", color_image)
            cv2.imshow("D435 IR (NIR)", ir_image)

            # Press q or ESC to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    preview_rgb_and_ir()
