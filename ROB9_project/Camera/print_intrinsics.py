import pyrealsense2 as rs

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # Grab one frame to get intrinsics
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    intr = color_frame.get_profile().as_video_stream_profile().get_intrinsics()

    print("Width:", intr.width)
    print("Height:", intr.height)
    print("fx:", intr.fx)
    print("fy:", intr.fy)
    print("cx:", intr.ppx)
    print("cy:", intr.ppy)
    print("coeffs (k1,k2,p1,p2,k3):", intr.coeffs)

    pipeline.stop()

if __name__ == "__main__":
    main()
