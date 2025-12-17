# UR5/compute_homography.py
import json
from pathlib import Path

import numpy as np
import cv2


HERE = Path(__file__).resolve().parent
UV_TCP_JSON = HERE / "uv_tcp_poses.json"
OUT_JSON = HERE / "homography_uv_to_base_xy.json"


def main():
    data = json.loads(UV_TCP_JSON.read_text())

    pts = data["points"]
    if len(pts) < 4:
        raise RuntimeError(f"Need at least 4 point correspondences, got {len(pts)}")

    # Build correspondences:
    #   src = image pixels (u,v)
    #   dst = robot base plane coordinates (x,y) from tcp_pose
    src_uv = []
    dst_xy = []
    labels = []

    for p in pts:
        u, v = p["uv"]
        x, y, z, rx, ry, rz = p["tcp_pose"]  # UR base frame TCP pose
        src_uv.append([float(u), float(v)])
        dst_xy.append([float(x), float(y)])
        labels.append(p["label"])

    src_uv = np.asarray(src_uv, dtype=np.float32)
    dst_xy = np.asarray(dst_xy, dtype=np.float32)

    # Robust solve using all points (RANSAC).
    # Threshold is in "dst units" (meters), so 0.01 = 1 cm.
    H, inliers = cv2.findHomography(src_uv, dst_xy, method=cv2.RANSAC, ransacReprojThreshold=0.01)
    if H is None:
        raise RuntimeError("cv2.findHomography failed. Check your point pairs.")

    Hinv = np.linalg.inv(H)

    # Quick reprojection error report
    uv_h = cv2.convertPointsToHomogeneous(src_uv).reshape(-1, 3)          # Nx3
    pred_xy_h = (H @ uv_h.T).T                                            # Nx3
    pred_xy = pred_xy_h[:, :2] / pred_xy_h[:, 2:3]                        # Nx2
    err = np.linalg.norm(pred_xy - dst_xy, axis=1)
    rmse = float(np.sqrt(np.mean(err ** 2)))

    result = {
        "robot_model": data.get("robot_model", ""),
        "source": str(UV_TCP_JSON),
        "labels": labels,
        "inliers": inliers.reshape(-1).astype(int).tolist() if inliers is not None else None,
        "H_uv_to_xy": H.tolist(),
        "H_xy_to_uv": Hinv.tolist(),
        "rmse_m": rmse,
        "note": "Maps image pixel (u,v) from the SAME view used during calibration -> robot base (x,y) on table plane."
    }

    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"[OK] Saved homography to: {OUT_JSON}")
    print(f"[OK] RMSE (meters): {rmse:.6f}")

    # Print per-point errors
    for lab, e in zip(labels, err):
        print(f"  {lab:20s} err = {e:.6f} m")


if __name__ == "__main__":
    main()
