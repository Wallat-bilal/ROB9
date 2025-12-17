import json
import numpy as np
import cv2
from pathlib import Path

H_JSON = Path(__file__).resolve().parent / "homography_uv_to_base_xy.json"

_h = json.loads(H_JSON.read_text())
H = np.asarray(_h["H_uv_to_xy"], dtype=np.float64)

def uv_to_base_xy(u: float, v: float):
    pt = np.array([[[u, v]]], dtype=np.float32)     # shape (1,1,2)
    out = cv2.perspectiveTransform(pt, H)           # shape (1,1,2)
    x, y = out[0, 0]
    return float(x), float(y)
