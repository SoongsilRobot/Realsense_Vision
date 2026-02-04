# ros/depth_estimator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pyrealsense2 as rs


@dataclass
class DepthConfig:
    sample_window: int = 3     # NxN median (odd 권장). 1이면 단일 픽셀
    max_m: float = 0.0         # >0이면 이보다 큰 값은 NaN 처리(선택)


class DepthEstimator:
    """
    역할: depth_frame + (cx,cy) -> distance(m)
    - ROS 모름
    - YOLO 모름
    """
    def __init__(self, cfg: DepthConfig):
        self.cfg = cfg

    def estimate_m(self, depth_frame: rs.depth_frame, cx: int, cy: int) -> float:
        n = int(self.cfg.sample_window)
        if n <= 1:
            d = float(depth_frame.get_distance(cx, cy))
            return self._sanitize(d)

        half = n // 2
        vals = []
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                x = cx + dx
                y = cy + dy
                if x < 0 or y < 0:
                    continue
                try:
                    d = float(depth_frame.get_distance(int(x), int(y)))
                except Exception:
                    continue
                if d > 0:
                    vals.append(d)

        if not vals:
            return float("nan")

        d_med = float(np.median(np.array(vals, dtype=np.float32)))
        return self._sanitize(d_med)

    def _sanitize(self, d: float) -> float:
        if not np.isfinite(d) or d <= 0:
            return float("nan")
        if self.cfg.max_m and self.cfg.max_m > 0 and d > float(self.cfg.max_m):
            return float("nan")
        return float(d)
