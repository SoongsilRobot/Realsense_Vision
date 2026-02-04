# ros/camera_reader.py
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pyrealsense2 as rs


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fps: int = 30


class RealSenseCameraReader:
    """
    역할: RealSense D405에서 (color_bgr, depth_frame) 읽기만 한다.
    - ROS publish 하지 않음
    - YOLO 모름
    """
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        self.profile = None

    def start(self, max_tries: int = 5, sleep_s: float = 0.8) -> None:
        config = rs.config()
        config.enable_stream(rs.stream.color, self.cfg.width, self.cfg.height, rs.format.bgr8, self.cfg.fps)
        config.enable_stream(rs.stream.depth, self.cfg.width, self.cfg.height, rs.format.z16, self.cfg.fps)

        last_err: Optional[Exception] = None
        for i in range(1, max_tries + 1):
            try:
                self.profile = self.pipeline.start(config)
                time.sleep(0.2)
                return
            except Exception as e:
                last_err = e
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
                time.sleep(sleep_s)

        raise RuntimeError(f"RealSense start failed after {max_tries} tries. Last error: {last_err}")

    def stop(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def read(self, timeout_ms: int = 1000) -> Optional[Tuple[np.ndarray, rs.depth_frame]]:
        """
        Returns:
          - color_bgr (H,W,3) uint8
          - depth_frame (realsense depth_frame aligned to color)
        """
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
            frames = self.align.process(frames)
            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            if not depth or not color:
                return None
            bgr = np.asanyarray(color.get_data())
            return bgr, depth
        except Exception:
            return None
