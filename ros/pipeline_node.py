# ros/pipeline_node.py

# 각 요소들 연결
# camera_reader.py : 카메라 읽기
# yolo_infer.py : yolo 모델 추론
# depth_estimator.py : 깊이 추정
# publishers.py : 토픽 발행

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from .topics import IMAGE_TOPIC, DIST_TOPIC, NODE_NAME
from .camera_reader import RealSenseCameraReader, CameraConfig
from .yolo_infer import YoloOnnxInfer, YoloConfig, Detection
from .depth_estimator import DepthEstimator, DepthConfig
from .publishers import Publishers


@dataclass
class PipelineConfig:
    onnx_path: str
    hz: float = 15.0
    annotated: bool = False

    # camera
    width: int = 640
    height: int = 480
    fps: int = 30

    # yolo
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.45

    # depth
    depth_window: int = 3
    depth_max_m: float = 0.0


class PipelineNode(Node):

    def __init__(self, cfg: PipelineConfig):
        super().__init__(NODE_NAME)

        # 구성 요소 생성
        self.camera = RealSenseCameraReader(CameraConfig(cfg.width, cfg.height, cfg.fps))
        self.camera.start(max_tries=5)

        self.yolo = YoloOnnxInfer(YoloConfig(
            onnx_path=cfg.onnx_path,
            imgsz=cfg.imgsz,
            conf_thres=cfg.conf,
            iou_thres=cfg.iou,
        ))

        self.depth = DepthEstimator(DepthConfig(
            sample_window=cfg.depth_window,
            max_m=cfg.depth_max_m,
        ))

        self.pubs = Publishers(self, IMAGE_TOPIC, DIST_TOPIC)
        self.annotated = bool(cfg.annotated)

        # timer
        self.timer = self.create_timer(1.0 / float(cfg.hz), self.loop)

        self.get_logger().info("PipelineNode started.")

    def loop(self):
        got = self.camera.read(timeout_ms=1000)
        if got is None:
            self.get_logger().warn("No frames.")
            return

        bgr, depth_frame = got


        det = None
        try:
            det = self.yolo.infer_top1(bgr)
        except Exception as e:
            self.get_logger().warn(f"YOLO infer failed: {e}")

        out_img = bgr
        dist_m = float("nan")

        cx, cy = 0.0, 0.0
        if det is not None:
            cx, cy = det.cxcy
            dist_m = self.depth.estimate_m(depth_frame, cx, cy)

            if self.annotated:
                out_img = self._draw(bgr, det, dist_m)

        # publish (토픽 2개 고정)
        self.pubs.publish_distance_m(dist_m, cx, cy)
        self.pubs.publish_image_bgr(out_img)

    def _draw(self, bgr: np.ndarray, det: Detection, dist_m: float) -> np.ndarray:
        img = bgr.copy()
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        cx, cy = det.cxcy

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

        if np.isfinite(dist_m):
            txt = f"cls={det.cls_id} conf={det.conf:.2f} d={dist_m:.3f}m"
        else:
            txt = f"cls={det.cls_id} conf={det.conf:.2f} d=NaN"
        cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img

    def destroy_node(self):
        try:
            self.camera.stop()
        except Exception:
            pass
        super().destroy_node()
