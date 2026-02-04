# ros/yolo_infer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import cv2
import onnxruntime as ort


@dataclass
class YoloConfig:
    onnx_path: str
    imgsz: int = 640
    conf_thres: float = 0.25
    iou_thres: float = 0.45


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    cls_id: int
    conf: float

    @property
    def cxcy(self) -> Tuple[int, int]:
        cx = int((self.x1 + self.x2) / 2)
        cy = int((self.y1 + self.y2) / 2)
        return cx, cy


def letterbox(im: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize+pad to new_shape, keep aspect. Returns (img, r, (padx, pady))."""
    h, w = im.shape[:2]
    nh, nw = new_shape
    r = min(nw / w, nh / h)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = nw - new_unpad[0], nh - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w, h) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)


def nms_xyxy(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    """OpenCV NMS. boxes_xyxy: (N,4) xyxy. Returns kept indices."""
    if boxes_xyxy.size == 0:
        return []
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    idxs = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh,
        scores=scores.tolist(),
        score_threshold=0.0,
        nms_threshold=float(iou_thres),
    )
    if len(idxs) == 0:
        return []
    return idxs.flatten().tolist()


def _normalize_ultralytics_output(output0: np.ndarray) -> np.ndarray:
    """
    Ultralytics 계열 ONNX에서 자주 보이는 출력 형태:
      - (1,84,8400) 또는 (1,8400,84)
    -> (N,M)으로 정규화 (예: 8400 x 84)
    """
    pred = np.array(output0)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[0] < pred.shape[1]:
        pred = pred.transpose(1, 0)
    return pred


class YoloOnnxInfer:
    """
    역할: BGR 이미지를 받아 YOLO ONNX로 탐지 결과(대표 1개)를 반환.
    - ROS 모름
    - RealSense 모름
    """
    def __init__(self, cfg: YoloConfig):
        self.cfg = cfg
        self.sess = ort.InferenceSession(cfg.onnx_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name

    def _preprocess(self, bgr: np.ndarray):
        img, r, (padx, pady) = letterbox(bgr, (self.cfg.imgsz, self.cfg.imgsz))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]  # (1,3,H,W)
        return x, r, padx, pady

    def infer_top1(self, bgr: np.ndarray) -> Optional[Detection]:
        H, W = bgr.shape[:2]
        x, r, padx, pady = self._preprocess(bgr)

        outs = self.sess.run(None, {self.in_name: x})
        pred = _normalize_ultralytics_output(outs[0])

        if pred.shape[1] <= 4:
            return None

        # 가정: [x,y,w,h] + class scores...
        boxes_xywh = pred[:, 0:4]
        cls_scores = pred[:, 4:]
        conf = cls_scores.max(axis=1)
        cls_id = cls_scores.argmax(axis=1)

        keep = conf > self.cfg.conf_thres
        boxes_xywh = boxes_xywh[keep]
        conf = conf[keep]
        cls_id = cls_id[keep]
        if boxes_xywh.shape[0] == 0:
            return None

        # xywh -> xyxy (letterbox 좌표계)
        x0, y0, w0, h0 = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = x0 - w0 / 2
        y1 = y0 - h0 / 2
        x2 = x0 + w0 / 2
        y2 = y0 + h0 / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        idxs = nms_xyxy(boxes_xyxy, conf, self.cfg.iou_thres)
        if not idxs:
            return None

        # NMS 후 최고 conf 1개 선택
        idxs = sorted(idxs, key=lambda i: float(conf[i]), reverse=True)
        i = idxs[0]

        # 원본 좌표계로 역변환 (padding 제거 후 r로 나누기)
        ox1 = (boxes_xyxy[i, 0] - padx) / r
        oy1 = (boxes_xyxy[i, 1] - pady) / r
        ox2 = (boxes_xyxy[i, 2] - padx) / r
        oy2 = (boxes_xyxy[i, 3] - pady) / r

        ox1 = float(np.clip(ox1, 0, W - 1))
        oy1 = float(np.clip(oy1, 0, H - 1))
        ox2 = float(np.clip(ox2, 0, W - 1))
        oy2 = float(np.clip(oy2, 0, H - 1))

        return Detection(ox1, oy1, ox2, oy2, int(cls_id[i]), float(conf[i]))
