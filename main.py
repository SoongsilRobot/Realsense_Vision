#!/usr/bin/env python3

# yolo 모델 어떤 거 사용할지 설정
# 화면 주사율, 프레임 크기 등 설정
# 흐름 담당은 main.py가 아닌 pipeline_node.py
# main.py는 기본적인 설정만 담당 후 pipeline_node.py에 인자 전달 및 노드 실행
# 실행 시: python main.py --onnx [모델경로] --hz 0.2

import argparse
import rclpy

from ros.pipeline_node import PipelineNode, PipelineConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True, help="Path to yolo26n.onnx")

    p.add_argument("--hz", type=float, default=15.0) # 0.2로 바꾸면 5초에 한번
    p.add_argument("--annotated", action="store_true")

    # camera
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)

    # yolo
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)

    # depth
    p.add_argument("--depth-window", type=int, default=3)
    p.add_argument("--depth-max-m", type=float, default=0.0)

    return p.parse_args()


def main():
    args = parse_args()

    cfg = PipelineConfig(
        onnx_path=args.onnx,
        hz=args.hz,
        annotated=args.annotated,

        width=args.width,
        height=args.height,
        fps=args.fps,

        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,

        depth_window=args.depth_window,
        depth_max_m=args.depth_max_m,
    )

    rclpy.init()
    node = PipelineNode(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
