#!/usr/bin/env python3
import argparse
import rclpy

from ros.pipeline_node import PipelineNode, PipelineConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True, help="Path to yolo11n.onnx")

    p.add_argument("--hz", type=float, default=15.0)
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
