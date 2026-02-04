# ros/publishers.py
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class Publishers:
    """
    역할: ROS publish만 담당 (Image, Float32)
    - 카메라 모름
    - YOLO 모름
    """
    def __init__(self, node: Node, image_topic: str, dist_topic: str):
        self.node = node
        self.bridge = CvBridge()
        self.pub_img = node.create_publisher(Image, image_topic, 10)
        self.pub_dist = node.create_publisher(Float32, dist_topic, 10)

    def publish_image_bgr(self, bgr: np.ndarray) -> None:
        msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        msg.header.stamp = self.node.get_clock().now().to_msg()
        self.pub_img.publish(msg)

    def publish_distance_m(self, dist_m: float) -> None:
        msg = Float32()
        msg.data = float(dist_m)  # NaN도 그대로 들어감
        self.pub_dist.publish(msg)
