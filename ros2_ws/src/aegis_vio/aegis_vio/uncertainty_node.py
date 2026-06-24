"""ROS2 scaffold node for uncertainty monitoring."""

import rclpy
from rclpy.node import Node


class UncertaintyNode(Node):
    def __init__(self):
        super().__init__("aegis_uncertainty_node")
        self.get_logger().info("AegisVIO uncertainty node started")


def main(args=None):
    rclpy.init(args=args)
    node = UncertaintyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
