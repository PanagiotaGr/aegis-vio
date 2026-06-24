"""ROS2 scaffold node for visual-inertial estimation."""

import rclpy
from rclpy.node import Node


class VIONode(Node):
    def __init__(self):
        super().__init__("aegis_vio_node")
        self.get_logger().info("AegisVIO estimator node started")


def main(args=None):
    rclpy.init(args=args)
    node = VIONode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
