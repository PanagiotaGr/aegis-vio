"""ROS2 scaffold node for uncertainty-aware navigation."""

import rclpy
from rclpy.node import Node


class NavigationNode(Node):
    def __init__(self):
        super().__init__("aegis_navigation_node")
        self.get_logger().info("AegisVIO navigation node started")


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
