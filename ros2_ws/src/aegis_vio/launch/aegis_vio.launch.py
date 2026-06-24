from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package="aegis_vio", executable="vio_node", name="aegis_vio_node"),
        Node(package="aegis_vio", executable="uncertainty_node", name="aegis_uncertainty_node"),
        Node(package="aegis_vio", executable="navigation_node", name="aegis_navigation_node"),
    ])
