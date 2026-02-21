from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    package_share = get_package_share_directory("misis_rbw_autonomy")
    default_config = os.path.join(package_share, "config", "rbw_strategy.yaml")

    config_arg = DeclareLaunchArgument(
        "config",
        default_value=default_config,
        description="Path to MISIS RBW autonomy parameters",
    )

    return LaunchDescription(
        [
            config_arg,
            Node(
                package="misis_rbw_autonomy",
                executable="autonomy_node",
                name="misis_rbw_autonomy",
                output="screen",
                parameters=[LaunchConfiguration("config")],
            ),
        ]
    )
