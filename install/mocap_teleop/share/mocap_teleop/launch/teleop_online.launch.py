# launch/teleop_mocap.launch.py — ONLINE mode
# ros2 launch mocap_teleop teleop_online.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('sampling_freq', default_value='240.0'),

        Node(
            package='mocap_teleop',
            executable='online_node.py',
            name='online_node',
            parameters=[{'sampling_freq': LaunchConfiguration('sampling_freq')}],
            output='screen',
        ),
        Node(
            package='mocap_teleop',
            executable='teleop_node.py',
            name='teleop',
            parameters=[{'sampling_freq': LaunchConfiguration('sampling_freq')}],
            output='screen',
        ),
    ])