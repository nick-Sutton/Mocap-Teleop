# launch/teleop_offline.launch.py — OFFLINE CSV replay mode
# ros2 launch mocap_teleop teleop_offline.launch.py \
#     input_file:=./data/Walk_backwards_000.csv

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('input_file',    default_value=''),
        DeclareLaunchArgument('sampling_freq', default_value='240.0'),

        # Replays CSV on same topics as mocap_driver_node
        Node(
            package='mocap_teleop',
            executable='offline_node.py',
            name='offline_node',
            parameters=[{
                'input_file':    LaunchConfiguration('input_file'),
                'sampling_freq': LaunchConfiguration('sampling_freq'),
            }],
            output='screen',
        ),

        # Teleop node unchanged — subscribes to same topics as online mode
        Node(
            package='mocap_teleop',
            executable='teleop_node.py',
            name='teleop',
            parameters=[{'sampling_freq': LaunchConfiguration('sampling_freq')}],
            output='screen',
        ),
    ])