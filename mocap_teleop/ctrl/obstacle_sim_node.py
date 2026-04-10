#!/usr/bin/env python3

"""
obstacle_sim_node.py — Ground-truth obstacle publisher for MuJoCo simulation.

Publishes static obstacle positions from the MuJoCo scene as a MarkerArray
on /obstacles in the odom (world) frame.  The teleop_node and cbf_safety_filter
consume this topic identically whether it comes from this node or from the
real RealSense camera node.

Obstacle positions and radii are provided as ROS2 parameters so they match
the MuJoCo scene XML without hard-coding values here.

Default parameters match the scene in go2_scene.xml:
  box at (2.0, 0.0), half-size 0.25 m → bounding circle radius 0.354 m

Usage
─────
  ros2 run mocap_teleop obstacle_sim_node

  # Override for a different scene:
  ros2 run mocap_teleop obstacle_sim_node \
      --ros-args \
      -p obstacle_x:=[1.5, -1.0] \
      -p obstacle_y:=[0.0,  1.5] \
      -p obstacle_r:=[0.354, 0.30]

Frame convention
────────────────
  Obstacles are published in the odom/world frame — the same frame used by
  CtrlInterface.get_robot_position() and the OptiTrack mocap origin.
  Ensure the MuJoCo world origin matches the odom frame origin.
"""

import math

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Vector3


# Default: one box at (2, 0) from go2_scene.xml
# Box half-size = 0.25 m → square 0.5×0.5 m → circumscribed circle r = √(0.25²+0.25²)
_DEFAULT_RADIUS = math.sqrt(0.25 ** 2 + 0.25 ** 2)   # ≈ 0.354 m


class ObstacleSimNode(Node):
    def __init__(self):
        super().__init__('obstacle_sim_node')

        # ── Parameters ────────────────────────────────────────────────────────
        # Lists of obstacle x, y centres and radii.  All three must be the
        # same length.  Coordinates are in the odom/world frame (metres).
        self.declare_parameter('obstacle_x', [2.0])
        self.declare_parameter('obstacle_y', [0.0])
        self.declare_parameter('obstacle_r', [_DEFAULT_RADIUS])
        self.declare_parameter('publish_hz',  10.0)
        self.declare_parameter('frame_id',   'odom')

        xs  = self.get_parameter('obstacle_x').get_parameter_value().double_array_value
        ys  = self.get_parameter('obstacle_y').get_parameter_value().double_array_value
        rs  = self.get_parameter('obstacle_r').get_parameter_value().double_array_value
        hz  = self.get_parameter('publish_hz').get_parameter_value().double_value
        self._frame_id = (self.get_parameter('frame_id')
                          .get_parameter_value().string_value)

        if not (len(xs) == len(ys) == len(rs)):
            raise ValueError(
                f'obstacle_x ({len(xs)}), obstacle_y ({len(ys)}), '
                f'obstacle_r ({len(rs)}) must all be the same length')

        self._obstacles = list(zip(xs, ys, rs))
        self.get_logger().info(
            f'Publishing {len(self._obstacles)} obstacle(s) at {hz:.0f} Hz '
            f'on /obstacles  frame={self._frame_id}')
        for x, y, r in self._obstacles:
            self.get_logger().info(f'  center=({x:.3f}, {y:.3f})  radius={r:.3f} m')

        # ── Publisher ─────────────────────────────────────────────────────────
        self._pub = self.create_publisher(MarkerArray, '/obstacles', 10)

        # Pre-build the static message — obstacles never move in this node
        self._msg = self._build_marker_array()

        self.create_timer(1.0 / hz, self._publish)

    def _build_marker_array(self) -> MarkerArray:
        """Build a MarkerArray of SPHERE markers, one per obstacle."""
        msg = MarkerArray()
        for i, (x, y, r) in enumerate(self._obstacles):
            m = Marker()
            m.header.frame_id = self._frame_id
            m.ns              = 'obstacles'
            m.id              = i
            m.type            = Marker.SPHERE
            m.action          = Marker.ADD

            m.pose.position.x    = float(x)
            m.pose.position.y    = float(y)
            m.pose.position.z    = 0.0
            m.pose.orientation.w = 1.0

            # scale.x = diameter so radius = scale.x / 2
            # (matches the convention in teleop_node._obstacles_cb)
            diameter = float(2.0 * r)
            m.scale = Vector3(x=diameter, y=diameter, z=diameter)

            # Semi-transparent red
            m.color = ColorRGBA(r=0.9, g=0.2, b=0.2, a=0.6)

            # Never expire
            m.lifetime.sec     = 0
            m.lifetime.nanosec = 0

            msg.markers.append(m)
        return msg

    def _publish(self) -> None:
        # Update stamp on every publish so rviz2 doesn't age out the markers
        now = self.get_clock().now().to_msg()
        for m in self._msg.markers:
            m.header.stamp = now
        self._pub.publish(self._msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
