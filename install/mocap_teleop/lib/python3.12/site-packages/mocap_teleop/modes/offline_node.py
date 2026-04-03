#!/usr/bin/env python3

"""
csv_replay_node.py — Offline CSV replay source node.

Reads a Motive-exported CSV and publishes HumanState messages at the
recorded sampling rate on the same topics as mocap_driver_node.

The teleop_node is completely unaware of whether it is receiving live
mocap data or a CSV replay — it subscribes to the same topics either way.

Usage
─────
ros2 launch mocap_teleop teleop_offline.launch.py \
    input_file:=./data/Walk_backwards_000.csv
"""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Bool
from mocap_teleop_msgs.msg import HumanState
import mocap_teleop.util.io_parser as io
from mocap_teleop.util.freq_meter import FrequencyMeter


# Body name → CSV column prefix mapping
BODY_COLS = {
    'Root':  'Waist',   # CSV uses 'Waist' for the root/pelvis segment
    'LFoot': 'LFoot',
    'RFoot': 'RFoot',
}


def _row_to_pos_quat(row: pd.Series, prefix: str):
    """Extract position [x,y,z] and quaternion [qx,qy,qz,qw] from a CSV row."""
    pos  = np.array([
        row[f'{prefix}:Position:X'],
        row[f'{prefix}:Position:Y'],
        row[f'{prefix}:Position:Z'],
    ])
    quat = np.array([
        row[f'{prefix}:Rotation:X'],
        row[f'{prefix}:Rotation:Y'],
        row[f'{prefix}:Rotation:Z'],
        row[f'{prefix}:Rotation:W'],
    ])
    return pos, quat


def _make_pose(stamp, frame_id: str,
               pos: np.ndarray, quat: np.ndarray) -> PoseStamped:
    ps                    = PoseStamped()
    ps.header.stamp       = stamp
    ps.header.frame_id    = frame_id
    ps.pose.position.x    = float(pos[0])
    ps.pose.position.y    = float(pos[1])
    ps.pose.position.z    = float(pos[2])
    ps.pose.orientation.x = float(quat[0])
    ps.pose.orientation.y = float(quat[1])
    ps.pose.orientation.z = float(quat[2])
    ps.pose.orientation.w = float(quat[3])
    return ps


def _make_twist(stamp, frame_id: str,
                lin: np.ndarray, ang: np.ndarray) -> TwistStamped:
    ts                 = TwistStamped()
    ts.header.stamp    = stamp
    ts.header.frame_id = frame_id
    ts.twist.linear.x  = float(lin[0])
    ts.twist.linear.y  = float(lin[1])
    ts.twist.linear.z  = float(lin[2])
    ts.twist.angular.x = float(ang[0])
    ts.twist.angular.y = float(ang[1])
    ts.twist.angular.z = float(ang[2])
    return ts


def _body_velocities(curr_pos, prev_pos, curr_quat, prev_quat, dt):
    linear_vel  = (curr_pos - prev_pos) / dt
    rot_curr    = Rotation.from_quat(curr_quat)
    rot_prev    = Rotation.from_quat(prev_quat)
    angular_vel = (rot_curr * rot_prev.inv()).as_rotvec() / dt
    return linear_vel, angular_vel


class OfflineNode(Node):
    def __init__(self):
        super().__init__('csv_replay')

        mocap_cfg = io.parse_mocap_config()

        self.declare_parameter('input_file',   '')
        self.declare_parameter('sampling_freq', float(mocap_cfg['camera_frequency']))

        input_file    = (self.get_parameter('input_file')
                         .get_parameter_value().string_value)
        sampling_freq = (self.get_parameter('sampling_freq')
                         .get_parameter_value().double_value)

        if not input_file:
            self.get_logger().error('input_file parameter is required')
            raise ValueError('input_file not set')

        self._dt  = 1.0 / sampling_freq
        self._df  = pd.read_csv(input_file)
        self._idx = 1   # frame 0 is used as prev; publishing starts at frame 1

        # Cache frame 0 as the initial previous state
        self._prev_data = {
            name: _row_to_pos_quat(self._df.iloc[0], col_prefix)
            for name, col_prefix in BODY_COLS.items()
        }

        self.get_logger().info(
            f'Loaded {len(self._df)} frames from {input_file}')

        # ── Publishers — identical topics to mocap_driver_node ────────────────
        self._human_state_pub = self.create_publisher(
            HumanState, '/mocap/human_state', 10)
        self._tracking_pub = self.create_publisher(
            Bool, '/mocap/tracking_valid', 10)

        # Frequency meter — measures actual replay publish rate
        self._freq_mocap = FrequencyMeter(window=100)

        # Wait for the teleop node to subscribe before starting replay,
        # so the full clip is seen regardless of teleop startup time.
        self._replay_timer = None
        self._wait_timer = self.create_timer(0.1, self._wait_for_subscriber)
        self.get_logger().info('Waiting for teleop node to subscribe...')

        # Log measured frequency at 1 Hz
        self.create_timer(1.0, self._log_freq_cb)

    def _wait_for_subscriber(self) -> None:
        """Poll until the teleop node has subscribed, then start the replay."""
        if self._human_state_pub.get_subscription_count() > 0:
            self._wait_timer.cancel()
            self.get_logger().info('Subscriber detected — starting CSV replay')
            self._replay_timer = self.create_timer(self._dt, self._publish_next_frame)

    def _publish_next_frame(self) -> None:
        if self._idx >= len(self._df):
            self.get_logger().info('CSV replay complete')
            # Raise to exit spin(); the finally block in main() handles cleanup.
            raise SystemExit

        row   = self._df.iloc[self._idx]
        stamp = self.get_clock().now().to_msg()

        msg              = HumanState()
        msg.header.stamp    = stamp
        msg.header.frame_id = 'mocap_world'
        msg.tracking_valid  = True

        for (attr_pose, attr_twist, body_name) in (
            ('root_pose',  'root_twist',  'Root'),
            ('lfoot_pose', 'lfoot_twist', 'LFoot'),
            ('rfoot_pose', 'rfoot_twist', 'RFoot'),
        ):
            col_prefix = BODY_COLS[body_name]
            curr_pos, curr_quat = _row_to_pos_quat(row, col_prefix)
            prev_pos, prev_quat = self._prev_data[body_name]

            lin_vel, ang_vel = _body_velocities(
                curr_pos, prev_pos, curr_quat, prev_quat, self._dt)

            setattr(msg, attr_pose,
                    _make_pose(stamp, 'mocap_world', curr_pos, curr_quat))
            setattr(msg, attr_twist,
                    _make_twist(stamp, 'mocap_world', lin_vel, ang_vel))

            self._prev_data[body_name] = (curr_pos, curr_quat)

        self._human_state_pub.publish(msg)
        self._tracking_pub.publish(Bool(data=True))
        self._freq_mocap.tick()

        self._idx += 1

    def _log_freq_cb(self) -> None:
        self.get_logger().info(
            f'[freq] mocap_publish={self._freq_mocap.hz:.1f} Hz'
        )


def main(args=None):
    rclpy.init(args=args)
    node = OfflineNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()