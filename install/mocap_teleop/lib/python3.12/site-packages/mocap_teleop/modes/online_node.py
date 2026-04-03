#!/usr/bin/env python3

"""
mocap_driver_node.py — NatNet → ROS2 bridge.

This node is the ONLY place where:
  1. OptiTrack Y-up coordinates are transformed to Z-up (ROS convention)
  2. Raw UDP NatNet data is converted to standard ROS2 message types

All downstream nodes receive data already in the mocap_world (Z-up) frame.
No coordinate transforms should occur anywhere else in the pipeline.

Published topics
────────────────
/mocap/human_state  cross_tele/HumanState   Full body state at mocap rate
/mocap/tracking_status  std_msgs/Bool        True when all 3 bodies tracked

Frame IDs
─────────
mocap_world : fixed world frame, Z-up, origin at first valid frame
"""
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from std_msgs.msg import Bool, Header
import mocap_teleop.util.io_parser as io

from natnet import DataDescriptions, DataFrame, NatNetClient
from mocap_teleop_msgs.msg import HumanState


# ── Coordinate transform helpers ──────────────────────────────────────────────
# OptiTrack/Motive: Y-up, Z-forward, X-right
# ROS convention:   Z-up, X-forward, Y-left
#
# Position:  x_ros = z_opti,  y_ros = -x_opti,  z_ros = y_opti
# Quaternion components transform consistently with the position axes.

def optitrack_pos_to_ros(x: float, y: float, z: float):
    return z, -x, y


def optitrack_quat_to_ros(rx: float, ry: float, rz: float, rw: float):
    """
    Rotate the quaternion to match the position axis swap.
    When swapping axes [x,y,z] → [z,-x,y] the quaternion components
    transform as [qx,qy,qz] → [qz,-qx,qy].
    """
    return rz, -rx, ry, rw


def compute_body_velocities(
    curr_pos: np.ndarray,
    prev_pos: np.ndarray,
    curr_quat: np.ndarray,
    prev_quat: np.ndarray,
    dt: float,
):
    """
    Compute linear and angular velocities by finite difference.
    curr/prev_quat are [qx, qy, qz, qw].
    Returns (linear_vel [3], angular_vel [3]) both in mocap_world frame.
    """
    linear_vel = (curr_pos - prev_pos) / dt

    rot_curr = Rotation.from_quat(curr_quat)
    rot_prev = Rotation.from_quat(prev_quat)
    rot_rel  = rot_curr * rot_prev.inv()
    angular_vel = rot_rel.as_rotvec() / dt

    return linear_vel, angular_vel


# ── NatNet tracker ────────────────────────────────────────────────────────────

class NatNetTracker:
    def __init__(self):
        # Stores (pos_ros, quat_ros) per body name — already transformed
        self.curr: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.prev: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.timestep:   float = 0.0
        self.prev_timestep: float = -1.0
        self.frame_idx:  int   = 0
        self.new_frame:  bool  = False

        # Get body ids
        mocap_cfg = io.parse_mocap_config()
        self.body_ids = {mocap_cfg['root_stream_id']: mocap_cfg['root_name'], 
                         mocap_cfg['left_foot_stream_id']: mocap_cfg['left_foot_name'], 
                         mocap_cfg['right_foot_stream_id']: mocap_cfg['right_foot_name']}

    def update_frame(self, data_frame: DataFrame) -> None:
        new_ts = data_frame.suffix.timestamp
        if new_ts == self.timestep:
            return

        self.prev_timestep = self.timestep
        self.timestep      = new_ts
        self.frame_idx    += 1
        self.prev          = dict(self.curr)

        for rb in data_frame.rigid_bodies:
            name = self.body_ids.get(rb.id_num)
            if name is None:
                continue
            x_ros, y_ros, z_ros = optitrack_pos_to_ros(*rb.pos)
            qx, qy, qz, qw     = optitrack_quat_to_ros(*rb.rot)
            self.curr[name] = (
                np.array([x_ros, y_ros, z_ros]),
                np.array([qx, qy, qz, qw]),
            )
        self.new_frame = True

    def ready(self) -> bool:
        return (len(self.curr) == 3
                and len(self.prev) == 3
                and self.prev_timestep > 0)


# ── ROS2 node ─────────────────────────────────────────────────────────────────

class NatnetNode(Node):
    def __init__(self):
        super().__init__('mocap_driver')

        self._tracker        = NatNetTracker()
        self._streaming_client = self._start_natnet()

        # Publishers
        self._human_state_pub = self.create_publisher(
            HumanState, '/mocap/human_state', 10)
        self._tracking_pub = self.create_publisher(
            Bool, '/mocap/tracking_valid', 10)

        # Poll NatNet at 500 Hz — mocap data arrives at ~240 Hz
        # The tracker deduplicates frames internally
        self.create_timer(0.002, self._poll_and_publish)

        self.get_logger().info('MocapDriverNode started')

    # ── NatNet setup ──────────────────────────────────────────────────────────

    def _start_natnet(self) -> NatNetClient:
        cfg = io.parse_network_config()

        client = NatNetClient(
            server_ip_address=cfg['server_address'],
            local_ip_address=cfg['local_address'],
            multicast_address=cfg['multicast_address'],
            command_port=cfg['command_port'],
            data_port=cfg['data_port'],
            use_multicast=cfg['use_multicast'],
        )
        client.on_data_frame_received_event.handlers.append(
            self._tracker.update_frame)
        client.on_data_description_received_event.handlers.append(
            lambda _: self.get_logger().info('NatNet descriptions received'))

        self.get_logger().info(
            f"Connecting to NatNet server at {cfg['server_address']}")
        return client

    # ── Timer callback ────────────────────────────────────────────────────────

    def _poll_and_publish(self) -> None:
        try:
            self._streaming_client.update_sync()
        except BlockingIOError:
            pass

        tracking_valid = self._tracker.ready()

        # Publish tracking status every tick
        self._tracking_pub.publish(Bool(data=tracking_valid))

        if not (tracking_valid and self._tracker.new_frame):
            return

        self._tracker.new_frame = False

        dt = self._tracker.timestep - self._tracker.prev_timestep
        if dt <= 0:
            return

        stamp = self.get_clock().now().to_msg()
        msg   = self._build_human_state(stamp, dt)
        self._human_state_pub.publish(msg)

    # ── Message building ──────────────────────────────────────────────────────

    def _build_human_state(self, stamp, dt: float) -> HumanState:
        msg              = HumanState()
        msg.header.stamp    = stamp
        msg.header.frame_id = 'mocap_world'
        msg.tracking_valid  = True

        for attr_pose, attr_twist, name in (
            ('root_pose',  'root_twist',  'Root'),
            ('lfoot_pose', 'lfoot_twist', 'LFoot'),
            ('rfoot_pose', 'rfoot_twist', 'RFoot'),
        ):
            curr_pos, curr_quat = self._tracker.curr[name]
            prev_pos, prev_quat = self._tracker.prev[name]

            lin_vel, ang_vel = compute_body_velocities(
                curr_pos, prev_pos, curr_quat, prev_quat, dt)

            setattr(msg, attr_pose,  self._make_pose(stamp, curr_pos, curr_quat))
            setattr(msg, attr_twist, self._make_twist(stamp, lin_vel, ang_vel))

        return msg

    @staticmethod
    def _make_pose(stamp, pos: np.ndarray, quat: np.ndarray) -> PoseStamped:
        ps              = PoseStamped()
        ps.header.stamp    = stamp
        ps.header.frame_id = 'mocap_world'
        ps.pose.position.x = float(pos[0])
        ps.pose.position.y = float(pos[1])
        ps.pose.position.z = float(pos[2])
        ps.pose.orientation.x = float(quat[0])
        ps.pose.orientation.y = float(quat[1])
        ps.pose.orientation.z = float(quat[2])
        ps.pose.orientation.w = float(quat[3])
        return ps

    @staticmethod
    def _make_twist(stamp, lin: np.ndarray, ang: np.ndarray) -> TwistStamped:
        ts              = TwistStamped()
        ts.header.stamp    = stamp
        ts.header.frame_id = 'mocap_world'
        ts.twist.linear.x  = float(lin[0])
        ts.twist.linear.y  = float(lin[1])
        ts.twist.linear.z  = float(lin[2])
        ts.twist.angular.x = float(ang[0])
        ts.twist.angular.y = float(ang[1])
        ts.twist.angular.z = float(ang[2])
        return ts

    def destroy_node(self):
        self.get_logger().info('Shutting down MocapDriverNode')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = NatnetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()