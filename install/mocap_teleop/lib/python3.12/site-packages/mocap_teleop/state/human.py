#!/usr/bin/env python3

"""
human.py — Human state container and gait feature extractor.

Stores the latest HumanState ROS2 message and computes derived gait features
for the classifier. No coordinate transforms happen here — all data arrives
already in the mocap_world frame from mocap_driver_node.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from mocap_teleop_msgs.msg import HumanState


class Human:
    def __init__(self, sampling_freq: float):
        self.dt = 1.0 / sampling_freq

        # Current and previous HumanState messages
        self.curr_state: HumanState | None = None
        self.prev_state: HumanState | None = None

        # Foot contact tracking
        self.last_left_contact_frame:  int | None = None
        self.last_right_contact_frame: int | None = None
        self.frames_since_last_contact: int = 0
        self.initialized: bool = False

    # ── State update ──────────────────────────────────────────────────────────

    def update(self, state: HumanState) -> None:
        """Called each frame with the latest HumanState message."""
        self.prev_state = self.curr_state
        self.curr_state = state

    def ready(self) -> bool:
        """True once we have both a current and previous state."""
        return self.curr_state is not None and self.prev_state is not None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _pos(pose_stamped) -> np.ndarray:
        """Extract [x, y, z] from a PoseStamped."""
        p = pose_stamped.pose.position
        return np.array([p.x, p.y, p.z])

    @staticmethod
    def _linear_vel(twist_stamped) -> np.ndarray:
        """Extract [vx, vy, vz] from a TwistStamped."""
        v = twist_stamped.twist.linear
        return np.array([v.x, v.y, v.z])

    @staticmethod
    def _angular_vel(twist_stamped) -> np.ndarray:
        """Extract [wx, wy, wz] from a TwistStamped."""
        w = twist_stamped.twist.angular
        return np.array([w.x, w.y, w.z])

    @staticmethod
    def _orientation_quat(pose_stamped) -> np.ndarray:
        """Extract [qx, qy, qz, qw] from a PoseStamped."""
        o = pose_stamped.pose.orientation
        return np.array([o.x, o.y, o.z, o.w])

    # ── Contact probability ───────────────────────────────────────────────────

    @staticmethod
    def calc_contact_probability(
        vel: np.ndarray,
        height: float,
        vel_threshold: float = 0.3,
        height_threshold: float = 0.05,
    ) -> float:
        vel_mag = np.linalg.norm(vel)
        vel_contact    = 1.0 - np.clip(vel_mag / vel_threshold,    0, 1)
        height_contact = 1.0 - np.clip(height  / height_threshold, 0, 1)
        return float(np.clip((vel_contact + height_contact) / 2, 0, 1))

    # ── Gait feature extraction ───────────────────────────────────────────────

    def extract_gait_features(self, features: dict, frame_idx: int) -> None:
        """
        Populate `features` dict with kinematic and gait features.
        Requires curr_state to be set — call update() first.
        """
        if self.curr_state is None:
            return

        left_pos  = self._pos(self.curr_state.lfoot_pose)
        right_pos = self._pos(self.curr_state.rfoot_pose)
        root_pos  = self._pos(self.curr_state.root_pose)

        # Foot heights relative to lowest foot
        min_foot_z   = min(left_pos[2], right_pos[2])
        left_height  = left_pos[2] - min_foot_z
        right_height = right_pos[2] - min_foot_z

        left_pos_rel  = left_pos  - root_pos
        right_pos_rel = right_pos - root_pos

        step_length = np.abs(left_pos[0] - right_pos[0])
        step_width  = np.abs(left_pos[1] - right_pos[1])
        step_height = np.abs(left_pos_rel[2] - right_pos_rel[2])

        left_lv  = self._linear_vel(self.curr_state.lfoot_twist)
        right_lv = self._linear_vel(self.curr_state.rfoot_twist)
        root_lv  = self._linear_vel(self.curr_state.root_twist)
        root_av  = self._angular_vel(self.curr_state.root_twist)

        left_contact_prob  = self.calc_contact_probability(left_lv,  left_height)
        right_contact_prob = self.calc_contact_probability(right_lv, right_height)

        # Contact frame tracking
        contact_threshold = 0.7
        if not self.initialized:
            if left_contact_prob > contact_threshold:
                self.last_left_contact_frame = frame_idx
                self.initialized = True
            if right_contact_prob > contact_threshold:
                self.last_right_contact_frame = frame_idx
                self.initialized = True

        if left_contact_prob  > contact_threshold:
            self.last_left_contact_frame  = frame_idx
        if right_contact_prob > contact_threshold:
            self.last_right_contact_frame = frame_idx

        if self.last_left_contact_frame is not None and self.last_right_contact_frame is not None:
            self.frames_since_last_contact = frame_idx - max(
                self.last_left_contact_frame, self.last_right_contact_frame)
        elif self.last_left_contact_frame is not None:
            self.frames_since_last_contact = frame_idx - self.last_left_contact_frame
        elif self.last_right_contact_frame is not None:
            self.frames_since_last_contact = frame_idx - self.last_right_contact_frame
        else:
            self.frames_since_last_contact = frame_idx

        # Root orientation — extract euler Z (yaw) component for classifier
        root_quat = self._orientation_quat(self.curr_state.root_pose)
        root_euler = Rotation.from_quat(root_quat).as_euler('xyz')

        # ── Kinematic features ────────────────────────────────────────────────
        features['root_position_x'] = root_pos[0]
        features['root_position_y'] = root_pos[1]
        features['root_position_z'] = root_pos[2]
        features['root_orientation_x'] = root_euler[0]
        features['root_orientation_y'] = root_euler[1]
        features['root_orientation_z'] = root_euler[2]
        features['root_linear_velocity_x']  = root_lv[0]
        features['root_linear_velocity_y']  = root_lv[1]
        features['root_linear_velocity_z']  = root_lv[2]
        features['root_angular_velocity_x'] = root_av[0]
        features['root_angular_velocity_y'] = root_av[1]
        features['root_angular_velocity_z'] = root_av[2]
        features['left_linear_velocity_x']  = left_lv[0]
        features['left_linear_velocity_y']  = left_lv[1]
        features['left_linear_velocity_z']  = left_lv[2]
        features['right_linear_velocity_x'] = right_lv[0]
        features['right_linear_velocity_y'] = right_lv[1]
        features['right_linear_velocity_z'] = right_lv[2]

        # ── Foot contact / gait features ──────────────────────────────────────
        features['left_pos_rel_x']  = left_pos_rel[0]
        features['left_pos_rel_y']  = left_pos_rel[1]
        features['left_pos_rel_z']  = left_pos_rel[2]
        features['right_pos_rel_x'] = right_pos_rel[0]
        features['right_pos_rel_y'] = right_pos_rel[1]
        features['right_pos_rel_z'] = right_pos_rel[2]
        features['step_length']         = step_length
        features['step_width']          = step_width
        features['step_height']         = step_height
        features['left_contact_prob']   = left_contact_prob
        features['right_contact_prob']  = right_contact_prob
        features['max_foot_height']     = max(left_pos_rel[2], right_pos_rel[2])

        if left_contact_prob > 0.5 and right_contact_prob > 0.5:
            features['support_type'] = 'double'
        elif left_contact_prob <= 0.5 and right_contact_prob <= 0.5:
            features['support_type'] = 'flight'
        else:
            features['support_type'] = 'single'