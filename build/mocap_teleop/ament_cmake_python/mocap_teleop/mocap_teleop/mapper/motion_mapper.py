#!/usr/bin/env python3

"""
motion_mapper.py — Feedforward + feedback controller for legged robot teleoperation.

Receives human state as ROS2 geometry_msgs types and robot odometry, outputs
cmd_vel [vx_body, vy_body, vrz] in the robot's body frame.

Coordinate frames
─────────────────
mocap_world : fixed world frame, Z-up. Human poses arrive in this frame.
odom        : robot odometry frame. Robot poses arrive in this frame.
base_link   : robot body frame. cmd_vel is expressed in this frame.

The one-time coordinate_offset aligns mocap_world origins with odom on
frame 1. After that, human_pos and robot_pos are comparable directly.
"""

import time

import numpy as np
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import PoseStamped


class MotionMapper:
    def __init__(self, sampling_freq: float):
        self.sampling_freq = sampling_freq
        self.dt = 1.0 / sampling_freq

        # ── Coordinate frame alignment ────────────────────────────────────────
        # [dx, dy, dyaw]: added to raw human position to align with robot odom.
        # Shared by reference with PerformanceMetrics — always update with [:].
        self.coordinate_offset    = np.array([0.0, 0.0, 0.0])
        self.offset_initialized   = False

        # ── Adaptive linear gains ─────────────────────────────────────────────
        self.Kp_close  = 0.8;  self.Kd_close  = 0.22
        self.Kp_medium = 1.8;  self.Kd_medium = 0.5
        self.Kp_far    = 2.0;  self.Kd_far    = 0.6

        # ── Adaptive angular gains ────────────────────────────────────────────
        self.Kp_ang_close  = 0.4;  self.Kd_ang_close  = 0.15
        self.Kp_ang_medium = 0.8;  self.Kd_ang_medium = 0.25
        self.Kp_ang_far    = 1.2;  self.Kd_ang_far    = 0.35

        self.HEADING_CLOSE  = np.radians(10)
        self.HEADING_MEDIUM = np.radians(30)
        self.HEADING_CAP    = np.radians(90)

        # ── Velocity limits ───────────────────────────────────────────────────
        self.MAX_LINEAR_VEL  = 1.5   # m/s
        self.MAX_ANGULAR_VEL = 0.8   # rad/s

        # ── Control thresholds ────────────────────────────────────────────────
        self.POSITION_TOLERANCE = 0.15   # m — stop correction below this
        self.CLOSE_DISTANCE     = 0.08   # m
        self.MEDIUM_DISTANCE    = 0.30   # m

        # ── Gait scaling ──────────────────────────────────────────────────────
        self.GAIT_VELOCITY_SCALES = {'stand': 0.0, 'walk': 1.0, 'jog': 1.0}
        self.CONFIDENCE_THRESHOLD = 0.6
        self.current_gait         = 'stand'

        # ── Internal state [x, y, yaw] ────────────────────────────────────────
        self.robot_pos = np.zeros(3)
        self.robot_vel = np.zeros(3)
        self.human_pos = np.zeros(3)
        self.human_vel = np.zeros(3)

        self.smoothed_speed:        float = 0.0
        self.first_update:          bool  = True
        self._last_pos_change_time: float = 0.0

    # ── State updates ─────────────────────────────────────────────────────────

    def update_robot_state(
        self,
        robot_pose: PoseStamped,
    ) -> None:
        """
        Update robot state from a nav_msgs/Odometry-derived PoseStamped.
        Velocity is estimated by finite difference.
        """
        prev_pos = self.robot_pos.copy()

        p = robot_pose.pose.position
        o = robot_pose.pose.orientation

        self.robot_pos[0] = p.x
        self.robot_pos[1] = p.y
        self.robot_pos[2] = self._quat_to_yaw(o.x, o.y, o.z, o.w)

        now = time.monotonic()
        if not self.first_update:
            delta    = self.robot_pos - prev_pos
            delta[2] = self._normalize_angle(delta[2])
            # Use actual elapsed time between position changes so that robot
            # velocity is correct even when telemetry updates slower than the
            # PD loop rate (e.g. 200 Hz telemetry polled at 1000 Hz).
            if np.any(np.abs(delta) > 1e-9):
                actual_dt = now - self._last_pos_change_time
                if actual_dt > 0:
                    self.robot_vel = delta / actual_dt
                self._last_pos_change_time = now
        else:
            self._last_pos_change_time = now

    def update_human_state(
        self,
        root_pose: PoseStamped,
    ) -> None:
        """
        Update human state from mocap_world-frame ROS2 messages.
        Coordinate offset is applied once offset_initialized is True.
        """
        p = root_pose.pose.position
        o = root_pose.pose.orientation

        human_x   = p.x
        human_y   = p.y
        human_yaw = self._quat_to_yaw(o.x, o.y, o.z, o.w)

        if self.offset_initialized:
            self.human_pos[0] = human_x   + self.coordinate_offset[0]
            self.human_pos[1] = human_y   + self.coordinate_offset[1]
            self.human_pos[2] = self._normalize_angle(
                human_yaw + self.coordinate_offset[2])
        else:
            self.human_pos[0] = human_x
            self.human_pos[1] = human_y
            self.human_pos[2] = human_yaw

        self.human_vel = self._extract_intended_motion()

    def _extract_intended_motion(self) -> np.ndarray:
        """
        Reconstruct smooth intended velocity from hip orientation + speed magnitude.
        Decouples gait oscillations from trajectory direction.

        Speed is derived from the previous self.human_vel magnitude (matching the
        original system). This keeps the feedforward contribution near zero and
        makes the controller behave as a position-PD follower, which is what the
        original mapper did in practice.
        """
        intended_heading = self.human_pos[2]
        raw_speed        = np.linalg.norm(self.human_vel[:2])

        alpha            = 0.5
        self.smoothed_speed = (alpha * raw_speed
                               + (1.0 - alpha) * self.smoothed_speed)

        vx = self.smoothed_speed * np.cos(intended_heading)
        vy = self.smoothed_speed * np.sin(intended_heading)
        return np.array([vx, vy, self.human_vel[2]])

    # ── Control ───────────────────────────────────────────────────────────────

    def compute_unified_control(self):
        """
        Feedforward + adaptive feedback PD controller.

        Returns
        -------
        cmd_vel        : np.ndarray [vx_body, vy_body, vrz]
        distance_error : float  metres
        heading_error  : float  radians
        """
        x_delta        = self.human_pos[0] - self.robot_pos[0]
        y_delta        = self.human_pos[1] - self.robot_pos[1]
        distance_error = np.hypot(x_delta, y_delta)
        heading_error  = self._normalize_angle(
            self.human_pos[2] - self.robot_pos[2])

        vel_error_x = self.human_vel[0] - self.robot_vel[0]
        vel_error_y = self.human_vel[1] - self.robot_vel[1]

        # Feedforward
        vx_ff    = self.human_vel[0]
        vy_ff    = self.human_vel[1]
        omega_ff = self.human_vel[2]

        # Adaptive linear gains
        kp, kd, lin_mode = self._linear_gains(distance_error)

        # Adaptive angular gains
        kp_ang, kd_ang, ang_mode = self._angular_gains(abs(heading_error))

        # Linear feedback
        vx_world = vx_ff + kp * x_delta        + kd * vel_error_x
        vy_world = vy_ff + kp * y_delta        + kd * vel_error_y

        # Angular control
        vrz_cmd = np.clip(
            omega_ff
            + kp_ang * heading_error
            + kd_ang * (self.human_vel[2] - self.robot_vel[2]),
            -self.MAX_ANGULAR_VEL, self.MAX_ANGULAR_VEL,
        )

        # Linear velocity clamp
        speed = np.hypot(vx_world, vy_world)
        if speed > self.MAX_LINEAR_VEL:
            scale    = self.MAX_LINEAR_VEL / speed
            vx_world *= scale
            vy_world *= scale

        # World → body frame (2-D yaw rotation only)
        vel_body = self._world_to_body(
            np.array([vx_world, vy_world]), self.robot_pos[2])

        return np.array([vel_body[0], vel_body[1], vrz_cmd]), distance_error, heading_error

    def select_gait(self, gait_prediction: str, confidence: float | None) -> str:
        if confidence is None or confidence < self.CONFIDENCE_THRESHOLD:
            return self.current_gait
        if gait_prediction != self.current_gait:
            self.current_gait = gait_prediction
        return self.current_gait

    def select_robot_command(
        self,
        gait_prediction:  str,
        confidence:       float | None,
        robot_pose:       PoseStamped,
        root_pose:        PoseStamped,
    ) -> np.ndarray:
        """
        Main control entry point — call at fixed rate (e.g. 240 Hz).

        Parameters
        ----------
        gait_prediction : classified gait label
        confidence      : classifier confidence [0, 1]
        robot_pose      : current robot PoseStamped (odom frame)
        root_pose       : human pelvis PoseStamped (mocap_world frame)

        Returns
        -------
        cmd_vel : np.ndarray [vx_body, vy_body, vrz]
        """
        self.update_robot_state(robot_pose)
        self.update_human_state(root_pose)

        # One-time coordinate offset: align human frame origin with robot odom origin.
        # Use [:] assignment so PerformanceMetrics (which holds the same array) sees it.
        if not self.offset_initialized:
            self.coordinate_offset[:] = self.robot_pos - self.human_pos
            # Apply immediately to current human position
            self.human_pos[0] += self.coordinate_offset[0]
            self.human_pos[1] += self.coordinate_offset[1]
            self.human_pos[2]  = self._normalize_angle(
                self.human_pos[2] + self.coordinate_offset[2])
            self.offset_initialized = True

        if self.first_update:
            self.first_update = False

        active_gait   = self.select_gait(gait_prediction, confidence)
        cmd_vel, distance_error, heading_error = self.compute_unified_control()

        velocity_scale = self.GAIT_VELOCITY_SCALES[active_gait]
        # Allow partial position correction even when standing but drifted.
        if active_gait == 'stand' and distance_error > self.POSITION_TOLERANCE:
            velocity_scale = 0.5
        cmd_vel *= velocity_scale

        return cmd_vel

    # ── Math helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
        """Extract yaw angle from a quaternion."""
        return np.arctan2(2.0 * (qw * qz + qx * qy),
                          1.0 - 2.0 * (qy**2 + qz**2))

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Wrap angle to [-π, π]."""
        return np.arctan2(np.sin(angle), np.cos(angle))

    @staticmethod
    def _world_to_body(world_vel: np.ndarray, heading: float) -> np.ndarray:
        """Rotate 2-D velocity from world frame into robot body frame."""
        c, s = np.cos(heading), np.sin(heading)
        return np.array([ c * world_vel[0] + s * world_vel[1],
                          -s * world_vel[0] + c * world_vel[1]])

    def _linear_gains(self, dist: float):
        if dist < self.CLOSE_DISTANCE:
            return self.Kp_close, self.Kd_close, "CLOSE"
        elif dist < self.MEDIUM_DISTANCE:
            a = (dist - self.CLOSE_DISTANCE) / (self.MEDIUM_DISTANCE - self.CLOSE_DISTANCE)
            return (self.Kp_close  + a * (self.Kp_medium - self.Kp_close),
                    self.Kd_close  + a * (self.Kd_medium - self.Kd_close),
                    "MEDIUM")
        else:
            a = (min(dist, 1.0) - self.MEDIUM_DISTANCE) / (1.0 - self.MEDIUM_DISTANCE)
            return (self.Kp_medium + a * (self.Kp_far - self.Kp_medium),
                    self.Kd_medium + a * (self.Kd_far - self.Kd_medium),
                    "FAR")

    def _angular_gains(self, abs_heading: float):
        if abs_heading < self.HEADING_CLOSE:
            return self.Kp_ang_close, self.Kd_ang_close, "CLOSE"
        elif abs_heading < self.HEADING_MEDIUM:
            a = ((abs_heading - self.HEADING_CLOSE)
                 / (self.HEADING_MEDIUM - self.HEADING_CLOSE))
            return (self.Kp_ang_close  + a * (self.Kp_ang_medium - self.Kp_ang_close),
                    self.Kd_ang_close  + a * (self.Kd_ang_medium - self.Kd_ang_close),
                    "MEDIUM")
        else:
            a = ((min(abs_heading, self.HEADING_CAP) - self.HEADING_MEDIUM)
                 / (self.HEADING_CAP - self.HEADING_MEDIUM))
            return (self.Kp_ang_medium + a * (self.Kp_ang_far - self.Kp_ang_medium),
                    self.Kd_ang_medium + a * (self.Kd_ang_far - self.Kd_ang_medium),
                    "FAR")

