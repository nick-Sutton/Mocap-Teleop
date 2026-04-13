#!/usr/bin/env python3

"""
motion_mapper.py — Proportional position-following controller for legged robot teleoperation.

Receives human state as ROS2 geometry_msgs types and robot odometry, outputs
cmd_vel [vx_body, vy_body, vrz] in the robot's body frame.

Control law
───────────
Simple proportional control on the position error between human and robot,
with velocity saturation:

    error_world = human_pos − robot_pos
    cmd_world   = Kp * error_world,  clipped to MAX_LINEAR_VEL
    cmd_body    = R(−yaw) * cmd_world

The velocity cap already provides the distance-dependent scaling that the
previous adaptive-gain approach achieved with 12 tuning parameters:
  far from human  → error large → cmd at v_max
  close to human  → error small → cmd proportionally reduced → no oscillation

Coordinate frames
─────────────────
mocap_world : fixed world frame, Z-up. Human poses arrive in this frame.
odom        : robot odometry frame. Robot poses arrive in this frame.
base_link   : robot body frame. cmd_vel is expressed in this frame.

The one-time coordinate_offset aligns mocap_world origins with odom on
the first call to select_robot_command.
"""

import numpy as np
from geometry_msgs.msg import PoseStamped


class MotionMapper:
    def __init__(self, sampling_freq: float):
        self.sampling_freq = sampling_freq
        self.dt = 1.0 / sampling_freq

        # ── Coordinate frame alignment ────────────────────────────────────────
        # [dx, dy, dyaw]: added to raw human position to align with robot odom.
        # Shared by reference with PerformanceMetrics — always update with [:].
        self.coordinate_offset  = np.array([0.0, 0.0, 0.0])
        self.offset_initialized = False

        # ── Control gains ─────────────────────────────────────────────────────
        # Kp: at POSITION_TOLERANCE the robot commands Kp * tolerance m/s.
        # With Kp=1.0 and tolerance=0.05 m that's 0.05 m/s — effectively a stop.
        # At 0.3 m error the robot commands 0.3 m/s = full speed.
        self.Kp     = 1.0   # linear  (m/s per m of position error)
        self.Kp_ang = 1.0   # angular (rad/s per rad of heading error)

        # ── Velocity limits ───────────────────────────────────────────────────
        self.MAX_LINEAR_VEL  = 0.3   # m/s  — real Go2 vx limit
        self.MAX_ANGULAR_VEL = 0.8   # rad/s

        # ── Dead-zones — suppress chatter when essentially co-located ─────────
        self.POSITION_TOLERANCE = 0.05          # m
        self.HEADING_TOLERANCE  = np.radians(5) # rad

        # ── Gait scaling ──────────────────────────────────────────────────────
        self.GAIT_VELOCITY_SCALES = {'stand': 0.0, 'walk': 1.0, 'jog': 1.0}
        self.CONFIDENCE_THRESHOLD = 0.6
        self.current_gait         = 'stand'

        # ── Internal state [x, y, yaw] ────────────────────────────────────────
        self.robot_pos = np.zeros(3)
        self.human_pos = np.zeros(3)

    # ── State updates ─────────────────────────────────────────────────────────

    def update_robot_state(self, robot_pose: PoseStamped) -> None:
        p = robot_pose.pose.position
        o = robot_pose.pose.orientation
        self.robot_pos[0] = p.x
        self.robot_pos[1] = p.y
        self.robot_pos[2] = self._quat_to_yaw(o.x, o.y, o.z, o.w)

    def update_human_state(self, root_pose: PoseStamped) -> None:
        p = root_pose.pose.position
        o = root_pose.pose.orientation
        human_yaw = self._quat_to_yaw(o.x, o.y, o.z, o.w)

        if self.offset_initialized:
            self.human_pos[0] = p.x + self.coordinate_offset[0]
            self.human_pos[1] = p.y + self.coordinate_offset[1]
            self.human_pos[2] = self._normalize_angle(
                human_yaw + self.coordinate_offset[2])
        else:
            self.human_pos[0] = p.x
            self.human_pos[1] = p.y
            self.human_pos[2] = human_yaw

    # ── Control ───────────────────────────────────────────────────────────────

    def compute_cmd_vel(self):
        """
        Proportional position-following controller.

        Returns
        -------
        cmd_vel        : np.ndarray [vx_body, vy_body, vrz]
        distance_error : float  metres
        heading_error  : float  radians
        """
        ex = self.human_pos[0] - self.robot_pos[0]
        ey = self.human_pos[1] - self.robot_pos[1]
        distance_error = np.hypot(ex, ey)
        heading_error  = self._normalize_angle(
            self.human_pos[2] - self.robot_pos[2])

        # Linear: P control with velocity saturation
        if distance_error > self.POSITION_TOLERANCE:
            vx_world = self.Kp * ex
            vy_world = self.Kp * ey
            speed    = np.hypot(vx_world, vy_world)
            if speed > self.MAX_LINEAR_VEL:
                scale    = self.MAX_LINEAR_VEL / speed
                vx_world *= scale
                vy_world *= scale
        else:
            vx_world = 0.0
            vy_world = 0.0

        # Angular: P control with rate saturation
        if abs(heading_error) > self.HEADING_TOLERANCE:
            vrz = np.clip(self.Kp_ang * heading_error,
                          -self.MAX_ANGULAR_VEL, self.MAX_ANGULAR_VEL)
        else:
            vrz = 0.0

        vel_body = self._world_to_body(
            np.array([vx_world, vy_world]), self.robot_pos[2])

        return np.array([vel_body[0], vel_body[1], vrz]), distance_error, heading_error

    def select_gait(self, gait_prediction: str, confidence: float | None) -> str:
        if confidence is None or confidence < self.CONFIDENCE_THRESHOLD:
            return self.current_gait
        if gait_prediction != self.current_gait:
            self.current_gait = gait_prediction
        return self.current_gait

    def select_robot_command(
        self,
        gait_prediction: str,
        confidence:      float | None,
        robot_pose:      PoseStamped,
        root_pose:       PoseStamped,
    ) -> np.ndarray:
        """
        Main control entry point — call at fixed rate (e.g. 1000 Hz).

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

        # One-time coordinate offset: align human frame origin with robot odom.
        # Use [:] assignment so PerformanceMetrics (which holds the same array) sees it.
        if not self.offset_initialized:
            self.coordinate_offset[:] = self.robot_pos - self.human_pos
            self.human_pos[0] += self.coordinate_offset[0]
            self.human_pos[1] += self.coordinate_offset[1]
            self.human_pos[2]  = self._normalize_angle(
                self.human_pos[2] + self.coordinate_offset[2])
            self.offset_initialized = True

        active_gait = self.select_gait(gait_prediction, confidence)
        cmd_vel, distance_error, _ = self.compute_cmd_vel()

        velocity_scale = self.GAIT_VELOCITY_SCALES[active_gait]
        # Allow partial correction when standing but drifted
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
        """Wrap angle to [−π, π]."""
        return np.arctan2(np.sin(angle), np.cos(angle))

    @staticmethod
    def _world_to_body(world_vel: np.ndarray, heading: float) -> np.ndarray:
        """Rotate 2-D velocity from world frame into robot body frame."""
        c, s = np.cos(heading), np.sin(heading)
        return np.array([ c * world_vel[0] + s * world_vel[1],
                          -s * world_vel[0] + c * world_vel[1]])
