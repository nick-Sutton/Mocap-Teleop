#!/usr/bin/env python3

"""
motion_mapper.py — PD position-following controller for legged robot teleoperation.

Receives human state as ROS2 geometry_msgs types and robot odometry, outputs
cmd_vel [vx_body, vy_body, vrz] in the robot's body frame.

Control law
───────────
PD control on the position error between human and robot:

    error_world  = human_pos − robot_pos
    d_error/dt   = (error − prev_error) / dt  ≈ v_human − v_robot  (feedforward)
    cmd_world    = Kp * error + Kd * d_error,  clipped to per-gait MAX_VEL
    cmd_body     = R(−yaw) * cmd_world

The derivative term acts as velocity feedforward: when the human is moving at
v_human m/s, d_error/dt ≈ v_human, so Kd * d_error adds a proportional
anticipatory component that reduces lag at the cost of sensitivity to noise.

Heading follows the same PD structure with independent gains.

Coordinate frames
─────────────────
mocap_world : fixed world frame, Z-up. Human poses arrive in this frame.
odom        : robot odometry frame. Robot poses arrive in this frame.
base_link   : robot body frame. cmd_vel is expressed in this frame.

The one-time coordinate_offset aligns mocap_world origins with odom on
the first call to select_robot_command.
"""

import time

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
        # Kp:     proportional — at 0.5 m error commands 2.0 m/s before D term.
        # Kd:     derivative   — adds velocity feedforward; Kd=0.4 means human
        #         walking at 1.3 m/s contributes +0.52 m/s to the command.
        # Kp_ang: angular proportional.
        # Kd_ang: angular derivative — damps heading oscillation.
        self.Kp     = 3.5   # m/s per m  — saturates to walk cap at ~0.43 m error
        self.Kd     = 0.4   # m/s per m/s — velocity feedforward
        self.Kp_ang = 2.0   # rad/s per rad
        # Kd_ang is now a feedforward gain on the human's direct angular rate
        # (from root_twist.angular.z) rather than a finite-difference derivative
        # of heading error.  The overall loop remains feedback-controlled via
        # Kp_ang; Kd_ang anticipates turns without differencing noisy error at
        # 350 Hz.  Kd_ang≈0.5 injects half the human turn rate as anticipation;
        # Kp_ang handles residual heading error.
        self.Kd_ang         = 0.5   # rad/s per rad/s (human angular rate feedforward)
        self.human_yaw_rate = 0.0   # rad/s, updated from root_twist.angular.z

        # ── Velocity limits ───────────────────────────────────────────────────
        # MAX_LINEAR_VEL: absolute cap across all gaits; used by the CBF filter.
        # MAX_VEL_BY_GAIT: per-gait cap applied inside compute_cmd_vel.
        self.MAX_LINEAR_VEL  = 3.7   # m/s — Go2 nominal max forward speed
        self.MAX_ANGULAR_VEL = 1.5   # rad/s — Go2 nominal max yaw rate
        self.MAX_VEL_BY_GAIT = {
            'stand': 1.0,   # enough to close gaps during brief stand phases
            'walk':  2.0,   # human walks ~1.3 m/s; headroom for gap closing
            'jog':   3.0,   # human jogs ~2-3 m/s; headroom without hitting hardware max
        }
        # Go2 lateral (vy body-frame) capability is much weaker than forward.
        # Hardware clips at 0.6 m/s internally — cap here too so the command
        # ratio is not distorted when the robot needs to move diagonally.
        self.MAX_LATERAL_VEL = 0.6   # m/s — Go2 nominal max lateral speed

        # ── Dead-zones — suppress chatter when essentially co-located ─────────
        self.POSITION_TOLERANCE = 0.05          # m
        self.HEADING_TOLERANCE  = np.radians(5) # rad

        # ── Gait state ────────────────────────────────────────────────────────
        self.CONFIDENCE_THRESHOLD   = 0.6
        # Minimum time (s) before accepting a gait change — prevents rapid
        # oscillation between gaits from a noisy classifier.
        self.GAIT_DWELL_S           = 1.0
        self.current_gait           = 'stand'
        self._gait_last_change_time = 0.0

        # ── Lookahead ─────────────────────────────────────────────────────────
        # Target the human's predicted position T seconds ahead rather than
        # their current position.  Directly compensates for the PD controller's
        # inherent following lag during constant-velocity motion.
        # Uses root_twist from the mocap driver (finite-differenced at 240 Hz)
        # — cleaner than differentiating the position error ourselves.
        self.LOOKAHEAD_S  = 0.25         # seconds; tune between 0.1–0.5
        self.human_vel    = np.zeros(2)  # [vx, vy] world frame, updated each call

        # ── Internal state [x, y, yaw] ────────────────────────────────────────
        self.robot_pos = np.zeros(3)
        self.human_pos = np.zeros(3)

        # ── Derivative state ──────────────────────────────────────────────────
        # Initialised to None; set to current error on the first compute call
        # so the D term is zero on the first frame (avoids a large spike).
        # _last_compute_time tracks actual elapsed time for accurate derivatives
        # regardless of loop jitter (nominal dt is unreliable at ~350 Hz actual).
        self._prev_ex:           float | None = None
        self._prev_ey:           float | None = None
        self._last_compute_time: float | None = None



    # ── State updates ─────────────────────────────────────────────────────────

    def update_robot_state(self, robot_pose: PoseStamped) -> None:
        p = robot_pose.pose.position
        o = robot_pose.pose.orientation
        self.robot_pos[0] = p.x
        self.robot_pos[1] = p.y
        self.robot_pos[2] = self._quat_to_yaw(o.x, o.y, o.z, o.w)

    def update_human_state(self, root_pose: PoseStamped,
                           root_twist=None) -> None:
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

        if root_twist is not None:
            self.human_vel[0]   = root_twist.twist.linear.x
            self.human_vel[1]   = root_twist.twist.linear.y
            self.human_yaw_rate = root_twist.twist.angular.z

    # ── Control ───────────────────────────────────────────────────────────────

    def compute_cmd_vel(self):
        """
        PD position-following controller.

        Returns
        -------
        cmd_vel        : np.ndarray [vx_body, vy_body, vrz]
        distance_error : float  metres
        heading_error  : float  radians
        """
        # ── Measure actual elapsed time for derivative accuracy ───────────────
        # self.dt (nominal) is unreliable: the loop runs at ~350 Hz while the
        # constructor receives 500 Hz, making the D term 44% too large if we use
        # self.dt.  Using wall-clock elapsed time keeps gains correct under jitter.
        now = time.monotonic()
        actual_dt = (now - self._last_compute_time) if self._last_compute_time is not None else self.dt
        actual_dt = max(actual_dt, 1e-4)   # guard against zero on first call
        self._last_compute_time = now

        # Lookahead: target where the human will be in LOOKAHEAD_S seconds.
        # human_vel is the mocap-driver finite difference — cleaner than the
        # D term derivative and requires no actual_dt accounting.
        target_x = self.human_pos[0] + self.LOOKAHEAD_S * self.human_vel[0]
        target_y = self.human_pos[1] + self.LOOKAHEAD_S * self.human_vel[1]

        ex = target_x - self.robot_pos[0]
        ey = target_y - self.robot_pos[1]
        distance_error = np.hypot(self.human_pos[0] - self.robot_pos[0],
                                  self.human_pos[1] - self.robot_pos[1])
        heading_error  = self._normalize_angle(
            self.human_pos[2] - self.robot_pos[2])

        # ── Linear PD ─────────────────────────────────────────────────────────
        if distance_error > self.POSITION_TOLERANCE:
            # Seed derivative state on first call so D term starts at zero.
            if self._prev_ex is None:
                self._prev_ex = ex
                self._prev_ey = ey

            d_ex = (ex - self._prev_ex) / actual_dt
            d_ey = (ey - self._prev_ey) / actual_dt

            vx_world = self.Kp * ex + self.Kd * d_ex
            vy_world = self.Kp * ey + self.Kd * d_ey

            speed   = np.hypot(vx_world, vy_world)
            max_vel = self.MAX_VEL_BY_GAIT.get(self.current_gait,
                                                self.MAX_LINEAR_VEL)
            if max_vel <= 0.0:
                vx_world = vy_world = 0.0
            elif speed > max_vel:
                scale    = max_vel / speed
                vx_world *= scale
                vy_world *= scale
        else:
            vx_world = vy_world = 0.0

        # Always advance derivative state so the next D term is well-conditioned
        # regardless of whether the deadzone fired this step.
        self._prev_ex = ex
        self._prev_ey = ey

        # ── Angular P + feedforward ───────────────────────────────────────────
        # Kp_ang closes the feedback loop on heading error.
        # Kd_ang * human_yaw_rate anticipates turns using the mocap angular
        # rate directly — cleaner than differencing heading_error at 350 Hz.
        if abs(heading_error) > self.HEADING_TOLERANCE:
            vrz = np.clip(
                self.Kp_ang * heading_error + self.Kd_ang * self.human_yaw_rate,
                -self.MAX_ANGULAR_VEL, self.MAX_ANGULAR_VEL)
        else:
            vrz = 0.0

        vel_body = self._world_to_body(
            np.array([vx_world, vy_world]), self.robot_pos[2])

        # Clamp lateral velocity to hardware limit — the Go2 is much less
        # stable moving sideways than forward; mpac clips internally anyway
        # but capping here keeps the vx/vy ratio from being distorted.
        vy_clamped = np.clip(vel_body[1], -self.MAX_LATERAL_VEL, self.MAX_LATERAL_VEL)

        return np.array([vel_body[0], vy_clamped, vrz]), distance_error, heading_error

    def select_gait(self, gait_prediction: str, confidence: float | None) -> str:
        if confidence is None or confidence < self.CONFIDENCE_THRESHOLD:
            return self.current_gait
        if gait_prediction != self.current_gait:
            now = time.monotonic()
            if now - self._gait_last_change_time >= self.GAIT_DWELL_S:
                self.current_gait           = gait_prediction
                self._gait_last_change_time = now
        return self.current_gait

    def select_robot_command(
        self,
        gait_prediction: str,
        confidence:      float | None,
        robot_pose:      PoseStamped,
        root_pose:       PoseStamped,
        root_twist=None,
    ) -> np.ndarray:
        """
        Main control entry point — call at fixed rate (e.g. 500 Hz).

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
        self.update_human_state(root_pose, root_twist)

        # One-time coordinate offset: align human frame origin with robot odom.
        # Use [:] assignment so PerformanceMetrics (which holds the same array) sees it.
        # Yaw offset is intentionally zero: the robot should face the same absolute
        # compass direction as the human at all times (teleportation model — if the
        # human faces North, the robot faces North, independent of their starting headings).
        if not self.offset_initialized:
            self.coordinate_offset[:] = self.robot_pos - self.human_pos
            # coordinate_offset[2] is the yaw offset between mocap and odom frames.
            # Including it means heading error starts at zero and the controller
            # tracks heading changes rather than absolute compass direction —
            # the robot still fully mimics the human's heading, without spending
            # the first frames correcting an artificial frame-misalignment artifact.
            self.human_pos[0] += self.coordinate_offset[0]
            self.human_pos[1] += self.coordinate_offset[1]
            self.human_pos[2]  = self._normalize_angle(
                self.human_pos[2] + self.coordinate_offset[2])
            self.offset_initialized = True

        self.select_gait(gait_prediction, confidence)
        cmd_vel, _, _ = self.compute_cmd_vel()
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
