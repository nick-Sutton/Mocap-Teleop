#!/usr/bin/env python3

"""
teleop_node.py — ROS2 teleoperation controller node.

Rate design
───────────
ROS2 executor      — SingleThreadedExecutor with only trivial callbacks:
                     subscriber (~240 Hz, store+flag) and log timer (1 Hz).
                     MultiThreadedExecutor mutex contention limited the
                     subscriber to ~17 Hz; SingleThreadedExecutor removes it.

Buffer-fill thread — Python thread: polls _new_mocap_frame at 1 kHz, runs
                     feature extraction + gc.add_frame() for each new frame.
                     Fills the TCN 60-frame window at the data rate (~240 Hz)
                     without holding up the subscriber callback.

Classifier thread  — Python thread: calls gc.infer() at ~13 Hz (capped by the
                     75ms TCN forward pass).  Sets _gait_prediction for PD.

PD thread          — Python thread at 500 Hz using deadline-sleep.
                     ROS2 Python timers cannot reliably enforce periods below
                     ~10 ms so a dedicated thread is required for 500 Hz.

FSM overview
────────────
IMITATE  — normal PD + CBF imitation control (default).
PLAN     — feasibility estimator detected low Q; A* routes robot around
            obstacle toward human's current position.
RESYNC   — Q recovered above Q_HIGH; waypoint follower walks robot back
            to within RESYNC_DIST_M of the human before re-entering IMITATE.

Transitions
  IMITATE → PLAN    : Q < Q_LOW for _FSM_CONFIRM_FRAMES consecutive ticks
  PLAN    → RESYNC  : Q > Q_HIGH (safe path now available)
  PLAN    → PLAN    : A* replans each tick as human moves
  RESYNC  → IMITATE : distance to human < RESYNC_DIST_M
  RESYNC  → PLAN    : Q < Q_LOW again (path still blocked)
"""

import enum
import os
import threading
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
from geometry_msgs.msg import Twist, TwistStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from std_msgs.msg import String, Bool
from visualization_msgs.msg import MarkerArray

from mocap_teleop_msgs.msg import HumanState
from mocap_teleop.state.human import Human
from mocap_teleop.mapper.motion_mapper import MotionMapper
from mocap_teleop.model.gait_classifier import GaitClassifier
from mocap_teleop.ctrl.ctrl_interface import CtrlInterface
from mocap_teleop.ctrl.cbf_safety_filter import CbfSafetyFilter
from mocap_teleop.ctrl.cbf_qp import Obstacle
from mocap_teleop.util.freq_meter import FrequencyMeter
from mocap_teleop.util.performance_metrics import PerformanceMetrics
import mocap_teleop.util.io_parser as io

_PD_RATE_HZ          = 500.0    # deadline-sleep enforces this reliably
_CLASSIFIER_RATE_HZ  = 20.0
_LOG_RATE_HZ         = 1.0
_MOCAP_TIMEOUT_S     = 0.5      # stop robot if no mocap for 500 ms

# ── FSM thresholds ────────────────────────────────────────────────────────────
_Q_LOW              = -30.0   # below this → PLAN (critic predicts failure)
_Q_HIGH             = -20.0   # above this → RESYNC (path is viable again)
_FSM_CONFIRM_FRAMES = 10      # consecutive low-Q ticks before entering PLAN
_RESYNC_DIST_M      = 0.4     # re-enter IMITATE once this close to human


class TeleopMode(enum.Enum):
    IMITATE = 'IMITATE'
    PLAN    = 'PLAN'
    RESYNC  = 'RESYNC'


class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_node')

        mocap_cfg    = io.parse_mocap_config()
        learning_cfg = io.parse_learning_config()
        logging_cfg  = io.parse_logging_config()

        self.declare_parameter('sampling_freq',
                               mocap_cfg['camera_frequency'])
        self.declare_parameter('model_path',
                               learning_cfg['gait_classifier_path'])

        sampling_freq = (self.get_parameter('sampling_freq')
                         .get_parameter_value().double_value)
        model_path    = (self.get_parameter('model_path')
                         .get_parameter_value().string_value)

        # ── Core objects ──────────────────────────────────────────────────────
        self.human  = Human(sampling_freq)
        self.mapper = MotionMapper(_PD_RATE_HZ)
        self.get_logger().info('Loading gait classifier...')
        self.gc     = GaitClassifier(model_path)
        self.get_logger().info('Gait classifier ready')

        # ── CBF safety filter ─────────────────────────────────────────────────
        cbf_model_path = learning_cfg.get('cbf_model_path', '')
        cbf_enabled    = bool(cbf_model_path) and os.path.isfile(cbf_model_path)
        if cbf_enabled:
            self.get_logger().info(f'Loading CBF α-net from {cbf_model_path}...')
            self._cbf_filter = CbfSafetyFilter(
                model_path = cbf_model_path,
                v_max      = self.mapper.MAX_LINEAR_VEL,
                enabled    = True,
            )
            self.get_logger().info('CBF safety filter ready')
        else:
            self._cbf_filter = None
            if cbf_model_path:
                self.get_logger().warn(
                    f'CBF model path set but file not found: {cbf_model_path}'
                    ' — safety filter disabled')
            else:
                self.get_logger().info(
                    'CBF model path not set — safety filter disabled')

        # ── Feasibility estimator (optional) ──────────────────────────────────
        critic_path = learning_cfg.get('critic_path', '')
        if bool(critic_path) and os.path.isfile(critic_path):
            from mocap_teleop.ctrl.feasibility_estimator import FeasibilityEstimator
            from mocap_teleop.planning.mppi_planner import MPPIPlanner
            self._feasibility = FeasibilityEstimator(critic_path)
            self._mppi        = MPPIPlanner()
            self.get_logger().info('Feasibility estimator ready — FSM enabled')
        else:
            self._feasibility = None
            self._mppi        = None
            if critic_path:
                self.get_logger().warn(
                    f'Critic path set but file not found: {critic_path}'
                    ' — FSM disabled')
            else:
                self.get_logger().info('Critic path not set — FSM disabled')

        # ── FSM state ─────────────────────────────────────────────────────────
        self._fsm_mode:         TeleopMode = TeleopMode.IMITATE
        self._fsm_low_q_count:  int        = 0   # consecutive low-Q ticks
        self._costmap                      = None  # OccupancyGrid | None

        # ── Stand up before starting control threads ──────────────────────────
        self.get_logger().info('Sending stand command — waiting 3 s for robot to stand...')
        CtrlInterface.stand(0, 0, 0)
        time.sleep(3.0)
        self.get_logger().info('Stand complete — starting control threads')

        # ── Shared state (written by sub/classifier, read by PD thread) ───────
        self._latest_human_state: HumanState | None = None
        self._new_mocap_frame:    bool              = False
        self._last_mocap_time:    float             = 0.0
        self._frame_idx:          int               = 0
        self._gait_prediction:    str | None        = None
        self._gait_confidence:    float             = 0.0
        self._last_cmd_vel:       np.ndarray | None = None
        self._running:            bool              = True
        # Obstacle list updated by /obstacles subscriber, consumed by PD thread.
        # List replacement is GIL-atomic in CPython — no lock needed.
        self._obstacles:          list              = []

        # ── Frequency meters ──────────────────────────────────────────────────
        self._freq_mocap      = FrequencyMeter(window=100)
        self._freq_classifier = FrequencyMeter(window=100)
        self._freq_pd         = FrequencyMeter(window=100)

        # ── Performance metrics ───────────────────────────────────────────────
        logs_dir = logging_cfg['logs_dir']
        self._metrics = PerformanceMetrics(
            coordinate_offset = self.mapper.coordinate_offset,
            logs_dir          = logs_dir,
            dt                = 1.0 / sampling_freq,
            nominal_rates     = {
                'mocap':      sampling_freq,
                'classifier': _CLASSIFIER_RATE_HZ,
                'pd':         _PD_RATE_HZ,
            },
        )

        # ── Subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(
            HumanState, '/mocap/human_state',
            self._human_state_cb, 10)
        self.create_subscription(
            Bool, '/mocap/tracking_valid',
            self._tracking_valid_cb, 10)
        if self._cbf_filter is not None:
            self.create_subscription(
                MarkerArray, '/obstacles',
                self._obstacles_cb, 10)
        if self._mppi is not None:
            self.create_subscription(
                OccupancyGridMsg, '/local_costmap',
                self._costmap_cb, 1)   # depth=1: only latest grid matters

        # ── Publishers ────────────────────────────────────────────────────────
        self._cmd_vel_pub = self.create_publisher(Twist,  '/cmd_vel',     10)
        self._gait_pub    = self.create_publisher(String, '/teleop/gait', 10)
        self._mode_pub    = self.create_publisher(String, '/teleop/mode', 10)

        # ── Timers ────────────────────────────────────────────────────────────
        self.create_timer(1.0 / _LOG_RATE_HZ, self._log_freq_cb)

        # ── Background threads ────────────────────────────────────────────────
        self._buf_thread = threading.Thread(
            target=self._buffer_fill_loop, name='buf_fill', daemon=True)
        self._buf_thread.start()
        self._cls_thread = threading.Thread(
            target=self._classifier_loop, name='classifier', daemon=True)
        self._cls_thread.start()
        self._pd_thread = threading.Thread(
            target=self._pd_loop, name='pd_control', daemon=True)
        self._pd_thread.start()

        self.get_logger().info('TeleopNode ready — waiting for data...')

    # ── Subscriber callbacks ──────────────────────────────────────────────────

    def _human_state_cb(self, msg: HumanState) -> None:
        """Truly minimal hot path — store msg and signal the buffer-fill thread."""
        self._latest_human_state = msg
        self._last_mocap_time    = time.monotonic()
        self.human.update(msg)
        self._new_mocap_frame = True
        self._freq_mocap.tick()

    def _tracking_valid_cb(self, msg: Bool) -> None:
        pass   # consumed by the mocap timeout mechanism

    def _obstacles_cb(self, msg: MarkerArray) -> None:
        """Convert incoming MarkerArray into a list of Obstacle objects.

        Expects SPHERE markers with:
          pose.position.{x,y} — obstacle center in odom frame
          scale.x             — sphere diameter (radius = scale.x / 2)

        List replacement is GIL-atomic so no lock is needed.
        """
        obs = []
        for marker in msg.markers:
            center = np.array([marker.pose.position.x,
                               marker.pose.position.y])
            radius = marker.scale.x / 2.0
            if radius > 0.01:   # ignore degenerate markers
                obs.append(Obstacle(center=center, radius=radius))
        self._obstacles = obs

    def _costmap_cb(self, msg: OccupancyGridMsg) -> None:
        """Convert incoming OccupancyGrid to a queryable OccupancyGrid wrapper.

        Runs in the ROS executor thread; assignment is GIL-atomic.
        """
        from mocap_teleop.planning.mppi_planner import OccupancyGrid
        self._costmap = OccupancyGrid(
            data       = np.array(msg.data, dtype=np.int8),
            width      = msg.info.width,
            height     = msg.info.height,
            resolution = msg.info.resolution,
            origin_x   = msg.info.origin.position.x,
            origin_y   = msg.info.origin.position.y,
        )

    # ── Buffer-fill thread ────────────────────────────────────────────────────

    def _buffer_fill_loop(self) -> None:
        """Dedicated thread: extract gait features and fill the TCN sliding
        window at the mocap data rate (~240 Hz).

        Keeping this work out of _human_state_cb means the subscriber callback
        stays trivial and can receive messages without delay.  The feature
        extraction + sklearn scaling step takes ~5–50 ms, far too slow to run
        inside the ROS2 executor callback.
        """
        while self._running:
            if self._new_mocap_frame and self.human.ready():
                self._new_mocap_frame = False
                self._frame_idx += 1
                features = {}
                self.human.extract_gait_features(features, self._frame_idx)
                self.gc.add_frame(features)
            time.sleep(0.001)   # 1 ms poll — processes each frame within 1 ms

    # ── Classifier thread (~13 Hz, limited by TCN inference time) ────────────

    def _classifier_loop(self) -> None:
        """Runs TCN inference as fast as the model allows (target 20 Hz)."""
        dt = 1.0 / _CLASSIFIER_RATE_HZ
        while self._running:
            t0 = time.monotonic()

            gait_prediction, confidence, _ = self.gc.infer()
            if gait_prediction is not None:
                self._gait_prediction = gait_prediction
                self._gait_confidence = confidence
                self._freq_classifier.tick()
                self._gait_pub.publish(String(data=gait_prediction))

            elapsed = time.monotonic() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ── PD control thread (500 Hz) ────────────────────────────────────────────

    def _pd_loop(self) -> None:
        """Dedicated thread at 500 Hz using a deadline-sleep pattern.

        A deadline sleep (sleep for the remaining time each iteration) reliably
        enforces the target rate.  ROS2 Python timers cannot enforce periods
        below ~10 ms due to GIL scheduling overhead.
        """
        _target_dt        = 1.0 / _PD_RATE_HZ   # 0.002 s
        _telemetry_warned = False
        _stopped          = False

        while self._running:
            _t0 = time.monotonic()

            if self._latest_human_state is not None:
                if time.monotonic() - self._last_mocap_time <= _MOCAP_TIMEOUT_S:
                    _stopped = False

                    if self._gait_prediction is None:
                        CtrlInterface.walk(vx=0, vy=0, vrz=0)
                    else:
                        try:
                            pos, quat = CtrlInterface.get_robot_pose()
                            _telemetry_warned = False
                        except Exception as e:
                            if not _telemetry_warned:
                                self.get_logger().warn(
                                    f'Robot telemetry unavailable: {e}  '
                                    f'— PD control disabled until resolved')
                                _telemetry_warned = True
                            pos = quat = None

                        if pos is not None:
                            robot_pose                    = PoseStamped()
                            robot_pose.header.stamp       = self.get_clock().now().to_msg()
                            robot_pose.header.frame_id    = 'odom'
                            robot_pose.pose.position.x    = float(pos[0])
                            robot_pose.pose.position.y    = float(pos[1])
                            robot_pose.pose.position.z    = float(pos[2])
                            robot_pose.pose.orientation.x = float(quat[0])
                            robot_pose.pose.orientation.y = float(quat[1])
                            robot_pose.pose.orientation.z = float(quat[2])
                            robot_pose.pose.orientation.w = float(quat[3])

                            from scipy.spatial.transform import Rotation as _R
                            yaw = _R.from_quat([
                                float(quat[0]), float(quat[1]),
                                float(quat[2]), float(quat[3]),
                            ]).as_euler('xyz')[2]

                            # ── Mapper PD command (always computed) ───────────
                            cmd_vel = self.mapper.select_robot_command(
                                gait_prediction = self._gait_prediction,
                                confidence      = self._gait_confidence,
                                robot_pose      = robot_pose,
                                root_pose       = self._latest_human_state.root_pose,
                                root_twist      = self._latest_human_state.root_twist,
                            )

                            # ── CBF safety filter ─────────────────────────────
                            if self._cbf_filter is not None:
                                u_safe = self._cbf_filter.filter(
                                    u_perf_body = np.array(cmd_vel[:2],
                                                           dtype=np.float64),
                                    robot_pos   = np.array(pos[:2],
                                                           dtype=np.float64),
                                    robot_yaw   = float(yaw),
                                    obstacles   = self._obstacles,
                                )
                            else:
                                u_safe = cmd_vel[:2].copy()

                            # ── FSM update ────────────────────────────────────
                            fsm_cmd = self._fsm_update(
                                robot_pos = np.array(pos[:2], dtype=np.float64),
                                robot_yaw = float(yaw),
                                vel_body  = np.array(cmd_vel[:2], dtype=np.float64),
                                u_safe    = u_safe,
                                cmd_vel   = cmd_vel,
                            )

                            vx_cmd, vy_cmd = float(fsm_cmd[0]), float(fsm_cmd[1])

                            CtrlInterface.walk(
                                vx  = vx_cmd,
                                vy  = vy_cmd,
                                vrz = float(fsm_cmd[2]),
                            )

                            _yaw = np.arctan2(
                                2.0 * (quat[3] * quat[2] + quat[0] * quat[1]),
                                1.0 - 2.0 * (quat[1]**2 + quat[2]**2))
                            _c, _s = np.cos(_yaw), np.sin(_yaw)
                            robot_twist_msg = TwistStamped()
                            robot_twist_msg.twist.linear.x  = float(_c * vx_cmd - _s * vy_cmd)
                            robot_twist_msg.twist.linear.y  = float(_s * vx_cmd + _c * vy_cmd)
                            robot_twist_msg.twist.angular.z = float(fsm_cmd[2])

                            self._metrics.log_frame(
                                human_state = self._latest_human_state,
                                robot_pose  = robot_pose,
                                robot_twist = robot_twist_msg,
                                gait        = self.mapper.current_gait,
                            )

                            twist_msg           = Twist()
                            twist_msg.linear.x  = float(fsm_cmd[0])
                            twist_msg.linear.y  = float(fsm_cmd[1])
                            twist_msg.angular.z = float(fsm_cmd[2])
                            self._cmd_vel_pub.publish(twist_msg)
                            self._last_cmd_vel = fsm_cmd
                            self._freq_pd.tick()

                elif not _stopped:
                    CtrlInterface.soft_stop()
                    self.get_logger().info('Mocap stream ended — soft stop sent')
                    _stopped = True
                    rclpy.shutdown()

            elapsed   = time.monotonic() - _t0
            remaining = _target_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

    # ── FSM logic ─────────────────────────────────────────────────────────────

    def _fsm_update(
        self,
        robot_pos: np.ndarray,   # [x, y]
        robot_yaw: float,
        vel_body:  np.ndarray,   # [vx, vy] body frame
        u_safe:    np.ndarray,   # CBF-filtered [vx, vy]
        cmd_vel:   np.ndarray,   # full [vx, vy, vrz] from mapper
    ) -> np.ndarray:             # [vx_body, vy_body, vrz] to send
        """Run one FSM tick and return the velocity command to execute."""

        # FSM disabled — pure imitation.
        if self._feasibility is None:
            return np.array([u_safe[0], u_safe[1], cmd_vel[2]])

        human_xy  = self.mapper.human_pos[:2].copy()
        obstacles = self._obstacles

        # ── Evaluate Q ───────────────────────────────────────────────────────
        q = self._feasibility.q_value(
            robot_pos  = robot_pos,
            robot_yaw  = robot_yaw,
            vel_body   = vel_body,
            goal_xy    = human_xy,
            obstacles  = obstacles,
            u_safe     = u_safe,
        )

        dist_to_human = np.linalg.norm(human_xy - robot_pos)

        # ── State transitions ────────────────────────────────────────────────
        prev_mode = self._fsm_mode

        if self._fsm_mode == TeleopMode.IMITATE:
            if q < _Q_LOW:
                self._fsm_low_q_count += 1
                if self._fsm_low_q_count >= _FSM_CONFIRM_FRAMES:
                    self._fsm_mode        = TeleopMode.PLAN
                    self._fsm_low_q_count = 0
                    self._mppi.reset()
                    self.get_logger().info(
                        f'FSM IMITATE→PLAN  Q={q:.2f}')
            else:
                self._fsm_low_q_count = 0

        elif self._fsm_mode == TeleopMode.PLAN:
            if q > _Q_HIGH:
                self._fsm_mode = TeleopMode.RESYNC
                self.get_logger().info(
                    f'FSM PLAN→RESYNC  Q={q:.2f}')

        elif self._fsm_mode == TeleopMode.RESYNC:
            if dist_to_human < _RESYNC_DIST_M:
                self._fsm_mode        = TeleopMode.IMITATE
                self._fsm_low_q_count = 0
                self.get_logger().info(
                    f'FSM RESYNC→IMITATE  dist={dist_to_human:.2f} m')
            elif q < _Q_LOW:
                self._fsm_mode = TeleopMode.PLAN
                self.get_logger().info(
                    f'FSM RESYNC→PLAN  Q={q:.2f}')

        if prev_mode != self._fsm_mode:
            self._mode_pub.publish(String(data=self._fsm_mode.value))

        # ── Generate command ─────────────────────────────────────────────────
        if self._fsm_mode == TeleopMode.IMITATE:
            return np.array([u_safe[0], u_safe[1], cmd_vel[2]])

        # PLAN or RESYNC: use MPPI.
        mppi_cmd = self._mppi.compute(
            robot_pos = robot_pos,
            robot_yaw = robot_yaw,
            goal_xy   = human_xy,
            grid      = self._costmap,   # None is safe — obstacle term skipped
        )

        # Still run CBF on the MPPI output for safety.
        if self._cbf_filter is not None:
            mp_safe = self._cbf_filter.filter(
                u_perf_body = np.array(mppi_cmd[:2], dtype=np.float64),
                robot_pos   = robot_pos,
                robot_yaw   = robot_yaw,
                obstacles   = obstacles,
            )
            return np.array([mp_safe[0], mp_safe[1], mppi_cmd[2]])

        return mppi_cmd

    # ── Frequency logging (1 Hz) ──────────────────────────────────────────────

    def _log_freq_cb(self) -> None:
        mocap_hz      = self._freq_mocap.hz
        classifier_hz = self._freq_classifier.hz
        pd_hz         = self._freq_pd.hz

        self._metrics.log_frequency(
            mocap_hz      = mocap_hz,
            classifier_hz = classifier_hz,
            pd_hz         = pd_hz,
        )

        gait_str = (f'{self._gait_prediction} ({self._gait_confidence:.2f})'
                    if self._gait_prediction else 'buffering')
        cmd_str = (f'vx={self._last_cmd_vel[0]:.3f} vy={self._last_cmd_vel[1]:.3f} vrz={self._last_cmd_vel[2]:.3f}'
                   if self._last_cmd_vel is not None else 'none')
        cbf_str = ''
        if self._cbf_filter is not None:
            stats   = self._cbf_filter.reset_counters()
            cbf_str = (f'  cbf={stats["filter_pct"]:.0f}%active'
                       f'  obs={len(self._obstacles)}')
        fsm_str = (f'  fsm={self._fsm_mode.value}'
                   if self._feasibility is not None else '')
        self.get_logger().info(
            f'[freq] mocap={mocap_hz:.1f} Hz  '
            f'classifier={classifier_hz:.1f} Hz  '
            f'pd_ctrl={pd_hz:.1f} Hz  '
            f'gait={gait_str}  cmd={cmd_str}{cbf_str}{fsm_str}'
        )

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        self.get_logger().info('Shutting down — stopping robot')
        self._running = False
        self._buf_thread.join(timeout=1.0)
        self._cls_thread.join(timeout=1.0)
        self._pd_thread.join(timeout=1.0)
        CtrlInterface.hard_stop()
        self._metrics.plot_metrics()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
