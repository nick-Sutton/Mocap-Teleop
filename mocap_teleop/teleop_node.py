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

PD thread          — Python thread at 1000 Hz using time.sleep().
"""

import os
import threading
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
from geometry_msgs.msg import Twist, PoseStamped
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

_PD_RATE_HZ          = 1000.0
_CLASSIFIER_RATE_HZ  = 20.0     # cap at realistic CPU inference rate
_LOG_RATE_HZ         = 1.0
_MOCAP_TIMEOUT_S     = 0.5      # stop robot if no mocap for 500 ms


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
        # Written by the subscriber callback (ROS2 executor thread) and read by
        # the PD thread — list replacement is GIL-atomic in CPython.
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
        # Obstacles published as a MarkerArray of SPHERE markers.
        # Each marker: position = obstacle center (odom frame),
        #              scale.x  = obstacle diameter (radius = scale.x / 2).
        # Publish to /obstacles from your perception node.
        if self._cbf_filter is not None:
            self.create_subscription(
                MarkerArray, '/obstacles',
                self._obstacles_cb, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self._cmd_vel_pub = self.create_publisher(Twist,  '/cmd_vel',     10)
        self._gait_pub    = self.create_publisher(String, '/teleop/gait', 10)

        # ── Timers ────────────────────────────────────────────────────────────
        self.create_timer(1.0 / _LOG_RATE_HZ, self._log_freq_cb)

        # ── Background threads — all heavy work runs here, not in executor ─────
        # Buffer-fill: extract features + update TCN window at mocap rate.
        self._buf_thread = threading.Thread(
            target=self._buffer_fill_loop, name='buf_fill', daemon=True)
        self._buf_thread.start()
        # Classifier: TCN inference at ~13 Hz (limited by 75ms forward pass).
        self._cls_thread = threading.Thread(
            target=self._classifier_loop, name='classifier', daemon=True)
        self._cls_thread.start()
        # PD: 1000 Hz control loop.
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

        This is the standard format from most ROS2 obstacle detection packages.
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
        """Runs TCN inference as fast as the model allows (target 20 Hz).

        Runs in a Python thread — NOT via ROS2 executor — so inference never
        blocks subscriber callbacks.  Publishers are thread-safe in ROS2.
        """
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

    # ── PD control loop (Python thread, 1000 Hz) ──────────────────────────────

    def _pd_loop(self) -> None:
        """Runs in a dedicated thread — not managed by the ROS2 executor."""
        dt = 1.0 / _PD_RATE_HZ
        _telemetry_warned = False
        _stopped = False   # track whether soft_stop has already been sent

        while self._running:
            t0 = time.monotonic()

            if self._latest_human_state is not None:
                if time.monotonic() - self._last_mocap_time <= _MOCAP_TIMEOUT_S:
                    _stopped = False   # mocap is live again — re-arm stop

                    # During classifier warm-up (buffer not full) keep the robot
                    # standing by sending zero velocity — same as old system.
                    if self._gait_prediction is None:
                        CtrlInterface.walk(vx=0, vy=0, vrz=0)
                    else:
                        try:
                            pos  = CtrlInterface.get_robot_position()
                            quat = CtrlInterface.get_robot_orientation()
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

                            cmd_vel = self.mapper.select_robot_command(
                                gait_prediction = self._gait_prediction,
                                confidence      = self._gait_confidence,
                                robot_pose      = robot_pose,
                                root_pose       = self._latest_human_state.root_pose,
                            )

                            # ── CBF safety filter ─────────────────────────────
                            # Intercept the translational command and deflect
                            # the robot around any detected obstacles before
                            # sending to the hardware.  vrz passes through.
                            if self._cbf_filter is not None:
                                from scipy.spatial.transform import Rotation as _R
                                yaw = _R.from_quat([
                                    float(quat[0]), float(quat[1]),
                                    float(quat[2]), float(quat[3]),
                                ]).as_euler('xyz')[2]
                                u_safe = self._cbf_filter.filter(
                                    u_perf_body = np.array(cmd_vel[:2],
                                                           dtype=np.float64),
                                    robot_pos   = np.array(pos[:2],
                                                           dtype=np.float64),
                                    robot_yaw   = float(yaw),
                                    obstacles   = self._obstacles,
                                )
                                vx_cmd, vy_cmd = float(u_safe[0]), float(u_safe[1])
                            else:
                                vx_cmd, vy_cmd = float(cmd_vel[0]), float(cmd_vel[1])

                            CtrlInterface.walk(
                                vx  = vx_cmd,
                                vy  = vy_cmd,
                                vrz = float(cmd_vel[2]),
                            )

                            self._metrics.log_frame(
                                human_state = self._latest_human_state,
                                robot_pose  = robot_pose,
                                gait        = self._gait_prediction,
                            )

                            twist_msg           = Twist()
                            twist_msg.linear.x  = float(cmd_vel[0])
                            twist_msg.linear.y  = float(cmd_vel[1])
                            twist_msg.angular.z = float(cmd_vel[2])
                            self._cmd_vel_pub.publish(twist_msg)
                            self._last_cmd_vel = cmd_vel
                            self._freq_pd.tick()
                elif not _stopped:
                    CtrlInterface.soft_stop()
                    self.get_logger().info('Mocap stream ended — soft stop sent')
                    _stopped = True
                    rclpy.shutdown()

            elapsed = time.monotonic() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

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
        self.get_logger().info(
            f'[freq] mocap={mocap_hz:.1f} Hz  '
            f'classifier={classifier_hz:.1f} Hz  '
            f'pd_ctrl={pd_hz:.1f} Hz  '
            f'gait={gait_str}  cmd={cmd_str}{cbf_str}'
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
    # SingleThreadedExecutor: only trivial callbacks remain in the executor
    # (subscriber + 1 Hz log timer).  All heavy work is in Python threads.
    # This eliminates MultiThreadedExecutor mutex contention that was limiting
    # the subscriber to ~17 Hz instead of 240 Hz.
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
