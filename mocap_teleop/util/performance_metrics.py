"""
performance_metrics.py — Per-frame logging and end-of-run analysis.

Accepts ROS2 message types directly at log time rather than holding live
references to mutable objects. All internal Pose/Twist references have been
replaced with geometry_msgs types.

Usage (from teleop_node or a dedicated metrics node)
─────────────────────────────────────────────────────
    metrics = PerformanceMetrics(
        coordinate_offset = mapper.coordinate_offset,
        logs_dir          = './log',
    )

    # Each control frame:
    metrics.log_frame(
        human_state  = latest_human_state,    # HumanState msg
        robot_pose   = robot_pose_stamped,     # PoseStamped from odom
        robot_twist  = robot_twist_stamped,    # TwistStamped from odom (optional)
        gait         = gait_prediction,
    )

    # End of run:
    metrics.plot_metrics()
"""

import csv
import os
import time
import yaml
from datetime import datetime

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import PoseStamped, TwistStamped
from mocap_teleop_msgs.msg import HumanState


# ── Gait color palette ────────────────────────────────────────────────────────

GAIT_COLORS = {
    'stand': '#E74C3C',
    'walk':  '#2ECC71',
    'jog':   '#3498DB',
}
DEFAULT_GAIT_COLOR = '#95A5A6'


def _gait_legend_patches():
    return [
        mpatches.Patch(color=GAIT_COLORS['stand'], label='Stand'),
        mpatches.Patch(color=GAIT_COLORS['walk'],  label='Walk'),
        mpatches.Patch(color=GAIT_COLORS['jog'],   label='Jog'),
    ]


def _plot_colored_segments(ax, x, y, gait_at_x, linewidth=2, alpha=0.9):
    x     = np.asarray(x)
    y     = np.asarray(y)
    gaits = np.asarray(gait_at_x)
    if len(gaits) != len(x):
        gaits = np.full(len(x), 'unknown')
    i = 0
    while i < len(x):
        current_gait = gaits[i]
        j = i + 1
        while j < len(x) and gaits[j] == current_gait:
            j += 1
        color = GAIT_COLORS.get(current_gait, DEFAULT_GAIT_COLOR)
        if j - i >= 1:
            ax.plot(x[i:j], y[i:j], color=color,
                    linewidth=linewidth, alpha=alpha)
        i = j


def _build_gait_lookup(gait_history: dict):
    if not gait_history:
        return lambda t: 'unknown'
    sorted_times = np.array(sorted(gait_history.keys()))
    sorted_gaits = [gait_history[t] for t in sorted_times]

    def lookup(t):
        idx = np.searchsorted(sorted_times, t, side='right') - 1
        idx = max(0, min(idx, len(sorted_gaits) - 1))
        return sorted_gaits[idx]
    return lookup


# ── Helpers to unpack ROS2 messages ──────────────────────────────────────────

def _pos(pose_stamped: PoseStamped) -> np.ndarray:
    p = pose_stamped.pose.position
    return np.array([p.x, p.y, p.z])

def _quat(pose_stamped: PoseStamped) -> np.ndarray:
    o = pose_stamped.pose.orientation
    return np.array([o.x, o.y, o.z, o.w])

def _lin_vel(twist_stamped: TwistStamped) -> np.ndarray:
    v = twist_stamped.twist.linear
    return np.array([v.x, v.y, v.z])

def _ang_vel(twist_stamped: TwistStamped) -> np.ndarray:
    w = twist_stamped.twist.angular
    return np.array([w.x, w.y, w.z])

def _quat_to_yaw(quat: np.ndarray) -> float:
    qx, qy, qz, qw = quat
    return float(np.arctan2(2.0 * (qw * qz + qx * qy),
                             1.0 - 2.0 * (qy**2 + qz**2)))


# ── Main class ────────────────────────────────────────────────────────────────

class PerformanceMetrics:
    def __init__(
        self,
        coordinate_offset: np.ndarray | None = None,
        logs_dir:          str               = './log',
        dt:                float             = 1.0 / 240.0,
        nominal_rates:     dict | None       = None,
    ):
        self.dt               = dt
        # Store a live reference so that when mapper.coordinate_offset[:] is
        # updated on the first control frame, metrics sees the correct value.
        self.coordinate_offset = (coordinate_offset
                                  if coordinate_offset is not None
                                  else np.zeros(3))
        self.logs_dir         = logs_dir
        self.start_time       = time.time()
        self.frame_count      = 0

        # Nominal component rates used to compute drift statistics
        _defaults = {'mocap': 240.0, 'classifier': 240.0, 'pd': 1000.0}
        self.nominal_rates: dict[str, float] = {**_defaults, **(nominal_rates or {})}

        # ── Per-frame error lists ─────────────────────────────────────────────
        self.distance_errors: list[float] = []
        self.velocity_errors: list[float] = []
        self.heading_errors:  list[float] = []
        self.error_times:     list[float] = []   # wall-clock seconds from start

        # ── Trajectory data for plots ─────────────────────────────────────────
        self.human_positions:  list[np.ndarray] = []   # [x, y] offset-corrected
        self.robot_positions:  list[np.ndarray] = []   # [x, y]
        self.lfoot_positions:  list[np.ndarray] = []   # [x, y, z]
        self.rfoot_positions:  list[np.ndarray] = []   # [x, y, z]
        self.lfoot_velocities: list[np.ndarray] = []   # [vx, vy, vz]
        self.rfoot_velocities: list[np.ndarray] = []   # [vx, vy, vz]

        # ── Gait tracking ─────────────────────────────────────────────────────
        self.gait_history:    dict[float, str]          = {}
        self.gait_transitions: list[tuple]              = []
        self._current_gait:   str | None                = None

        # ── Starting positions (set on first frame) ───────────────────────────
        self._human_start: np.ndarray | None = None
        self._robot_start: np.ndarray | None = None

        # ── Output paths ──────────────────────────────────────────────────────
        os.makedirs(logs_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._run_dir = os.path.join(logs_dir, f"run_{ts}")
        os.makedirs(self._run_dir, exist_ok=True)

        self._csv_path      = os.path.join(self._run_dir, f"metrics_{ts}.csv")
        self._summary_path  = os.path.join(self._run_dir, f"summary_{ts}.txt")
        self._stats_path    = os.path.join(self._run_dir, f"statistics_{ts}.yaml")
        self._freq_csv_path = os.path.join(self._run_dir, f"frequency_{ts}.csv")

        # Frequency samples — one entry per log_frequency() call (typically 1 Hz)
        self._freq_times:      list[float] = []
        self._freq_mocap:      list[float] = []
        self._freq_classifier: list[float] = []
        self._freq_pd:         list[float] = []

        self._init_csv()
        self._init_freq_csv()

    # ── CSV ───────────────────────────────────────────────────────────────────

    def _init_csv(self):
        headers = [
            'time_s', 'frame',
            'position_error_m', 'x_error_m', 'y_error_m',
            'velocity_error_ms',
            'heading_error_deg',
            'gait',
        ]
        with open(self._csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=headers).writeheader()

    def _init_freq_csv(self) -> None:
        headers = ['time_s', 'mocap_hz', 'classifier_hz', 'pd_ctrl_hz']
        with open(self._freq_csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=headers).writeheader()

    # ── Frequency logging ─────────────────────────────────────────────────────

    def log_frequency(
        self,
        mocap_hz:      float,
        classifier_hz: float,
        pd_hz:         float,
    ) -> None:
        """
        Record one frequency sample for each pipeline component.

        Call this at a fixed rate (e.g. 1 Hz) throughout the run.  The sample
        is appended to the in-memory lists and written to frequency_{ts}.csv
        immediately so data is preserved even if the run is interrupted.

        Parameters
        ----------
        mocap_hz      : measured rate of incoming mocap frames (Hz)
        classifier_hz : measured rate of gait classifier calls (Hz)
        pd_hz         : measured rate of PD controller calls (Hz)
        """
        t = time.time() - self.start_time
        self._freq_times.append(t)
        self._freq_mocap.append(mocap_hz)
        self._freq_classifier.append(classifier_hz)
        self._freq_pd.append(pd_hz)

        row = {
            'time_s':        round(t, 3),
            'mocap_hz':      round(mocap_hz, 2),
            'classifier_hz': round(classifier_hz, 2),
            'pd_ctrl_hz':    round(pd_hz, 2),
        }
        with open(self._freq_csv_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=row.keys()).writerow(row)

    # ── Per-frame logging ─────────────────────────────────────────────────────

    def log_frame(
        self,
        human_state:  HumanState,
        robot_pose:   PoseStamped,
        robot_twist:  TwistStamped | None = None,
        gait:         str | None          = None,
    ) -> None:
        """
        Log one control frame. Call this every iteration of the control loop.

        Parameters
        ----------
        human_state : latest HumanState message from mocap
        robot_pose  : robot PoseStamped from odometry
        robot_twist : robot TwistStamped from odometry (used for velocity error)
        gait        : active gait label string
        """
        t = time.time() - self.start_time
        self.frame_count += 1

        # ── Positions ─────────────────────────────────────────────────────────
        raw_human_xy = _pos(human_state.root_pose)[:2]
        human_xy     = raw_human_xy + self.coordinate_offset[:2]
        robot_xy     = _pos(robot_pose)[:2]

        # Initialise starting positions on first frame
        if self._human_start is None:
            self._human_start = raw_human_xy.copy()
            self._robot_start = robot_xy.copy()

        self.human_positions.append(human_xy)
        self.robot_positions.append(robot_xy)

        lfoot = _pos(human_state.lfoot_pose)
        rfoot = _pos(human_state.rfoot_pose)
        self.lfoot_positions.append(lfoot)
        self.rfoot_positions.append(rfoot)
        self.lfoot_velocities.append(_lin_vel(human_state.lfoot_twist))
        self.rfoot_velocities.append(_lin_vel(human_state.rfoot_twist))

        # ── Position error ────────────────────────────────────────────────────
        pos_diff  = human_xy - robot_xy
        pos_error = float(np.linalg.norm(pos_diff))
        self.distance_errors.append(pos_error)

        # ── Heading error ─────────────────────────────────────────────────────
        raw_human_yaw = _quat_to_yaw(_quat(human_state.root_pose))
        human_yaw     = float(np.arctan2(
            np.sin(raw_human_yaw + self.coordinate_offset[2]),
            np.cos(raw_human_yaw + self.coordinate_offset[2])))
        robot_yaw     = _quat_to_yaw(_quat(robot_pose))
        heading_error = abs(float(np.arctan2(
            np.sin(human_yaw - robot_yaw),
            np.cos(human_yaw - robot_yaw))))
        self.heading_errors.append(heading_error)

        # ── Velocity error ────────────────────────────────────────────────────
        human_vel = _lin_vel(human_state.root_twist)
        if robot_twist is not None:
            robot_vel = _lin_vel(robot_twist)
        else:
            robot_vel = np.zeros(3)
        vel_error = float(np.linalg.norm(human_vel - robot_vel))
        self.velocity_errors.append(vel_error)

        self.error_times.append(t)

        # ── Gait recording ────────────────────────────────────────────────────
        if gait is not None:
            self._record_gait(t, gait)

        # ── CSV row ───────────────────────────────────────────────────────────
        row = {
            'time_s':           round(t, 4),
            'frame':            self.frame_count,
            'position_error_m': round(pos_error, 4),
            'x_error_m':        round(pos_diff[0], 4),
            'y_error_m':        round(pos_diff[1], 4),
            'velocity_error_ms': round(vel_error, 4),
            'heading_error_deg': round(np.degrees(heading_error), 3),
            'gait':             gait or '',
        }
        with open(self._csv_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=row.keys()).writerow(row)

    # ── Gait tracking ─────────────────────────────────────────────────────────

    def _record_gait(self, t: float, gait: str) -> None:
        if self._current_gait is None:
            self._current_gait = gait
        elif gait != self._current_gait:
            self.gait_transitions.append((t, self._current_gait, gait))
            self._current_gait = gait
        self.gait_history[t] = gait

    # ── Summary statistics ────────────────────────────────────────────────────

    def compute_summary_statistics(self) -> dict:
        if not self.distance_errors:
            return {'overall': {'total_frames': 0}}

        pos  = np.array(self.distance_errors)
        vel  = np.array(self.velocity_errors)
        head = np.array(self.heading_errors)

        def rmse(a): return float(np.sqrt(np.mean(a**2)))
        def pct(a, thresh): return float(np.mean(a <= thresh) * 100)

        stats = {
            'overall': {
                'duration_seconds':     float(self.frame_count * self.dt),
                'total_frames':         self.frame_count,
                'num_gait_transitions': len(self.gait_transitions),
            },
            'position_tracking': {
                'mean_m':             float(np.mean(pos)),
                'std_m':              float(np.std(pos)),
                'median_m':           float(np.median(pos)),
                'max_m':              float(np.max(pos)),
                'rmse_m':             rmse(pos),
                'p95_m':              float(np.percentile(pos, 95)),
                'p99_m':              float(np.percentile(pos, 99)),
                'accuracy_5cm_pct':   pct(pos, 0.05),
                'accuracy_10cm_pct':  pct(pos, 0.10),
                'accuracy_20cm_pct':  pct(pos, 0.20),
            },
            'velocity_tracking': {
                'mean_ms':            float(np.mean(vel)),
                'std_ms':             float(np.std(vel)),
                'median_ms':          float(np.median(vel)),
                'max_ms':             float(np.max(vel)),
                'rmse_ms':            rmse(vel),
                'accuracy_01ms_pct':  pct(vel, 0.1),
                'accuracy_02ms_pct':  pct(vel, 0.2),
                'accuracy_05ms_pct':  pct(vel, 0.5),
            },
            'heading_tracking': {
                'mean_deg':           float(np.degrees(np.mean(head))),
                'std_deg':            float(np.degrees(np.std(head))),
                'max_deg':            float(np.degrees(np.max(head))),
                'rmse_deg':           float(np.degrees(rmse(head))),
                'accuracy_5deg_pct':  pct(head, np.radians(5)),
                'accuracy_10deg_pct': pct(head, np.radians(10)),
                'accuracy_20deg_pct': pct(head, np.radians(20)),
            },
            'gait_transitions': [
                {'time_s': float(t), 'from': old, 'to': new}
                for t, old, new in self.gait_transitions
            ],
        }

        # Per-gait breakdown
        gait_lookup  = _build_gait_lookup(self.gait_history)
        ts           = np.array(self.error_times)
        gait_arr     = np.array([gait_lookup(t) for t in ts])
        per_gait     = {}
        for g in ['stand', 'walk', 'jog']:
            mask = gait_arr == g
            if not np.any(mask):
                continue
            pg = pos[mask]; vg = vel[mask]; hg = head[mask]
            per_gait[g] = {
                'frames':            int(np.sum(mask)),
                'duration_s':        float(np.sum(mask) * self.dt),
                'position_rmse_m':   rmse(pg),
                'position_mean_m':   float(np.mean(pg)),
                'velocity_rmse_ms':  rmse(vg),
                'heading_rmse_deg':  float(np.degrees(rmse(hg))),
                'pos_10cm_pct':      pct(pg, 0.10),
            }
        stats['per_gait'] = per_gait

        # ── Component frequency statistics ────────────────────────────────────
        if self._freq_times:
            def _freq_stats(samples: list[float], nominal: float) -> dict:
                arr = np.array(samples)
                tol = 0.05 * nominal   # ±5 % of nominal
                return {
                    'nominal_hz':      nominal,
                    'mean_hz':         float(np.mean(arr)),
                    'std_hz':          float(np.std(arr)),
                    'min_hz':          float(np.min(arr)),
                    'max_hz':          float(np.max(arr)),
                    'within_5pct':     float(np.mean(np.abs(arr - nominal) <= tol) * 100),
                }

            stats['component_frequency'] = {
                'mocap':      _freq_stats(self._freq_mocap,      self.nominal_rates['mocap']),
                'classifier': _freq_stats(self._freq_classifier, self.nominal_rates['classifier']),
                'pd_ctrl':    _freq_stats(self._freq_pd,         self.nominal_rates['pd']),
                'num_samples': len(self._freq_times),
                'duration_s':  float(self._freq_times[-1]),
            }

        return stats

    # ── Save summary text ─────────────────────────────────────────────────────

    def save_summary(self) -> None:
        stats = self.compute_summary_statistics()
        if stats['overall'].get('total_frames', 0) == 0:
            print("[METRICS] No data to save")
            return

        with open(self._stats_path, 'w') as f:
            yaml.dump(stats, f, default_flow_style=False, sort_keys=False)

        with open(self._summary_path, 'w') as f:
            def w(s=''): f.write(s + '\n')
            w('=' * 80)
            w('TELEOP PERFORMANCE METRICS SUMMARY')
            w('=' * 80); w()

            o = stats['overall']
            w('OVERALL:'); w('-' * 40)
            w(f"  Duration:          {o['duration_seconds']:.2f} s")
            w(f"  Frames:            {o['total_frames']}")
            w(f"  Gait transitions:  {o['num_gait_transitions']}"); w()

            p = stats['position_tracking']
            w('POSITION:'); w('-' * 40)
            w(f"  Mean ± Std:  {p['mean_m']*100:.2f} ± {p['std_m']*100:.2f} cm")
            w(f"  RMSE:        {p['rmse_m']*100:.2f} cm")
            w(f"  Median:      {p['median_m']*100:.2f} cm")
            w(f"  Max:         {p['max_m']*100:.2f} cm")
            w(f"  ≤ 5 cm:      {p['accuracy_5cm_pct']:.1f}%")
            w(f"  ≤ 10 cm:     {p['accuracy_10cm_pct']:.1f}%")
            w(f"  ≤ 20 cm:     {p['accuracy_20cm_pct']:.1f}%"); w()

            v = stats['velocity_tracking']
            w('VELOCITY:'); w('-' * 40)
            w(f"  Mean ± Std:  {v['mean_ms']:.3f} ± {v['std_ms']:.3f} m/s")
            w(f"  RMSE:        {v['rmse_ms']:.3f} m/s")
            w(f"  ≤ 0.1 m/s:   {v['accuracy_01ms_pct']:.1f}%")
            w(f"  ≤ 0.2 m/s:   {v['accuracy_02ms_pct']:.1f}%"); w()

            h = stats['heading_tracking']
            w('HEADING:'); w('-' * 40)
            w(f"  Mean ± Std:  {h['mean_deg']:.2f} ± {h['std_deg']:.2f}°")
            w(f"  RMSE:        {h['rmse_deg']:.2f}°")
            w(f"  ≤ 5°:        {h['accuracy_5deg_pct']:.1f}%")
            w(f"  ≤ 10°:       {h['accuracy_10deg_pct']:.1f}%"); w()

            if stats.get('per_gait'):
                w('PER-GAIT:'); w('-' * 40)
                for g, gd in stats['per_gait'].items():
                    w(f"  {g.upper()} ({gd['duration_s']:.1f}s, {gd['frames']} frames)")
                    w(f"    Pos RMSE: {gd['position_rmse_m']*100:.2f} cm  "
                      f"Vel RMSE: {gd['velocity_rmse_ms']:.3f} m/s  "
                      f"Head RMSE: {gd['heading_rmse_deg']:.2f}°")
                    w(f"    ≤10cm: {gd['pos_10cm_pct']:.1f}%")
                w()

            if stats['gait_transitions']:
                w('TRANSITIONS:'); w('-' * 40)
                for tr in stats['gait_transitions']:
                    w(f"  t={tr['time_s']:.2f}s: {tr['from']} → {tr['to']}")

            if stats.get('component_frequency'):
                w(); w('COMPONENT FREQUENCY:'); w('-' * 40)
                cf = stats['component_frequency']
                w(f"  Samples logged: {cf['num_samples']}  "
                  f"(over {cf['duration_s']:.1f} s)")
                for key, label in [('mocap', 'Mocap'),
                                   ('classifier', 'Classifier'),
                                   ('pd_ctrl', 'PD Ctrl')]:
                    s = cf[key]
                    w(f"  {label:<12} nominal={s['nominal_hz']:.0f} Hz  "
                      f"mean={s['mean_hz']:.1f}  std={s['std_hz']:.1f}  "
                      f"min={s['min_hz']:.1f}  max={s['max_hz']:.1f}  "
                      f"within±5%={s['within_5pct']:.1f}%")

            w(); w('=' * 80)

        print(f"[METRICS] Saved summary:    {self._summary_path}")
        print(f"[METRICS] Saved statistics: {self._stats_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────

    def plot_metrics(self) -> None:
        plt.close('all')
        self._plot_position()
        self._plot_tracking()
        self._plot_gait_features()
        self._plot_frequencies()
        self.save_summary()
        self._print_quick_summary()

    def _plot_position(self) -> None:
        if not self.distance_errors:
            return

        ts   = np.array(self.error_times)
        pos  = np.array(self.distance_errors)
        hpos = np.array(self.human_positions)   # [N, 2]
        rpos = np.array(self.robot_positions)   # [N, 2]

        gait_lookup = _build_gait_lookup(self.gait_history)
        gait_arr    = np.array([gait_lookup(t) for t in ts])

        # Displacements from start for path plot
        h_disp = hpos - hpos[0]
        r_disp = rpos - rpos[0]

        src_mag = np.linalg.norm(h_disp, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_err = np.where(src_mag > 1e-6, (pos / src_mag) * 100, 0)

        patches = _gait_legend_patches()
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        ax = axes[0, 0]
        _plot_colored_segments(ax, ts, pos * 100, gait_arr)
        ax.set(xlabel='Time (s)', ylabel='Position Error (cm)',
               title='Total Position Error')
        ax.legend(handles=patches, fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        _plot_colored_segments(ax, ts, (hpos - rpos)[:, 0] * 100, gait_arr, alpha=0.9)
        _plot_colored_segments(ax, ts, (hpos - rpos)[:, 1] * 100, gait_arr, alpha=0.45)
        ax.set(xlabel='Time (s)', ylabel='Error (cm)', title='Position Error by Axis')
        ax.legend(handles=patches, fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        _plot_colored_segments(ax, ts, h_disp[:, 0], gait_arr, alpha=0.9)
        _plot_colored_segments(ax, ts, h_disp[:, 1], gait_arr, alpha=0.45)
        ax.set(xlabel='Time (s)', ylabel='Displacement (m)', title='Human Displacement')
        ax.legend(handles=patches, fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        _plot_colored_segments(ax, ts, r_disp[:, 0], gait_arr, alpha=0.9)
        _plot_colored_segments(ax, ts, r_disp[:, 1], gait_arr, alpha=0.45)
        ax.set(xlabel='Time (s)', ylabel='Displacement (m)', title='Robot Displacement')
        ax.legend(handles=patches, fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[2, 0]
        _plot_colored_segments(ax, h_disp[:, 0], h_disp[:, 1], gait_arr, linewidth=2)
        ax.plot(r_disp[:, 0], r_disp[:, 1], color='#2C3E50', lw=2, alpha=0.7)
        ax.plot(*h_disp[0], 'o', color='#E74C3C', ms=10)
        ax.plot(*h_disp[-1], '^', color='#E74C3C', ms=10)
        ax.plot(*r_disp[0], 'o', color='#2C3E50', ms=10)
        ax.plot(*r_disp[-1], '^', color='#2C3E50', ms=10)
        from matplotlib.lines import Line2D
        ax.set(xlabel='X (m)', ylabel='Y (m)', title='2D Path Comparison')
        ax.legend(handles=patches + [Line2D([0],[0], color='#2C3E50', lw=2,
                  label='Robot')], fontsize=8)
        ax.grid(True, alpha=0.3); ax.axis('equal')

        ax = axes[2, 1]
        _plot_colored_segments(ax, ts, pct_err, gait_arr)
        ax.set(xlabel='Time (s)', ylabel='% Error',
               title='Position Error as % of Human Movement')
        ax.legend(handles=patches, fontsize=8); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self._run_dir, 'position_plots.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[PLOT] {path}")

    def _plot_tracking(self) -> None:
        if not self.distance_errors:
            return

        ts       = np.array(self.error_times)
        pos      = np.array(self.distance_errors)
        vel      = np.array(self.velocity_errors)
        head_deg = np.degrees(np.array(self.heading_errors))

        gait_lookup = _build_gait_lookup(self.gait_history)
        gait_arr    = np.array([gait_lookup(t) for t in ts])

        window = max(1, int(2.0 / self.dt))

        def rolling_rmse(arr, w):
            out = np.full(len(arr), np.nan)
            for i in range(w - 1, len(arr)):
                out[i] = np.sqrt(np.mean(arr[i - w + 1:i + 1]**2))
            return out

        pos_roll  = rolling_rmse(pos,      window)
        vel_roll  = rolling_rmse(vel,      window)
        head_roll = rolling_rmse(head_deg, window)

        patches = _gait_legend_patches()
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Tracking Accuracy Metrics', fontsize=14, fontweight='bold')

        from matplotlib.lines import Line2D as L2D

        def rmse_legend(val, unit):
            return [L2D([0],[0], color='#2C3E50', lw=2, label='Rolling RMSE (2s)'),
                    L2D([0],[0], color='k', lw=1.5, ls='--',
                        label=f'Global RMSE = {val:.3g} {unit}')]

        ax = axes[0, 0]
        _plot_colored_segments(ax, ts, pos * 100, gait_arr, alpha=0.35)
        ax.plot(ts, pos_roll * 100, color='#2C3E50', lw=2, zorder=5)
        g = float(np.sqrt(np.mean(pos**2)))
        ax.axhline(g * 100, color='k', ls='--', lw=1.5)
        ax.set(xlabel='Time (s)', ylabel='Error (cm)', title='Position Error & Rolling RMSE')
        ax.legend(handles=patches + rmse_legend(g * 100, 'cm'), fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        _plot_colored_segments(ax, ts, vel, gait_arr, alpha=0.35)
        ax.plot(ts, vel_roll, color='#2C3E50', lw=2, zorder=5)
        g = float(np.sqrt(np.mean(vel**2)))
        ax.axhline(g, color='k', ls='--', lw=1.5)
        ax.set(xlabel='Time (s)', ylabel='Error (m/s)', title='Velocity Error & Rolling RMSE')
        ax.legend(handles=patches + rmse_legend(g, 'm/s'), fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        _plot_colored_segments(ax, ts, head_deg, gait_arr, alpha=0.35)
        ax.plot(ts, head_roll, color='#2C3E50', lw=2, zorder=5)
        g = float(np.degrees(np.sqrt(np.mean(np.radians(head_deg)**2))))
        ax.axhline(g, color='k', ls='--', lw=1.5)
        ax.set(xlabel='Time (s)', ylabel='Error (°)', title='Heading Error & Rolling RMSE')
        ax.legend(handles=patches + rmse_legend(g, '°'), fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        def _norm(a): return a / (np.nanmax(a) + 1e-9)
        ax.plot(ts, _norm(pos_roll),  color='#E74C3C', lw=2, label='Position (norm)')
        ax.plot(ts, _norm(vel_roll),  color='#2ECC71', lw=2, label='Velocity (norm)')
        ax.plot(ts, _norm(head_roll), color='#3498DB', lw=2, label='Heading (norm)')
        ax.set(xlabel='Time (s)', ylabel='Normalised RMSE',
               title='Relative Tracking Performance', ylim=(0, 1.1))
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[2, 0]
        sorted_pos = np.sort(pos) * 100
        cdf = np.arange(1, len(sorted_pos) + 1) / len(sorted_pos) * 100
        ax.plot(sorted_pos, cdf, color='#8E44AD', lw=2)
        for thr, col in [(5, '#E74C3C'), (10, '#E67E22'), (20, '#27AE60')]:
            p = float(np.mean(pos * 100 <= thr) * 100)
            ax.axvline(thr, color=col, ls='--', alpha=0.7, lw=1.5)
            ax.text(thr + 0.3, 8, f'{p:.0f}%\n≤{thr}cm', fontsize=8,
                    color=col, va='bottom')
        ax.set(xlabel='Position Error (cm)', ylabel='Cumulative % of Frames',
               title='CDF of Position Error')
        ax.grid(True, alpha=0.3)

        ax = axes[2, 1]
        unique_gaits = [g for g in ['stand', 'walk', 'jog']
                        if np.any(gait_arr == g)]
        box_data = [pos[gait_arr == g] * 100 for g in unique_gaits]
        if box_data:
            bp = ax.boxplot(box_data, patch_artist=True, labels=unique_gaits,
                            showfliers=True,
                            flierprops=dict(marker='o', ms=3, alpha=0.3))
            for patch, g in zip(bp['boxes'], unique_gaits):
                patch.set_facecolor(GAIT_COLORS.get(g, DEFAULT_GAIT_COLOR))
                patch.set_alpha(0.7)
        ax.set(xlabel='Gait', ylabel='Position Error (cm)',
               title='Position Error by Gait')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        path = os.path.join(self._run_dir, 'tracking_metrics.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[PLOT] {path}")

    def _plot_gait_features(self) -> None:
        if not self.lfoot_positions:
            return

        ts    = np.array(self.error_times)
        hpos  = np.array(self.human_positions)
        lf    = np.array(self.lfoot_positions)
        rf    = np.array(self.rfoot_positions)
        lfv   = np.array(self.lfoot_velocities)
        rfv   = np.array(self.rfoot_velocities)

        gait_lookup = _build_gait_lookup(self.gait_history)
        gait_arr    = np.array([gait_lookup(t) for t in ts])
        patches     = _gait_legend_patches()

        fig = plt.figure(figsize=(15, 12))

        ax = plt.subplot(4, 2, 1)
        _plot_colored_segments(ax, hpos[:, 0], hpos[:, 1], gait_arr, linewidth=2)
        ax.set(xlabel='X (m)', ylabel='Y (m)', title='Human Base Position')
        ax.legend(handles=patches, fontsize=8); ax.grid(True, alpha=0.3)

        ax = plt.subplot(4, 2, 2)
        _plot_colored_segments(ax, ts, lf[:, 2], gait_arr, alpha=0.9)
        _plot_colored_segments(ax, ts, rf[:, 2], gait_arr, alpha=0.45)
        ax.set(xlabel='Time (s)', ylabel='Z (m)', title='Foot Z Positions (L/R)')
        ax.legend(handles=patches, fontsize=8); ax.grid(True, alpha=0.3)

        ax = plt.subplot(4, 2, 3)
        _plot_colored_segments(ax, ts, lf[:, 2], gait_arr, linewidth=2)
        ax.set(xlabel='Time (s)', ylabel='Z (m)', title='Left Foot Z')
        ax.legend(handles=patches, fontsize=8); ax.grid(True, alpha=0.3)

        ax = plt.subplot(4, 2, 4)
        _plot_colored_segments(ax, ts, rf[:, 2], gait_arr, linewidth=2)
        ax.set(xlabel='Time (s)', ylabel='Z (m)', title='Right Foot Z')
        ax.legend(handles=patches, fontsize=8); ax.grid(True, alpha=0.3)

        ax = plt.subplot(4, 2, 5)
        for i, a in enumerate([0.9, 0.6, 0.35]):
            _plot_colored_segments(ax, ts, lfv[:, i], gait_arr, alpha=a)
        ax.set(xlabel='Time (s)', ylabel='Velocity (m/s)', title='Left Foot Linear Velocity')
        ax.legend(handles=patches, fontsize=8); ax.grid(True, alpha=0.3)

        ax = plt.subplot(4, 2, 6)
        for i, a in enumerate([0.9, 0.6, 0.35]):
            _plot_colored_segments(ax, ts, rfv[:, i], gait_arr, alpha=a)
        ax.set(xlabel='Time (s)', ylabel='Velocity (m/s)', title='Right Foot Linear Velocity')
        ax.legend(handles=patches, fontsize=8); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self._run_dir, 'gait_plots.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[PLOT] {path}")

    def _plot_frequencies(self) -> None:
        if not self._freq_times:
            return

        ts  = np.array(self._freq_times)
        moc = np.array(self._freq_mocap)
        cls = np.array(self._freq_classifier)
        pd  = np.array(self._freq_pd)

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.suptitle('Component Frequency Over Time', fontsize=14, fontweight='bold')

        specs = [
            (axes[0], moc, self.nominal_rates['mocap'],      'Mocap',      '#3498DB'),
            (axes[1], cls, self.nominal_rates['classifier'], 'Classifier', '#2ECC71'),
            (axes[2], pd,  self.nominal_rates['pd'],         'PD Ctrl',   '#E74C3C'),
        ]

        for ax, data, nominal, label, color in specs:
            tol = 0.05 * nominal
            ax.plot(ts, data, color=color, lw=1.5, label=f'{label} measured')
            ax.axhline(nominal,          color='k',    lw=1.5, ls='--', label=f'Nominal ({nominal:.0f} Hz)')
            ax.axhline(nominal + tol,    color='gray', lw=1.0, ls=':',  label='±5% band')
            ax.axhline(nominal - tol,    color='gray', lw=1.0, ls=':')
            ax.fill_between(ts, nominal - tol, nominal + tol, alpha=0.1, color=color)
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(label)
            ax.legend(fontsize=8, loc='lower right')
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        path = os.path.join(self._run_dir, 'frequency_plots.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[PLOT] {path}")

    def _print_quick_summary(self) -> None:
        stats = self.compute_summary_statistics()
        if not stats['overall'].get('total_frames'):
            return
        p = stats['position_tracking']
        v = stats['velocity_tracking']
        h = stats['heading_tracking']
        print("\n" + "=" * 70)
        print("QUICK SUMMARY")
        print("=" * 70)
        print(f"  Position RMSE:  {p['rmse_m']*100:.2f} cm   "
              f"(≤5cm: {p['accuracy_5cm_pct']:.0f}%  "
              f"≤10cm: {p['accuracy_10cm_pct']:.0f}%)")
        print(f"  Velocity RMSE:  {v['rmse_ms']:.3f} m/s  "
              f"(≤0.2m/s: {v['accuracy_02ms_pct']:.0f}%)")
        print(f"  Heading RMSE:   {h['rmse_deg']:.2f}°      "
              f"(≤10°: {h['accuracy_10deg_pct']:.0f}%)")
        print(f"  Transitions:    {stats['overall']['num_gait_transitions']}")
        if stats.get('component_frequency'):
            cf = stats['component_frequency']
            print(f"  Mocap freq:     {cf['mocap']['mean_hz']:.1f} ± "
                  f"{cf['mocap']['std_hz']:.1f} Hz  "
                  f"(nominal {cf['mocap']['nominal_hz']:.0f} Hz, "
                  f"within±5%: {cf['mocap']['within_5pct']:.1f}%)")
            print(f"  Classifier:     {cf['classifier']['mean_hz']:.1f} ± "
                  f"{cf['classifier']['std_hz']:.1f} Hz  "
                  f"(nominal {cf['classifier']['nominal_hz']:.0f} Hz, "
                  f"within±5%: {cf['classifier']['within_5pct']:.1f}%)")
            print(f"  PD ctrl:        {cf['pd_ctrl']['mean_hz']:.1f} ± "
                  f"{cf['pd_ctrl']['std_hz']:.1f} Hz  "
                  f"(nominal {cf['pd_ctrl']['nominal_hz']:.0f} Hz, "
                  f"within±5%: {cf['pd_ctrl']['within_5pct']:.1f}%)")
        print("=" * 70 + "\n")