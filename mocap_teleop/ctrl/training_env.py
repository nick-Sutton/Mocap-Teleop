#!/usr/bin/env python3

"""
training_env.py — Go2 MuJoCo training environment for AM-CBF.

Wraps the unitree_mujoco Python bridge to provide a gym-style interface for
the DDPG training loop.  mpac runs as a separate C++ process connected via DDS.

Usage
─────
    env = Go2TrainingEnv(use_viewer=False)
    env.startup()          # stand + stabilise (call once after mpac is running)
    env.reset([x, y])      # teleport + wait for mpac to restabilise
    state = env.get_state()
    ...
    env.close()

Coordinate frames
─────────────────
    World frame : MuJoCo global frame (X forward, Y left, Z up at startup)
    Body frame  : robot base_link frame; forward = robot nose direction

State dict keys
───────────────
    pos_xy   : (2,) world-frame position [x, y]
    yaw      : float, world-frame heading in radians
    vel_body : (2,) body-frame linear velocity [vx, vy]
"""

from __future__ import annotations

import os
import sys
import time
import threading
from typing import Dict

import mujoco
import mujoco.viewer
import numpy as np

# ── DDS bridge path setup ─────────────────────────────────────────────────────
# The unitree_sdk2py_bridge lives inside mpac_mujoco/simulate_python/.
# We resolve it relative to THIS file (mocap_teleop/ctrl/), going up two
# levels to the project root and then into mpac_mujoco.
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
_BRIDGE_DIR   = os.path.join(_PROJECT_ROOT, 'mpac_mujoco', 'simulate_python')

if _BRIDGE_DIR not in sys.path:
    sys.path.insert(0, _BRIDGE_DIR)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge

# ── Go2 constants ─────────────────────────────────────────────────────────────
# Standing height and joint configuration from go2.xml keyframe "home":
#   qpos = [0, 0, 0.27,  1 0 0 0,  0 0.9 -1.8  ×4]
_STANDING_Z   = 0.27          # metres above floor
_HOME_JOINTS  = np.array([    # 12 joint angles in model order
    0.0,  0.9, -1.8,          # FL: hip-x, hip-y, knee
    0.0,  0.9, -1.8,          # FR
    0.0,  0.9, -1.8,          # RL
    0.0,  0.9, -1.8,          # RR
], dtype=np.float64)

# Number of DOF before the leg joints (freejoint = 6 qvel, 7 qpos)
_QPOS_JOINTS_START = 7   # qpos[7:19] = leg joints
_QVEL_LIN_START    = 3   # qvel[3:6]  = body-frame linear velocity (freejoint)

# base_link body index in the MuJoCo model (0 = world, 1 = base_link)
_BASE_LINK_IDX = 1

# Stabilisation wait after a reset (seconds)
# mpac transitions: stand command → hold for this long → ready
_STABILISE_S = 2.5

# ── DDS initialisation guard ──────────────────────────────────────────────────
_dds_initialised = False
_dds_lock        = threading.Lock()


def _ensure_dds(domain_id: int, interface: str) -> None:
    global _dds_initialised
    with _dds_lock:
        if not _dds_initialised:
            ChannelFactoryInitialize(domain_id, interface)
            _dds_initialised = True


# ── Main class ────────────────────────────────────────────────────────────────

class Go2TrainingEnv:
    """
    Thin wrapper around the unitree_mujoco Python bridge for RL training.

    Parameters
    ----------
    scene_xml   : path to the MuJoCo scene XML (defaults to the project's
                  mpac_mujoco/unitree_robots/go2/scene.xml)
    domain_id   : DDS domain id (must match mpac)
    interface   : network interface for DDS (usually 'lo' for loopback)
    use_viewer  : if True, launch the passive MuJoCo viewer (debug mode)
    sim_dt      : MuJoCo physics timestep (seconds)
    """

    def __init__(
        self,
        scene_xml:  str | None = None,
        domain_id:  int        = 1,
        interface:  str        = 'lo',
        use_viewer: bool       = False,
        sim_dt:     float      = 0.001,
    ):
        if scene_xml is None:
            scene_xml = os.path.join(
                _PROJECT_ROOT, 'mpac_mujoco', 'unitree_robots', 'go2', 'scene.xml')

        self._use_viewer = use_viewer
        self._running    = False

        # ── Load model ────────────────────────────────────────────────────────
        self.mj_model              = mujoco.MjModel.from_xml_path(scene_xml)
        self.mj_data               = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = sim_dt
        self._lock                 = threading.Lock()

        # ── Pending-reset state ───────────────────────────────────────────────
        # The reset is *queued* here and applied inside _sim_loop so that
        # mj_data is never written from two threads at the same time.
        self._pending_qpos:   np.ndarray | None = None
        self._reset_done:     threading.Event   = threading.Event()

        # ── Viewer obstacle / goal overlay ───────────────────────────────────
        # Set by the training/rollout loop each episode; the viewer loop draws
        # semi-transparent cylinders for obstacles and a green sphere for goal.
        self._vis_obstacles: list           = []
        self._vis_goal:      np.ndarray | None = None

        # Capture the home qpos from the "home" keyframe
        key_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, key_id)
        self._home_qpos = self.mj_data.qpos.copy()
        self._home_qvel = self.mj_data.qvel.copy()  # all zeros typically

        # ── DDS + bridge ──────────────────────────────────────────────────────
        _ensure_dds(domain_id, interface)
        self._bridge = UnitreeSdk2Bridge(self.mj_model, self.mj_data)

        # ── Optional viewer ───────────────────────────────────────────────────
        self._viewer = None
        if use_viewer:
            self._viewer = mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def startup(self, ctrl: 'CtrlInterface') -> None:
        """
        Start the physics loop and wait for mpac to bring the robot to a
        standing configuration.  Call once after mpac is already running.

        Parameters
        ----------
        ctrl : a CtrlInterface instance for sending stand/walk commands
        """
        self._running = True
        self._sim_thread = threading.Thread(
            target=self._sim_loop, daemon=True, name='go2_sim')
        self._sim_thread.start()

        if self._viewer is not None:
            self._viewer_thread = threading.Thread(
                target=self._viewer_loop, daemon=True, name='go2_viewer')
            self._viewer_thread.start()

        # Command the robot to stand and wait for it to stabilise
        print('[Go2TrainingEnv] Sending stand command, waiting for stabilisation...')
        ctrl.stand()
        time.sleep(_STABILISE_S)
        print('[Go2TrainingEnv] Ready.')

    def close(self) -> None:
        """
        Stop the simulation loop, drain the bridge threads, then close the
        viewer.  Joining threads before Python's GC runs prevents the
        RecurrentThread callbacks from accessing freed mj_data (segfault).
        """
        self._running = False

        # Stop bridge RecurrentThreads (they call mj_data).
        # RecurrentThread.Wait(timeout) sets __quit=True and waits.
        for attr in ('lowStateThread', 'HighStateThread', 'WirelessControllerThread'):
            thread = getattr(self._bridge, attr, None)
            if thread is not None:
                try:
                    thread.Wait(0.5)
                except Exception:
                    pass

        # Wait for our own sim loop to exit.
        if hasattr(self, '_sim_thread'):
            self._sim_thread.join(timeout=1.0)

        if self._viewer is not None:
            self._viewer.close()

    # ── Episode reset ─────────────────────────────────────────────────────────

    def reset(self, start_xy: np.ndarray, heading: float = 0.0,
              ctrl: 'CtrlInterface | None' = None) -> None:
        """
        Teleport the robot to a new (x, y) position and wait for mpac to
        restabilise at the new location.

        Parameters
        ----------
        start_xy : (2,) target position in world frame
        heading  : desired yaw in radians (default 0 = facing +x)
        ctrl     : CtrlInterface used to send stand command after teleport.
                   If None, only the physics state is reset (useful for the
                   first reset before startup() is called).
        """
        # Build the new qpos: copy home, override base position + heading
        new_qpos       = self._home_qpos.copy()
        new_qpos[0]    = float(start_xy[0])
        new_qpos[1]    = float(start_xy[1])
        new_qpos[2]    = _STANDING_Z
        # Quaternion for heading rotation about Z: [qw, qx, qy, qz]
        half            = heading / 2.0
        new_qpos[3]    = np.cos(half)   # qw
        new_qpos[4]    = 0.0            # qx
        new_qpos[5]    = 0.0            # qy
        new_qpos[6]    = np.sin(half)   # qz

        # Queue the new state — _sim_loop applies it on the next tick
        # while it holds _lock, preventing concurrent bridge-thread reads
        # from seeing a partially-written mj_data.
        self._reset_done.clear()
        self._pending_qpos = new_qpos          # atomic Python assignment (GIL)
        self._reset_done.wait(timeout=0.5)     # wait for sim_loop to confirm

        if ctrl is not None:
            ctrl.stand()
            time.sleep(_STABILISE_S)

    # ── State readback ────────────────────────────────────────────────────────

    def get_state(self) -> Dict[str, np.ndarray | float]:
        """
        Return the current robot state.

        Returns
        -------
        dict with:
          pos_xy   : (2,) world-frame position [x, y]
          yaw      : float, world-frame heading (radians)
          vel_body : (2,) body-frame linear velocity [vx, vy]
        """
        with self._lock:
            pos_xy   = self.mj_data.xpos[_BASE_LINK_IDX, :2].copy()
            xquat    = self.mj_data.xquat[_BASE_LINK_IDX].copy()  # [qw,qx,qy,qz]
            # qvel[3:6] = body-frame linear velocity for the freejoint
            vel_body = self.mj_data.qvel[_QVEL_LIN_START : _QVEL_LIN_START + 2].copy()

        yaw = np.arctan2(
            2.0 * (xquat[0] * xquat[3] + xquat[1] * xquat[2]),
            1.0 - 2.0 * (xquat[2] ** 2 + xquat[3] ** 2),
        )
        return {'pos_xy': pos_xy, 'yaw': float(yaw), 'vel_body': vel_body}

    # ── Internal simulation loop ──────────────────────────────────────────────

    def _sim_loop(self) -> None:
        """Physics loop — runs at sim_dt in a background thread."""
        dt = self.mj_model.opt.timestep
        while self._running:
            step_start = time.perf_counter()
            with self._lock:
                # Apply any queued reset before the next physics step.
                # Doing it here (under the lock) prevents concurrent writes
                # from reset() racing with bridge-thread reads.
                if self._pending_qpos is not None:
                    self.mj_data.qpos[:] = self._pending_qpos
                    self.mj_data.qvel[:] = 0.0
                    mujoco.mj_forward(self.mj_model, self.mj_data)
                    self._pending_qpos = None
                    self._reset_done.set()
                mujoco.mj_step(self.mj_model, self.mj_data)
            remaining = dt - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)

    def set_vis_obstacles(self, obstacles: list) -> None:
        """
        Set the obstacle list to render in the viewer each frame.
        Call this once per episode from the training / rollout loop.
        Thread-safe — Python list assignment is atomic under the GIL.
        """
        self._vis_obstacles = list(obstacles)

    def set_vis_goal(self, goal: np.ndarray | None) -> None:
        """
        Set the goal position to render as a green sphere in the viewer.
        Pass None to clear.  Thread-safe — Python assignment is atomic under GIL.
        """
        self._vis_goal = None if goal is None else np.asarray(goal, dtype=np.float64)

    def _viewer_loop(self) -> None:
        """Viewer sync loop — 50 fps with obstacle and goal overlay."""
        _OBS_HEIGHT   = 1.0    # visual cylinder height (m)
        _OBS_RGBA     = np.array([1.0, 0.35, 0.1, 0.45], dtype=np.float32)  # orange, semi-transparent
        _GOAL_RADIUS  = 0.15   # visual sphere radius (m)
        _GOAL_RGBA    = np.array([0.1, 0.9, 0.1, 0.9], dtype=np.float32)    # green
        _MAT_IDENTITY = np.eye(3, dtype=np.float64).flatten()

        while self._running and self._viewer.is_running():
            obs_list = self._vis_obstacles   # snapshots (GIL-safe)
            goal     = self._vis_goal

            with self._viewer.lock():
                self._viewer.user_scn.ngeom = 0

                # ── Obstacle cylinders ────────────────────────────────────────
                for obs in obs_list:
                    n = self._viewer.user_scn.ngeom
                    if n >= self._viewer.user_scn.maxgeom:
                        break
                    mujoco.mjv_initGeom(
                        self._viewer.user_scn.geoms[n],
                        type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
                        size  = np.array([obs.radius, _OBS_HEIGHT / 2.0, 0.0]),
                        pos   = np.array([obs.center[0], obs.center[1], _OBS_HEIGHT / 2.0]),
                        mat   = _MAT_IDENTITY,
                        rgba  = _OBS_RGBA,
                    )
                    self._viewer.user_scn.ngeom += 1

                # ── Goal sphere ───────────────────────────────────────────────
                if goal is not None:
                    n = self._viewer.user_scn.ngeom
                    if n < self._viewer.user_scn.maxgeom:
                        mujoco.mjv_initGeom(
                            self._viewer.user_scn.geoms[n],
                            type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                            size  = np.array([_GOAL_RADIUS, 0.0, 0.0]),
                            pos   = np.array([goal[0], goal[1], _GOAL_RADIUS]),
                            mat   = _MAT_IDENTITY,
                            rgba  = _GOAL_RGBA,
                        )
                        self._viewer.user_scn.ngeom += 1

            with self._lock:
                self._viewer.sync()
            time.sleep(0.02)
