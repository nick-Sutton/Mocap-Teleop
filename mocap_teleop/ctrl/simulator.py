#!/usr/bin/env python3

"""
simulator.py — 2D kinematic simulator for offline CBF training.

The robot is modelled as a pure kinematic point mass:

    ẋ = u   (world frame)

where x = [px, py] ∈ ℝ² and u = [vx, vy] is the body-frame velocity
rotated to the world frame by the current heading.

This gives us a cheap, differentiable environment to train the α-net
entirely in numpy/torch without needing MuJoCo or a real robot.
At deployment the α-net is dropped in front of CtrlInterface.walk()
and sees the real mocap command instead of the PD controller output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from mocap_teleop.ctrl.cbf_qp import (Obstacle, body_to_world, world_to_body,
                                       cbf_value as _cbf_value,
                                       ROBOT_HALF_LENGTH, ROBOT_HALF_WIDTH)


# ── Episode ───────────────────────────────────────────────────────────────────

@dataclass
class Episode:
    """One training episode: a start position, a goal, and a set of obstacles."""
    start:     np.ndarray         # (2,) world frame
    goal:      np.ndarray         # (2,) world frame
    obstacles: List[Obstacle] = field(default_factory=list)


# ── Kinematic robot ───────────────────────────────────────────────────────────

class KinematicSim:
    """
    2D kinematic point-mass robot.

    The control input is a body-frame velocity [vx, vy].
    Heading (yaw) is updated separately or derived from motion direction.
    """

    def __init__(self, dt: float = 0.05):
        self.dt:  float      = dt
        self.pos: np.ndarray = np.zeros(2, dtype=np.float64)
        self.yaw: float      = 0.0

    def reset(self, pos: np.ndarray, yaw: float = 0.0) -> None:
        self.pos = np.asarray(pos, dtype=np.float64).copy()
        self.yaw = float(yaw)

    def step(self, u_body: np.ndarray) -> None:
        """Integrate one step.  u_body = [vx, vy] in body frame."""
        u_world   = body_to_world(u_body, self.yaw)
        self.pos += u_world * self.dt

    def heading_toward(self, target: np.ndarray) -> float:
        """Yaw angle pointing from current position to target."""
        d = target - self.pos
        return float(np.arctan2(d[1], d[0]))

    def cbf_value(self, obs: Obstacle) -> float:
        """Elliptical h(x) = (pos−obs.center)ᵀ Q (pos−obs.center) − 1  ≥ 0 means safe."""
        return _cbf_value(self.pos, obs.center, self.yaw, obs.radius,
                          ROBOT_HALF_LENGTH, ROBOT_HALF_WIDTH)

    def is_safe(self, obs: Obstacle, margin: float = 0.0) -> bool:
        return self.cbf_value(obs) >= -margin


# ── Performance controller (u_perf) ──────────────────────────────────────────

class PDController:
    """
    Simple proportional controller driving the robot toward a goal.

    Used as the stand-in for the mocap command during training.
    At deployment this is replaced by the real mocap-derived velocity.
    """

    def __init__(self, kp: float = 1.5, v_max: float = 1.5):
        self.kp    = kp
        self.v_max = v_max

    def __call__(
        self,
        pos:  np.ndarray,
        goal: np.ndarray,
        yaw:  float,
    ) -> np.ndarray:
        """
        Compute body-frame velocity command toward goal.

        Returns
        -------
        u_perf : (2,) [vx, vy] in body frame
        """
        err_world = goal - pos
        u_world   = self.kp * err_world

        speed = np.linalg.norm(u_world)
        if speed > self.v_max:
            u_world = u_world * self.v_max / speed

        return world_to_body(u_world, yaw)


# ── Episode sampler ───────────────────────────────────────────────────────────

def sample_episode(
    n_obstacles:  int   = 1,
    arena_half:   float = 3.0,
    min_r:        float = 0.3,
    max_r:        float = 0.8,
    min_sg_dist:  float = 1.5,
    obs_spread:   float = 0.4,
) -> Episode:
    """
    Sample a random training episode.

    Obstacles are placed near the straight-line path from start to goal
    so the robot has to actively avoid them rather than going around.

    For multi-obstacle episodes, obstacles are guaranteed not to mutually
    overlap (with enough clearance for the robot to pass between any pair),
    preventing infeasible CBF-QP configurations where constraints point in
    irreconcilable directions.

    Parameters
    ----------
    n_obstacles  : number of circular obstacles
    arena_half   : half-width of the square sampling arena (m)
    min_r        : minimum obstacle radius (m)
    max_r        : maximum obstacle radius (m)
    min_sg_dist  : minimum required distance between start and goal (m)
    obs_spread   : std-dev of the lateral scatter around the straight-line path (m)
    """
    # With multiple obstacles reduce max radius so they don't collectively
    # block all corridors (3 × 0.8m obstacles can wall off a short path).
    effective_max_r = max_r if n_obstacles == 1 else min(max_r, 0.5)

    # Robot diameter — minimum gap we must leave between any two obstacles
    # so the robot can physically fit through.
    _ROBOT_DIAM  = 2.0 * ROBOT_HALF_LENGTH + 0.3   # add 0.3 m clearance

    # Rejection-sample start/goal pair with sufficient separation
    while True:
        start = np.random.uniform(-arena_half, arena_half, 2)
        goal  = np.random.uniform(-arena_half, arena_half, 2)
        if np.linalg.norm(goal - start) >= min_sg_dist:
            break

    # Sample obstacles, rejection-sampling until:
    #   1. Start and goal are both clear of all obstacles.
    #   2. No two obstacles overlap (robot can pass between every pair).
    # This guarantees h(x_0) > 0 and the CBF-QP is never trivially infeasible
    # due to conflicting constraint gradients from packed obstacles.
    _SAFE_MARGIN = 0.3   # metres — minimum h value required at start and goal
    for _ in range(500):
        obstacles = []
        for _ in range(n_obstacles):
            t      = np.random.uniform(0.20, 0.80)
            center = start + t * (goal - start) + np.random.randn(2) * obs_spread
            radius = np.random.uniform(min_r, effective_max_r)
            obstacles.append(Obstacle(center=center, radius=radius))

        # Reject if robot starts or ends inside (or too close to) any obstacle
        start_safe = all(
            _cbf_value(start, obs.center, 0.0, obs.radius,
                       ROBOT_HALF_LENGTH, ROBOT_HALF_WIDTH) > _SAFE_MARGIN
            for obs in obstacles
        )
        goal_safe = all(
            _cbf_value(goal, obs.center, 0.0, obs.radius,
                       ROBOT_HALF_LENGTH, ROBOT_HALF_WIDTH) > _SAFE_MARGIN
            for obs in obstacles
        )

        # Reject if any two obstacles are so close the robot can't pass between
        obs_passable = True
        for i in range(len(obstacles)):
            for j in range(i + 1, len(obstacles)):
                gap = (np.linalg.norm(obstacles[i].center - obstacles[j].center)
                       - obstacles[i].radius - obstacles[j].radius)
                if gap < _ROBOT_DIAM:
                    obs_passable = False
                    break
            if not obs_passable:
                break

        if start_safe and goal_safe and obs_passable:
            break

    return Episode(start=start, goal=goal, obstacles=obstacles)


# ── Trajectory rollout (numpy, no gradients) ─────────────────────────────────

def rollout_numpy(
    episode:  Episode,
    u_policy,               # callable(pos, goal, yaw) → u_body (2,)
    T:        int   = 160,
    dt:       float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Roll out a trajectory with an arbitrary policy (no gradient tracking).

    Useful for visualisation and sanity checks.

    Returns
    -------
    positions : (T+1, 2) trajectory including the start position
    h_values  : (T+1, n_obs) CBF values along the trajectory
    """
    sim = KinematicSim(dt=dt)
    sim.reset(episode.start)

    n_obs     = len(episode.obstacles)
    positions = np.zeros((T + 1, 2))
    h_values  = np.zeros((T + 1, n_obs))

    positions[0] = sim.pos
    for i, obs in enumerate(episode.obstacles):
        h_values[0, i] = sim.cbf_value(obs)

    for t in range(T):
        yaw = sim.heading_toward(episode.goal)
        u   = u_policy(sim.pos, episode.goal, yaw)
        sim.step(u)

        positions[t + 1] = sim.pos
        for i, obs in enumerate(episode.obstacles):
            h_values[t + 1, i] = sim.cbf_value(obs)

    return positions, h_values
