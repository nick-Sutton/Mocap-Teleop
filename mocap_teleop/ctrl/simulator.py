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
from typing import List, Optional, Tuple

import numpy as np

from mocap_teleop.ctrl.cbf_qp import (Obstacle, body_to_world, world_to_body,
                                       cbf_value as _cbf_value,
                                       ROBOT_HALF_LENGTH, ROBOT_HALF_WIDTH)


# ── Episode ───────────────────────────────────────────────────────────────────

@dataclass
class Episode:
    """One training episode: a start position, a goal, and a set of obstacles.

    For teleop episodes (human_dir is not None):
      human_dir — unit vector of the human's walking direction
      v_human   — human's constant forward speed (m/s)
      goal      — human's final position: start + v_human * T * dt * human_dir

    human_pos(t) = start + t * dt * v_human * human_dir
    """
    start:     np.ndarray             # (2,) world frame
    goal:      np.ndarray             # (2,) world frame
    obstacles: List[Obstacle]         = field(default_factory=list)
    human_dir: Optional[np.ndarray]  = None   # set for teleop episodes
    v_human:   float                  = 1.5   # human speed (m/s)


# ── Kinematic robot ───────────────────────────────────────────────────────────

class KinematicSim:
    """
    2D first-order-lag robot (2nd-order system).

    The control input is a body-frame velocity command u_cmd = [vx, vy].
    Actual velocity v_actual tracks u_cmd through a first-order lag:

        τ · v̇_actual = u_cmd − v_actual   →   Euler: v += (u − v) * dt/τ

    τ = 0.25 s is derived from the Go2 mpac controller acceleration limits:
        τ = v_max / a_max = 0.3 m/s / 1.2 m/s² ≈ 0.25 s

    This gives the correct 2nd-order structure for ECBF training:
    h(p) has relative degree 2 w.r.t. u_cmd (through v_actual → ṗ → h).
    """

    def __init__(self, dt: float = 0.05, tau: float = 0.25):
        self.dt:  float      = dt
        self.tau: float      = tau
        self.pos: np.ndarray = np.zeros(2, dtype=np.float64)
        self.vel: np.ndarray = np.zeros(2, dtype=np.float64)   # world frame
        self.yaw: float      = 0.0

    def reset(self, pos: np.ndarray, yaw: float = 0.0) -> None:
        self.pos = np.asarray(pos, dtype=np.float64).copy()
        self.vel = np.zeros(2, dtype=np.float64)   # start from rest each episode
        self.yaw = float(yaw)

    def step(self, u_body: np.ndarray) -> None:
        """Integrate one step.  u_body = [vx, vy] command in body frame."""
        u_world   = body_to_world(u_body, self.yaw)
        self.vel += (u_world - self.vel) * (self.dt / self.tau)
        self.pos += self.vel * self.dt

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

def _obstacles_valid(
    obstacles:   List[Obstacle],
    start:       np.ndarray,
    goal:        np.ndarray,
    safe_margin: float,
    min_gap:     float,
) -> bool:
    """Return True if start/goal are clear and no two obstacles are too close."""
    for obs in obstacles:
        if _cbf_value(start, obs.center, 0.0, obs.radius,
                      ROBOT_HALF_LENGTH, ROBOT_HALF_WIDTH) <= safe_margin:
            return False
        if _cbf_value(goal, obs.center, 0.0, obs.radius,
                      ROBOT_HALF_LENGTH, ROBOT_HALF_WIDTH) <= safe_margin:
            return False
    for i in range(len(obstacles)):
        for j in range(i + 1, len(obstacles)):
            gap = (np.linalg.norm(obstacles[i].center - obstacles[j].center)
                   - obstacles[i].radius - obstacles[j].radius)
            if gap < min_gap:
                return False
    return True


def sample_episode(
    n_obstacles:  int   = 1,
    arena_half:   float = 3.0,
    min_r:        float = 0.3,
    max_r:        float = 0.8,
    min_sg_dist:  float = 1.5,
    obs_spread:   float = 0.4,
    force_slalom: bool  = False,
) -> Episode:
    """
    Sample a random training episode.

    Episode types
    ─────────────
    n_obstacles = 1  →  single on-path obstacle (original design).

    n_obstacles ≥ 2  →  randomly chooses between two types each call:

      Slalom (70 % of multi-obstacle episodes)
        Obstacles are placed at evenly-spaced positions along the path,
        alternating LEFT and RIGHT of the centre line.  The robot must
        navigate an S-curve: its lateral displacement after avoiding
        obstacle k determines its approach angle to obstacle k+1.
        Being too aggressive on obstacle k (high α_k) pushes the robot
        further off-centre and makes obstacle k+1 harder to avoid.
        This is the geometric coupling that forces the network to learn
        environment-dependent α rather than a constant maximum.

      Simple (30 % of multi-obstacle episodes)
        One on-path obstacle + remaining obstacles placed randomly in the
        arena.  Keeps the training distribution diverse and provides signal
        for "non-threatening" obstacles (network should learn high α).

    All episodes guarantee start/goal safety and inter-obstacle clearance.

    Parameters
    ----------
    n_obstacles  : number of circular obstacles
    arena_half   : half-width of the square sampling arena (m)
    min_r        : minimum obstacle radius (m)
    max_r        : maximum obstacle radius (m)
    min_sg_dist  : minimum required distance between start and goal (m)
    obs_spread   : lateral noise added to slalom obstacle positions (m)
    """
    _SAFE_MARGIN = 0.3
    _MIN_GAP     = 2.0 * ROBOT_HALF_LENGTH + 0.2   # robot fits between any pair

    # Rejection-sample start/goal pair with sufficient separation
    while True:
        start = np.random.uniform(-arena_half, arena_half, 2)
        goal  = np.random.uniform(-arena_half, arena_half, 2)
        if np.linalg.norm(goal - start) >= min_sg_dist:
            break

    path_vec  = goal - start
    path_dir  = path_vec / np.linalg.norm(path_vec)
    path_perp = np.array([-path_dir[1], path_dir[0]])   # 90° left of path

    # Episode type probabilities for n_obstacles >= 2:
    #   slalom (70%): alternating-side obstacles — the key geometry that
    #                 creates non-monotonic α-performance coupling.
    #   simple (30%): one on-path + rest random — keeps distribution diverse.
    rng = np.random.random()
    if force_slalom:
        episode_type = 'slalom'
    elif n_obstacles >= 2:
        episode_type = 'slalom' if rng < 0.7 else 'simple'
    else:
        episode_type = 'simple'

    for _ in range(1000):
        obstacles: List[Obstacle] = []

        if episode_type == 'simple':
            # ── Simple: one on-path + rest random ─────────────────────────
            t      = np.random.uniform(0.25, 0.75)
            center = start + t * path_vec + np.random.randn(2) * obs_spread
            radius = np.random.uniform(min_r, max_r)
            obstacles.append(Obstacle(center=center, radius=radius))
            for _ in range(n_obstacles - 1):
                c = np.random.uniform(-arena_half, arena_half, 2)
                r = np.random.uniform(min_r, max_r)
                obstacles.append(Obstacle(center=c, radius=r))

        else:
            # ── Slalom: alternating-side obstacles along path ─────────────
            n_slalom = min(n_obstacles, 3)
            ts       = np.linspace(0.25, 0.75, n_slalom)
            for i, t in enumerate(ts):
                radius  = np.random.uniform(min_r, max_r)
                side    = 1.0 if i % 2 == 0 else -1.0
                lateral = side * (radius + ROBOT_HALF_LENGTH
                                  + np.random.uniform(0.05, 0.30))
                center  = (start + t * path_vec
                           + lateral * path_perp
                           + np.random.randn(2) * 0.1)
                obstacles.append(Obstacle(center=center, radius=radius))
            for _ in range(n_obstacles - n_slalom):
                c = np.random.uniform(-arena_half, arena_half, 2)
                r = np.random.uniform(min_r, max_r)
                obstacles.append(Obstacle(center=c, radius=r))

        if _obstacles_valid(obstacles, start, goal, _SAFE_MARGIN, _MIN_GAP):
            return Episode(start=start, goal=goal, obstacles=obstacles)

    # Fallback: relax gap check, keep count and start/goal safety
    while True:
        t      = np.random.uniform(0.25, 0.75)
        center = start + t * path_vec + np.random.randn(2) * obs_spread
        radius = np.random.uniform(min_r, max_r)
        obstacles = [Obstacle(center=center, radius=radius)]
        for _ in range(n_obstacles - 1):
            c = np.random.uniform(-arena_half, arena_half, 2)
            r = np.random.uniform(min_r, max_r)
            obstacles.append(Obstacle(center=c, radius=r))
        if all(_cbf_value(start, o.center, 0.0, o.radius,
                          ROBOT_HALF_LENGTH, ROBOT_HALF_WIDTH) > _SAFE_MARGIN
               for o in obstacles) and \
           all(_cbf_value(goal,  o.center, 0.0, o.radius,
                          ROBOT_HALF_LENGTH, ROBOT_HALF_WIDTH) > _SAFE_MARGIN
               for o in obstacles):
            return Episode(start=start, goal=goal, obstacles=obstacles)


# ── Teleop episode sampler ────────────────────────────────────────────────────

def sample_teleop_episode(
    n_obstacles:  int   = 1,
    arena_half:   float = 3.0,
    min_r:        float = 0.3,
    max_r:        float = 0.8,
    v_human:      float = 1.5,
    T:            int   = 100,
    dt:           float = 0.05,
) -> Episode:
    """
    Sample a teleop training episode.

    The human walks in a straight line at constant speed v_human.  The robot
    starts co-located with the human and must track the human while avoiding
    obstacles the human ignores.

    Episode types
    ─────────────
      corridor (60 %, n_obstacles ≥ 2):
        Two obstacles straddle the human path at the same along-path position
        forming a narrow passage (gap = 1.0–1.8 × robot width).  The robot
        must pass between them while staying close to the human trajectory.
        High α on both walls → each wall's CBF pushes the robot toward the
        other wall → lateral oscillation → large deviation from human path.
        Low α → late response → collision.
        Moderate α → smooth threading.
        This is GENUINELY non-monotonic: no single α is universally optimal.

      single (40 %, always for n_obstacles = 1):
        One obstacle at δ ∈ (0.72, 0.98)×(a+r) from the human path.
        The δ ≥ 0.72×(a+r) lower bound ensures geometric feasibility: the
        required lateral speed at h=0 is v·√(1−δ²/r²) ≤ v_max.
        Provides diverse single-obstacle geometry.

    Returns
    -------
    Episode with human_dir and v_human set.
      goal = start + v_human * T * dt * human_dir  (human's final position)
    """
    _SAFE_MARGIN = 0.3
    path_length  = v_human * T * dt   # e.g. 1.5 m/s × 100 × 0.05 s = 7.5 m

    while True:
        # Human walks from a random start in a random direction.
        # No goal-in-arena check: obstacles are placed along the path (within
        # the first 85 % of path_length ≈ 6 m from start), so the goal can
        # be outside the arena without affecting the training signal.
        start     = np.random.uniform(-arena_half, arena_half, 2)
        angle     = np.random.uniform(0, 2 * np.pi)
        human_dir = np.array([np.cos(angle), np.sin(angle)])
        perp      = np.array([-human_dir[1], human_dir[0]])   # 90° left
        goal      = start + path_length * human_dir

        for _ in range(500):
            valid     = True
            obs_list: List[Obstacle] = []

            # Episode type:
            #   corridor (60 % when n_obstacles ≥ 2): two obstacles straddling
            #     the human path at the same along-path position, forming a
            #     narrow passage.  The robot MUST pass between them.
            #     High α on both walls → oscillation → large lateral deviation.
            #     Low α on both walls → late response → collision.
            #     Moderate α → smooth threading → stays with human.
            #     This is genuinely non-monotonic in the teleop lateral loss.
            #   single (40 %, or always for n_obstacles = 1): one obstacle at
            #     δ ∈ (0.72, 0.98)×safe_r.  Provides single-obstacle geometry.
            use_corridor = (n_obstacles >= 2 and np.random.random() < 0.60)

            if use_corridor:
                t_frac  = np.random.uniform(0.20, 0.70)
                on_path = start + t_frac * path_length * human_dir
                r_l     = np.random.uniform(min_r, max_r)
                r_r     = np.random.uniform(min_r, max_r)
                # gap_half: half the clear width between surface-to-surface.
                # Must exceed ROBOT_HALF_WIDTH so the robot can fit through.
                gap_half = ROBOT_HALF_WIDTH * np.random.uniform(1.0, 1.8)
                obs_list.append(Obstacle(
                    center=on_path + (gap_half + r_l) * perp, radius=r_l))
                obs_list.append(Obstacle(
                    center=on_path - (gap_half + r_r) * perp, radius=r_r))
                # Fill remaining slots with single-type obstacles
                for _ in range(n_obstacles - 2):
                    r2     = np.random.uniform(min_r, max_r)
                    sr2    = ROBOT_HALF_LENGTH + r2
                    t2     = np.random.uniform(0.15, 0.85)
                    side   = np.random.choice([-1.0, 1.0])
                    delta  = np.random.uniform(0.72 * sr2, 0.98 * sr2)
                    obs_list.append(Obstacle(
                        center=start + t2 * path_length * human_dir
                               + side * delta * perp,
                        radius=r2))
            else:
                for _ in range(n_obstacles):
                    r      = np.random.uniform(min_r, max_r)
                    sr     = ROBOT_HALF_LENGTH + r
                    t_frac = np.random.uniform(0.15, 0.85)
                    side   = np.random.choice([-1.0, 1.0])
                    delta  = np.random.uniform(0.72 * sr, 0.98 * sr)
                    obs_list.append(Obstacle(
                        center=start + t_frac * path_length * human_dir
                               + side * delta * perp,
                        radius=r))

            # All obstacles must be safe from the robot's start position
            for obs in obs_list:
                if _cbf_value(start, obs.center, 0.0, obs.radius,
                              ROBOT_HALF_LENGTH, ROBOT_HALF_WIDTH) <= _SAFE_MARGIN:
                    valid = False
                    break

            if valid:
                return Episode(start=start, goal=goal, obstacles=obs_list,
                               human_dir=human_dir, v_human=v_human)

        # Fallback: relax start-safety and try again


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
