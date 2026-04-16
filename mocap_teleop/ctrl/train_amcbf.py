#!/usr/bin/env python3

"""
train_amcbf.py — DDPG training for the AM-CBF κ-function.

Implements Algorithm 1 from:
  Chriat & Sun, "AM-CBF: Adaptive Multi-step CBF for Safe Reinforcement
  Learning," IEEE RA-L 2023.

Overview
────────
The κ-net learns a class-K function κ(h) so that the CBF constraint

    ∇h(x) · u  ≥  −κ(h(x))

keeps the robot safe while allowing maximum performance.  DDPG's critic
provides ∂Q/∂u_safe at each single step, so gradients flow back through
just the one-step differentiable QP into κ — no BPTT through dynamics.

Training pipeline
─────────────────
  1. Actor μ(s) → u_RL          P-controller toward goal (fixed, not trained)
  2. κ-net κ(h) → kappa values  one per obstacle
  3. Dykstra QP filters u_RL → u_safe  (differentiable)
  4. Execute u_safe via CtrlInterface.walk()
  5. Observe next state, compute reward
  6. Store (s, u_safe, r, s') in replay buffer
  7. Critic update: min MSE( Q(s,a), r + γ Q'(s',μ'(s')) )
  8. κ update: max  Q(s, QP(μ(s), κ(h)))   — gradient through Dykstra QP

Usage
─────
  # Start mpac first, then:
  python -m mocap_teleop.ctrl.train_amcbf

  python -m mocap_teleop.ctrl.train_amcbf \\
      --episodes 500 --steps 200 --out kappa_net.pth --viewer
"""

from __future__ import annotations

import argparse
import collections
import os
import random
import time
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mocap_teleop.ctrl.kappa_net import KappaNet
from mocap_teleop.ctrl.training_env import Go2TrainingEnv
from mocap_teleop.ctrl.cbf_qp import (
    CBFQP, Obstacle, cbf_value, cbf_gradient,
    body_to_world, world_to_body,
    ROBOT_HALF_LENGTH,
)
from mocap_teleop.ctrl.ctrl_interface import CtrlInterface

# ── Hyperparameters ───────────────────────────────────────────────────────────

_V_MAX          = 1.5      # m/s — nominal walk speed cap
_KP_NOMINAL     = 1.5      # P-gain for goal-reaching nominal policy
_KP_YAW         = 2.0      # P-gain for heading controller (rad/s per rad)
_VRZ_MAX        = 1.2      # max yaw rate (rad/s)
_SAFETY_WEIGHT  = 1.0      # reward penalty weight for h < 0 violations
_STEP_PENALTY   = 0.005    # constant per-step cost (encourages minimum-time)
_MAX_OBS        = 3        # max obstacles per episode
_ARENA_HALF     = 3.0      # half-width of square sampling arena (m)
_OBS_R_MIN      = 0.3      # min obstacle radius (m)
_OBS_R_MAX      = 0.8      # max obstacle radius (m)
_MIN_SG_DIST    = 1.5      # min start-to-goal distance (m)
_GOAL_RADIUS    = 0.3      # episode ends when robot is within this of goal (m)
_ARENA_DIAG     = 2.0 * _ARENA_HALF * np.sqrt(2.0)   # max possible dist in arena (m)

_T              = 200      # max steps per episode
_CTRL_DT        = 0.05     # seconds between control steps (20 Hz)

_BUFFER_SIZE    = 50_000   # replay buffer capacity
_BATCH_SIZE     = 64
_GAMMA          = 0.99     # discount factor
_TAU_TARGET     = 0.7      # soft target update coefficient (paper Table I)
_LR_CRITIC      = 1e-3     # critic learning rate
_LR_KAPPA       = 1e-3     # κ-net learning rate

_OU_THETA       = 0.15     # Ornstein-Uhlenbeck noise θ
_OU_SIGMA       = 0.2      # Ornstein-Uhlenbeck noise σ

_WARMUP_STEPS   = 1000     # collect this many steps before first update
_UPDATE_EVERY   = 1        # update networks every N steps

# State dimension: [d_goal(2), vel_body(2), per_obs: d_obs(2)+r(1)+h(1)] × MAX_OBS
_STATE_DIM      = 4 + _MAX_OBS * 4

# Normalisation scales
_D_SCALE        = _ARENA_HALF * 2   # metres
_H_SCALE        = 5.0               # CBF value upper clamp for normalisation
_R_SCALE        = 1.0               # obstacle radius scale


# ── Critic network ────────────────────────────────────────────────────────────

class CriticNet(nn.Module):
    """Q(s, a) — action injected into the second hidden layer."""

    def __init__(self, state_dim: int = _STATE_DIM, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden + 2, 64)    # +2 for action [vx, vy]
        self.fc3 = nn.Linear(64, 1)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(torch.cat([x, action], dim=-1)))
        return self.fc3(x)   # (B, 1)


# ── Replay buffer ─────────────────────────────────────────────────────────────

class Transition(NamedTuple):
    state:       np.ndarray   # (STATE_DIM,)
    action:      np.ndarray   # (2,)  u_safe in body frame
    reward:      float
    next_state:  np.ndarray   # (STATE_DIM,)
    done:        bool
    # Raw geometry for recomputing the QP during κ update
    obs_centers: np.ndarray   # (MAX_OBS, 2) world frame, padded with zeros
    obs_radii:   np.ndarray   # (MAX_OBS,)   padded with large values
    n_obs:       int          # actual obstacle count
    robot_pos:   np.ndarray   # (2,) world frame at this step
    yaw:         float        # robot heading (radians)
    # Same for next state (needed for target Q computation)
    obs_centers_next: np.ndarray
    obs_radii_next:   np.ndarray
    robot_pos_next:   np.ndarray
    yaw_next:         float


class ReplayBuffer:
    def __init__(self, capacity: int = _BUFFER_SIZE):
        self._buf: collections.deque[Transition] = collections.deque(
            maxlen=capacity)

    def push(self, t: Transition) -> None:
        self._buf.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self._buf, batch_size)

    def __len__(self) -> int:
        return len(self._buf)


# ── Ornstein-Uhlenbeck noise ──────────────────────────────────────────────────

class OUNoise:
    def __init__(self, size: int, theta: float = _OU_THETA,
                 sigma: float = _OU_SIGMA):
        self.size  = size
        self.theta = theta
        self.sigma = sigma
        self.state = np.zeros(size)

    def reset(self) -> None:
        self.state = np.zeros(self.size)

    def sample(self) -> np.ndarray:
        self.state += (
            -self.theta * self.state
            + self.sigma * np.random.randn(self.size)
        )
        return self.state.copy()


# ── State builder ─────────────────────────────────────────────────────────────

def build_state(
    pos_xy:    np.ndarray,
    yaw:       float,
    vel_body:  np.ndarray,
    goal_xy:   np.ndarray,
    obstacles: List[Obstacle],
) -> np.ndarray:
    """
    Build the flat state vector for the critic.

    Layout: [d_goal_body(2), vel_body(2), obs0(4), obs1(4), obs2(4)]
    Padded obstacle slots have h=_H_SCALE (normalised to 1.0 → very safe).
    """
    # Goal direction in body frame
    d_goal_world = goal_xy - pos_xy
    c, s         = np.cos(yaw), np.sin(yaw)
    d_goal_body  = np.array([
        c * d_goal_world[0] + s * d_goal_world[1],
        -s * d_goal_world[0] + c * d_goal_world[1],
    ])

    feats = np.zeros(_STATE_DIM, dtype=np.float32)
    feats[0:2] = d_goal_body  / _D_SCALE
    feats[2:4] = vel_body[:2] / _V_MAX

    for i in range(_MAX_OBS):
        base = 4 + i * 4
        if i < len(obstacles):
            obs = obstacles[i]
            d_world = obs.center - pos_xy
            d_body  = np.array([
                c * d_world[0] + s * d_world[1],
                -s * d_world[0] + c * d_world[1],
            ])
            h = cbf_value(pos_xy, obs.center, yaw, obs.radius)
            feats[base + 0] = d_body[0]            / _D_SCALE
            feats[base + 1] = d_body[1]            / _D_SCALE
            feats[base + 2] = obs.radius           / _R_SCALE
            feats[base + 3] = np.clip(h, -1.0, _H_SCALE) / _H_SCALE
        else:
            # Padded slot: mark as very safe (h at ceiling)
            feats[base + 3] = 1.0

    return feats


# ── Nominal P-controller ──────────────────────────────────────────────────────

def nominal_policy(
    pos_xy:   np.ndarray,
    yaw:      float,
    goal_xy:  np.ndarray,
    v_max:    float = _V_MAX,
    kp:       float = _KP_NOMINAL,
) -> np.ndarray:
    """P-controller: drives toward goal in body frame, clipped to v_max."""
    err_world = goal_xy - pos_xy
    u_world   = kp * err_world
    speed     = np.linalg.norm(u_world)
    if speed > v_max:
        u_world = u_world * v_max / speed
    return world_to_body(u_world, yaw)   # (2,) body frame


def nominal_yaw_rate(
    pos_xy:  np.ndarray,
    yaw:     float,
    goal_xy: np.ndarray,
) -> float:
    """
    P-controller for heading: turn to face the goal direction.
    Returns vrz (rad/s), clipped to ±_VRZ_MAX.
    Stops turning when very close to goal (avoids spinning in place).
    """
    err = goal_xy - pos_xy
    if np.linalg.norm(err) < _GOAL_RADIUS:
        return 0.0
    target_yaw = np.arctan2(err[1], err[0])
    yaw_err    = target_yaw - yaw
    # Wrap to [-π, π]
    yaw_err    = np.arctan2(np.sin(yaw_err), np.cos(yaw_err))
    return float(np.clip(_KP_YAW * yaw_err, -_VRZ_MAX, _VRZ_MAX))


# ── Episode sampler ───────────────────────────────────────────────────────────

def _sample_episode(
    n_obs:       int   = _MAX_OBS,
    arena_half:  float = _ARENA_HALF,
    min_sg_dist: float = _MIN_SG_DIST,
) -> Tuple[np.ndarray, np.ndarray, List[Obstacle]]:
    """
    Sample a random start, goal, and obstacle set.

    Obstacles are placed on or near the direct path (start→goal) so the
    nominal P-controller must engage the CBF.  Start and goal are guaranteed
    clear of all obstacles.
    """
    _SAFE_MARGIN = 0.5   # h must exceed this at start/goal

    # Rejection-sample start/goal
    for _ in range(2000):
        start = np.random.uniform(-arena_half, arena_half, 2)
        goal  = np.random.uniform(-arena_half, arena_half, 2)
        if np.linalg.norm(goal - start) >= min_sg_dist:
            break

    path_vec  = goal - start
    path_dir  = path_vec / (np.linalg.norm(path_vec) + 1e-8)
    path_perp = np.array([-path_dir[1], path_dir[0]])

    for _ in range(1000):
        obstacles: List[Obstacle] = []
        valid = True

        for i in range(n_obs):
            r      = np.random.uniform(_OBS_R_MIN, _OBS_R_MAX)
            t      = np.random.uniform(0.2, 0.8)
            side   = 1.0 if i % 2 == 0 else -1.0
            # Place just off-path so P-controller must deflect
            lateral = side * (ROBOT_HALF_LENGTH + r + np.random.uniform(0.05, 0.25))
            center  = start + t * path_vec + lateral * path_perp + \
                      np.random.randn(2) * 0.1
            obstacles.append(Obstacle(center=center, radius=r))

        # Check start and goal are safe from all obstacles
        for obs in obstacles:
            if cbf_value(start, obs.center, 0.0, obs.radius) <= _SAFE_MARGIN:
                valid = False
                break
            if cbf_value(goal, obs.center, 0.0, obs.radius) <= _SAFE_MARGIN:
                valid = False
                break

        if valid:
            return start, goal, obstacles

    # Fallback: single on-path obstacle
    r      = np.random.uniform(_OBS_R_MIN, _OBS_R_MAX)
    t      = np.random.uniform(0.3, 0.7)
    center = start + t * path_vec
    return start, goal, [Obstacle(center=center, radius=r)]


def _pack_obs(obstacles: List[Obstacle]) -> Tuple[np.ndarray, np.ndarray]:
    """Pack obstacle list into fixed-size arrays (padded to MAX_OBS)."""
    centers = np.zeros((_MAX_OBS, 2))
    radii   = np.full(_MAX_OBS, 999.0)   # large radius → very safe
    for i, obs in enumerate(obstacles[:_MAX_OBS]):
        centers[i] = obs.center
        radii[i]   = obs.radius
    return centers, radii


# ── QP helpers ────────────────────────────────────────────────────────────────

def _build_cbf_matrices(
    pos_xy:    np.ndarray,
    yaw:       float,
    obstacles: List[Obstacle],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute CBF constraint matrices for Dykstra QP.

    Returns
    -------
    A_cbf : (n_obs, 2)  gradient rows ∂h_i/∂p
    h_vals : (n_obs,)   barrier values h_i
    """
    if not obstacles:
        return np.zeros((0, 2)), np.zeros(0)

    grads  = np.vstack([
        cbf_gradient(pos_xy, obs.center, yaw, obs.radius)
        for obs in obstacles
    ])                                                # (n, 2)
    h_vals = np.array([
        cbf_value(pos_xy, obs.center, yaw, obs.radius)
        for obs in obstacles
    ])                                                # (n,)
    return grads, h_vals


# ── Soft target update ────────────────────────────────────────────────────────

def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


# ── Main training loop ────────────────────────────────────────────────────────

def train(
    n_episodes:  int  = 300,
    n_steps:     int  = _T,
    out_path:    str  = 'kappa_net.pth',
    use_viewer:  bool = False,
    device_str:  str  = 'cpu',
) -> KappaNet:
    """
    Run DDPG training and return the trained κ-net.

    Parameters
    ----------
    n_episodes  : number of training episodes
    n_steps     : max steps per episode
    out_path    : where to save the final κ-net weights
    use_viewer  : launch MuJoCo viewer for debugging
    device_str  : torch device ('cpu' or 'cuda')
    """
    device = torch.device(device_str)

    # ── Networks ──────────────────────────────────────────────────────────────
    kappa_net    = KappaNet(hidden_dim=7).to(device)
    critic       = CriticNet(_STATE_DIM).to(device)
    critic_tgt   = CriticNet(_STATE_DIM).to(device)
    critic_tgt.load_state_dict(critic.state_dict())

    opt_kappa    = optim.Adam(kappa_net.parameters(), lr=_LR_KAPPA)
    opt_critic   = optim.Adam(critic.parameters(),    lr=_LR_CRITIC)

    # ── Environment ───────────────────────────────────────────────────────────
    ctrl = CtrlInterface()
    env  = Go2TrainingEnv(use_viewer=use_viewer)
    env.startup(ctrl)

    # ── QP solver ─────────────────────────────────────────────────────────────
    cbf_qp = CBFQP(v_max=_V_MAX, max_obstacles=_MAX_OBS)

    # ── Replay buffer + noise ──────────────────────────────────────────────────
    buffer   = ReplayBuffer(_BUFFER_SIZE)
    ou_noise = OUNoise(2)

    total_steps = 0

    for ep in range(n_episodes):
        # Sample new episode geometry
        start, goal, obstacles = _sample_episode()
        obs_centers, obs_radii = _pack_obs(obstacles)
        env.set_vis_obstacles(obstacles)

        # Teleport robot to start
        env.reset(start, heading=0.0, ctrl=ctrl)
        # Switch mpac to walk mode so it responds to velocity commands
        ctrl.walk(vx=0, vy=0, vrz=0)
        time.sleep(0.3)

        ou_noise.reset()
        state_dict = env.get_state()
        ep_reward  = 0.0
        ep_min_h   = float('inf')

        for step in range(n_steps):
            pos_xy   = state_dict['pos_xy']
            yaw      = state_dict['yaw']
            vel_body = state_dict['vel_body']

            # ── Build state & nominal action ──────────────────────────────────
            state_vec = build_state(pos_xy, yaw, vel_body, goal, obstacles)
            u_rl      = nominal_policy(pos_xy, yaw, goal) + ou_noise.sample()
            u_rl      = np.clip(u_rl, -_V_MAX, _V_MAX)

            # ── κ filter ──────────────────────────────────────────────────────
            vrz           = nominal_yaw_rate(pos_xy, yaw, goal)
            A_cbf, h_vals = _build_cbf_matrices(pos_xy, yaw, obstacles)
            if len(h_vals) > 0:
                h_t      = torch.tensor(h_vals, dtype=torch.float32,
                                        device=device)
                with torch.no_grad():
                    kappa_vals = kappa_net(h_t).cpu().numpy()
                # Forward-invariance cap: κ(h) ≤ h/dt ensures h cannot go
                # negative in one step (discrete-time CBF condition).
                kappa_vals = np.minimum(
                    kappa_vals,
                    np.maximum(h_vals, 0.0) / _CTRL_DT)
                b_cbf = -kappa_vals   # (n_obs,)

                # Rotate u_rl to world frame for QP
                u_rl_world = body_to_world(u_rl, yaw)
                u_rl_t     = torch.tensor(u_rl_world[None],
                                          dtype=torch.float32, device=device)
                A_t        = torch.tensor(A_cbf[None],
                                          dtype=torch.float32, device=device)
                b_t        = torch.tensor(b_cbf[None],
                                          dtype=torch.float32, device=device)
                with torch.no_grad():
                    u_safe_world = cbf_qp.solve_differentiable_batch(
                        u_rl_t, A_t, b_t)[0].cpu().numpy()
                u_safe_body = world_to_body(u_safe_world, yaw)

                ep_min_h = min(ep_min_h, float(h_vals.min()))
            else:
                u_safe_body = u_rl

            # ── Execute command ───────────────────────────────────────────────
            ctrl.walk(vx=float(u_safe_body[0]),
                      vy=float(u_safe_body[1]), vrz=vrz)
            time.sleep(_CTRL_DT)

            # ── Observe next state ────────────────────────────────────────────
            next_dict     = env.get_state()
            pos_xy_next   = next_dict['pos_xy']
            yaw_next      = next_dict['yaw']
            vel_body_next = next_dict['vel_body']

            dist_to_goal  = np.linalg.norm(pos_xy_next - goal)
            done          = dist_to_goal < _GOAL_RADIUS or step == n_steps - 1

            # Reward: negative normalised distance + safety penalty for h < 0.
            # Distance term bounded in [-1, 0]; safety penalty adds up to
            # -_SAFETY_WEIGHT per obstacle per step when fully penetrated.
            reward = -float(dist_to_goal) / _ARENA_DIAG - _STEP_PENALTY
            if len(h_vals) > 0:
                reward += _SAFETY_WEIGHT * float(np.sum(np.minimum(h_vals, 0.0)))
            ep_reward += reward

            # ── Next state obs for target QP ──────────────────────────────────
            obs_centers_next, obs_radii_next = _pack_obs(obstacles)

            next_state_vec = build_state(
                pos_xy_next, yaw_next, vel_body_next, goal, obstacles)

            # ── Store transition ──────────────────────────────────────────────
            buffer.push(Transition(
                state       = state_vec,
                action      = u_safe_body.astype(np.float32),
                reward      = reward,
                next_state  = next_state_vec,
                done        = done,
                obs_centers = obs_centers.astype(np.float32),
                obs_radii   = obs_radii.astype(np.float32),
                n_obs       = len(obstacles),
                robot_pos   = pos_xy.astype(np.float32),
                yaw         = yaw,
                obs_centers_next = obs_centers_next.astype(np.float32),
                obs_radii_next   = obs_radii_next.astype(np.float32),
                robot_pos_next   = pos_xy_next.astype(np.float32),
                yaw_next         = yaw_next,
            ))

            state_dict  = next_dict
            total_steps += 1

            # ── Network updates ───────────────────────────────────────────────
            if (len(buffer) >= _WARMUP_STEPS and
                    total_steps % _UPDATE_EVERY == 0):
                _update(buffer, kappa_net, critic, critic_tgt,
                        opt_kappa, opt_critic, cbf_qp, device)

            if done:
                break

        # Soft target update once per episode (τ=0.7 per-step would collapse
        # target → online in ~10 steps; once per episode is the paper's intent)
        if len(buffer) >= _WARMUP_STEPS:
            _soft_update(critic_tgt, critic, _TAU_TARGET)

        print(f'[ep {ep+1:4d}/{n_episodes}]  '
              f'reward={ep_reward:7.1f}  '
              f'min_h={ep_min_h:+.3f}  '
              f'buf={len(buffer)}')

    # ── Save ──────────────────────────────────────────────────────────────────
    ctrl.soft_stop()
    env.close()
    kappa_net.save(out_path)
    return kappa_net


# ── Network update step ───────────────────────────────────────────────────────

def _update(
    buffer:     ReplayBuffer,
    kappa_net:  KappaNet,
    critic:     CriticNet,
    critic_tgt: CriticNet,
    opt_kappa:  optim.Optimizer,
    opt_critic: optim.Optimizer,
    cbf_qp:     CBFQP,
    device:     torch.device,
) -> None:
    """One gradient step for critic and κ-net from a sampled mini-batch."""
    batch = buffer.sample(_BATCH_SIZE)

    # ── Unpack batch ──────────────────────────────────────────────────────────
    states      = torch.tensor(np.array([t.state      for t in batch]),
                               dtype=torch.float32, device=device)
    actions     = torch.tensor(np.array([t.action     for t in batch]),
                               dtype=torch.float32, device=device)
    rewards     = torch.tensor([t.reward     for t in batch],
                               dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.tensor(np.array([t.next_state for t in batch]),
                               dtype=torch.float32, device=device)
    dones       = torch.tensor([t.done       for t in batch],
                               dtype=torch.float32, device=device).unsqueeze(1)

    obs_centers = torch.tensor(np.array([t.obs_centers for t in batch]),
                               dtype=torch.float32, device=device)  # (B,MAX_OBS,2)
    obs_radii   = torch.tensor(np.array([t.obs_radii   for t in batch]),
                               dtype=torch.float32, device=device)  # (B,MAX_OBS)
    robot_pos   = torch.tensor(np.array([t.robot_pos   for t in batch]),
                               dtype=torch.float32, device=device)  # (B,2)
    yaws        = torch.tensor([t.yaw for t in batch],
                               dtype=torch.float32, device=device)  # (B,)

    obs_centers_next = torch.tensor(
        np.array([t.obs_centers_next for t in batch]),
        dtype=torch.float32, device=device)
    obs_radii_next = torch.tensor(
        np.array([t.obs_radii_next for t in batch]),
        dtype=torch.float32, device=device)
    robot_pos_next = torch.tensor(
        np.array([t.robot_pos_next for t in batch]),
        dtype=torch.float32, device=device)
    yaws_next = torch.tensor([t.yaw_next for t in batch],
                             dtype=torch.float32, device=device)

    B = len(batch)

    # ── Critic target: y = r + γ Q'(s', u_safe_next) ─────────────────────────
    with torch.no_grad():
        # Nominal action at next state — use the same batch helper as the κ update
        # to keep train/target consistent (correct _D_SCALE unscaling + magnitude clip)
        u_rl_next_body = _nominal_batch(next_states)

        # Build CBF constraints for next state and filter nominal action
        A_next, b_next = _build_cbf_batch(
            robot_pos_next, yaws_next, obs_centers_next, obs_radii_next,
            kappa_net, device)

        # Rotate to world frame, project, rotate back
        u_rl_next_world = _body_to_world_batch(u_rl_next_body, yaws_next)
        u_safe_next_world = cbf_qp.solve_differentiable_batch(
            u_rl_next_world, A_next, b_next)
        u_safe_next_body = _world_to_body_batch(u_safe_next_world, yaws_next)

        y = rewards + _GAMMA * critic_tgt(next_states, u_safe_next_body) * (1.0 - dones)

    # ── Critic update ─────────────────────────────────────────────────────────
    q_pred  = critic(states, actions)
    loss_q  = nn.functional.mse_loss(q_pred, y)

    opt_critic.zero_grad()
    loss_q.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    opt_critic.step()

    # ── κ update: maximise Q(s, u_safe(μ(s), κ(h))) ──────────────────────────
    # Gradients flow: Q → u_safe_body → Dykstra QP → b_cbf = -κ(h) → κ-net
    u_rl_body = _nominal_batch(states)                    # (B,2) body frame
    u_rl_world = _body_to_world_batch(u_rl_body, yaws)

    A_cur, b_cur = _build_cbf_batch(
        robot_pos, yaws, obs_centers, obs_radii,
        kappa_net, device)                                # b_cur is differentiable

    u_safe_world = cbf_qp.solve_differentiable_batch(u_rl_world, A_cur, b_cur)
    u_safe_body  = _world_to_body_batch(u_safe_world, yaws)

    # Freeze critic parameters during actor/κ update
    for p in critic.parameters():
        p.requires_grad_(False)

    loss_kappa = -critic(states.detach(), u_safe_body).mean()

    opt_kappa.zero_grad()
    loss_kappa.backward()
    nn.utils.clip_grad_norm_(kappa_net.parameters(), 1.0)
    opt_kappa.step()

    for p in critic.parameters():
        p.requires_grad_(True)

    # Target network update is done once per episode (see train loop),
    # not here — τ=0.7 per-step would collapse target→online in ~10 steps.


# ── Batch helpers ─────────────────────────────────────────────────────────────

def _body_to_world_batch(v_body: torch.Tensor, yaws: torch.Tensor) -> torch.Tensor:
    """Rotate (B,2) body-frame vectors to world frame using per-row yaws."""
    c = torch.cos(yaws)    # (B,)
    s = torch.sin(yaws)
    vx = c * v_body[:, 0] - s * v_body[:, 1]
    vy = s * v_body[:, 0] + c * v_body[:, 1]
    return torch.stack([vx, vy], dim=1)   # (B,2)


def _world_to_body_batch(v_world: torch.Tensor, yaws: torch.Tensor) -> torch.Tensor:
    """Rotate (B,2) world-frame vectors to body frame using per-row yaws."""
    c = torch.cos(yaws)
    s = torch.sin(yaws)
    vx =  c * v_world[:, 0] + s * v_world[:, 1]
    vy = -s * v_world[:, 0] + c * v_world[:, 1]
    return torch.stack([vx, vy], dim=1)   # (B,2)


def _nominal_batch(states: torch.Tensor) -> torch.Tensor:
    """
    Batch nominal policy from state vector.

    The goal direction is the first two elements of the state (d_goal_body /
    _D_SCALE), so u_RL ∝ d_goal_body gives a P-controller in body frame.
    """
    d_goal_body = states[:, 0:2] * _D_SCALE   # (B,2) body frame, unscaled
    u_rl        = _KP_NOMINAL * d_goal_body
    speed       = u_rl.norm(dim=1, keepdim=True).clamp(min=1e-6)
    scale       = torch.clamp(speed, max=_V_MAX) / speed
    return u_rl * scale   # (B,2) body frame, clipped to v_max


def _build_cbf_batch(
    robot_pos:   torch.Tensor,    # (B, 2)
    yaws:        torch.Tensor,    # (B,)
    obs_centers: torch.Tensor,    # (B, MAX_OBS, 2)
    obs_radii:   torch.Tensor,    # (B, MAX_OBS)
    kappa_net:   KappaNet,
    device:      torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build batched CBF constraint matrices A (B, MAX_OBS, 2) and
    b = -κ(h) (B, MAX_OBS) for the differentiable QP.

    Padded obstacle slots (radius > 100) get h = _H_SCALE → κ is large →
    b = -κ is large-negative → constraint is trivially satisfied.
    """
    B = robot_pos.shape[0]

    # Compute h for each obstacle: h = |d|²/(a+r)² − 1
    # d = robot_pos - obs_center (B, MAX_OBS, 2)
    d      = robot_pos.unsqueeze(1) - obs_centers           # (B,MAX_OBS,2)
    a_r    = (ROBOT_HALF_LENGTH + obs_radii).unsqueeze(-1)  # (B,MAX_OBS,1)
    h_vals = (d * d).sum(-1) / (a_r.squeeze(-1) ** 2) - 1  # (B,MAX_OBS)

    # Gradient: ∂h/∂p = 2 * d / (a+r)²   →   shape (B,MAX_OBS,2)
    A_cbf = 2.0 * d / (a_r ** 2)                           # (B,MAX_OBS,2)

    # κ(h) — differentiable; flatten for network, then reshape
    h_flat     = h_vals.reshape(-1)                         # (B*MAX_OBS,)
    kappa_flat = kappa_net(h_flat)                          # (B*MAX_OBS,)
    # No cap here — let gradients flow freely so the safety penalty in the
    # reward teaches κ to respect the invariance bound.  The cap is applied
    # at execution time (train rollout + deployment) to guarantee h ≥ 0.
    kappa_vals = kappa_flat.reshape(B, _MAX_OBS)            # (B,MAX_OBS)

    b_cbf = -kappa_vals                                     # (B,MAX_OBS)

    return A_cbf, b_cbf


# ── Rollout (eval) loop ───────────────────────────────────────────────────────

def rollout(
    model_path:  str  = 'kappa_net.pth',
    n_episodes:  int  = 20,
    n_steps:     int  = _T,
    use_viewer:  bool = False,
    device_str:  str  = 'cpu',
) -> None:
    """
    Evaluate a trained κ-net without any gradient updates.

    Logs per-episode: reward, goal reached, min_h, steps taken.
    Prints aggregate stats (goal-reach rate, mean reward, constraint
    violation rate) at the end.
    """
    device = torch.device(device_str)

    kappa_net = KappaNet(hidden_dim=7).to(device)
    kappa_net.load(model_path, device=device_str)
    kappa_net.eval()

    ctrl    = CtrlInterface()
    env     = Go2TrainingEnv(use_viewer=use_viewer)
    cbf_qp  = CBFQP(v_max=_V_MAX, max_obstacles=_MAX_OBS)
    env.startup(ctrl)

    reached   = 0
    ep_rewards: List[float] = []
    min_hs:     List[float] = []
    viol_steps  = 0   # steps where h < 0
    total_steps = 0

    for ep in range(n_episodes):
        start, goal, obstacles = _sample_episode()
        env.set_vis_obstacles(obstacles)
        env.reset(start, heading=0.0, ctrl=ctrl)
        ctrl.walk(vx=0, vy=0, vrz=0)
        time.sleep(0.3)

        state_dict = env.get_state()
        ep_reward  = 0.0
        ep_min_h   = float('inf')
        ep_steps   = 0

        for step in range(n_steps):
            pos_xy   = state_dict['pos_xy']
            yaw      = state_dict['yaw']
            vel_body = state_dict['vel_body']

            u_rl = nominal_policy(pos_xy, yaw, goal)   # no noise in eval
            vrz  = nominal_yaw_rate(pos_xy, yaw, goal)

            A_cbf, h_vals = _build_cbf_matrices(pos_xy, yaw, obstacles)
            if len(h_vals) > 0:
                ep_min_h = min(ep_min_h, float(h_vals.min()))
                if h_vals.min() < 0:
                    viol_steps += 1

                with torch.no_grad():
                    h_t        = torch.tensor(h_vals, dtype=torch.float32, device=device)
                    kappa_vals = kappa_net(h_t).cpu().numpy()
                # Forward-invariance cap
                kappa_vals = np.minimum(
                    kappa_vals,
                    np.maximum(h_vals, 0.0) / _CTRL_DT)
                b_cbf = -kappa_vals

                u_rl_world = body_to_world(u_rl, yaw)
                u_rl_t = torch.tensor(u_rl_world[None], dtype=torch.float32, device=device)
                A_t    = torch.tensor(A_cbf[None],      dtype=torch.float32, device=device)
                b_t    = torch.tensor(b_cbf[None],      dtype=torch.float32, device=device)
                with torch.no_grad():
                    u_safe_world = cbf_qp.solve_differentiable_batch(
                        u_rl_t, A_t, b_t)[0].cpu().numpy()
                u_safe_body = world_to_body(u_safe_world, yaw)
            else:
                u_safe_body = u_rl

            ctrl.walk(vx=float(u_safe_body[0]),
                      vy=float(u_safe_body[1]), vrz=vrz)
            time.sleep(_CTRL_DT)

            state_dict    = env.get_state()
            dist_to_goal  = np.linalg.norm(state_dict['pos_xy'] - goal)
            ep_reward    += -float(dist_to_goal) / _ARENA_DIAG
            ep_steps     += 1
            total_steps  += 1

            if dist_to_goal < _GOAL_RADIUS:
                reached += 1
                break

        if ep_min_h == float('inf'):
            ep_min_h = 0.0
        ep_rewards.append(ep_reward)
        min_hs.append(ep_min_h)

        goal_tag = 'GOAL' if ep_steps < n_steps else '    '
        print(f'[ep {ep+1:3d}/{n_episodes}] {goal_tag}  '
              f'reward={ep_reward:7.2f}  min_h={ep_min_h:+.3f}  '
              f'steps={ep_steps:3d}')

    ctrl.soft_stop()
    env.close()

    viol_pct = 100.0 * viol_steps / max(total_steps, 1)
    print()
    print('── Rollout summary ──────────────────────────────')
    print(f'  goal reach rate : {reached}/{n_episodes} '
          f'({100*reached/n_episodes:.0f}%)')
    print(f'  mean reward     : {np.mean(ep_rewards):.2f}')
    print(f'  mean min_h      : {np.mean(min_hs):+.3f}')
    print(f'  h<0 steps       : {viol_steps}/{total_steps} '
          f'({viol_pct:.1f}%)')
    print('─────────────────────────────────────────────────')


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train or evaluate the AM-CBF κ-net on Go2 MuJoCo sim')
    parser.add_argument('--rollout',  action='store_true',
                        help='evaluate a saved model instead of training')
    parser.add_argument('--model',    type=str, default='kappa_net.pth',
                        help='κ-net weights to load for --rollout')
    parser.add_argument('--episodes', type=int, default=300,
                        help='number of episodes')
    parser.add_argument('--steps',    type=int, default=_T,
                        help='max steps per episode')
    parser.add_argument('--out',      type=str, default='kappa_net.pth',
                        help='output path for κ-net weights (training only)')
    parser.add_argument('--viewer',   action='store_true',
                        help='launch MuJoCo viewer for debugging')
    parser.add_argument('--device',   type=str, default='cpu',
                        help='torch device (cpu or cuda)')
    args = parser.parse_args()

    print('Make sure mpac is running before proceeding.')
    input('Press Enter when mpac is up and ready...')

    if args.rollout:
        print(f'AM-CBF rollout  model={args.model}  episodes={args.episodes}')
        rollout(
            model_path  = args.model,
            n_episodes  = args.episodes,
            use_viewer  = args.viewer,
            device_str  = args.device,
        )
    else:
        print('AM-CBF training starting')
        print(f'  episodes={args.episodes}, steps/ep={args.steps}')
        print(f'  output={args.out}, viewer={args.viewer}')
        train(
            n_episodes = args.episodes,
            n_steps    = args.steps,
            out_path   = args.out,
            use_viewer = args.viewer,
            device_str = args.device,
        )


if __name__ == '__main__':
    main()
