#!/usr/bin/env python3

"""
train.py — Offline training loop for the learned CBF α-net.

Implements Algorithm 1 from:
  "Learning Differentiable Safety-Critical Control using Control Barrier
   Functions for Generalization to Novel Environments"
  Ma, Zhang, Tomizuka, Sreenath — UC Berkeley

The α-net is trained end-to-end by backpropagating a trajectory performance
loss through the differentiable CBF-QP (cvxpylayers) into the network weights.

Gradient flow
─────────────
  θ  ──►  α_net  ──►  α_i  ──►  b_i = −α_i
                                   │
                              cvxpylayers QP
                                   │
                                  u*  ──►  pos_{t+1} = pos_t + R·u*·dt
                                                │
                                        loss = Σ‖pos_t − goal‖²
                                                │
                              ◄────────────────── backprop

Robot state (pos, yaw) is treated as a constant input to both α_net and the
QP constraint normals — only α carries gradients w.r.t. θ.  This matches
the paper's formulation where x is an observed quantity, not a learnable one.

Usage
─────
  python -m mocap_teleop.ctrl.train                 # default settings
  python -m mocap_teleop.ctrl.train --iters 200 --envs 30 --out alpha_net.pth
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn

from mocap_teleop.ctrl.alpha_net import AlphaNet
from mocap_teleop.ctrl.cbf_qp import CBFQP, Obstacle, ellipse_Q, ROBOT_HALF_LENGTH, ROBOT_HALF_WIDTH
from mocap_teleop.ctrl.simulator import Episode, PDController, sample_episode

# ── Hyperparameters ───────────────────────────────────────────────────────────

_MAX_OBSTACLES  = 3       # max obstacles per episode (must match CBFQP.max_obstacles)
_T              = 100     # rollout length (steps)
_DT             = 0.05    # simulation timestep (s)  →  5 s per episode
_N_ITERS        = 150     # training iterations
_N_ENVS         = 30      # environments sampled per iteration
_LR             = 1e-3    # Adam learning rate
_SLACK_WEIGHT   = 100.0   # λ_δ — penalty for CBF constraint violations
_CHECKPOINT_DIR = '.'     # directory to save checkpoints


# ── Per-episode rollout (differentiable) ─────────────────────────────────────

def _rollout(
    alpha_net: AlphaNet,
    cbf_qp:    CBFQP,
    pd_ctrl:   PDController,
    episode:   Episode,
) -> torch.Tensor:
    """
    Roll out one episode with the differentiable CBF-QP and return the loss.

    Loss = Σ_t ‖pos_t − goal‖²  +  λ_δ · Σ_t Σ_i relu(−h_i(pos_t))²

    The second term softly penalises any CBF-constraint violation that slips
    through during training (equivalent to a quadratic slack penalty).

    Returns
    -------
    loss : scalar tensor, differentiable w.r.t. alpha_net.parameters()
    """
    pos    = torch.tensor(episode.start,  dtype=torch.float32)
    x0_t   = pos.clone()
    goal_t = torch.tensor(episode.goal,   dtype=torch.float32)

    n_obs  = len(episode.obstacles)
    loss   = torch.zeros(1)

    for _ in range(_T):
        pos_np = pos.detach().numpy()

        # Yaw: always face the goal (training simplification)
        yaw = float(np.arctan2(
            episode.goal[1] - pos_np[1],
            episode.goal[0] - pos_np[0],
        ))

        # Nominal command from PD controller (stand-in for mocap at train time)
        u_perf_np = pd_ctrl(pos_np, episode.goal, yaw)
        u_perf_t  = torch.tensor(u_perf_np, dtype=torch.float32)

        # ── α-net forward pass (one call per obstacle) ─────────────────────
        alphas: List[torch.Tensor] = []
        for obs in episode.obstacles:
            feat  = AlphaNet.build_input(
                p_obs   = torch.tensor(obs.center,         dtype=torch.float32),
                r       = torch.tensor([obs.radius],       dtype=torch.float32),
                p_robot = pos.detach(),   # state is observed, not differentiated
                yaw     = torch.tensor([yaw],              dtype=torch.float32),
                x0      = x0_t,
                u_perf  = u_perf_t,
            )
            alphas.append(alpha_net(feat).squeeze())

        # Pad inactive slots to max_obstacles
        pad = [torch.ones(1).squeeze()] * (_MAX_OBSTACLES - n_obs)
        alphas_t = torch.stack((alphas + pad)[:_MAX_OBSTACLES])

        # ── Differentiable CBF-QP ──────────────────────────────────────────
        u_safe = cbf_qp.solve_differentiable(
            u_perf_body_t = u_perf_t,
            obstacles     = episode.obstacles,
            p_robot_t     = pos.detach(),   # constant in constraint normals
            yaw           = yaw,
            alphas_t      = alphas_t,
        )

        # ── Kinematic integration (differentiable through u_safe) ──────────
        c, s   = float(np.cos(yaw)), float(np.sin(yaw))
        R      = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
        pos    = pos + (R @ u_safe) * _DT

        # ── Performance loss ───────────────────────────────────────────────
        loss = loss + torch.sum((pos - goal_t) ** 2)

        # ── Soft CBF violation penalty (elliptical h) ─────────────────────
        for obs in episode.obstacles:
            p_obs = torch.tensor(obs.center, dtype=torch.float32)
            Q_t   = torch.tensor(
                ellipse_Q(yaw, obs.radius, ROBOT_HALF_LENGTH, ROBOT_HALF_WIDTH),
                dtype=torch.float32,
            )
            d = pos - p_obs
            h = d @ Q_t @ d - 1.0
            loss = loss + _SLACK_WEIGHT * torch.relu(-h) ** 2

    return loss


# ── Main training loop ────────────────────────────────────────────────────────

def train(
    n_iters:        int  = _N_ITERS,
    n_envs:         int  = _N_ENVS,
    n_obstacles:    int  = 1,
    lr:             float = _LR,
    checkpoint_dir: str  = _CHECKPOINT_DIR,
    out_name:       str  = 'alpha_net.pth',
) -> AlphaNet:
    """
    Train the α-net and return the trained model.

    Parameters
    ----------
    n_iters        : number of training iterations
    n_envs         : environments sampled per iteration
    n_obstacles    : obstacles per episode (≤ _MAX_OBSTACLES)
    lr             : Adam learning rate
    checkpoint_dir : directory for periodic checkpoints
    out_name       : filename for the final saved weights
    """
    assert n_obstacles <= _MAX_OBSTACLES, (
        f"n_obstacles={n_obstacles} exceeds _MAX_OBSTACLES={_MAX_OBSTACLES}. "
        f"Increase _MAX_OBSTACLES and CBFQP(max_obstacles=...) together."
    )

    alpha_net = AlphaNet(hidden_dim=64)
    cbf_qp    = CBFQP(v_max=1.5, max_obstacles=_MAX_OBSTACLES)
    pd_ctrl   = PDController(kp=1.5, v_max=1.5)
    optimizer = torch.optim.Adam(alpha_net.parameters(), lr=lr)

    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Training α-net  |  iters={n_iters}  envs/iter={n_envs}  "
          f"obstacles={n_obstacles}  T={_T}  dt={_DT}s")
    print(f"{'Iter':>6}  {'Loss (mean)':>12}  {'Min h (safety)':>16}")
    print("-" * 42)

    for it in range(n_iters):
        optimizer.zero_grad()
        total_loss = torch.zeros(1)

        for _ in range(n_envs):
            episode = sample_episode(n_obstacles=n_obstacles)
            total_loss = total_loss + _rollout(alpha_net, cbf_qp, pd_ctrl, episode)

        mean_loss = total_loss / n_envs
        mean_loss.backward()
        optimizer.step()

        # ── Logging ───────────────────────────────────────────────────────
        if it % 10 == 0 or it == n_iters - 1:
            min_h = _eval_min_cbf(alpha_net, cbf_qp, pd_ctrl, n_eval=20,
                                  n_obstacles=n_obstacles)
            print(f"{it:>6}  {mean_loss.item():>12.3f}  {min_h:>16.4f}")

        # ── Checkpoint ────────────────────────────────────────────────────
        if (it + 1) % 50 == 0:
            ckpt = os.path.join(checkpoint_dir, f'alpha_net_iter{it+1}.pth')
            alpha_net.save(ckpt)

    # Final save
    final_path = os.path.join(checkpoint_dir, out_name)
    alpha_net.save(final_path)
    print(f"\nSaved → {final_path}")

    return alpha_net


# ── Evaluation helper ─────────────────────────────────────────────────────────

@torch.no_grad()
def _eval_min_cbf(
    alpha_net:   AlphaNet,
    cbf_qp:      CBFQP,
    pd_ctrl:     PDController,
    n_eval:      int = 20,
    n_obstacles: int = 1,
) -> float:
    """
    Roll out n_eval episodes with OSQP (fast backend) and return the minimum
    CBF value seen across all steps and obstacles.  Negative means a collision.
    """
    from mocap_teleop.ctrl.simulator import KinematicSim

    sim     = KinematicSim(dt=_DT)
    min_h   = float('inf')

    for _ in range(n_eval):
        episode = sample_episode(n_obstacles=n_obstacles)
        sim.reset(episode.start)
        x0 = episode.start.copy()

        for _ in range(_T):
            pos_np  = sim.pos
            yaw     = sim.heading_toward(episode.goal)
            sim.yaw = yaw   # keep sim orientation in sync with heading

            u_perf_np = pd_ctrl(pos_np, episode.goal, yaw)
            u_perf_t  = torch.tensor(u_perf_np, dtype=torch.float32)

            alphas: List[float] = []
            for obs in episode.obstacles:
                feat = AlphaNet.build_input(
                    p_obs   = torch.tensor(obs.center,   dtype=torch.float32),
                    r       = torch.tensor([obs.radius], dtype=torch.float32),
                    p_robot = torch.tensor(pos_np,       dtype=torch.float32),
                    yaw     = torch.tensor([yaw],        dtype=torch.float32),
                    x0      = torch.tensor(x0,           dtype=torch.float32),
                    u_perf  = u_perf_t,
                )
                alphas.append(float(alpha_net(feat).item()))

            u_safe_np = cbf_qp.solve_fast(
                u_perf_body = u_perf_np,
                obstacles   = episode.obstacles,
                p_robot     = pos_np,
                yaw         = yaw,
                alphas      = alphas,
            )
            sim.step(u_safe_np)

            for obs in episode.obstacles:
                min_h = min(min_h, sim.cbf_value(obs))

    return min_h


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train the CBF α-net')
    parser.add_argument('--iters',      type=int,   default=_N_ITERS)
    parser.add_argument('--envs',       type=int,   default=_N_ENVS)
    parser.add_argument('--obstacles',  type=int,   default=1)
    parser.add_argument('--lr',         type=float, default=_LR)
    parser.add_argument('--out',        type=str,   default='alpha_net.pth')
    parser.add_argument('--ckpt-dir',   type=str,   default=_CHECKPOINT_DIR)
    args = parser.parse_args()

    train(
        n_iters        = args.iters,
        n_envs         = args.envs,
        n_obstacles    = args.obstacles,
        lr             = args.lr,
        checkpoint_dir = args.ckpt_dir,
        out_name       = args.out,
    )


if __name__ == '__main__':
    main()
