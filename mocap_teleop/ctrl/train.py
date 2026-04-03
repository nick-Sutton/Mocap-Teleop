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

from mocap_teleop.ctrl.alpha_net import AlphaNet
from mocap_teleop.ctrl.cbf_qp import (CBFQP, Obstacle, ellipse_Q,
                                       ROBOT_HALF_LENGTH, ROBOT_HALF_WIDTH)
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


# ── Batched rollout (differentiable, GPU-compatible) ─────────────────────────

def _rollout_batch(
    alpha_net: AlphaNet,
    cbf_qp:    CBFQP,
    episodes:  List[Episode],
    device:    torch.device,
) -> torch.Tensor:
    """
    Roll out a full batch of episodes in parallel and return the mean loss.

    All per-timestep operations are vectorised over the batch dimension B so
    the GPU is properly utilised.  The only Python loop is over time (T steps).

    Loss per episode = Σ_t ‖pos_t − goal‖²  +  λ · Σ_t relu(−h_i(pos_t))²

    Returns
    -------
    mean_loss : scalar tensor, differentiable w.r.t. alpha_net.parameters()
    """
    B     = len(episodes)
    n_obs = len(episodes[0].obstacles)   # same across all episodes in a batch

    # ── Pre-extract episode data onto device ──────────────────────────────
    pos    = torch.tensor(
        np.stack([ep.start for ep in episodes]), dtype=torch.float32, device=device)
    x0     = pos.clone()                                              # (B, 2)
    goals  = torch.tensor(
        np.stack([ep.goal  for ep in episodes]), dtype=torch.float32, device=device)

    # Obstacle geometry: (B, n_obs, 2) and (B, n_obs)
    centers = torch.tensor(
        np.array([[obs.center for obs in ep.obstacles] for ep in episodes]),
        dtype=torch.float32, device=device)
    radii   = torch.tensor(
        np.array([[obs.radius for obs in ep.obstacles] for ep in episodes]),
        dtype=torch.float32, device=device)

    loss = torch.zeros(1, device=device)

    for _ in range(_T):
        pos_np = pos.detach().cpu().numpy()           # (B, 2)

        # ── Yaw: face the goal ────────────────────────────────────────────
        d_np  = goals.detach().cpu().numpy() - pos_np # (B, 2)
        yaws  = np.arctan2(d_np[:, 1], d_np[:, 0])   # (B,)

        # ── Rotation matrices R (body→world): (B, 2, 2) ──────────────────
        c = np.cos(yaws); s = np.sin(yaws)
        R_np = np.stack([np.stack([ c, -s], axis=1),
                         np.stack([ s,  c], axis=1)], axis=1)  # (B, 2, 2)
        R_t  = torch.tensor(R_np, dtype=torch.float32, device=device)

        # ── u_perf in world frame (batched PD controller) ─────────────────
        err_world  = d_np * 1.5                                # kp=1.5, (B,2)
        speed      = np.linalg.norm(err_world, axis=1, keepdims=True)
        mask       = speed > cbf_qp.v_max
        u_perf_world_np = np.where(
            mask, err_world * cbf_qp.v_max / (speed + 1e-8), err_world)
        u_perf_world_t = torch.tensor(
            u_perf_world_np, dtype=torch.float32, device=device)  # (B, 2)

        # u_perf in body frame for α-net feature (R^T @ u_world)
        u_perf_body_t = torch.einsum(
            'bij,bj->bi', R_t.transpose(1, 2), u_perf_world_t)   # (B, 2)

        # ── Elliptical CBF gradient A_cbf: (B, n_obs, 2) ─────────────────
        # Q(yaw, r) = R·diag(1/(a+r)², 1/(b+r)²)·Rᵀ  per (env, obstacle)
        r_np   = radii.detach().cpu().numpy()                     # (B, n_obs)
        a, b_  = ROBOT_HALF_LENGTH, ROBOT_HALF_WIDTH
        la     = 1.0 / (a  + r_np) ** 2                          # (B, n_obs)
        lb     = 1.0 / (b_ + r_np) ** 2
        # Expand R for each obstacle: (B, 1, 2, 2) → broadcast (B, n_obs, 2, 2)
        R_exp  = R_np[:, np.newaxis, :, :]                        # (B,1,2,2)
        Lambda = np.zeros((*r_np.shape, 2, 2))                    # (B,n_obs,2,2)
        Lambda[..., 0, 0] = la
        Lambda[..., 1, 1] = lb
        Q_np   = R_exp @ Lambda @ R_exp.transpose(0, 1, 3, 2)    # (B,n_obs,2,2)

        # d = pos − center: (B, n_obs, 2)
        d_obs_np = (pos_np[:, np.newaxis, :]
                    - centers.detach().cpu().numpy())
        # grad = 2·Q·d: einsum over (2,2)×(2,)→(2,)
        A_cbf_np = 2.0 * np.einsum('bnij,bnj->bni', Q_np, d_obs_np)
        A_cbf_t  = torch.tensor(A_cbf_np, dtype=torch.float32, device=device)

        # ── α-net: (B × n_obs) features in one forward pass ──────────────
        # Build (B, n_obs, 10) feature tensor then reshape to (B*n_obs, 10)
        p_robot_exp = pos.detach().unsqueeze(1).expand(-1, n_obs, -1)  # (B,n_obs,2)
        yaw_exp     = torch.tensor(yaws, dtype=torch.float32,
                                   device=device).unsqueeze(1).unsqueeze(2).expand(
                                       -1, n_obs, 1)                   # (B,n_obs,1)
        x0_exp      = x0.unsqueeze(1).expand(-1, n_obs, -1)            # (B,n_obs,2)
        u_exp       = u_perf_body_t.unsqueeze(1).expand(-1, n_obs, -1) # (B,n_obs,2)

        feat = AlphaNet.build_input(
            p_obs   = centers,                     # (B, n_obs, 2)
            r       = radii.unsqueeze(-1),         # (B, n_obs, 1)
            p_robot = p_robot_exp,
            yaw     = yaw_exp,
            x0      = x0_exp,
            u_perf  = u_exp,
        ).reshape(B * n_obs, AlphaNet.INPUT_DIM if hasattr(AlphaNet, 'INPUT_DIM')
                  else 10)

        alphas_flat = alpha_net(feat).reshape(B, n_obs)   # (B, n_obs)
        b_cbf_t     = -alphas_flat                         # (B, n_obs)

        # ── Batched differentiable CBF-QP ─────────────────────────────────
        u_safe_world = cbf_qp.solve_differentiable_batch(
            u_perf_world_t, A_cbf_t, b_cbf_t)             # (B, 2)

        # ── Kinematic integration ─────────────────────────────────────────
        pos = pos + u_safe_world * _DT                     # (B, 2)

        # ── Performance loss ──────────────────────────────────────────────
        loss = loss + torch.sum((pos - goals) ** 2)

        # ── Soft CBF violation penalty ────────────────────────────────────
        Q_t    = torch.tensor(Q_np, dtype=torch.float32, device=device) # (B,n_obs,2,2)
        d_obs  = pos.unsqueeze(1) - centers                              # (B,n_obs,2)
        h_vals = torch.einsum('bni,bnij,bnj->bn', d_obs, Q_t, d_obs) - 1.0
        loss   = loss + _SLACK_WEIGHT * torch.sum(torch.relu(-h_vals) ** 2)

    return loss / B


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

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alpha_net = AlphaNet(hidden_dim=64).to(device)
    cbf_qp    = CBFQP(v_max=1.5, max_obstacles=_MAX_OBSTACLES)
    optimizer = torch.optim.Adam(alpha_net.parameters(), lr=lr)

    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Training α-net  |  device={device}  iters={n_iters}  "
          f"envs/iter={n_envs}  obstacles={n_obstacles}  T={_T}  dt={_DT}s")
    print(f"{'Iter':>6}  {'Loss (mean)':>12}  {'Min h (safety)':>16}")
    print("-" * 42)

    for it in range(n_iters):
        optimizer.zero_grad()

        episodes  = [sample_episode(n_obstacles=n_obstacles) for _ in range(n_envs)]
        mean_loss = _rollout_batch(alpha_net, cbf_qp, episodes, device)
        mean_loss.backward()
        optimizer.step()

        # ── Logging ───────────────────────────────────────────────────────
        if it % 10 == 0 or it == n_iters - 1:
            min_h = _eval_min_cbf(alpha_net, cbf_qp, n_eval=20,
                                  n_obstacles=n_obstacles, device=device)
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
    n_eval:      int          = 20,
    n_obstacles: int          = 1,
    device:      torch.device = torch.device('cpu'),
) -> float:
    """
    Roll out n_eval episodes with OSQP (fast backend) and return the minimum
    CBF value seen across all steps and obstacles.  Negative means a collision.
    """
    from mocap_teleop.ctrl.simulator import KinematicSim

    pd_eval = PDController(kp=1.5, v_max=1.5)
    sim     = KinematicSim(dt=_DT)
    min_h   = float('inf')

    for _ in range(n_eval):
        episode = sample_episode(n_obstacles=n_obstacles)
        sim.reset(episode.start)
        x0 = episode.start.copy()

        for _ in range(_T):
            pos_np  = sim.pos
            yaw     = sim.heading_toward(episode.goal)
            sim.yaw = yaw

            u_perf_np = pd_eval(pos_np, episode.goal, yaw)
            u_perf_t  = torch.tensor(u_perf_np, dtype=torch.float32, device=device)

            alphas: List[float] = []
            for obs in episode.obstacles:
                feat = AlphaNet.build_input(
                    p_obs   = torch.tensor(obs.center,   dtype=torch.float32, device=device),
                    r       = torch.tensor([obs.radius], dtype=torch.float32, device=device),
                    p_robot = torch.tensor(pos_np,       dtype=torch.float32, device=device),
                    yaw     = torch.tensor([yaw],        dtype=torch.float32, device=device),
                    x0      = torch.tensor(x0,           dtype=torch.float32, device=device),
                    u_perf  = u_perf_t,
                )
                alphas.append(float(alpha_net(feat).cpu().item()))

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
