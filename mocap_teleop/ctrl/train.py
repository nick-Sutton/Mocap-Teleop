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
_LR             = 3e-4    # Adam learning rate
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
    goals  = torch.tensor(
        np.stack([ep.goal  for ep in episodes]), dtype=torch.float32, device=device)

    # Obstacle geometry: (B, n_obs, 2) and (B, n_obs)
    centers = torch.tensor(
        np.array([[obs.center for obs in ep.obstacles] for ep in episodes]),
        dtype=torch.float32, device=device)
    radii   = torch.tensor(
        np.array([[obs.radius for obs in ep.obstacles] for ep in episodes]),
        dtype=torch.float32, device=device)

    # Per-episode normalisation: divide by initial squared distance to goal so
    # that hard episodes (far start-goal) don't dominate the gradient signal.
    dist0_sq = torch.sum((pos - goals) ** 2, dim=1).clamp(min=1e-3)  # (B,)

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

        # ── Circular CBF gradient A_cbf: (B, n_obs, 2) ──────────────────
        # Q = I/(a+r)²  →  h = |d|²/(a+r)² − 1  (yaw-independent)
        #
        # The previous yaw-dependent Q = R·diag(1/(a+r)², 1/(b+r)²)·Rᵀ caused
        # h to change as the robot turned (∂h/∂yaw ≠ 0), and that heading-rate
        # term is not controlled by the QP.  With circular Q the rotation R
        # cancels (R·(la·I)·Rᵀ = la·I), so h depends only on |d|² and the
        # CBF constraint fully controls ḣ.
        r_np   = radii.detach().cpu().numpy()                     # (B, n_obs)
        a      = ROBOT_HALF_LENGTH
        la     = 1.0 / (a + r_np) ** 2                           # (B, n_obs)
        # Q = la·I  (scalar per (batch, obstacle), broadcast over 2×2)
        Q_np                = np.zeros((*r_np.shape, 2, 2))       # (B,n_obs,2,2)
        Q_np[..., 0, 0]     = la
        Q_np[..., 1, 1]     = la

        # d = pos − center: (B, n_obs, 2)
        d_obs_np = (pos_np[:, np.newaxis, :]
                    - centers.detach().cpu().numpy())
        # grad = 2·Q·d: einsum over (2,2)×(2,)→(2,)
        A_cbf_np = 2.0 * np.einsum('bnij,bnj->bni', Q_np, d_obs_np)
        A_cbf_t  = torch.tensor(A_cbf_np, dtype=torch.float32, device=device)

        # ── α-net: (B × n_obs) features in one forward pass ──────────────
        # All features are relative to the robot body frame so the network
        # generalises across arbitrary world positions and headings.
        #
        # Features (8-D per obstacle):
        #   d_to_obs_body : obstacle centre relative to robot, in body frame
        #   r             : obstacle radius
        #   h             : current CBF value (how close to safety boundary)
        #   u_perf_body   : nominal velocity in body frame (direction of intent)
        #   d_to_goal     : goal relative to robot, in body frame (navigation urgency)
        #
        # Using absolute world coords (p_obs, p_robot) would cause the network
        # to memorise obstacle locations in the training arena instead of
        # learning the geometry of avoidance — useless for deployment.

        pos_det   = pos.detach()                                          # (B, 2)
        # R^T: world → body; einsum 'bji,bnj->bni' transposes the 2×2 block
        # d_to_obs in world frame: (B, n_obs, 2)
        d_to_obs_w = centers - pos_det.unsqueeze(1)                       # (B, n_obs, 2)
        # Rotate to body frame
        d_to_obs_body_t = torch.einsum(
            'bji,bnj->bni', R_t, d_to_obs_w)                             # (B, n_obs, 2)

        # h for features from detached pos (constant input to network)
        Q_t_now = torch.tensor(Q_np, dtype=torch.float32, device=device)
        d_now_det = pos_det.unsqueeze(1) - centers                        # (B, n_obs, 2)
        h_feat    = (torch.einsum('bni,bnij,bnj->bn', d_now_det, Q_t_now, d_now_det)
                     - 1.0).unsqueeze(-1)                                 # (B, n_obs, 1)

        # d_to_goal in body frame: (B, n_obs, 2)
        d_to_goal_w   = goals.detach() - pos_det                          # (B, 2)
        d_to_goal_b   = torch.einsum('bji,bj->bi', R_t, d_to_goal_w)     # (B, 2)
        d_to_goal_exp = d_to_goal_b.unsqueeze(1).expand(-1, n_obs, -1)   # (B, n_obs, 2)

        u_exp = u_perf_body_t.unsqueeze(1).expand(-1, n_obs, -1)          # (B, n_obs, 2)

        feat = AlphaNet.build_input(
            d_to_obs_body = d_to_obs_body_t,        # (B, n_obs, 2)
            r             = radii.unsqueeze(-1),    # (B, n_obs, 1)
            h             = h_feat,                 # (B, n_obs, 1)
            u_perf        = u_exp,                  # (B, n_obs, 2)
            d_to_goal     = d_to_goal_exp,          # (B, n_obs, 2)
        ).reshape(B * n_obs, AlphaNet.INPUT_DIM)

        alphas_flat = alpha_net(feat).reshape(B, n_obs)   # (B, n_obs)

        # Standard CBF class K constraint: ∂h/∂x · u ≥ −α · h(x)
        # RHS scales with current barrier value so constraint tightens
        # as the robot approaches the obstacle boundary (h → 0).
        #
        # h_now uses pos WITHOUT detach so gradient flows back through the
        # trajectory: dL/dα includes the cumulative effect of α on future
        # positions, not just the immediate QP correction.
        # h for the QP RHS — NOT detached so gradient flows through trajectory
        d_now   = pos.unsqueeze(1) - centers               # (B, n_obs, 2)  [no detach]
        h_now   = (torch.einsum('bni,bnij,bnj->bn', d_now, Q_t_now, d_now)
                   - 1.0)                                  # (B, n_obs)
        b_cbf_t = -alphas_flat * h_now                     # (B, n_obs)  [no clamp]

        # ── Batched differentiable CBF-QP ─────────────────────────────────
        u_safe_world = cbf_qp.solve_differentiable_batch(
            u_perf_world_t, A_cbf_t, b_cbf_t)             # (B, 2)

        # ── Kinematic integration ─────────────────────────────────────────
        pos = pos + u_safe_world * _DT                     # (B, 2)

        # ── Performance loss (normalised per episode) ─────────────────────
        per_ep = torch.sum((pos - goals) ** 2, dim=1) / dist0_sq  # (B,)
        loss   = loss + torch.sum(per_ep)

        # ── Soft CBF violation penalty with safety margin (normalised) ───────
        # relu(margin − h)² penalises h < margin, not just h < 0.
        # This creates a gradient that pushes α DOWN before the robot reaches
        # the physical boundary, giving a stable equilibrium at h ≈ margin
        # rather than h ≈ 0.  Without this, the performance gradient drives α
        # up until the robot reaches h = 0 exactly — which is unsafe in practice
        # due to discrete-time integration error.
        _H_MARGIN = 0.05
        Q_t    = torch.tensor(Q_np, dtype=torch.float32, device=device) # (B,n_obs,2,2)
        d_obs  = pos.unsqueeze(1) - centers                              # (B,n_obs,2)
        h_vals = torch.einsum('bni,bnij,bnj->bn', d_obs, Q_t, d_obs) - 1.0
        viol   = _SLACK_WEIGHT * torch.sum(
            torch.relu(_H_MARGIN - h_vals) ** 2, dim=1) / dist0_sq    # (B,)
        loss   = loss + torch.sum(viol)

    return loss / B


# ── Main training loop ────────────────────────────────────────────────────────

def train(
    n_iters:        int   = _N_ITERS,
    n_envs:         int   = _N_ENVS,
    n_obstacles:    int   = 1,
    lr:             float = _LR,
    checkpoint_dir: str   = _CHECKPOINT_DIR,
    out_name:       str   = 'alpha_net.pth',
    resume:         str   = '',
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
    resume         : path to a checkpoint to warm-restart from; the LR
                     schedule is reset to its initial value so training
                     continues with full step sizes from the loaded weights
    """
    assert n_obstacles <= _MAX_OBSTACLES, (
        f"n_obstacles={n_obstacles} exceeds _MAX_OBSTACLES={_MAX_OBSTACLES}. "
        f"Increase _MAX_OBSTACLES and CBFQP(max_obstacles=...) together."
    )

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alpha_net = AlphaNet(hidden_dim=64).to(device)
    if resume:
        alpha_net.load(resume, device=str(device))
        alpha_net.train()   # load() sets eval() — switch back for training
        print(f"Resumed from {resume}  (LR schedule reset to {lr:.2e})")
    cbf_qp    = CBFQP(v_max=1.5, max_obstacles=_MAX_OBSTACLES)
    optimizer = torch.optim.Adam(alpha_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_iters, eta_min=lr * 0.01)

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
        # Raised from 1.0: full BPTT through h_now produces larger gradient
        # norms; clipping too aggressively at 1.0 crushes the signal.
        torch.nn.utils.clip_grad_norm_(alpha_net.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        # ── Logging ───────────────────────────────────────────────────────
        if it % 10 == 0 or it == n_iters - 1:
            min_h = _eval_min_cbf(alpha_net, cbf_qp, n_eval=50,
                                  n_obstacles=n_obstacles, device=device)
            print(f"{it:>6}  {mean_loss.item():>12.3f}  {min_h:>16.4f}"
                  f"   lr={scheduler.get_last_lr()[0]:.2e}")

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

        for _ in range(_T):
            pos_np  = sim.pos
            yaw     = sim.heading_toward(episode.goal)
            sim.yaw = yaw

            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s], [s, c]])   # body → world; R^T is world → body

            u_perf_np = pd_eval(pos_np, episode.goal, yaw)
            u_perf_t  = torch.tensor(u_perf_np, dtype=torch.float32, device=device)

            # d_to_goal in body frame — same for all obstacles this step
            d_to_goal_body = torch.tensor(
                R.T @ (episode.goal - pos_np), dtype=torch.float32, device=device)

            alphas: List[float] = []
            for obs in episode.obstacles:
                a_half = ROBOT_HALF_LENGTH
                d_w    = obs.center - pos_np                               # world frame
                d_to_obs_body = R.T @ d_w                                  # body frame
                h_val  = float(np.dot(d_w, d_w) / (a_half + obs.radius)**2 - 1.0)

                feat = AlphaNet.build_input(
                    d_to_obs_body = torch.tensor(d_to_obs_body, dtype=torch.float32, device=device),
                    r             = torch.tensor([obs.radius],  dtype=torch.float32, device=device),
                    h             = torch.tensor([h_val],       dtype=torch.float32, device=device),
                    u_perf        = u_perf_t,
                    d_to_goal     = d_to_goal_body,
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
    parser.add_argument('--resume',     type=str,   default='',
                        help='path to checkpoint for warm restart (weights loaded, LR reset)')
    args = parser.parse_args()

    train(
        n_iters        = args.iters,
        n_envs         = args.envs,
        n_obstacles    = args.obstacles,
        lr             = args.lr,
        checkpoint_dir = args.ckpt_dir,
        out_name       = args.out,
        resume         = args.resume,
    )


if __name__ == '__main__':
    main()
