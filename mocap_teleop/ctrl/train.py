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
                                       world_to_body,
                                       ROBOT_HALF_LENGTH, ROBOT_HALF_WIDTH)
from mocap_teleop.ctrl.simulator import (Episode, PDController,
                                          sample_episode, sample_teleop_episode)

# ── Hyperparameters ───────────────────────────────────────────────────────────

_MAX_OBSTACLES  = 3       # max obstacles per episode (must match CBFQP.max_obstacles)
_T              = 200     # rollout length (steps)  →  10 s per episode at dt=0.05
_DT             = 0.05    # simulation timestep (s)
_TAU            = 0.25    # first-order lag time constant (s): v_max/a_max = 0.3/1.2
_V_MAX          = 0.3     # real Go2 vx limit (m/s); vy limit is 0.15 but use 0.3 for sym
_ARENA_HALF     = 2.0     # arena half-width for episode sampling (m) — matches real space
_MIN_SG_DIST    = 1.0     # minimum start-goal distance (m)
_N_ITERS        = 150     # training iterations
_N_ENVS         = 30      # environments sampled per iteration
_LR             = 3e-4    # Adam learning rate
_SLACK_WEIGHT        = 1000.0  # λ_δ — penalty for CBF constraint violations (goal mode)
_SLACK_WEIGHT_TELEOP = 500.0   # higher because path_length_sq ≈ 56 >> typical dist0_sq ≈ 9
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

    # Velocity state (world frame) — starts at rest each episode
    vel = torch.zeros(B, 2, dtype=torch.float32, device=device)

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

        # Actual velocity in body frame for α-net feature: R^T @ vel_world
        v_act_body_t = torch.einsum('bji,bj->bi', R_t, vel.detach())     # (B, 2)
        v_act_exp    = v_act_body_t.unsqueeze(1).expand(-1, n_obs, -1)   # (B, n_obs, 2)

        feat = AlphaNet.build_input(
            d_to_obs_body = d_to_obs_body_t,        # (B, n_obs, 2)
            r             = radii.unsqueeze(-1),    # (B, n_obs, 1)
            h             = h_feat,                 # (B, n_obs, 1)
            u_perf        = u_exp,                  # (B, n_obs, 2)
            d_to_goal     = d_to_goal_exp,          # (B, n_obs, 2)
            v_actual_body = v_act_exp,              # (B, n_obs, 2)
        ).reshape(B * n_obs, AlphaNet.INPUT_DIM)

        k_flat = alpha_net(feat).reshape(B, n_obs, 2)  # (B, n_obs, 2): [k1, k2]
        k1_flat = k_flat[..., 0]                        # (B, n_obs)
        k2_flat = k_flat[..., 1]                        # (B, n_obs)

        # ECBF constraint: A·u ≥ −k1·ḣ − k2·h
        # ḣ = (∂h/∂p) · v_actual; A_cbf_t already IS ∂h/∂p = 2Qd
        h_dot_t = torch.einsum('bni,bi->bn', A_cbf_t, vel.detach())  # (B, n_obs)

        # h for the QP RHS — NOT detached so gradient flows through trajectory
        d_now   = pos.unsqueeze(1) - centers               # (B, n_obs, 2)  [no detach]
        h_now   = (torch.einsum('bni,bnij,bnj->bn', d_now, Q_t_now, d_now)
                   - 1.0)                                  # (B, n_obs)
        b_cbf_t = -k1_flat * h_dot_t - k2_flat * h_now    # (B, n_obs)  ECBF RHS

        # ── Batched differentiable CBF-QP ─────────────────────────────────
        u_safe_world = cbf_qp.solve_differentiable_batch(
            u_perf_world_t, A_cbf_t, b_cbf_t)             # (B, 2)

        # ── First-order lag velocity integration ──────────────────────────
        # vel carries gradients so dL/dk propagates through future positions
        vel = vel + (u_safe_world - vel) * (_DT / _TAU)
        pos = pos + vel * _DT                     # (B, 2)

        # ── Performance loss (normalised per episode) ─────────────────────
        per_ep = torch.sum((pos - goals) ** 2, dim=1) / dist0_sq  # (B,)
        loss   = loss + torch.sum(per_ep)

        # ── Soft CBF violation penalty with safety margin (normalised) ───────
        # relu(margin − h)² penalises h < margin, not just h < 0.
        # Divided by n_obs so the penalty stays the same magnitude regardless
        # of obstacle count — without this, 2 obstacles → 2× safety gradient,
        # which overpowers the performance gradient and stalls learning.
        _H_MARGIN = 0.10   # raised: penalise closer approaches before h goes negative
        Q_t    = torch.tensor(Q_np, dtype=torch.float32, device=device) # (B,n_obs,2,2)
        d_obs  = pos.unsqueeze(1) - centers                              # (B,n_obs,2)
        h_vals = torch.einsum('bni,bnij,bnj->bn', d_obs, Q_t, d_obs) - 1.0
        viol   = (_SLACK_WEIGHT / n_obs) * torch.sum(
            torch.relu(_H_MARGIN - h_vals) ** 2, dim=1) / dist0_sq    # (B,)
        loss   = loss + torch.sum(viol)

    return loss / B


# ── Teleop rollout (differentiable) ──────────────────────────────────────────

def _rollout_batch_teleop(
    alpha_net: AlphaNet,
    cbf_qp:    CBFQP,
    episodes:  List[Episode],
    device:    torch.device,
) -> torch.Tensor:
    """
    Teleop rollout: the human walks straight at constant speed; the robot
    tries to track the human trajectory while avoiding obstacles.

    Loss per episode = Σ_t ‖pos(t) − human_pos(t)‖² / path_length²
                     + λ · Σ_t relu(margin − h_i(pos(t)))²  / path_length²

    where human_pos(t) = start + t · dt · v_human · human_dir.

    Key differences from _rollout_batch
    ─────────────────────────────────────
    • u_perf is constant: v_human · human_dir (human command, not PD).
    • Yaw is constant: direction of human travel.
    • d_to_goal feature → d_to_human(t) = human_pos(t) − pos(t) (dynamic).
    • Normalisation uses path_length² instead of initial distance² (robot and
      human start co-located, so dist0 = 0).
    """
    B     = len(episodes)
    n_obs = len(episodes[0].obstacles)

    pos = torch.tensor(
        np.stack([ep.start for ep in episodes]),
        dtype=torch.float32, device=device)                         # (B, 2)
    vel = torch.zeros(B, 2, dtype=torch.float32, device=device)    # world frame velocity
    centers = torch.tensor(
        np.array([[obs.center for obs in ep.obstacles] for ep in episodes]),
        dtype=torch.float32, device=device)                         # (B, n_obs, 2)
    radii = torch.tensor(
        np.array([[obs.radius for obs in ep.obstacles] for ep in episodes]),
        dtype=torch.float32, device=device)                         # (B, n_obs)

    # Human trajectory parameters (constant per episode)
    human_dirs = torch.tensor(
        np.stack([ep.human_dir for ep in episodes]),
        dtype=torch.float32, device=device)                         # (B, 2)
    v_humans = torch.tensor(
        [ep.v_human for ep in episodes],
        dtype=torch.float32, device=device)                         # (B,)

    # u_perf = v_human * human_dir (constant, no PD controller)
    u_perf_world_t = v_humans.unsqueeze(-1) * human_dirs           # (B, 2)

    # Yaw = human direction (constant — human command sets heading)
    yaws_np = np.arctan2(
        [ep.human_dir[1] for ep in episodes],
        [ep.human_dir[0] for ep in episodes])
    c_y = np.cos(yaws_np); s_y = np.sin(yaws_np)
    R_np = np.stack([np.stack([ c_y, -s_y], axis=1),
                     np.stack([ s_y,  c_y], axis=1)], axis=1)      # (B, 2, 2)
    R_t  = torch.tensor(R_np, dtype=torch.float32, device=device)  # constant

    # u_perf in body frame for α-net feature (R^T @ u_world)
    u_perf_body_t = torch.einsum(
        'bij,bj->bi', R_t.transpose(1, 2), u_perf_world_t)         # (B, 2)
    u_exp = u_perf_body_t.unsqueeze(1).expand(-1, n_obs, -1)       # (B, n_obs, 2)

    # CBF shape matrix Q = I/(a+r)² — yaw-independent, computed once
    r_np = radii.detach().cpu().numpy()
    a    = ROBOT_HALF_LENGTH
    la   = 1.0 / (a + r_np) ** 2
    Q_np = np.zeros((*r_np.shape, 2, 2))
    Q_np[..., 0, 0] = la
    Q_np[..., 1, 1] = la
    Q_t  = torch.tensor(Q_np, dtype=torch.float32, device=device)  # (B, n_obs, 2, 2)

    # Normalise by squared path length so loss is dimensionless
    path_length_sq = (v_humans * _T * _DT) ** 2                    # (B,)
    path_length_sq = path_length_sq.clamp(min=1e-3)

    # Human's starting position (same as robot start, no gradient needed)
    human_start = pos.detach().clone()                              # (B, 2)

    # Perpendicular to human direction — lateral deviation is α-dependent.
    # Forward deviation (robot falling behind) is α-independent: the robot
    # always moves at v_max, just sideways during avoidance.  Including the
    # forward component adds noise without gradient signal.
    perp_t = torch.stack(
        [-human_dirs[:, 1], human_dirs[:, 0]], dim=1)              # (B, 2)

    # Normalise by half-path-length squared: at the midpoint a 1 m lateral
    # deviation gives loss ≈ 1.  Keeps loss magnitude interpretable.
    lat_norm_sq = (v_humans * _T * _DT / 2.0) ** 2                 # (B,)
    lat_norm_sq = lat_norm_sq.clamp(min=1e-3)

    loss = torch.zeros(1, device=device)

    for t in range(_T):
        # Human position at end of this step (for loss) and start (for feature)
        human_pos_t   = (human_start
                         + (t + 1) * _DT * u_perf_world_t.detach()) # (B, 2)
        human_pos_now = (human_start
                         + t * _DT * u_perf_world_t.detach())       # (B, 2)

        # ── CBF gradients ─────────────────────────────────────────────────
        pos_np   = pos.detach().cpu().numpy()
        d_obs_np = (pos_np[:, np.newaxis, :]
                    - centers.detach().cpu().numpy())
        A_cbf_np = 2.0 * np.einsum('bnij,bnj->bni', Q_np, d_obs_np)
        A_cbf_t  = torch.tensor(A_cbf_np, dtype=torch.float32, device=device)

        # ── α-net features (relative body frame) ─────────────────────────
        pos_det       = pos.detach()

        d_to_obs_w    = centers - pos_det.unsqueeze(1)             # (B, n_obs, 2)
        d_to_obs_body = torch.einsum(
            'bji,bnj->bni', R_t, d_to_obs_w)                      # (B, n_obs, 2)

        d_now_det = pos_det.unsqueeze(1) - centers
        h_feat    = (torch.einsum('bni,bnij,bnj->bn',
                                  d_now_det, Q_t, d_now_det) - 1.0
                     ).unsqueeze(-1)                               # (B, n_obs, 1)

        # d_to_human in body frame — dynamic lateral-deviation signal
        d_to_human_w   = human_pos_now - pos_det                  # (B, 2)
        d_to_human_b   = torch.einsum(
            'bji,bj->bi', R_t, d_to_human_w)                     # (B, 2)
        d_to_human_exp = d_to_human_b.unsqueeze(1).expand(
            -1, n_obs, -1)                                        # (B, n_obs, 2)

        v_act_body   = torch.einsum('bji,bj->bi', R_t, vel.detach())        # (B, 2)
        v_act_exp    = v_act_body.unsqueeze(1).expand(-1, n_obs, -1)       # (B, n_obs, 2)

        feat = AlphaNet.build_input(
            d_to_obs_body = d_to_obs_body,
            r             = radii.unsqueeze(-1),
            h             = h_feat,
            u_perf        = u_exp,
            d_to_goal     = d_to_human_exp,
            v_actual_body = v_act_exp,
        ).reshape(B * n_obs, AlphaNet.INPUT_DIM)

        k_flat  = alpha_net(feat).reshape(B, n_obs, 2)             # (B, n_obs, 2)
        k1_flat = k_flat[..., 0]                                    # (B, n_obs)
        k2_flat = k_flat[..., 1]                                    # (B, n_obs)

        # ── ECBF QP RHS ───────────────────────────────────────────────────
        h_dot_t = torch.einsum('bni,bi->bn', A_cbf_t, vel.detach())  # (B, n_obs)
        d_now   = pos.unsqueeze(1) - centers
        h_now   = (torch.einsum('bni,bnij,bnj->bn', d_now, Q_t, d_now)
                   - 1.0)                                          # (B, n_obs)
        b_cbf_t = -k1_flat * h_dot_t - k2_flat * h_now

        # ── Solve ECBF-QP ─────────────────────────────────────────────────
        u_safe_world = cbf_qp.solve_differentiable_batch(
            u_perf_world_t, A_cbf_t, b_cbf_t)                     # (B, 2)

        # ── First-order lag velocity integration ──────────────────────────
        vel = vel + (u_safe_world - vel) * (_DT / _TAU)
        pos = pos + vel * _DT

        # ── Lateral-only teleop loss, gated on CBF activity ───────────────
        #
        # Option 2: only count steps where at least one obstacle is active
        # (h < _H_ACTIVE).  Steps far from all obstacles contribute zero
        # α-gradient — excluding them removes uninformative noise from the
        # gradient.
        #
        # Option 3: lateral deviation only — project (pos − human_pos) onto
        # perp (perpendicular to human walking direction).  The forward
        # component is α-independent (robot and human both walk at v_max
        # along the path); only the lateral push-away is α-dependent.
        #
        # Together these two changes make the gradient signal tight:
        # every term in the sum is both (a) near an obstacle and (b) measures
        # only the α-dependent component of deviation.

        # ── Lateral-only teleop loss ──────────────────────────────────────────
        # Project (pos − human_pos) onto perp (perpendicular to human dir).
        # Forward deviation is α-independent; lateral push-away is α-dependent.
        lat_dev = ((pos - human_pos_t) * perp_t).sum(dim=1)        # (B,)
        per_ep  = lat_dev ** 2 / lat_norm_sq
        loss    = loss + torch.sum(per_ep)

        # ── Safety penalty (normalised by n_obs) ─────────────────────────────
        _H_MARGIN = 0.10
        d_obs  = pos.unsqueeze(1) - centers
        h_vals = (torch.einsum('bni,bnij,bnj->bn', d_obs, Q_t, d_obs) - 1.0)
        viol   = (_SLACK_WEIGHT_TELEOP / n_obs) * torch.sum(
            torch.relu(_H_MARGIN - h_vals) ** 2, dim=1) / lat_norm_sq
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
    teleop:         bool  = False,
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
    teleop         : if True, use teleop loss (deviation from moving human)
                     instead of static goal-reaching loss
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
    cbf_qp    = CBFQP(v_max=_V_MAX, max_obstacles=_MAX_OBSTACLES)
    optimizer = torch.optim.Adam(alpha_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_iters, eta_min=lr * 0.01)

    os.makedirs(checkpoint_dir, exist_ok=True)

    mode_str = "teleop" if teleop else "goal-reaching"
    print(f"Training α-net  |  device={device}  iters={n_iters}  "
          f"envs/iter={n_envs}  obstacles={n_obstacles}  T={_T}  dt={_DT}s"
          f"  mode={mode_str}")
    print(f"{'Iter':>6}  {'Loss (mean)':>12}  {'Min h':>8}  {'α mean':>8}  {'α std':>8}")
    print("-" * 56)

    _rollout = _rollout_batch_teleop if teleop else _rollout_batch

    def _sample(n_obstacles):
        if teleop:
            return sample_teleop_episode(n_obstacles=n_obstacles,
                                         arena_half=_ARENA_HALF)
        return sample_episode(n_obstacles=n_obstacles,
                              arena_half=_ARENA_HALF,
                              min_sg_dist=_MIN_SG_DIST)

    for it in range(n_iters):
        optimizer.zero_grad()

        episodes  = [_sample(n_obstacles) for _ in range(n_envs)]
        mean_loss = _rollout(alpha_net, cbf_qp, episodes, device)
        mean_loss.backward()
        # Raised from 1.0: full BPTT through h_now produces larger gradient
        # norms; clipping too aggressively at 1.0 crushes the signal.
        torch.nn.utils.clip_grad_norm_(alpha_net.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        # ── Logging ───────────────────────────────────────────────────────
        if it % 10 == 0 or it == n_iters - 1:
            min_h, a_mean, a_std = _eval_min_cbf(
                alpha_net, cbf_qp, n_eval=50,
                n_obstacles=n_obstacles, teleop=teleop, device=device)
            print(f"{it:>6}  {mean_loss.item():>12.3f}  {min_h:>8.4f}"
                  f"  {a_mean:>8.3f}  {a_std:>8.3f}"
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
    teleop:      bool         = False,
    device:      torch.device = torch.device('cpu'),
) -> tuple:
    """
    Roll out n_eval episodes with OSQP (fast backend) and return:
        (min_h, alpha_mean, alpha_std)

    min_h      : minimum CBF value seen — negative means collision
    alpha_mean : mean α over all steps/obstacles — indicates aggressiveness
    alpha_std  : std of α — indicates environment-dependent adaptation
                 (near-zero means network outputs constant α regardless of env)
    """
    from mocap_teleop.ctrl.simulator import KinematicSim

    pd_eval    = PDController(kp=1.5, v_max=_V_MAX)
    sim        = KinematicSim(dt=_DT, tau=_TAU)
    min_h      = float('inf')
    all_k1: List[float] = []
    all_k2: List[float] = []

    def _sample_ep(n_obs):
        if teleop:
            return sample_teleop_episode(n_obstacles=n_obs, arena_half=_ARENA_HALF)
        return sample_episode(n_obstacles=n_obs, arena_half=_ARENA_HALF,
                              min_sg_dist=_MIN_SG_DIST)

    for _ in range(n_eval):
        episode  = _sample_ep(n_obstacles)
        sim.reset(episode.start)
        human_pos = episode.start.copy()

        for step in range(_T):
            pos_np = sim.pos
            v_act_world = sim.vel   # world-frame actual velocity

            if teleop:
                yaw = float(np.arctan2(episode.human_dir[1], episode.human_dir[0]))
                sim.yaw = yaw
                c, s = np.cos(yaw), np.sin(yaw)
                R    = np.array([[c, -s], [s, c]])
                u_perf_np  = world_to_body(
                    episode.v_human * episode.human_dir, yaw)
                human_pos  = episode.start + step * _DT * episode.v_human * episode.human_dir
                d_to_ref   = R.T @ (human_pos - pos_np)
            else:
                yaw = sim.heading_toward(episode.goal)
                sim.yaw = yaw
                c, s = np.cos(yaw), np.sin(yaw)
                R    = np.array([[c, -s], [s, c]])
                u_perf_np = pd_eval(pos_np, episode.goal, yaw)
                d_to_ref  = R.T @ (episode.goal - pos_np)

            u_perf_t      = torch.tensor(u_perf_np,  dtype=torch.float32, device=device)
            d_to_ref_t    = torch.tensor(d_to_ref,   dtype=torch.float32, device=device)
            v_act_body_np = R.T @ v_act_world
            v_act_body_t  = torch.tensor(v_act_body_np, dtype=torch.float32, device=device)

            k_vals: List[tuple] = []
            for obs in episode.obstacles:
                d_w           = obs.center - pos_np
                d_to_obs_body = R.T @ d_w
                h_val         = float(
                    np.dot(d_w, d_w) / (ROBOT_HALF_LENGTH + obs.radius)**2 - 1.0)

                feat = AlphaNet.build_input(
                    d_to_obs_body = torch.tensor(
                        d_to_obs_body, dtype=torch.float32, device=device),
                    r             = torch.tensor(
                        [obs.radius], dtype=torch.float32, device=device),
                    h             = torch.tensor(
                        [h_val],      dtype=torch.float32, device=device),
                    u_perf        = u_perf_t,
                    d_to_goal     = d_to_ref_t,
                    v_actual_body = v_act_body_t,
                )
                k = alpha_net(feat).cpu().detach().numpy().flatten()  # [k1, k2]
                k_vals.append((float(k[0]), float(k[1])))
                all_k1.append(float(k[0]))
                all_k2.append(float(k[1]))

            u_safe_np = cbf_qp.solve_fast(
                u_perf_body=u_perf_np, obstacles=episode.obstacles,
                p_robot=pos_np, yaw=yaw, k_vals=k_vals,
                v_actual_body=R.T @ v_act_world)
            sim.step(u_safe_np)

            for obs in episode.obstacles:
                min_h = min(min_h, sim.cbf_value(obs))

    k1_arr = np.array(all_k1)
    k2_arr = np.array(all_k2)
    # Report k2 mean/std (analogous to old α) for the logging line
    return min_h, float(k2_arr.mean()), float(k2_arr.std())


# ── Baseline comparison ───────────────────────────────────────────────────────

def evaluate(
    model_path:  str,
    n_eval:      int  = 500,
    n_obstacles: int  = 1,
    slalom_only: bool = False,
    teleop:      bool = False,
) -> None:
    """
    Compare the learned α-net against fixed-α baselines.

    Reports per-policy:
      mean_loss  — average normalised trajectory cost
                   goal-reaching: Σ‖pos−goal‖²/dist0²
                   teleop:        Σ‖pos−human_pos(t)‖²/path_length²
      min_h      — minimum CBF value seen (negative = collision)
      coll_rate  — fraction of episodes with at least one h < 0 step
      mean_alpha — average α output (for diagnosing aggression)
    """
    from mocap_teleop.ctrl.simulator import KinematicSim

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alpha_net = AlphaNet(hidden_dim=64).to(device)
    alpha_net.load(model_path, device=str(device))

    cbf_qp  = CBFQP(v_max=_V_MAX, max_obstacles=_MAX_OBSTACLES)
    pd_eval = PDController(kp=1.5, v_max=_V_MAX)
    sim     = KinematicSim(dt=_DT, tau=_TAU)

    # Pre-sample episodes so every policy sees the same scenarios
    if teleop:
        episodes = [sample_teleop_episode(n_obstacles=n_obstacles,
                                           arena_half=_ARENA_HALF)
                    for _ in range(n_eval)]
    else:
        episodes = [sample_episode(n_obstacles=n_obstacles,
                                   arena_half=_ARENA_HALF,
                                   min_sg_dist=_MIN_SG_DIST,
                                   force_slalom=slalom_only)
                    for _ in range(n_eval)]

    def _run_policy(k_fn):
        """k_fn(pos_np, yaw, R, obs, u_perf_body_t, ref_body, v_act_body_t) → (k1, k2)"""
        total_loss   = 0.0
        global_min_h = float('inf')
        n_collisions = 0
        alpha_by_idx: dict = {}

        for ep in episodes:
            sim.reset(ep.start)

            if teleop:
                norm_sq = max((ep.v_human * _T * _DT) ** 2, 1e-3)
            else:
                norm_sq = max(float(np.sum((ep.start - ep.goal) ** 2)), 1e-3)

            ep_loss  = 0.0
            ep_min_h = float('inf')

            for step in range(_T):
                pos_np = sim.pos

                if teleop:
                    yaw = float(np.arctan2(ep.human_dir[1], ep.human_dir[0]))
                    sim.yaw = yaw   # must be set so sim.step() rotates u correctly
                    c, s = np.cos(yaw), np.sin(yaw)
                    R    = np.array([[c, -s], [s, c]])
                    u_perf_np = world_to_body(
                        ep.v_human * ep.human_dir, yaw)        # body frame
                    human_pos = (ep.start
                                 + step * _DT * ep.v_human * ep.human_dir)
                    ref_body  = R.T @ (human_pos - pos_np)     # d_to_human
                else:
                    yaw = sim.heading_toward(ep.goal)
                    sim.yaw = yaw
                    c, s = np.cos(yaw), np.sin(yaw)
                    R    = np.array([[c, -s], [s, c]])
                    u_perf_np = pd_eval(pos_np, ep.goal, yaw)  # body frame
                    ref_body  = R.T @ (ep.goal - pos_np)       # d_to_goal

                u_perf_t     = torch.tensor(u_perf_np, dtype=torch.float32, device=device)
                v_act_body_np = R.T @ sim.vel
                v_act_body_t  = torch.tensor(v_act_body_np, dtype=torch.float32, device=device)

                k_vals = []
                for ki, obs in enumerate(ep.obstacles):
                    kv = k_fn(pos_np, yaw, R, obs, u_perf_t, ref_body, v_act_body_t)
                    k_vals.append(kv)
                    alpha_by_idx.setdefault(ki, []).append(kv[1])  # log k2 (position gain)

                u_safe_np = cbf_qp.solve_fast(
                    u_perf_body=u_perf_np, obstacles=ep.obstacles,
                    p_robot=pos_np, yaw=yaw, k_vals=k_vals,
                    v_actual_body=v_act_body_np)
                sim.step(u_safe_np)

                if teleop:
                    human_pos_next = (ep.start
                                      + (step + 1) * _DT
                                      * ep.v_human * ep.human_dir)
                    ep_loss += float(
                        np.sum((sim.pos - human_pos_next) ** 2)) / norm_sq
                else:
                    ep_loss += float(
                        np.sum((sim.pos - ep.goal) ** 2)) / norm_sq

                for obs in ep.obstacles:
                    ep_min_h = min(ep_min_h, sim.cbf_value(obs))

            total_loss   += ep_loss
            global_min_h  = min(global_min_h, ep_min_h)
            if ep_min_h < 0:
                n_collisions += 1

        mean_loss  = total_loss / n_eval
        coll_rate  = n_collisions / n_eval
        all_k2     = [a for vals in alpha_by_idx.values() for a in vals]
        mean_alpha = float(np.mean(all_k2)) if all_k2 else float('nan')
        std_alpha  = float(np.std(all_k2))  if all_k2 else float('nan')
        per_obs    = {k: float(np.mean(v))
                      for k, v in sorted(alpha_by_idx.items())}
        return mean_loss, global_min_h, coll_rate, mean_alpha, std_alpha, per_obs

    # ── Learned [k1, k2] ──────────────────────────────────────────────────────
    @torch.no_grad()
    def learned_k(pos_np, _yaw, R, obs, u_perf_t, ref_body, v_act_body_t):
        d_w           = obs.center - pos_np
        d_to_obs_body = R.T @ d_w
        h_val         = float(
            np.dot(d_w, d_w) / (ROBOT_HALF_LENGTH + obs.radius)**2 - 1.0)
        feat = AlphaNet.build_input(
            d_to_obs_body = torch.tensor(
                d_to_obs_body, dtype=torch.float32, device=device),
            r             = torch.tensor(
                [obs.radius], dtype=torch.float32, device=device),
            h             = torch.tensor(
                [h_val],      dtype=torch.float32, device=device),
            u_perf        = u_perf_t,
            d_to_goal     = torch.tensor(
                ref_body,    dtype=torch.float32, device=device),
            v_actual_body = v_act_body_t,
        )
        k = alpha_net(feat).cpu().numpy().flatten()
        return (float(k[0]), float(k[1]))

    mode_str = "teleop" if teleop else "goal-reaching"
    print(f"\nEvaluating on {n_eval} episodes  "
          f"(obstacles={n_obstacles}  mode={mode_str})\n")
    print(f"{'Policy':<20}  {'Mean loss':>10}  {'Min h':>8}  {'Coll%':>7}"
          f"  {'Mean k2':>8}  {'Std k2':>7}  Per-obstacle k2")
    print("-" * 82)

    loss, min_h, coll, mean_a, std_a, per_obs = _run_policy(learned_k)
    per_str = "  ".join(f"obs{k}={v:.2f}" for k, v in per_obs.items())
    print(f"{'learned [k1,k2]':<20}  {loss:>10.3f}  {min_h:>8.4f}  {coll*100:>6.1f}%"
          f"  {mean_a:>8.3f}  {std_a:>7.3f}  {per_str}")

    for fixed_k in [0.5, 1.0, 2.0, 5.0, 8.0, 10.0]:
        def fixed_k_fn(*_, k=fixed_k):
            return (k, k)   # k1 = k2 = fixed value
        loss, min_h, coll, mean_a, std_a, per_obs = _run_policy(fixed_k_fn)
        per_str = "  ".join(f"obs{k}={v:.2f}" for k, v in per_obs.items())
        print(f"{'fixed k1=k2='+str(fixed_k):<20}  {loss:>10.3f}  {min_h:>8.4f}  "
              f"{coll*100:>6.1f}%  {mean_a:>8.3f}  {std_a:>7.3f}  {per_str}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate the CBF α-net')
    sub = parser.add_subparsers(dest='cmd')

    # ── train subcommand (default behaviour) ──────────────────────────────────
    tr = sub.add_parser('train', help='train the α-net (default)')
    tr.add_argument('--iters',     type=int,   default=_N_ITERS)
    tr.add_argument('--envs',      type=int,   default=_N_ENVS)
    tr.add_argument('--obstacles', type=int,   default=1)
    tr.add_argument('--lr',        type=float, default=_LR)
    tr.add_argument('--out',       type=str,   default='alpha_net.pth')
    tr.add_argument('--ckpt-dir',  type=str,   default=_CHECKPOINT_DIR)
    tr.add_argument('--resume',    type=str,   default='',
                    help='checkpoint for warm restart (weights loaded, LR reset)')
    tr.add_argument('--teleop',    action='store_true',
                    help='train with teleop loss (deviation from moving human)')

    # ── eval subcommand ───────────────────────────────────────────────────────
    ev = sub.add_parser('eval', help='compare learned α against fixed-α baselines')
    ev.add_argument('model',          type=str,
                    help='path to saved α-net checkpoint (e.g. alpha_net_v2.pth)')
    ev.add_argument('--episodes',     type=int,            default=500)
    ev.add_argument('--obstacles',    type=int,            default=1)
    ev.add_argument('--slalom-only',   action='store_true',
                    help='evaluate on slalom episodes only')
    ev.add_argument('--teleop',        action='store_true',
                    help='evaluate with teleop loss (deviation from moving human)')

    # ── fallback: no subcommand → train with old-style flat args ─────────────
    parser.add_argument('--iters',     type=int,   default=_N_ITERS)
    parser.add_argument('--envs',      type=int,   default=_N_ENVS)
    parser.add_argument('--obstacles', type=int,   default=1)
    parser.add_argument('--lr',        type=float, default=_LR)
    parser.add_argument('--out',       type=str,   default='alpha_net.pth')
    parser.add_argument('--ckpt-dir',  type=str,   default=_CHECKPOINT_DIR)
    parser.add_argument('--resume',    type=str,   default='')
    args = parser.parse_args()

    if args.cmd == 'eval':
        evaluate(args.model, n_eval=args.episodes, n_obstacles=args.obstacles,
                 slalom_only=args.slalom_only, teleop=args.teleop)
    else:
        train(
            n_iters        = args.iters,
            n_envs         = args.envs,
            n_obstacles    = args.obstacles,
            lr             = args.lr,
            checkpoint_dir = args.ckpt_dir,
            out_name       = args.out,
            resume         = args.resume,
            teleop         = getattr(args, 'teleop', False),
        )


if __name__ == '__main__':
    main()
