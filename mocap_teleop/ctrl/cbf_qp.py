#!/usr/bin/env python3

"""
cbf_qp.py — CBF-QP safety filter with two solver backends.

The QP is solved in the world (odom) frame so the CBF gradient is simple.
Results are transformed back into the robot body frame before returning,
matching the convention of CtrlInterface.walk(vx, vy, vrz).

Robot footprint
───────────────
The robot is modelled as an ellipse with half-axes (a, b) in body frame:
  a = ROBOT_HALF_LENGTH  (forward/backward)
  b = ROBOT_HALF_WIDTH   (lateral)

The obstacle is a circle with radius r.  The safe region is their Minkowski
complement: the robot centre must stay outside the combined ellipse
  (a + r) × (b + r)  oriented with the robot heading.

Problem (world frame, relative-degree 1):
─────────────────────────────────────────
  Q(yaw, r) = R(yaw) · diag(1/(a+r)², 1/(b+r)²) · R(yaw)ᵀ

  h_i(x) = (p_robot − p_obs_i)ᵀ Q_i (p_robot − p_obs_i) − 1  ≥ 0

  ∂h_i/∂x = 2 · Q_i · (p_robot − p_obs_i)  ∈ ℝ²

  min  ||u - u_perf||²
   u
  s.t. (∂h_i/∂x)ᵀ · u ≥ −α_i    for each obstacle i
       −v_max ≤ u_j ≤ v_max      box constraint

Two backends:
  solve_fast()           — OSQP,        use at runtime (1 kHz budget)
  solve_differentiable() — cvxpylayers, use during offline training only

Frame conventions:
  All positions are in world/odom frame.
  u_perf and the returned u_safe are both in robot body frame.
  The rotation between body ↔ world is handled internally using yaw.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
import osqp

import torch
from qpth.qp import QPFunction


# ── Go2 robot dimensions (with safety margin) ────────────────────────────────
# Lying-down footprint from docs: 76 cm × 31 cm  →  half-axes 38 cm × 15.5 cm
# Rounded up ~10 cm each axis for leg swing, sensor noise, and conservatism.
ROBOT_HALF_LENGTH = 0.50   # metres, forward/backward  (a)
ROBOT_HALF_WIDTH  = 0.20   # metres, lateral            (b)


# ── Obstacle representation ───────────────────────────────────────────────────

@dataclass
class Obstacle:
    """A circular obstacle in the world frame."""
    center: np.ndarray   # shape (2,), world-frame (x, y)
    radius: float


# ── Rotation helpers ──────────────────────────────────────────────────────────

def _R(yaw: float) -> np.ndarray:
    """2×2 rotation matrix: body → world."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s],
                     [s,  c]])


def body_to_world(v_body: np.ndarray, yaw: float) -> np.ndarray:
    return _R(yaw) @ v_body


def world_to_body(v_world: np.ndarray, yaw: float) -> np.ndarray:
    return _R(yaw).T @ v_world


# ── Elliptical CBF helpers ────────────────────────────────────────────────────

def ellipse_Q(yaw: float, r: float,
              a: float = ROBOT_HALF_LENGTH,
              b: float = ROBOT_HALF_WIDTH) -> np.ndarray:
    """
    Shape matrix Q for the combined robot-ellipse + obstacle-circle barrier.

    Q(yaw, r) = R(yaw) · diag(1/(a+r)², 1/(b+r)²) · R(yaw)ᵀ
    """
    R  = _R(yaw)
    La = np.diag([1.0 / (a + r) ** 2, 1.0 / (b + r) ** 2])
    return R @ La @ R.T


def cbf_value(p_robot: np.ndarray, p_obs: np.ndarray,
              yaw: float, r: float,
              a: float = ROBOT_HALF_LENGTH,
              b: float = ROBOT_HALF_WIDTH) -> float:
    """h(x) = (p_robot − p_obs)ᵀ Q (p_robot − p_obs) − 1."""
    d = p_robot - p_obs
    Q = ellipse_Q(yaw, r, a, b)
    return float(d @ Q @ d) - 1.0


def cbf_gradient(p_robot: np.ndarray, p_obs: np.ndarray,
                 yaw: float, r: float,
                 a: float = ROBOT_HALF_LENGTH,
                 b: float = ROBOT_HALF_WIDTH) -> np.ndarray:
    """∂h/∂x = 2 · Q(yaw, r) · (p_robot − p_obs), shape (2,)."""
    Q = ellipse_Q(yaw, r, a, b)
    return 2.0 * Q @ (p_robot - p_obs)


# ── Main class ────────────────────────────────────────────────────────────────

class CBFQP:
    """
    CBF-QP safety filter.

    Parameters
    ----------
    v_max         : maximum linear speed (m/s), applied as a box constraint
    max_obstacles : maximum number of obstacles used by the differentiable
                    (cvxpylayers) backend.  The fast (OSQP) backend handles
                    any number natively.
    """

    def __init__(self,
                 v_max:             float = 1.5,
                 max_obstacles:     int   = 5,
                 robot_half_length: float = ROBOT_HALF_LENGTH,
                 robot_half_width:  float = ROBOT_HALF_WIDTH):
        self.v_max             = v_max
        self.max_obstacles     = max_obstacles
        self.robot_half_length = robot_half_length
        self.robot_half_width  = robot_half_width

    # ── Fast backend (OSQP) ───────────────────────────────────────────────────

    def solve_fast(
        self,
        u_perf_body: np.ndarray,
        obstacles:   List[Obstacle],
        p_robot:     np.ndarray,
        yaw:         float,
        alphas:      List[float],
    ) -> np.ndarray:
        """
        Solve the CBF-QP with OSQP.  Use this at runtime.

        Parameters
        ----------
        u_perf_body : (2,) nominal velocity in body frame [vx, vy]
        obstacles   : list of Obstacle (world frame)
        p_robot     : (2,) robot position in world frame
        yaw         : robot heading (rad)
        alphas      : scalar α_i per obstacle (must have same length as obstacles)

        Returns
        -------
        u_safe_body : (2,) safe velocity in body frame [vx, vy]
                      Falls back to u_perf_body if the solver fails.
        """
        u_perf_world = body_to_world(u_perf_body, yaw)

        # ── QP matrices ───────────────────────────────────────────────────────
        # min  0.5·uᵀ P u + qᵀ u     (P = 2I, q = -2·u_perf  →  ||u-u_perf||²)
        P = sp.eye(2, format='csc') * 2.0
        q = -2.0 * u_perf_world

        # ── Constraint matrix ─────────────────────────────────────────────────
        # CBF rows:  (∂h_i/∂x) · u ≥ −α_i   →   lower bound = −α_i
        # Box rows:  −v_max ≤ u_j ≤ v_max
        n = len(obstacles)
        if n > 0:
            A_cbf = np.vstack([
                cbf_gradient(p_robot, obs.center, yaw, obs.radius,
                             self.robot_half_length, self.robot_half_width)
                for obs in obstacles
            ])                                          # (n, 2)
            l_cbf = np.array([-a for a in alphas])     # (n,) lower bounds

            A_full = sp.csc_matrix(np.vstack([A_cbf, np.eye(2)]))
            l_full = np.concatenate([l_cbf,          [-self.v_max, -self.v_max]])
            u_full = np.concatenate([np.full(n, np.inf), [self.v_max,  self.v_max]])
        else:
            A_full = sp.eye(2, format='csc')
            l_full = np.array([-self.v_max, -self.v_max])
            u_full = np.array([ self.v_max,  self.v_max])

        # ── Solve ─────────────────────────────────────────────────────────────
        solver = osqp.OSQP()
        solver.setup(
            P, q, A_full, l_full, u_full,
            warm_starting = True,
            verbose       = False,
            eps_abs       = 1e-4,
            eps_rel       = 1e-4,
            max_iter      = 1000,
        )
        result = solver.solve()

        if result.info.status in ('solved', 'solved_inaccurate'):
            u_world = result.x
        else:
            u_world = u_perf_world   # fallback: pass nominal command through

        return world_to_body(u_world, yaw)

    # ── Differentiable backend (qpth) ────────────────────────────────────────

    def solve_differentiable_batch(
        self,
        u_perf_world_t: torch.Tensor,
        A_cbf_t:        torch.Tensor,
        b_cbf_t:        torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched differentiable CBF-QP using qpth.  Use during offline training.

        All inputs and outputs are in the world frame.  The caller is
        responsible for rotating u_perf into world frame before calling and
        rotating the result back to body frame afterwards.

        QP solved per batch element:
          min  ‖u − u_perf‖²
           u
          s.t.  A_cbf_i · u ≥ b_cbf_i    (CBF constraints, one row per obstacle)
                −v_max ≤ u_j ≤ v_max      (box)

        qpth uses the convention  G·u ≤ h, so CBF rows are negated.
        A slack variable δ is added as a third optimisation variable so the
        QP is always feasible — exactly as in the paper (eq. 11).  δ is
        penalised with _ZETA so it stays near zero at optimum.

        Parameters
        ----------
        u_perf_world_t : (B, 2)        nominal velocity, world frame
        A_cbf_t        : (B, n_obs, 2) CBF constraint normals  ∂h/∂x
        b_cbf_t        : (B, n_obs)    lower bounds = −α_i

        Returns
        -------
        u_safe_world : (B, 2)  differentiable w.r.t. b_cbf_t (→ α-net weights)
        """
        from qpth.qp import QPFunction

        _ZETA = 1e2   # slack penalty: large enough δ≈0, small enough QP is well-scaled

        B     = u_perf_world_t.shape[0]
        n_obs = A_cbf_t.shape[1]
        dev   = u_perf_world_t.device

        # ── Optimisation variable: [u (2), δ (1)] ────────────────────────────
        # Cost: ‖u − u_perf‖² + ζ·δ² = uᵀu − 2·u_perfᵀu + ζ·δ²
        # In [u; δ] coords: Q_aug = diag(2I, 2ζ), p_aug = [-2·u_perf; 0]
        Q_aug = torch.zeros(B, 3, 3, device=dev)
        Q_aug[:, 0, 0] = 2.0
        Q_aug[:, 1, 1] = 2.0
        Q_aug[:, 2, 2] = 2.0 * _ZETA
        # Small regularisation for numerical stability
        Q_aug = Q_aug + 1e-6 * torch.eye(3, device=dev).unsqueeze(0)

        p_aug = torch.zeros(B, 3, device=dev)
        p_aug[:, :2] = -2.0 * u_perf_world_t

        # ── Inequality constraints G·[u;δ] ≤ h ───────────────────────────────
        # CBF (relaxed): −A_cbf·u + (−1)·δ ≤ −b_cbf   [n_obs rows]
        # Box:           ±I·u + 0·δ ≤ v_max            [4 rows]
        # δ ≥ 0:         0·u − δ ≤ 0                   [1 row]
        G_cbf  = torch.zeros(B, n_obs, 3, device=dev)
        G_cbf[:, :, :2] = -A_cbf_t                    # −A_cbf on u
        G_cbf[:, :,  2] = -1.0                        # −δ (relaxation)
        h_cbf  = -b_cbf_t                             # (B, n_obs)

        I2     = torch.eye(2, device=dev).unsqueeze(0).expand(B, -1, -1)
        G_box  = torch.zeros(B, 4, 3, device=dev)
        G_box[:, :2, :2] =  I2
        G_box[:, 2:, :2] = -I2
        h_box  = torch.full((B, 4), self.v_max, device=dev)

        G_slack       = torch.zeros(B, 1, 3, device=dev)
        G_slack[:, 0, 2] = -1.0                       # −δ ≤ 0  (δ ≥ 0)
        h_slack        = torch.zeros(B, 1, device=dev)

        G = torch.cat([G_cbf, G_box, G_slack], dim=1)   # (B, n_obs+5, 3)
        h = torch.cat([h_cbf, h_box, h_slack], dim=1)   # (B, n_obs+5)

        # Empty equality constraints
        A_eq = torch.zeros(B, 0, 3, device=dev)
        b_eq = torch.zeros(B, 0,    device=dev)

        sol = QPFunction(verbose=False)(Q_aug, p_aug, G, h, A_eq, b_eq)  # (B, 3)
        return sol[:, :2]   # discard δ, return u in world frame
