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
from typing import List, Tuple, Optional

import numpy as np
import scipy.sparse as sp
import osqp

import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


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
        self._layer: Optional[CvxpyLayer] = None   # built lazily

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

    # ── Differentiable backend (cvxpylayers) ──────────────────────────────────

    def _get_layer(self) -> CvxpyLayer:
        """Build (once) the differentiable QP layer for max_obstacles.

        Includes a scalar slack variable δ (paper eq. 11) that relaxes the
        CBF constraints when they are mutually infeasible.  A large penalty ζ
        keeps δ ≈ 0 at optimum so the constraints are effectively enforced,
        while still ensuring the QP always has a feasible solution and
        gradients always flow back to the α-net during training.
        """
        if self._layer is not None:
            return self._layer

        _ZETA = 1e2   # slack penalty — large enough that δ ≈ 0 at optimum

        n            = self.max_obstacles
        u            = cp.Variable(2)
        delta        = cp.Variable(nonneg=True)   # scalar slack ≥ 0
        u_perf_param = cp.Parameter(2)
        A_param      = cp.Parameter((n, 2))       # CBF constraint normals
        b_param      = cp.Parameter(n)            # lower bounds (= −α_i, padded)

        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(u - u_perf_param) + _ZETA * cp.square(delta)),
            [
                A_param @ u >= b_param - delta,   # relaxed CBF constraints
                u <=  self.v_max,
                u >= -self.v_max,
            ],
        )
        assert prob.is_dpp(), "QP must be DPP for cvxpylayers"

        self._layer = CvxpyLayer(
            prob,
            parameters=[u_perf_param, A_param, b_param],
            variables=[u, delta],
        )
        return self._layer

    def solve_differentiable(
        self,
        u_perf_body_t: torch.Tensor,
        obstacles:     List[Obstacle],
        p_robot_t:     torch.Tensor,
        yaw:           float,
        alphas_t:      torch.Tensor,
    ) -> torch.Tensor:
        """
        Solve the CBF-QP with cvxpylayers.  Use during offline training only.

        Obstacles beyond max_obstacles are ignored.  Fewer than max_obstacles
        are padded with inactive constraints (a = 0, b = −1e6).

        Parameters
        ----------
        u_perf_body_t : (2,)              nominal velocity, body frame
        obstacles     : list of Obstacle  (world frame, numpy)
        p_robot_t     : (2,)              robot position, world frame (tensor)
        yaw           : float             robot heading (rad) — treated as constant
        alphas_t      : (max_obstacles,)  α values from AlphaNet, padded if needed

        Returns
        -------
        u_safe_body_t : (2,)  safe velocity in body frame — differentiable w.r.t. alphas_t
        """
        # Rotation matrix (constant — yaw not differentiated)
        c, s = np.cos(yaw), np.sin(yaw)
        R = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)

        u_perf_world = R @ u_perf_body_t   # (2,)

        n = self.max_obstacles
        A_rows: List[torch.Tensor] = []
        b_vals: List[torch.Tensor] = []

        for i in range(n):
            if i < len(obstacles):
                obs   = obstacles[i]
                p_obs = torch.tensor(obs.center, dtype=torch.float32)
                # Elliptical CBF gradient: 2·Q(yaw,r)·(p_robot − p_obs)
                Q_np  = ellipse_Q(yaw, obs.radius,
                                  self.robot_half_length, self.robot_half_width)
                Q_t   = torch.tensor(Q_np, dtype=torch.float32)
                grad  = 2.0 * Q_t @ (p_robot_t - p_obs)   # (2,)
                A_rows.append(grad)
                b_vals.append(-alphas_t[i])
            else:
                # Inactive: constraint is always satisfied
                A_rows.append(torch.zeros(2))
                b_vals.append(torch.tensor(-1e6))

        A_t = torch.stack(A_rows)   # (n, 2)
        b_t = torch.stack(b_vals)   # (n,)

        (u_world, _delta) = self._get_layer()(u_perf_world, A_t, b_t)

        return R.T @ u_world.to(R.dtype)   # transform back to body frame
