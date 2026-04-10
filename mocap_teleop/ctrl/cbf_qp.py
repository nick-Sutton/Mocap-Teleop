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

def ellipse_Q(_yaw: float, r: float,
              a: float = ROBOT_HALF_LENGTH,
              _b: float = ROBOT_HALF_WIDTH) -> np.ndarray:
    """
    Shape matrix Q — yaw-independent circular approximation.

    Q = I / (a+r)²   →   h = |pos−center|² / (a+r)² − 1

    Using the robot's maximum semi-axis (a, the forward length) as the uniform
    clearance radius gives a conservative safe set that holds for any heading.

    The previous elliptical formulation Q = R(yaw)·diag(1/(a+r)², 1/(b+r)²)·R(yaw)ᵀ
    caused h to change as the robot turned (always facing the goal), because
    ḣ = ∂h/∂pos·u + ∂h/∂yaw·ẏaw.  The CBF-QP only controls the first term;
    the uncontrolled heading-rate term was pushing h negative during navigation
    even with a maximally conservative α ≈ 0.  The circular Q eliminates the
    yaw term entirely (∂h/∂yaw = 0).
    """
    R_sq = (a + r) ** 2
    return np.eye(2) / R_sq


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
        u_perf_body:   np.ndarray,
        obstacles:     List[Obstacle],
        p_robot:       np.ndarray,
        yaw:           float,
        k_vals:        List[Tuple[float, float]],
        v_actual_body: np.ndarray,
    ) -> np.ndarray:
        """
        Solve the ECBF-QP with OSQP.  Use this at runtime.

        ECBF constraint for a 2nd-order (first-order lag) system:
            A · u_cmd ≥ −k1 · ḣ − k2 · h
        where ḣ = (∂h/∂p) · v_actual.

        Parameters
        ----------
        u_perf_body   : (2,) nominal velocity command in body frame [vx, vy]
        obstacles     : list of Obstacle (world frame)
        p_robot       : (2,) robot position in world frame
        yaw           : robot heading (rad)
        k_vals        : list of (k1, k2) per obstacle — ECBF gains from α-net
        v_actual_body : (2,) actual robot velocity in body frame

        Returns
        -------
        u_safe_body : (2,) safe velocity command in body frame [vx, vy]
                      Falls back to u_perf_body if the solver fails.
        """
        u_perf_world  = body_to_world(u_perf_body,   yaw)
        v_actual_world = body_to_world(v_actual_body, yaw)

        # ── QP matrices ───────────────────────────────────────────────────────
        # min  0.5·uᵀ P u + qᵀ u     (P = 2I, q = -2·u_perf  →  ||u-u_perf||²)
        P = sp.eye(2, format='csc') * 2.0
        q = -2.0 * u_perf_world

        # ── Constraint matrix ─────────────────────────────────────────────────
        # ECBF rows:  (∂h_i/∂p) · u_cmd ≥ −k1_i · ḣ_i − k2_i · h_i
        # where ḣ_i = (∂h_i/∂p) · v_actual
        # Box rows:   −v_max ≤ u_j ≤ v_max
        n = len(obstacles)
        if n > 0:
            grads = np.vstack([
                cbf_gradient(p_robot, obs.center, yaw, obs.radius,
                             self.robot_half_length, self.robot_half_width)
                for obs in obstacles
            ])                                          # (n, 2)
            h_vals = np.array([
                cbf_value(p_robot, obs.center, yaw, obs.radius,
                          self.robot_half_length, self.robot_half_width)
                for obs in obstacles
            ])
            # ḣ_i = ∂h_i/∂p · v_actual  (world frame dot product)
            h_dots = grads @ v_actual_world              # (n,)
            rhs = np.array([-k[0] * hd - k[1] * h
                             for k, hd, h in zip(k_vals, h_dots, h_vals)])  # (n,)
            # Normalise rows for consistent conditioning with training
            norms = np.linalg.norm(grads, axis=1, keepdims=True).clip(min=1e-6)
            A_cbf = grads / norms
            l_cbf = rhs   / norms.squeeze()                       # (n,)

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

    # ── Differentiable backend (Dykstra projection) ──────────────────────────

    def solve_differentiable_batch(
        self,
        u_perf_world_t: torch.Tensor,
        A_cbf_t:        torch.Tensor,
        b_cbf_t:        torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched differentiable CBF-QP via Dykstra's alternating projection.
        Use during offline training.

        All inputs and outputs are in the world frame.  The caller is
        responsible for rotating u_perf into world frame before calling and
        rotating the result back to body frame afterwards.

        Solves per batch element:
          min  ‖u − u_perf‖²
           u
          s.t.  A_cbf_i · u ≥ b_cbf_i    (CBF half-space per obstacle)
                −v_max ≤ u_j ≤ v_max      (box)

        Implemented as Dykstra's alternating projection onto the intersection
        of convex sets.  Each projection is a closed-form PyTorch op →
        fully differentiable via autograd, GPU-native, no external solver.

        Parameters
        ----------
        u_perf_world_t : (B, 2)        nominal velocity, world frame
        A_cbf_t        : (B, n_obs, 2) CBF constraint normals  ∂h/∂x
        b_cbf_t        : (B, n_obs)    lower bounds = −α_i

        Returns
        -------
        u_safe_world : (B, 2)  differentiable w.r.t. b_cbf_t (→ α-net weights)
        """
        B     = u_perf_world_t.shape[0]
        n_obs = A_cbf_t.shape[1]
        dev   = u_perf_world_t.device

        # ── Exact halfspace projection (Dykstra, one pass per constraint) ───────
        #
        # Each iteration projects u onto the halfspace {u : a_i·u ≥ b_i}.
        # For a single obstacle this is exact in one pass.  For multiple
        # obstacles the alternating-projection sequence converges (Dykstra),
        # though one pass is a reasonable approximation.
        #
        # relu(b − a·u) is the exact violation; zero when the constraint is
        # already satisfied, positive when violated.  When zero, no correction
        # is applied and no gradient flows through the QP step — which is
        # correct.  The gradient still reaches α-net via the trajectory:
        # h_now is in the computation graph (fix-1), so positions accumulated
        # from active-constraint steps carry gradient signal to later steps
        # even when those later steps are constraint-inactive.
        #
        # Crucially, this makes the training solver identical to OSQP for the
        # single-obstacle case, eliminating the train/eval mismatch that made
        # the softplus approach fail.

        n_cbf   = A_cbf_t.shape[1]
        u       = u_perf_world_t.clone()   # (B, 2)
        v_max_t = torch.tensor(self.v_max, dtype=u.dtype, device=dev)

        # Dykstra's alternating projection over all constraint sets.
        # For n_cbf=1 one cycle is exact.  For n_cbf>1 each cycle tightens
        # the solution; 15 cycles gives near-exact results for ≤5 constraints
        # in 2D at negligible cost (pure PyTorch ops, fully GPU-native).
        # The incremental correction vectors p (one per constraint) are the
        # standard Dykstra correction that ensures convergence to the true
        # projection onto the constraint intersection, not just the last set.
        p = torch.zeros(u.shape[0], n_cbf, 2, device=dev, dtype=u.dtype)  # (B, n_cbf, 2)

        for _ in range(15):
            for i in range(n_cbf):
                y    = u + p[:, i, :]                            # Dykstra correction
                a    = A_cbf_t[:, i, :]                          # (B, 2)
                b    = b_cbf_t[:, i]                             # (B,)
                dot  = (a * y).sum(-1)                           # (B,)
                a_sq = (a * a).sum(-1).clamp(min=1e-8)           # (B,)
                viol = torch.nn.functional.relu(b - dot)         # exact violation
                y_proj = y + (viol / a_sq).unsqueeze(-1) * a
                p[:, i, :] = p[:, i, :] + u - y_proj            # update correction
                u = y_proj

            # Box constraint projection inside the cycle so it participates
            # in the Dykstra convergence rather than being a hard override at end
            u = u.clamp(-v_max_t, v_max_t)

        return u
