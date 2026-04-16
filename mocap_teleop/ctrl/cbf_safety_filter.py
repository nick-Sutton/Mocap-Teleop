#!/usr/bin/env python3

"""
cbf_safety_filter.py — AM-CBF safety filter for deployment.

Sits between the motion mapper (u_perf) and the robot controller (CtrlInterface).
At each control tick:

    1. Receive nominal command u_perf_body from MotionMapper
    2. Receive obstacle list from perception (world frame)
    3. κ-net evaluates κ(h_i) per obstacle from barrier value h_i
    4. OSQP solves the CBF-QP:  ∇h_i · u  ≥  −κ(h_i)
    5. Return u_safe_body

The vrz (yaw rate) command is always passed through unchanged.  The CBF
only filters translational (vx, vy) commands.

Usage
─────
    filt = CbfSafetyFilter(model_path='kappa_net.pth')
    ...
    u_safe = filt.filter(
        u_perf_body = cmd_vel[:2],   # (vx, vy) body frame from MotionMapper
        robot_pos   = pos_xy,        # (2,) world frame
        robot_yaw   = yaw,           # float, radians
        obstacles   = obstacle_list, # List[Obstacle], world frame
    )
    CtrlInterface.walk(vx=u_safe[0], vy=u_safe[1], vrz=cmd_vel[2])
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from mocap_teleop.ctrl.kappa_net import KappaNet
from mocap_teleop.ctrl.cbf_qp import CBFQP, Obstacle, cbf_value


class CbfSafetyFilter:
    """
    AM-CBF safety filter using a trained κ-net.

    Parameters
    ----------
    model_path    : path to saved κ-net weights (.pth file)
    v_max         : maximum translational speed (m/s)
    max_obstacles : max obstacles to consider per step (closest are used)
    device        : torch device ('cpu' or 'cuda')
    enabled       : if False, filter is a passthrough (useful for A/B testing)
    """

    def __init__(
        self,
        model_path:    str,
        v_max:         float = 1.5,
        max_obstacles: int   = 3,
        device:        str   = 'cpu',
        enabled:       bool  = True,
    ):
        self.enabled       = enabled
        self.max_obstacles = max_obstacles
        self._device       = torch.device(device)

        self._kappa_net = KappaNet(hidden_dim=7).to(self._device)
        self._kappa_net.load(model_path, device=device)

        self._cbf_qp = CBFQP(v_max=v_max, max_obstacles=max_obstacles)

        # Diagnostic counters (reset each second by the node)
        self.n_filtered = 0
        self.n_total    = 0

    def filter(
        self,
        u_perf_body: np.ndarray,
        robot_pos:   np.ndarray,
        robot_yaw:   float,
        obstacles:   List[Obstacle],
    ) -> np.ndarray:
        """
        Apply the AM-CBF safety filter to a nominal body-frame velocity command.

        Parameters
        ----------
        u_perf_body : (2,) [vx, vy] nominal velocity in robot body frame
        robot_pos   : (2,) robot position in world/odom frame
        robot_yaw   : robot heading (radians), world frame
        obstacles   : list of Obstacle (center, radius) in world/odom frame

        Returns
        -------
        u_safe_body : (2,) safe velocity in robot body frame.
                      Equals u_perf_body when no obstacles are nearby or
                      the filter is disabled.
        """
        self.n_total += 1

        if not self.enabled or len(obstacles) == 0:
            return u_perf_body.copy()

        obs        = self._select_closest(obstacles, robot_pos)
        kappa_vals = self._compute_kappa(robot_pos, robot_yaw, obs)

        u_safe_body = self._cbf_qp.solve_fast_cbf(
            u_perf_body = u_perf_body,
            obstacles   = obs,
            p_robot     = robot_pos,
            yaw         = robot_yaw,
            kappa_vals  = kappa_vals,
        )

        if np.linalg.norm(u_safe_body - u_perf_body) > 1e-3:
            self.n_filtered += 1

        return u_safe_body

    def reset_counters(self) -> dict:
        """Return and reset diagnostic counters."""
        stats = {
            'n_filtered': self.n_filtered,
            'n_total':    self.n_total,
            'filter_pct': 100.0 * self.n_filtered / max(self.n_total, 1),
        }
        self.n_filtered = 0
        self.n_total    = 0
        return stats

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _select_closest(
        self,
        obstacles: List[Obstacle],
        robot_pos: np.ndarray,
    ) -> List[Obstacle]:
        """Return up to max_obstacles closest obstacles."""
        if len(obstacles) <= self.max_obstacles:
            return obstacles
        dists = [np.linalg.norm(obs.center - robot_pos) for obs in obstacles]
        idx   = np.argsort(dists)[: self.max_obstacles]
        return [obstacles[i] for i in idx]

    @torch.no_grad()
    def _compute_kappa(
        self,
        robot_pos: np.ndarray,
        robot_yaw: float,
        obstacles: List[Obstacle],
    ) -> List[float]:
        """Evaluate κ(h_i) for each obstacle."""
        h_vals = np.array([
            cbf_value(robot_pos, obs.center, robot_yaw, obs.radius)
            for obs in obstacles
        ], dtype=np.float32)

        h_tensor   = torch.tensor(h_vals, device=self._device)
        kappa_vals = self._kappa_net(h_tensor).cpu().numpy()
        return [float(k) for k in kappa_vals]
