#!/usr/bin/env python3

"""
cbf_safety_filter.py — Learned CBF safety filter for deployment.

Sits between the motion mapper (u_perf) and the robot controller (CtrlInterface).
At each control tick:

    1. Receive nominal command u_perf_body from MotionMapper
    2. Receive obstacle list from perception (world frame)
    3. α-net predicts per-obstacle α from relative state features
    4. OSQP solves the CBF-QP to find u_safe_body
    5. Return u_safe_body — robot follows this instead of u_perf_body

The vrz (yaw rate) command is always passed through unchanged.  The CBF
only filters translational (vx, vy) commands.

Usage
─────
    filt = CbfSafetyFilter(model_path='alpha_net_v5.pth')
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

import os
from typing import List

import numpy as np
import torch

from mocap_teleop.ctrl.alpha_net import AlphaNet
from mocap_teleop.ctrl.cbf_qp import (CBFQP, Obstacle, world_to_body,
                                       ROBOT_HALF_LENGTH)


class CbfSafetyFilter:
    """
    Learned CBF safety filter.

    Parameters
    ----------
    model_path     : path to saved α-net weights (.pth file)
    v_max          : maximum translational speed, same as during training (m/s)
    max_obstacles  : maximum number of obstacles to consider per step.
                     If more are detected, the closest ones are used.
    device         : torch device ('cpu' or 'cuda')
    enabled        : if False, filter is a passthrough (useful for A/B testing)
    """

    def __init__(
        self,
        model_path:    str,
        v_max:         float = 1.5,
        max_obstacles: int   = 5,
        device:        str   = 'cpu',
        enabled:       bool  = True,
    ):
        self.enabled       = enabled
        self.max_obstacles = max_obstacles
        self._device       = torch.device(device)

        self._alpha_net = AlphaNet(hidden_dim=64).to(self._device)
        self._alpha_net.load(model_path, device=device)
        # load() already calls eval() — confirm
        self._alpha_net.eval()

        self._cbf_qp = CBFQP(v_max=v_max, max_obstacles=max_obstacles)

        # Diagnostic counters (reset each second by the node)
        self.n_filtered  = 0   # steps where u_safe ≠ u_perf (filter was active)
        self.n_total     = 0   # total steps processed

    def filter(
        self,
        u_perf_body: np.ndarray,
        robot_pos:   np.ndarray,
        robot_yaw:   float,
        obstacles:   List[Obstacle],
    ) -> np.ndarray:
        """
        Apply the CBF safety filter to a nominal body-frame velocity command.

        Parameters
        ----------
        u_perf_body : (2,) [vx, vy] nominal velocity in robot body frame
        robot_pos   : (2,) robot position in world/odom frame
        robot_yaw   : robot heading (radians), world frame
        obstacles   : list of Obstacle (center, radius) in world/odom frame

        Returns
        -------
        u_safe_body : (2,) safe velocity in robot body frame
                      Equals u_perf_body when no obstacle is nearby or
                      the filter is disabled.
        """
        self.n_total += 1

        if not self.enabled or len(obstacles) == 0:
            return u_perf_body.copy()

        # Use the closest obstacles (up to max_obstacles) — far obstacles
        # have h >> 0 so their CBF constraints are inactive anyway, but
        # excluding them keeps the QP small and fast.
        obs = self._select_closest(obstacles, robot_pos)

        alphas = self._compute_alphas(u_perf_body, robot_pos, robot_yaw, obs)

        u_safe_body = self._cbf_qp.solve_fast(
            u_perf_body = u_perf_body,
            obstacles   = obs,
            p_robot     = robot_pos,
            yaw         = robot_yaw,
            alphas      = alphas,
        )

        # Count as "filtered" if the command changed meaningfully
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
        """Return up to max_obstacles obstacles sorted by distance."""
        if len(obstacles) <= self.max_obstacles:
            return obstacles
        dists = [np.linalg.norm(obs.center - robot_pos) for obs in obstacles]
        idx   = np.argsort(dists)[: self.max_obstacles]
        return [obstacles[i] for i in idx]

    @torch.no_grad()
    def _compute_alphas(
        self,
        u_perf_body: np.ndarray,
        robot_pos:   np.ndarray,
        robot_yaw:   float,
        obstacles:   List[Obstacle],
    ) -> List[float]:
        """Run α-net for each obstacle and return a list of scalar α values."""
        c, s = np.cos(robot_yaw), np.sin(robot_yaw)
        R    = np.array([[c, -s], [s, c]])   # body → world;  R^T = world → body

        alphas = []
        for obs in obstacles:
            d_world       = obs.center - robot_pos
            d_to_obs_body = R.T @ d_world
            h_val         = float(
                np.dot(d_world, d_world) / (ROBOT_HALF_LENGTH + obs.radius) ** 2
                - 1.0
            )

            feat = AlphaNet.build_input(
                d_to_obs_body = torch.tensor(
                    d_to_obs_body, dtype=torch.float32, device=self._device),
                r             = torch.tensor(
                    [obs.radius],  dtype=torch.float32, device=self._device),
                h             = torch.tensor(
                    [h_val],       dtype=torch.float32, device=self._device),
                u_perf        = torch.tensor(
                    u_perf_body,   dtype=torch.float32, device=self._device),
                d_to_goal     = torch.tensor(
                    d_to_obs_body, dtype=torch.float32, device=self._device),
                # NOTE: d_to_goal at deployment = d_to_obs_body as a proxy.
                # The ideal value is (human_pos − robot_pos) in body frame,
                # which can be wired in once the human position is passed here.
            )
            alphas.append(float(self._alpha_net(feat).cpu().item()))

        return alphas
