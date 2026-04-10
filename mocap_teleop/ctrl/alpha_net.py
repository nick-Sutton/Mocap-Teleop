#!/usr/bin/env python3

"""
alpha_net.py — Learned ECBF gains for the CBF-QP safety filter.

Maps per-obstacle environment info + robot state → [k1, k2] > 0, which are
the exponential CBF gains for a 2nd-order (first-order lag) system:

    ḣ ≥ −k2·h    (position-based gain, like 1st-order α)
    ḧ ≥ −k1·ḣ   (velocity anticipation — compensates for lag τ)

Combined CBF-QP constraint:   A·u ≥ −k1·ḣ − k2·h

All features are expressed relative to the robot body frame so the network
generalises across arbitrary world positions and headings — absolute world
coordinates would cause the network to memorise obstacle locations in the
training arena rather than learning the geometry of avoidance.

Input (per obstacle, 10-D):
    [d_obs_x, d_obs_y,      obstacle centre relative to robot, body frame (m)
     r,                     obstacle radius  (m)
     h,                     current CBF value h(x) = |d|²/(a+r)² − 1
     vx_perf, vy_perf,      nominal (mocap) velocity command, body frame (m/s)
     g_x, g_y,              goal relative to robot, body frame (m)
     vx_act, vy_act]        actual robot velocity, body frame (m/s)

Output (per obstacle, 2-D):
    [k1, k2]  both > 0, enforced by ReLU + ε offset
"""

import torch
import torch.nn as nn

INPUT_DIM = 10

# ── Feature normalisation constants ──────────────────────────────────────────
# These are applied inside build_input so training and deployment are always
# consistent.  Values chosen to bring all features into roughly [-1, 1]:
#   distances    : arena half-width ≈ 2 m
#   radius       : midpoint of training range [0.3, 0.8] m
#   h            : clamped to [-1, 2] then scaled by range 3
#   cmd velocity : v_max = 0.3 m/s (Go2 real limit)
#   act velocity : same scale as cmd (actual ≤ cmd by lag)
_D_SCALE    = 3.0   # distance features (d_obs_body, d_to_goal)
_R_SCALE    = 0.5   # obstacle radius
_H_CLAMP    = 2.0   # clamp h above this — CBF is inactive for h > _H_CLAMP
_H_SCALE    = 3.0   # range of clamped h: [-1, _H_CLAMP] → divide by this
_V_SCALE    = 0.3   # command velocity features (u_perf)
_V_ACT_SCALE = 0.3  # actual velocity features (v_actual_body)


class AlphaNet(nn.Module):
    INPUT_DIM = INPUT_DIM   # expose as class attribute for external reshaping
    """Two-hidden-layer MLP outputting [k1, k2] > 0 for a single obstacle."""

    def __init__(self, hidden_dim: int = 64, eps: float = 1e-3):
        """
        Parameters
        ----------
        hidden_dim : width of each hidden layer
        eps        : minimum output value — guarantees k1, k2 > 0 strictly
        """
        super().__init__()
        self.eps = eps
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),   # outputs [k1, k2]
        )
        # Initialise to [k1, k2] ≈ [1.0, 1.0] with small input-dependent variation.
        # Both gains start near 1.0: conservative enough to guarantee safety from
        # step 1 while leaving meaningful gradient signal for both safety and performance.
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.normal_(last.weight, std=0.01)
        nn.init.constant_(last.bias, 1.0)   # both outputs bias=1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (..., INPUT_DIM)  — batch of obstacle-state feature vectors

        Returns
        -------
        k_vals : (..., 2)  — [k1, k2], both strictly positive
                  k1: velocity anticipation gain (compensates for lag τ)
                  k2: position-based gain (equivalent to 1st-order α)
        """
        return torch.relu(self.net(x)) + self.eps

    @staticmethod
    def build_input(
        d_to_obs_body:  torch.Tensor,
        r:              torch.Tensor,
        h:              torch.Tensor,
        u_perf:         torch.Tensor,
        d_to_goal:      torch.Tensor,
        v_actual_body:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience helper: concatenate features into the 10-D input vector.

        All arguments may be batched along an arbitrary leading batch dimension
        as long as shapes are broadcast-compatible.

        Parameters
        ----------
        d_to_obs_body  : (..., 2)  obstacle centre relative to robot, in body frame
        r              : (..., 1)  obstacle radius (m)
        h              : (..., 1)  current CBF value h(x) = |d|²/(a+r)² − 1
        u_perf         : (..., 2)  nominal velocity (vx, vy) in body frame (m/s)
        d_to_goal      : (..., 2)  goal relative to robot, in body frame (m)
        v_actual_body  : (..., 2)  actual robot velocity in body frame (m/s)

        Returns
        -------
        features : (..., INPUT_DIM)  — all features normalised to [-1, 1]
        """
        h_clamped = h.clamp(min=-1.0, max=_H_CLAMP)
        return torch.cat([
            d_to_obs_body  / _D_SCALE,
            r              / _R_SCALE,
            h_clamped      / _H_SCALE,
            u_perf         / _V_SCALE,
            d_to_goal      / _D_SCALE,
            v_actual_body  / _V_ACT_SCALE,
        ], dim=-1)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str = 'cpu') -> None:
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()
