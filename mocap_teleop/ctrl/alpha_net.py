#!/usr/bin/env python3

"""
alpha_net.py — Learned class-K function for the CBF-QP safety filter.

Maps per-obstacle environment info + robot state → scalar α > 0, which
controls how aggressively the CBF-QP pushes the robot away from each
obstacle.  A larger α allows the robot to get closer before the safety
constraint activates; a smaller α is more conservative.

All features are expressed relative to the robot body frame so the network
generalises across arbitrary world positions and headings — absolute world
coordinates would cause the network to memorise obstacle locations in the
training arena rather than learning the geometry of avoidance.

Input (per obstacle, 8-D):
    [d_obs_x, d_obs_y,      obstacle centre relative to robot, body frame (m)
     r,                     obstacle radius  (m)
     h,                     current CBF value h(x) = |d|²/(a+r)² − 1
     vx_perf, vy_perf,      nominal (mocap) velocity command, body frame (m/s)
     g_x, g_y]              goal relative to robot, body frame (m)

Output (per obstacle, scalar):
    α > 0   enforced by ReLU + ε offset
"""

import torch
import torch.nn as nn

INPUT_DIM = 8


class AlphaNet(nn.Module):
    INPUT_DIM = INPUT_DIM   # expose as class attribute for external reshaping
    """Two-hidden-layer MLP outputting α > 0 for a single obstacle."""

    def __init__(self, hidden_dim: int = 64, eps: float = 1e-3):
        """
        Parameters
        ----------
        hidden_dim : width of each hidden layer
        eps        : minimum output value — guarantees α > 0 strictly
        """
        super().__init__()
        self.eps = eps
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialise to α ≈ 1.0 with small input-dependent variation.
        #
        # The previous zeros_(weight) + zeros_(bias) initialisation caused a
        # permanent dead-relu: net(x) = 0 → relu(0) = 0, relu'(0) = 0 →
        # every gradient was multiplied by zero → no parameter ever updated.
        #
        # With bias = 1.0 and small weight: net(x) ≈ 1 + tiny_variation,
        # relu'(net) = 1, gradients flow through all layers from step 1.
        # α ≈ 1.0 is a reasonable starting point — the CBF allows moderate
        # approach, so the performance loss has a meaningful gradient w.r.t. α,
        # and the safety penalty keeps h from going negative.
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.normal_(last.weight, std=0.01)
        nn.init.constant_(last.bias, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (..., INPUT_DIM)  — batch of obstacle-state feature vectors

        Returns
        -------
        alpha : (..., 1)  — strictly positive scalar per obstacle
        """
        return torch.relu(self.net(x)) + self.eps

    @staticmethod
    def build_input(
        d_to_obs_body: torch.Tensor,
        r:             torch.Tensor,
        h:             torch.Tensor,
        u_perf:        torch.Tensor,
        d_to_goal:     torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience helper: concatenate features into the 8-D input vector.

        All arguments may be batched along an arbitrary leading batch dimension
        as long as shapes are broadcast-compatible.

        Parameters
        ----------
        d_to_obs_body : (..., 2)  obstacle centre relative to robot, in body frame
        r             : (..., 1)  obstacle radius (m)
        h             : (..., 1)  current CBF value h(x) = |d|²/(a+r)² − 1
        u_perf        : (..., 2)  nominal velocity (vx, vy) in body frame (m/s)
        d_to_goal     : (..., 2)  goal relative to robot, in body frame (m)

        Returns
        -------
        features : (..., INPUT_DIM)
        """
        return torch.cat([d_to_obs_body, r, h, u_perf, d_to_goal], dim=-1)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str = 'cpu') -> None:
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()
