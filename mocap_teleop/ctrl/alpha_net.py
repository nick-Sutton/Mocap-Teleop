#!/usr/bin/env python3

"""
alpha_net.py — Learned class-K function for the CBF-QP safety filter.

Maps per-obstacle environment info + robot state → scalar α > 0, which
controls how aggressively the CBF-QP pushes the robot away from each
obstacle.  A larger α allows the robot to get closer before the safety
constraint activates; a smaller α is more conservative.

Input (per obstacle, 10-D):
    [p_obs_x, p_obs_y,      obstacle centre  (world frame, m)
     r,                     obstacle radius   (m)
     p_robot_x, p_robot_y,  robot position    (world frame, m)
     yaw_robot,             robot heading     (rad)
     x0_x, x0_y,           robot position at episode start (world frame, m)
     vx_perf, vy_perf]      nominal (mocap) velocity command (m/s)

Output (per obstacle, scalar):
    α > 0   enforced by ReLU + ε offset
"""

import torch
import torch.nn as nn

INPUT_DIM = 10


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
        # Start output near zero so α ≈ ε initially.
        # This makes the CBF maximally conservative at the start of training —
        # the constraint is always tight, so gradients always flow through the QP.
        # The network then learns to increase α where it is safe to do so.
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

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
        p_obs:    torch.Tensor,
        r:        torch.Tensor,
        p_robot:  torch.Tensor,
        yaw:      torch.Tensor,
        x0:       torch.Tensor,
        u_perf:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience helper: concatenate features into the 10-D input vector.

        All arguments may be batched along an arbitrary leading batch dimension
        as long as shapes are broadcast-compatible.

        Parameters
        ----------
        p_obs   : (..., 2)  obstacle centre in world frame (x, y)
        r       : (..., 1)  obstacle radius
        p_robot : (..., 2)  robot position in world frame (x, y)
        yaw     : (..., 1)  robot heading (rad)
        x0      : (..., 2)  robot position at episode start (x, y)
        u_perf  : (..., 2)  nominal velocity command (vx, vy) in body frame

        Returns
        -------
        features : (..., INPUT_DIM)
        """
        return torch.cat([p_obs, r, p_robot, yaw, x0, u_perf], dim=-1)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str = 'cpu') -> None:
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()
