#!/usr/bin/env python3

"""
kappa_net.py — Class-K barrier function network for AM-CBF.

Implements the κ-function from:
  Chriat & Sun, "AM-CBF: Adaptive Multi-step CBF for Safe Reinforcement
  Learning," IEEE RA-L 2023.

The κ-net must satisfy the class-K conditions:
  (a) κ(0) = 0
  (b) κ is monotonically increasing

Both are guaranteed by construction:
  (a) No bias terms — all-zero input propagates to all-zero output through
      every linear + ReLU layer, so κ(0) = 0 automatically.
  (b) All weight matrices are passed through abs() at forward time, enforcing
      non-negative weights.  Non-negative weights + ReLU activations give a
      non-decreasing function of h.

Architecture: two hidden layers of width 7 (Table II, Chriat & Sun 2023).
Input:  h ∈ ℝ  (scalar CBF value for one obstacle)
Output: κ(h) ∈ ℝ≥0
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbsLinear(nn.Module):
    """
    Linear layer with non-negative weights: y = |W| · x  (no bias).

    Using abs(W) rather than softplus/exp reparametrisation lets weights
    freely explore negative values during SGD while the effective weight
    is always ≥ 0, avoiding the positive-initialisation trap.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.uniform_(self.weight, 0.01, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.abs())


class KappaNet(nn.Module):
    """
    Class-K function κ(h) used as the CBF decay rate.

    The QP safety constraint is:  ∇h(x) · u  ≥  −κ(h(x))

    Larger κ(h) for larger h means the CBF allows faster velocity when the
    robot is far from the obstacle and demands slowing as h → 0.

    Parameters
    ----------
    hidden_dim : hidden layer width (7 per paper Table II)
    """

    def __init__(self, hidden_dim: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            AbsLinear(1, hidden_dim), nn.ReLU(),
            AbsLinear(hidden_dim, hidden_dim), nn.ReLU(),
            AbsLinear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h : (...,) tensor of CBF values (any shape)

        Returns
        -------
        kappa_h : (...,) same shape, κ(0) = 0, κ(h) ≥ 0
        """
        shape = h.shape
        out = self.net(h.reshape(-1, 1))   # (N, 1)
        return out.reshape(shape)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f'[KappaNet] saved → {path}')

    def load(self, path: str, device: str = 'cpu') -> 'KappaNet':
        self.load_state_dict(
            torch.load(path, map_location=device, weights_only=True))
        self.eval()
        return self
