"""V1 policy network — single MLP, sigmoid head, N-1 savings-rate outputs.

Input dimension is 1 (TFP) + (N-1) (cohort wealths) = N.
Output dimension is N-1 savings rates.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model import N


class PolicyNet(nn.Module):
    def __init__(self, hidden: int = 128, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        in_dim = 1 + (N - 1)
        out_dim = N - 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, Z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([Z.unsqueeze(-1), a], dim=-1)
        s = torch.sigmoid(self.net(x))
        s = s.clamp(self.eps, 1 - self.eps)
        return s
