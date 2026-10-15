"""V0 policy network — single MLP, sigmoid head, savings-rate output."""
from __future__ import annotations

import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    def __init__(self, hidden: int = 64, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, 2),
        )

    def forward(self, Z: torch.Tensor, am: torch.Tensor, ao: torch.Tensor):
        x = torch.stack([Z, am, ao], dim=-1)
        out = torch.sigmoid(self.net(x))
        sy = out[..., 0].clamp(self.eps, 1 - self.eps)
        sm = out[..., 1].clamp(self.eps, 1 - self.eps)
        return sy, sm
