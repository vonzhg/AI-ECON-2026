"""V3 policy network.

Input  : (Z, k^1..k^{N-1}, b^1..b^{N-1})  — dimension 1 + 2(N-1)
Outputs:
   s_K   (N-1 sigmoid)         capital savings rates
   b_next (N-1 mean-zero)      next-period bond face values; sum=0 by construction
   p_b   (1 sigmoid → range)   endogenous bond price

The market-clearing layer subtracts the cohort-mean from the raw bond demand,
so the resulting `b_next` always satisfies sum_j b_next^j = 0.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model import N, P


class PolicyNet(nn.Module):
    def __init__(self, hidden: int = 192, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        in_dim = 1 + 2 * (N - 1)
        out_dim = (N - 1) + (N - 1) + 1
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, Z: torch.Tensor, k: torch.Tensor, b: torch.Tensor):
        x = torch.cat([Z.unsqueeze(-1), k, b], dim=-1)
        out = self.body(x)
        idx_s = N - 1
        idx_b = idx_s + (N - 1)
        s_K_raw = out[..., :idx_s]
        b_raw = out[..., idx_s:idx_b]
        p_raw = out[..., idx_b]

        s_K = torch.sigmoid(s_K_raw).clamp(self.eps, 1 - self.eps)
        # Bond demand: tanh × scale, then mean-subtract for market clearing.
        b_tanh = torch.tanh(b_raw) * P["b_scale"]
        b_next = b_tanh - b_tanh.mean(dim=-1, keepdim=True)
        # Bond price: sigmoid mapped into (p_b_min, p_b_max).
        p_b = P["p_b_min"] + (P["p_b_max"] - P["p_b_min"]) * torch.sigmoid(p_raw)
        return s_K, b_next, p_b
