"""V0 — three-period OLG primitives.

Calibration, prices, cohort budgets, and the cloud-state utilities are
all kept in one short module so the V1 author has one file to read and
extend, not five.
"""
from __future__ import annotations

import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


P: dict = dict(
    alpha=1 / 3,
    beta=0.85,
    gamma=2.0,
    delta=0.30,
    eps_y=0.6,
    eps_m=1.0,
    Z_lo=0.95,
    Z_hi=1.05,
    P_persist=0.80,
)

L: float = P["eps_y"] + P["eps_m"]


def make_z_vals(p: dict = P) -> torch.Tensor:
    return torch.tensor([p["Z_lo"], p["Z_hi"]], device=device, dtype=torch.float32)


def make_p_mat(p: dict = P) -> torch.Tensor:
    pp = p["P_persist"]
    return torch.tensor(
        [[pp, 1 - pp], [1 - pp, pp]],
        device=device, dtype=torch.float32,
    )


Z_VALS: torch.Tensor = make_z_vals()
P_MAT: torch.Tensor = make_p_mat()


def refresh_aggregates() -> None:
    """Rebuild Z_VALS and P_MAT after mutating P (e.g. shock-off check)."""
    global Z_VALS, P_MAT
    Z_VALS = make_z_vals()
    P_MAT = make_p_mat()


def prices(Z: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    KL = K / L
    r = P["alpha"] * Z * KL.pow(P["alpha"] - 1) - P["delta"]
    w = (1 - P["alpha"]) * Z * KL.pow(P["alpha"])
    return r, w


def cohort_decisions(
    Z: torch.Tensor,
    am: torch.Tensor,
    ao: torch.Tensor,
    sy: torch.Tensor,
    sm: torch.Tensor,
) -> dict:
    K = am + ao
    r, w = prices(Z, K)
    inc_y = w * P["eps_y"]
    am_next = sy * inc_y
    cy = inc_y - am_next
    inc_m = w * P["eps_m"] + (1 + r) * am
    ao_next = sm * inc_m
    cm = inc_m - ao_next
    co = (1 + r) * ao
    return dict(
        cy=cy, cm=cm, co=co,
        am_next=am_next, ao_next=ao_next,
        r=r, w=w, K=K,
    )


def init_cloud(N: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_idx = torch.randint(0, 2, (N,), device=device)
    am = 0.05 + 0.30 * torch.rand(N, device=device)
    ao = 0.10 + 0.50 * torch.rand(N, device=device)
    return z_idx, am, ao


@torch.no_grad()
def step_cloud(
    z_idx: torch.Tensor, am: torch.Tensor, ao: torch.Tensor, net,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Z = Z_VALS[z_idx]
    sy, sm = net(Z, am, ao)
    out = cohort_decisions(Z, am, ao, sy, sm)
    probs = P_MAT[z_idx]
    u = torch.rand(z_idx.shape[0], device=device)
    z_idx_next = (u > probs[:, 0]).long()
    return z_idx_next, out["am_next"], out["ao_next"]


def replace_random(z_idx: torch.Tensor, am: torch.Tensor, ao: torch.Tensor, frac: float = 0.05):
    n = z_idx.shape[0]
    k = max(1, int(frac * n))
    pos = torch.randperm(n, device=device)[:k]
    zi_new, am_new, ao_new = init_cloud(k)
    z_idx = z_idx.clone(); z_idx[pos] = zi_new
    am = am.clone(); am[pos] = am_new
    ao = ao.clone(); ao[pos] = ao_new
    return z_idx, am, ao
