"""V1 — seven-cohort OLG primitives.

Generalises V0 from 3 cohorts to N=7. Cohort age 0 is born with zero wealth;
cohort age N-1 is retired and consumes everything (no savings). The policy
network outputs N-1 savings rates, one per non-retired cohort.

State at time t: (Z_t, a^1_t, ..., a^{N-1}_t) — N-1 cohort wealth values plus TFP.
"""
from __future__ import annotations

import math

import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


N: int = 7
TAU: float = 72.0 / N  # ≈ 10.286 yearly sub-periods per generational period


P: dict = dict(
    N=N,
    alpha=1 / 3,
    beta_yearly=0.97,
    gamma=2.0,
    delta_yearly=0.06,
    Z_lo=0.95,
    Z_hi=1.05,
    P_persist=0.80,
    tau=TAU,
)
P["beta"] = P["beta_yearly"] ** TAU
P["delta"] = 1.0 - (1.0 - P["delta_yearly"]) ** TAU


# Hump-shaped labour profile across the N cohorts (peak in middle age,
# zero in retirement). Length must equal N.
EPS_PROFILE: list[float] = [0.7, 0.9, 1.0, 1.05, 1.0, 0.9, 0.0]
assert len(EPS_PROFILE) == N

L: float = sum(EPS_PROFILE)


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
EPS: torch.Tensor = torch.tensor(EPS_PROFILE, device=device, dtype=torch.float32)


def refresh_aggregates() -> None:
    global Z_VALS, P_MAT, EPS, L
    Z_VALS = make_z_vals()
    P_MAT = make_p_mat()
    EPS = torch.tensor(EPS_PROFILE, device=device, dtype=torch.float32)
    L = float(EPS.sum().item())


def prices(Z: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    KL = K / L
    r = P["alpha"] * Z * KL.pow(P["alpha"] - 1) - P["delta"]
    w = (1 - P["alpha"]) * Z * KL.pow(P["alpha"])
    return r, w


def cohort_decisions(
    Z: torch.Tensor,
    a: torch.Tensor,        # shape (batch, N-1) — wealth of cohorts age 1..N-1
    s: torch.Tensor,        # shape (batch, N-1) — savings rates of cohorts age 0..N-2
) -> dict:
    """Compute consumption and next-period wealth for every cohort."""
    K = a.sum(dim=-1)                                                # aggregate capital
    r, w = prices(Z, K)                                              # scalar-per-batch
    r_b = r.unsqueeze(-1)
    w_b = w.unsqueeze(-1)

    # Income by cohort age 0..N-1 (shape (batch, N))
    inc = torch.zeros(*Z.shape, N, device=Z.device, dtype=Z.dtype)
    inc[..., 0] = w * EPS[0]
    inc[..., 1:N] = w_b * EPS[1:N] + r_b.add(1) * a                  # cohorts age 1..N-1

    # Cohort age N-1 retires and consumes everything; cohorts 0..N-2 save fraction s.
    a_next = s * inc[..., :N - 1]                                    # next-period wealth of cohorts age 1..N-1
    c = torch.zeros_like(inc)
    c[..., :N - 1] = (1 - s) * inc[..., :N - 1]
    c[..., N - 1] = inc[..., N - 1]                                  # retired consumes all

    return dict(c=c, a_next=a_next, r=r, w=w, K=K, inc=inc)


def init_cloud(N_states: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Random states. Returns (z_idx, a) with a shape (N_states, N-1)."""
    z_idx = torch.randint(0, 2, (N_states,), device=device)
    # Wealth distribution per cohort: log-spaced midpoints with noise.
    base = torch.tensor(
        [0.05 * (j + 1) for j in range(N - 1)],
        device=device, dtype=torch.float32,
    )                                                                # ≈ rising in age
    noise = 0.5 * torch.rand(N_states, N - 1, device=device) + 0.75  # in [0.75, 1.25]
    a = base.unsqueeze(0) * noise
    return z_idx, a


@torch.no_grad()
def step_cloud(z_idx: torch.Tensor, a: torch.Tensor, net):
    Z = Z_VALS[z_idx]
    s = net(Z, a)
    out = cohort_decisions(Z, a, s)
    probs = P_MAT[z_idx]
    u = torch.rand(z_idx.shape[0], device=device)
    z_idx_next = (u > probs[:, 0]).long()
    return z_idx_next, out["a_next"]


def replace_random(z_idx: torch.Tensor, a: torch.Tensor, frac: float = 0.05):
    n = z_idx.shape[0]
    k = max(1, int(frac * n))
    pos = torch.randperm(n, device=device)[:k]
    zi_new, a_new = init_cloud(k)
    z_idx = z_idx.clone(); z_idx[pos] = zi_new
    a = a.clone(); a[pos] = a_new
    return z_idx, a
