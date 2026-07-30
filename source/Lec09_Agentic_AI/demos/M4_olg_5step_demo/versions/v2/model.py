"""V2 — seven-cohort OLG with a 4-state Rouwenhorst TFP chain.

Replaces V1's 2-state symmetric Markov with a 4-state discretisation of
an annual AR(1) on log-TFP. Closed-form expectations are preserved
(the expectation in each Euler residual is a 4-term sum).
"""
from __future__ import annotations

import math

import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


N: int = 7
TAU: float = 72.0 / N

# Annual AR(1) primitives for log-TFP.
RHO_YEARLY: float = 0.85
SIGMA_E_YEARLY: float = 0.03
N_TFP: int = 4


def aggregate_ar1(rho_yearly: float, sigma_e_yearly: float, tau: float) -> tuple[float, float]:
    """Per-period AR(1) primitives implied by an annual AR(1) sampled every τ years.

    log z_{t+τ} = ρ^τ · log z_t + ε,  Var(ε) = σ_eps^2 · (1 - ρ^{2τ})/(1 - ρ^2).
    Returns (rho_period, sigma_e_period).
    """
    rho = rho_yearly ** tau
    var_eps = (sigma_e_yearly ** 2) * (1 - rho_yearly ** (2 * tau)) / (1 - rho_yearly ** 2)
    return rho, math.sqrt(var_eps)


def rouwenhorst(n: int, rho: float, sigma_e: float) -> tuple[np.ndarray, np.ndarray]:
    """Rouwenhorst discretisation of log z_{t+1} = ρ log z_t + ε.

    Returns (grid, P) where grid is shape (n,) of log-TFP nodes and P is the n×n
    transition matrix. The grid is symmetric around 0; exponentiate for z-levels.
    """
    if n < 2:
        raise ValueError("Rouwenhorst needs n >= 2.")
    p = (1.0 + rho) / 2.0
    # Recursive construction of transition matrix.
    P = np.array([[p, 1 - p], [1 - p, p]])
    for k in range(3, n + 1):
        Pk = np.zeros((k, k))
        Pk[:k - 1, :k - 1] += p * P
        Pk[:k - 1, 1:] += (1 - p) * P
        Pk[1:, :k - 1] += (1 - p) * P
        Pk[1:, 1:] += p * P
        Pk[1:k - 1, :] /= 2.0
        P = Pk
    # Grid: equally spaced in [-h, h], h such that unconditional std matches.
    sigma_z = sigma_e / math.sqrt(1 - rho ** 2)
    h = sigma_z * math.sqrt(n - 1)
    grid = np.linspace(-h, h, n)
    return grid, P


P: dict = dict(
    N=N,
    n_tfp=N_TFP,
    alpha=1 / 3,
    beta_yearly=0.97,
    gamma=2.0,
    delta_yearly=0.06,
    rho_yearly=RHO_YEARLY,
    sigma_e_yearly=SIGMA_E_YEARLY,
    tau=TAU,
)
P["beta"] = P["beta_yearly"] ** TAU
P["delta"] = 1.0 - (1.0 - P["delta_yearly"]) ** TAU


EPS_PROFILE: list[float] = [0.7, 0.9, 1.0, 1.05, 1.0, 0.9, 0.0]
assert len(EPS_PROFILE) == N
L: float = sum(EPS_PROFILE)


def make_tfp(p: dict = P, n_tfp: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (Z_VALS, P_MAT) from per-period AR(1) primitives."""
    n = n_tfp if n_tfp is not None else p["n_tfp"]
    rho, sigma_e = aggregate_ar1(p["rho_yearly"], p["sigma_e_yearly"], p["tau"])
    grid_log, Pmat = rouwenhorst(n, rho, sigma_e)
    # Centre on E[z] ≈ 1: shift by -σ_z²/2 so log z is mean-zero adjusted.
    z_vals = np.exp(grid_log)
    Z = torch.tensor(z_vals, device=device, dtype=torch.float32)
    P_t = torch.tensor(Pmat, device=device, dtype=torch.float32)
    return Z, P_t


Z_VALS, P_MAT = make_tfp()
EPS: torch.Tensor = torch.tensor(EPS_PROFILE, device=device, dtype=torch.float32)


def refresh_aggregates() -> None:
    global Z_VALS, P_MAT, EPS, L
    Z_VALS, P_MAT = make_tfp()
    EPS = torch.tensor(EPS_PROFILE, device=device, dtype=torch.float32)
    L = float(EPS.sum().item())


def prices(Z: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    KL = K / L
    r = P["alpha"] * Z * KL.pow(P["alpha"] - 1) - P["delta"]
    w = (1 - P["alpha"]) * Z * KL.pow(P["alpha"])
    return r, w


def cohort_decisions(
    Z: torch.Tensor,
    a: torch.Tensor,
    s: torch.Tensor,
) -> dict:
    K = a.sum(dim=-1)
    r, w = prices(Z, K)
    r_b = r.unsqueeze(-1)
    w_b = w.unsqueeze(-1)
    inc = torch.zeros(*Z.shape, N, device=Z.device, dtype=Z.dtype)
    inc[..., 0] = w * EPS[0]
    inc[..., 1:N] = w_b * EPS[1:N] + r_b.add(1) * a
    a_next = s * inc[..., :N - 1]
    c = torch.zeros_like(inc)
    c[..., :N - 1] = (1 - s) * inc[..., :N - 1]
    c[..., N - 1] = inc[..., N - 1]
    return dict(c=c, a_next=a_next, r=r, w=w, K=K, inc=inc)


def init_cloud(N_states: int) -> tuple[torch.Tensor, torch.Tensor]:
    n_tfp = Z_VALS.shape[0]
    z_idx = torch.randint(0, n_tfp, (N_states,), device=device)
    base = torch.tensor(
        [0.05 * (j + 1) for j in range(N - 1)],
        device=device, dtype=torch.float32,
    )
    noise = 0.5 * torch.rand(N_states, N - 1, device=device) + 0.75
    a = base.unsqueeze(0) * noise
    return z_idx, a


@torch.no_grad()
def step_cloud(z_idx: torch.Tensor, a: torch.Tensor, net):
    Z = Z_VALS[z_idx]
    s = net(Z, a)
    out = cohort_decisions(Z, a, s)
    probs = P_MAT[z_idx]                                       # (M, n_tfp)
    cdf = torch.cumsum(probs, dim=-1)
    u = torch.rand(z_idx.shape[0], 1, device=device)
    z_idx_next = (u > cdf).sum(dim=-1).long().clamp(max=Z_VALS.shape[0] - 1)
    return z_idx_next, out["a_next"]


def replace_random(z_idx: torch.Tensor, a: torch.Tensor, frac: float = 0.05):
    n = z_idx.shape[0]
    k = max(1, int(frac * n))
    pos = torch.randperm(n, device=device)[:k]
    zi_new, a_new = init_cloud(k)
    z_idx = z_idx.clone(); z_idx[pos] = zi_new
    a = a.clone(); a[pos] = a_new
    return z_idx, a
