"""V4 — V3 plus convex capital adjustment cost ψ_K/2 · (k_next - k)².

The adjustment cost is paid out of consumption today. The capital Euler
equation gains a marginal-adjustment term:
    u'(c_t) · (1 + ψ_K · (k_next - k_t))  =  β · E[(1+r_{t+1}) · u'(c_{t+1})].

Setting ψ_K = 0 recovers V3 exactly.
"""
from __future__ import annotations

import math

import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


N: int = 7
TAU: float = 72.0 / N
N_TFP: int = 4

RHO_YEARLY: float = 0.85
SIGMA_E_YEARLY: float = 0.03


def aggregate_ar1(rho_yearly: float, sigma_e_yearly: float, tau: float) -> tuple[float, float]:
    rho = rho_yearly ** tau
    var_eps = (sigma_e_yearly ** 2) * (1 - rho_yearly ** (2 * tau)) / (1 - rho_yearly ** 2)
    return rho, math.sqrt(var_eps)


def rouwenhorst(n: int, rho: float, sigma_e: float) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("Rouwenhorst needs n >= 2.")
    p = (1.0 + rho) / 2.0
    P = np.array([[p, 1 - p], [1 - p, p]])
    for k in range(3, n + 1):
        Pk = np.zeros((k, k))
        Pk[:k - 1, :k - 1] += p * P
        Pk[:k - 1, 1:] += (1 - p) * P
        Pk[1:, :k - 1] += (1 - p) * P
        Pk[1:, 1:] += p * P
        Pk[1:k - 1, :] /= 2.0
        P = Pk
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
    b_min=-0.05,
    b_scale=0.10,
    p_b_min=0.55,
    p_b_max=0.95,
    psi_K=0.50,                # capital adjustment cost coefficient (V4 default)
)
P["beta"] = P["beta_yearly"] ** TAU
P["delta"] = 1.0 - (1.0 - P["delta_yearly"]) ** TAU


EPS_PROFILE: list[float] = [0.7, 0.9, 1.0, 1.05, 1.0, 0.9, 0.0]
assert len(EPS_PROFILE) == N
L: float = sum(EPS_PROFILE)


def make_tfp(p: dict = P, n_tfp: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    n = n_tfp if n_tfp is not None else p["n_tfp"]
    rho, sigma_e = aggregate_ar1(p["rho_yearly"], p["sigma_e_yearly"], p["tau"])
    grid_log, Pmat = rouwenhorst(n, rho, sigma_e)
    z_vals = np.exp(grid_log)
    return (torch.tensor(z_vals, device=device, dtype=torch.float32),
            torch.tensor(Pmat, device=device, dtype=torch.float32))


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
    k: torch.Tensor,
    b: torch.Tensor,
    s_K: torch.Tensor,
    b_next: torch.Tensor,
    p_b: torch.Tensor,
    *,
    bonds_off: bool = False,
    psi_K: float | None = None,
) -> dict:
    if psi_K is None:
        psi_K = P["psi_K"]
    K = k.sum(dim=-1)
    r, w = prices(Z, K)
    r_b = r.unsqueeze(-1)
    w_b = w.unsqueeze(-1)

    inc = torch.zeros(*Z.shape, N, device=Z.device, dtype=Z.dtype)
    inc[..., 0] = w * EPS[0]
    if bonds_off:
        inc[..., 1:N] = w_b * EPS[1:N] + r_b.add(1) * k
    else:
        inc[..., 1:N] = w_b * EPS[1:N] + r_b.add(1) * k + b

    bond_cost = (p_b.unsqueeze(-1) * b_next) if not bonds_off else torch.zeros_like(s_K)

    k_next = s_K * inc[..., :N - 1]

    # Adjustment cost relative to last period's capital wealth k.
    # Cohort age 0 enters with k=0; for cohorts age j ≥ 1 the relevant comparison
    # is between their next-period saving (k_next^{j+1}) and their current
    # entering wealth (k^j_t).  We approximate by aligning indices so that
    # cohort age j today (saving into k^{j+1}) compares against k^j_t
    # (their own entering wealth) for j ≥ 1, and against 0 for j = 0.
    k_aligned = torch.cat([torch.zeros(*Z.shape, 1, device=Z.device, dtype=Z.dtype), k[..., :N - 2]], dim=-1)
    delta_k = k_next - k_aligned                                           # (M, N-1)
    adj_cost = 0.5 * psi_K * delta_k ** 2

    c = torch.zeros_like(inc)
    c[..., :N - 1] = (1 - s_K) * inc[..., :N - 1] - bond_cost - adj_cost
    c[..., N - 1] = inc[..., N - 1]
    return dict(c=c, k_next=k_next, b_next=(b_next if not bonds_off else torch.zeros_like(b_next)),
                r=r, w=w, K=K, inc=inc, p_b=p_b, bond_cost=bond_cost,
                adj_cost=adj_cost, delta_k=delta_k)


def init_cloud(N_states: int, *, b_init: float = 0.0):
    n_tfp = Z_VALS.shape[0]
    z_idx = torch.randint(0, n_tfp, (N_states,), device=device)
    base = torch.tensor(
        [0.05 * (j + 1) for j in range(N - 1)],
        device=device, dtype=torch.float32,
    )
    noise = 0.5 * torch.rand(N_states, N - 1, device=device) + 0.75
    k = base.unsqueeze(0) * noise
    b = torch.full((N_states, N - 1), b_init, device=device, dtype=torch.float32)
    return z_idx, k, b


@torch.no_grad()
def step_cloud(z_idx: torch.Tensor, k: torch.Tensor, b: torch.Tensor, net,
               *, bonds_off: bool = False, psi_K: float | None = None):
    Z = Z_VALS[z_idx]
    s_K, b_next, p_b = net(Z, k, b)
    out = cohort_decisions(Z, k, b, s_K, b_next, p_b, bonds_off=bonds_off, psi_K=psi_K)
    probs = P_MAT[z_idx]
    cdf = torch.cumsum(probs, dim=-1)
    u = torch.rand(z_idx.shape[0], 1, device=device)
    z_idx_next = (u > cdf).sum(dim=-1).long().clamp(max=Z_VALS.shape[0] - 1)
    return z_idx_next, out["k_next"], out["b_next"]


def replace_random(z_idx: torch.Tensor, k: torch.Tensor, b: torch.Tensor, frac: float = 0.05):
    n = z_idx.shape[0]
    m = max(1, int(frac * n))
    pos = torch.randperm(n, device=device)[:m]
    zi_new, k_new, b_new = init_cloud(m)
    z_idx = z_idx.clone(); z_idx[pos] = zi_new
    k = k.clone(); k[pos] = k_new
    b = b.clone(); b[pos] = b_new
    return z_idx, k, b


def fb_residual(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y - torch.sqrt(x ** 2 + y ** 2 + 1e-12)
