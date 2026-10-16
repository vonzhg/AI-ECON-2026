"""Session 4 lab -- heterogeneous-agent solvers in plain numpy.

Blocks A-C of the lab: the household problem by the endogenous grid method, the invariant
distribution by Young's (2010) lottery, and the stationary equilibrium by bisection on the
interest rate, with the checks a solution has to pass before anyone believes it.
The other blocks build on this file: session04_nn.py (D, E), session04_ks.py (F), session04_paths.py (G).

Descended from the solver of the instructor's quantitative-macro course, rewritten around a
parameter object (no module-level calibration), with after-tax brackets and a sign-change test
in every root-finder. Imports: numpy only.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from math import comb

import numpy as np


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    """Annual calibration used by every Aiyagari, transition and life-cycle block."""
    alpha: float = 0.33       # capital share
    beta: float = 0.96        # discount factor
    delta: float = 0.08       # depreciation
    sigma: float = 2.0        # CRRA
    rho_e: float = 0.90       # persistence of log labour productivity
    sd_innov: float = 0.20    # s.d. of its innovation
    n_e: int = 7              # Rouwenhorst states
    a_min: float = 0.0        # borrowing limit a' >= a_min
    a_max: float = 150.0
    n_a: int = 200
    curv: float = 2.5         # grid curvature: dense near the constraint
    tau_k: float = 0.0        # tax on capital income, rebated lump-sum

    @property
    def sd_uncond(self) -> float:
        return self.sd_innov / np.sqrt(1.0 - self.rho_e ** 2)


def u(c, sigma):
    return np.log(c) if sigma == 1.0 else c ** (1.0 - sigma) / (1.0 - sigma)


def rouwenhorst(n: int, rho: float, sd_uncond: float):
    """Rouwenhorst chain for log e' = rho log e + innovation. Returns (levels with mean one, Pi, pi)."""
    p = (1.0 + rho) / 2.0
    Pi = np.array([[p, 1 - p], [1 - p, p]])
    for k in range(3, n + 1):
        new = np.zeros((k, k))
        new[:-1, :-1] += p * Pi
        new[:-1, 1:] += (1 - p) * Pi
        new[1:, :-1] += (1 - p) * Pi
        new[1:, 1:] += p * Pi
        new[1:-1, :] /= 2.0
        Pi = new
    psi = sd_uncond * np.sqrt(n - 1)
    pi = np.array([comb(n - 1, i) for i in range(n)], dtype=float) / 2.0 ** (n - 1)
    e = np.exp(np.linspace(-psi, psi, n))
    return e / (pi @ e), Pi, pi


def make_grid(p: Params) -> np.ndarray:
    return p.a_min + (p.a_max - p.a_min) * np.linspace(0.0, 1.0, p.n_a) ** p.curv


def firm(r: float, p: Params):
    """Capital per unit of labour and the wage at a pre-tax net return r (labour is normalised to one)."""
    kl = (p.alpha / (r + p.delta)) ** (1.0 / (1.0 - p.alpha))
    return kl, (1.0 - p.alpha) * kl ** p.alpha


def r_upper(p: Params) -> float:
    """The household's patience bounds the AFTER-tax return: (1 - tau_k) r < 1/beta - 1."""
    return (1.0 / p.beta - 1.0) / (1.0 - p.tau_k)


# --------------------------------------------------------------------------------------
# Block A -- the household, by the endogenous grid method
# --------------------------------------------------------------------------------------
def egm_step(c_next, r_net_next, r_net, w, transfer, grid, e, Pi, p: Params):
    """One backward step: tomorrow's consumption policy -> today's (consumption, savings)."""
    R_next, R = 1.0 + r_net_next, 1.0 + r_net
    Euc = (c_next ** (-p.sigma)) @ Pi.T                       # E[u'(c') | e] at a' = grid
    c_end = (p.beta * R_next * Euc) ** (-1.0 / p.sigma)        # invert the Euler equation
    a_end = (c_end + grid[:, None] - w * e[None, :] - transfer) / R   # the endogenous grid
    coh = R * grid[:, None] + w * e[None, :] + transfer        # cash on hand today
    c = np.empty_like(c_next)
    for q in range(e.size):
        c[:, q] = np.interp(grid, a_end[:, q], c_end[:, q])
        binds = grid < a_end[0, q]                              # below it the constraint binds
        c[binds, q] = coh[binds, q] - p.a_min
    ap = np.clip(coh - c, p.a_min, grid[-1])
    return coh - ap, ap


def solve_household(r_net, w, transfer, grid, e, Pi, p: Params, c_init=None, tol=1e-9, max_iter=5000):
    """Stationary household problem at fixed prices. Returns (a', c, sweeps, last change)."""
    coh = (1.0 + r_net) * grid[:, None] + w * e[None, :] + transfer
    c = np.maximum(coh - p.a_min if c_init is None else c_init.copy(), 1e-10)
    diff, it = np.inf, 0
    for it in range(1, max_iter + 1):
        c_new, ap = egm_step(c, r_net, r_net, w, transfer, grid, e, Pi, p)
        diff = float(np.abs(c_new - c).max())
        c = c_new
        if diff < tol:
            break
    return ap, c, it, diff


def euler_residuals(ap, c, r_net, w, transfer, grid, e, Pi, p: Params, n_off=3):
    """log10 |1 - c_Euler/c| at OFF-grid asset points where the constraint is slack.

    Returns (assets, residual matrix, slack mask). Points are placed strictly between grid nodes.
    """
    R = 1.0 + r_net
    fr = (np.arange(1, n_off + 1) / (n_off + 1.0))[None, :]
    a_off = (grid[:-1, None] + fr * np.diff(grid)[:, None]).ravel()
    res = np.full((a_off.size, e.size), np.nan)
    slack = np.zeros_like(res, dtype=bool)
    for q in range(e.size):
        c_here = np.interp(a_off, grid, c[:, q])
        ap_here = R * a_off + w * e[q] + transfer - c_here
        Euc = np.zeros_like(a_off)
        for qn in range(e.size):
            Euc += Pi[q, qn] * np.interp(ap_here, grid, c[:, qn]) ** (-p.sigma)
        c_euler = (p.beta * R * Euc) ** (-1.0 / p.sigma)
        slack[:, q] = (ap_here > p.a_min + 1e-8) & (ap_here < grid[-1])
        res[:, q] = np.log10(np.abs(1.0 - c_euler / c_here) + 1e-16)
    return a_off, res, slack


# --------------------------------------------------------------------------------------
# Block B -- from a policy to a population
# --------------------------------------------------------------------------------------
def lottery(ap, grid):
    """Young (2010): split the mass at a' between the two neighbouring nodes, preserving the mean."""
    a1 = np.clip(ap, grid[0], grid[-1])
    j = np.clip(np.searchsorted(grid, a1, side="right") - 1, 0, grid.size - 2)
    return j, (a1 - grid[j]) / (grid[j + 1] - grid[j])


def push_forward(lam, j, w_hi, Pi):
    """One application of the operator on measures: lambda_t -> lambda_{t+1}."""
    na, ne = lam.shape
    out = np.zeros_like(lam)
    for q in range(ne):
        m = lam[:, q]
        tmp = (np.bincount(j[:, q], weights=m * (1.0 - w_hi[:, q]), minlength=na)
               + np.bincount(j[:, q] + 1, weights=m * w_hi[:, q], minlength=na))
        out += np.outer(tmp, Pi[q, :])
    return out


def transition_matrix(j, w_hi, Pi):
    """The same operator as a dense matrix over (a, e) pairs, row-stochastic."""
    na, ne = j.shape
    P = np.zeros((na * ne, na * ne))
    rows = np.arange(na)
    for q in range(ne):
        for qn in range(ne):
            np.add.at(P, (rows * ne + q, j[:, q] * ne + qn), (1.0 - w_hi[:, q]) * Pi[q, qn])
            np.add.at(P, (rows * ne + q, (j[:, q] + 1) * ne + qn), w_hi[:, q] * Pi[q, qn])
    return P


def stationary_distribution(ap, grid, Pi, pi, method="solve", lam_init=None, tol=1e-11, max_iter=200000):
    """Invariant distribution over (a, e): by a linear solve, or by iterating the operator."""
    na, ne = ap.shape
    j, w_hi = lottery(ap, grid)
    if method == "solve":
        A = transition_matrix(j, w_hi, Pi).T - np.eye(na * ne)
        A[-1, :] = 1.0                                   # replace one equation by sum = 1
        b = np.zeros(na * ne); b[-1] = 1.0
        lam = np.maximum(np.linalg.solve(A, b), 0.0).reshape(na, ne)
        return lam / lam.sum(), 1, 0.0
    lam = np.zeros((na, ne)); lam[0, :] = pi
    if lam_init is not None:
        lam = lam_init.copy()
    diff, it = np.inf, 0
    for it in range(1, max_iter + 1):
        new = push_forward(lam, j, w_hi, Pi)
        diff = float(np.abs(new - lam).max())
        lam = new
        if diff < tol:
            break
    return lam / lam.sum(), it, diff


def gini(values, mass) -> float:
    m = np.asarray(mass, dtype=float).ravel(); x = np.asarray(values, dtype=float).ravel()
    order = np.argsort(x); x, m = x[order], m[order] / m.sum()
    S = np.cumsum(m * x)
    if S[-1] <= 0:
        return 0.0
    L = np.concatenate([[0.0], S / S[-1]]); F = np.concatenate([[0.0], np.cumsum(m)])
    return float(1.0 - np.sum((F[1:] - F[:-1]) * (L[1:] + L[:-1])))


def top_share(values, mass, top=0.10) -> float:
    m = np.asarray(mass, dtype=float).ravel(); x = np.asarray(values, dtype=float).ravel()
    order = np.argsort(x)[::-1]; x, m = x[order], m[order] / m.sum()
    cum = np.cumsum(m); k = np.searchsorted(cum, top)
    held = (m[:k] * x[:k]).sum() + (top - (cum[k - 1] if k else 0.0)) * x[k]
    return float(held / (m * x).sum())


# --------------------------------------------------------------------------------------
# Block C -- the stationary equilibrium
# --------------------------------------------------------------------------------------
def economy(p: Params):
    e, Pi, pi = rouwenhorst(p.n_e, p.rho_e, p.sd_uncond)
    return make_grid(p), e, Pi, pi


def capital_supply(r, p: Params, grid, e, Pi, pi, c_init=None, dist="iterate", lam_init=None):
    """Households' capital at a pre-tax return r, with the tax rebated at the firm's demand."""
    kl, w = firm(r, p)
    r_net, transfer = (1.0 - p.tau_k) * r, p.tau_k * r * kl
    ap, c, it, diff = solve_household(r_net, w, transfer, grid, e, Pi, p, c_init=c_init)
    lam, it_d, _ = stationary_distribution(ap, grid, Pi, pi, method=dist, lam_init=lam_init)
    Ks = float((lam.sum(axis=1) * grid).sum())
    return dict(r=r, r_net=r_net, w=w, transfer=transfer, K_demand=kl, K_supply=Ks, ap=ap, c=c, lam=lam,
                egm_sweeps=it, dist_sweeps=it_d)


def solve_equilibrium(p: Params, r_lo=None, r_hi=None, tol=1e-7, dist="iterate", verbose=False):
    """Bisection on the pre-tax return. Refuses to run without a sign change at the ends.

    The policy and the distribution are warm-started from the previous price. dist="solve" uses the dense
    linear solve instead of iteration: the same fixed point, slower at this size.
    """
    t0 = time.time()
    grid, e, Pi, pi = economy(p)
    r_lo = -p.delta + 0.02 if r_lo is None else r_lo
    r_hi = r_upper(p) - 1e-5 if r_hi is None else r_hi
    lo = capital_supply(r_lo, p, grid, e, Pi, pi, dist=dist)
    hi = capital_supply(r_hi, p, grid, e, Pi, pi, dist=dist)
    d_lo, d_hi = lo["K_supply"] - lo["K_demand"], hi["K_supply"] - hi["K_demand"]
    if not (d_lo < 0.0 < d_hi):
        raise RuntimeError(f"no sign change on [{r_lo:.5f}, {r_hi:.5f}]: excess supply {d_lo:.4f}, {d_hi:.4f}")
    sol, n = hi, 0
    while r_hi - r_lo > tol:
        n += 1
        sol = capital_supply(0.5 * (r_lo + r_hi), p, grid, e, Pi, pi, c_init=sol["c"], dist=dist, lam_init=sol["lam"])
        d = sol["K_supply"] - sol["K_demand"]
        if verbose:
            print(f"   {n:2d}  r = {sol['r']:.6f}   excess supply = {d:+.6f}")
        if d > 0:
            r_hi = sol["r"]
        else:
            r_lo = sol["r"]
    sol.update(grid=grid, e=e, Pi=Pi, pi=pi, bisections=n, seconds=time.time() - t0, params=p,
               clearing_residual=abs(sol["K_supply"] - sol["K_demand"]) / sol["K_demand"],
               bracket=(r_lo, r_hi), sign_change=(d_lo, d_hi))
    return sol


def five_checks(sol) -> dict:
    """The five things to check before believing a stationary equilibrium."""
    p, grid, lam = sol["params"], sol["grid"], sol["lam"]
    a_off, res, slack = euler_residuals(sol["ap"], sol["c"], sol["r_net"], sol["w"], sol["transfer"],
                                        grid, sol["e"], sol["Pi"], p)
    wts = np.repeat(lam[:-1, :] / 3.0, 3, axis=0)            # mass near each off-grid point
    return dict(
        mass_error=float(abs(lam.sum() - 1.0)),
        mass_at_top=float(lam[-1, :].sum()),
        euler_log10_max=float(np.nanmax(np.where(slack, res, np.nan))),
        euler_log10_mean_weighted=float(np.nansum(np.where(slack, res, 0.0) * wts) / np.sum(wts * slack)),
        clearing_residual=float(sol["clearing_residual"]),
        share_constrained=float(lam[0, :].sum()),
        gini_wealth=gini(np.repeat(grid[:, None], lam.shape[1], 1), lam),
        gini_earnings=gini(sol["e"], sol["pi"]),
        top10_share=top_share(np.repeat(grid[:, None], lam.shape[1], 1), lam, 0.10),
        top1_share=top_share(np.repeat(grid[:, None], lam.shape[1], 1), lam, 0.01),
    )
