"""Session 4 lab -- paths and cohorts, in plain numpy.

Block G of the lab: the transition path of the infinitely-lived economy after an unanticipated,
permanent tax on capital income, with the welfare arithmetic that a comparison of two steady states
gets wrong. The stationary life-cycle (overlapping-generations) economy is here too, with a
pay-as-you-go pension whose budget is balanced every period by a payroll tax.

Descended from the solver of the instructor's quantitative-macro course and built on
`session04_ha`: a parameter object, numpy only, and a sign-change test in every root-finder.
In finite lives 1/beta - 1 is NOT an upper bound on the interest rate, so the life-cycle bracket is wide.

    python3 session04_paths.py            live defaults, prints every gate
    python3 session04_paths.py --record   full-budget runs, writes recorded_paths.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, replace

import numpy as np

from session04_ha import (Params, economy, egm_step, firm, gini, lottery, push_forward,
                          r_upper, solve_equilibrium, solve_household, stationary_distribution, u)

HERE = os.path.dirname(os.path.abspath(__file__))
RECORD_FILE = os.path.join(HERE, "recorded_paths.json")


# --------------------------------------------------------------------------------------
# Welfare arithmetic
# --------------------------------------------------------------------------------------
def steady_state_value(c, ap, grid, Pi, p: Params, tol=1e-12, max_iter=50000):
    """Value of following a stationary policy for ever: iterate V = u(c) + beta E[V(a', e')]."""
    flow = u(c, p.sigma)
    V = flow / (1.0 - p.beta)
    for _ in range(max_iter):
        EV = V @ Pi.T
        cont = np.empty_like(V)
        for q in range(Pi.shape[0]):
            cont[:, q] = np.interp(ap[:, q], grid, EV[:, q])
        V_new = flow + p.beta * cont
        done = float(np.abs(V_new - V).max()) < tol
        V = V_new
        if done:
            break
    return V


def values_along(c_path, ap_path, grid, Pi, p: Params, V_final):
    """Values along a path, backward from V_T = V_final: V_t = u(c_t) + beta E[V_{t+1}(a'_t, e')]."""
    T = len(c_path) - 1
    V = [None] * (T + 1)
    V[T] = V_final
    for t in range(T - 1, -1, -1):
        EV = V[t + 1] @ Pi.T
        cont = np.empty_like(EV)
        for q in range(Pi.shape[0]):
            cont[:, q] = np.interp(ap_path[t][:, q], grid, EV[:, q])
        V[t] = u(c_path[t], p.sigma) + p.beta * cont
    return V


def cev(V_new, V_old, p: Params):
    """Consumption-equivalent variation g: scale the OLD consumption plan by (1 + g) to match V_new.

    CRRA: V(c (1+g)) = (1+g)^(1-sigma) V(c), so g = (V_new / V_old)^(1/(1-sigma)) - 1. With sigma > 1
    both values are negative, the ratio is positive, and a less negative V_new gives g > 0.
    Log utility: V(c (1+g)) = V(c) + log(1+g) / (1-beta).
    """
    V_new, V_old = np.asarray(V_new, dtype=float), np.asarray(V_old, dtype=float)
    if p.sigma == 1.0:
        return np.exp((1.0 - p.beta) * (V_new - V_old)) - 1.0
    ratio = V_new / V_old
    if np.any(ratio <= 0.0):
        raise ValueError("values of opposite sign: the CRRA formula does not apply")
    return ratio ** (1.0 / (1.0 - p.sigma)) - 1.0


def half_life(path, x0, x1) -> float:
    """Periods until the path has covered half of the distance from x0 to x1 (linear in between)."""
    if abs(x1 - x0) < 1e-14:
        return float("nan")
    frac = (np.asarray(path) - x0) / (x1 - x0)
    idx = np.where(frac >= 0.5)[0]
    if idx.size == 0:
        return float("nan")
    k = int(idx[0])
    if k == 0:
        return 0.0
    return float(k - 1 + (0.5 - frac[k - 1]) / (frac[k] - frac[k - 1]))


def _decile_means(g, lam, n=10):
    """Mean of g within each wealth decile of the distribution lam (mass split exactly at the edges)."""
    mass_a = lam.sum(axis=1)
    g_a = (g * lam).sum(axis=1) / np.maximum(mass_a, 1e-300)
    cum = np.concatenate([[0.0], np.cumsum(mass_a)])
    out = []
    for d in range(n):
        lo, hi = d / n, (d + 1) / n
        share = np.clip(np.minimum(cum[1:], hi) - np.maximum(cum[:-1], lo), 0.0, None)
        out.append(float((share * g_a).sum() / share.sum()))
    return out


# --------------------------------------------------------------------------------------
# Block G -- the transition path
# --------------------------------------------------------------------------------------
def path_prices(K, p: Params, tau_k: float):
    """Prices and the rebate along a path of capital (labour is one): r_t, (1 - tau_k) r_t, w_t, tau_k r_t K_t."""
    r = p.alpha * K ** (p.alpha - 1.0) - p.delta
    return r, (1.0 - tau_k) * r, (1.0 - p.alpha) * K ** p.alpha, tau_k * r * K


def backward_policies(r_net, w, transfer, c_final, grid, e, Pi, p: Params):
    """Households backward: from the terminal consumption policy to period 0, one EGM step per period."""
    T = len(r_net) - 1
    c_path, ap_path = [None] * (T + 1), [None] * (T + 1)
    c_path[T] = c_final
    for t in range(T - 1, -1, -1):
        c_path[t], ap_path[t] = egm_step(c_path[t + 1], r_net[t + 1], r_net[t], w[t], transfer[t],
                                         grid, e, Pi, p)
    return c_path, ap_path


def forward_capital(lam0, ap_path, grid, Pi, K0, c_path=None):
    """Distribution forward from lam0: the capital households supply, lambda_T, and (given the
    consumption policies) aggregate consumption in periods 0..T-1."""
    T = len(ap_path) - 1
    Ks, C = np.empty(T + 1), np.full(T + 1, np.nan)
    Ks[0] = K0
    lam = lam0
    for t in range(T):
        if c_path is not None:
            C[t] = float((lam * c_path[t]).sum())
        j, w_hi = lottery(ap_path[t], grid)
        lam = push_forward(lam, j, w_hi, Pi)
        Ks[t + 1] = float((lam.sum(axis=1) * grid).sum())
    return Ks, lam, C


def steady_state_after_tax(p_new: Params, ss_old: dict, dist: str = "iterate") -> dict:
    """The stationary equilibrium under the tax, from `solve_equilibrium` (sign change asserted there).

    A tax on capital income lowers capital and raises the pre-tax return, but not beyond the rate that
    restores the old AFTER-tax return: there households would hold about as much as before while firms
    want less. So [r_old, r_old / (1 - tau_k)] is tried first -- both ends are cheap to evaluate, being
    far from the patience bound -- and the wide after-tax bracket is the fallback if the sign test refuses.
    """
    if p_new.tau_k > 0.0:
        try:
            return solve_equilibrium(p_new, r_lo=ss_old["r"], r_hi=ss_old["r"] / (1.0 - p_new.tau_k), dist=dist)
        except RuntimeError:
            pass
    return solve_equilibrium(p_new, dist=dist)


def transition_path(p: Params = Params(), tau_k: float = 0.20, T: int = 150, damping: float = 0.5,
                    tol: float = 1e-5, max_outer: int = 300, ss_old=None, ss_new=None, K_init=None,
                    dist: str = "iterate", verbose: bool = False) -> dict:
    """Perfect-foresight path after an unanticipated, permanent tax on capital income.

    Timing. Up to t = -1 the economy rests in the stationary equilibrium with no tax. At the start of
    t = 0 the tax tau_k is announced, applies at once and stays for ever. K_0 and the distribution
    lambda_0 are predetermined: they are the old steady state's. In EVERY period t >= 0 households earn
    (1 - tau_k) r_t on their assets and receive the rebate tau_k r_t K_t; firms pay r_t and w_t at K_t.
    Households know the whole path from t = 0 on. Period T behaves as the new steady state.

    Algorithm. Guess {K_t}; prices from K_t; households backward from the new steady state's policy;
    the distribution forward from lambda_0; compare the capital households supply with the guess;
    damped update; stop when max_t |K^s_t - K_t| / K_t < tol.
    """
    t0 = time.time()
    p_old, p_new = replace(p, tau_k=0.0), replace(p, tau_k=tau_k)
    if ss_old is None:
        ss_old = solve_equilibrium(p_old, dist=dist)
    if ss_new is None:
        ss_new = steady_state_after_tax(p_new, ss_old, dist=dist)
    t_ss = time.time() - t0
    grid, e, Pi = ss_old["grid"], ss_old["e"], ss_old["Pi"]
    K_old, K_new = ss_old["K_supply"], ss_new["K_supply"]
    lam0 = ss_old["lam"]

    tt = np.arange(T + 1)
    if K_init is None:
        K = K_new + (K_old - K_new) * np.exp(-tt / 12.0)
    else:                                                  # warm start from a shorter path
        K = np.concatenate([K_init, np.full(max(0, T + 1 - len(K_init)), K_init[-1])])[:T + 1].copy()
    K[0] = K_old

    gap, outer, history = np.inf, 0, []
    t1 = time.time()
    for outer in range(1, max_outer + 1):
        r, r_net, w, transfer = path_prices(K, p_new, tau_k)
        c_path, ap_path = backward_policies(r_net, w, transfer, ss_new["c"], grid, e, Pi, p_new)
        Ks, lam_T, C = forward_capital(lam0, ap_path, grid, Pi, K_old, c_path)
        resid = np.abs(Ks - K) / K
        gap = float(resid.max())
        history.append(gap)
        if verbose:
            print(f"   outer {outer:3d}   max |K^s - K| / K = {gap:.3e}")
        if gap < tol:
            break
        K = (1.0 - damping) * K + damping * Ks
        K[0] = K_old
    converged = gap < tol
    t_path = time.time() - t1

    r, r_net, w, transfer = path_prices(K, p_new, tau_k)
    Y = K ** p_new.alpha
    # goods add up: C_t + K_{t+1} = (1 - delta) K_t + Y_t, an identity only if budgets, rebates and the
    # lottery are all consistent (and no mass is clipped at the top of the grid)
    resources = np.abs(C[:-1] + Ks[1:] - (1.0 - p_new.delta) * K[:-1] - Y[:-1]) / Y[:-1]
    out = dict(params=p_new, tau_k=tau_k, T=T, damping=damping, tol=tol, outer_iterations=outer,
               converged=bool(converged), gap_history=history,
               K=K, K_supply=Ks, r=r, r_net=r_net, w=w, transfer=transfer,
               residual=resid, max_residual=gap, C=C, Y=Y,
               resource_residual=resources, max_resource_residual=float(resources.max()),
               lam_T=lam_T, terminal_L1=float(np.abs(lam_T - ss_new["lam"]).sum()),
               terminal_K_gap=float(abs(K[-1] - K_new) / K_new),
               K_old=K_old, K_new=K_new, K_change_pct=100.0 * (K_new / K_old - 1.0),
               r_old=ss_old["r"], r_new=ss_new["r"], r_net_old=ss_old["r_net"], r_net_new=ss_new["r_net"],
               w_old=ss_old["w"], w_new=ss_new["w"],
               half_life_K=half_life(K, K_old, K_new),
               ss_old=ss_old, ss_new=ss_new, c_path=c_path, ap_path=ap_path,
               seconds_steady_states=t_ss, seconds_path=t_path, seconds=time.time() - t0)
    return out


def transition_welfare(tr: dict) -> dict:
    """Who gains. CEV at t = 0 for every (a, e) along the path, against staying in the old steady state --
    and the number a comparison of the two steady states would have reported instead."""
    p, ss0, ss1 = tr["params"], tr["ss_old"], tr["ss_new"]
    grid, Pi = ss0["grid"], ss0["Pi"]
    lam0, lam1 = ss0["lam"], ss1["lam"]
    V_old = steady_state_value(ss0["c"], ss0["ap"], grid, Pi, p)
    V_new = steady_state_value(ss1["c"], ss1["ap"], grid, Pi, p)
    V_path = values_along(tr["c_path"], tr["ap_path"], grid, Pi, p, V_new)

    g = cev(V_path[0], V_old, p)                        # state by state, the path included
    g_same_state = cev(V_new, V_old, p)                 # same (a, e), the path ignored
    W_old, W_new, W_path = float((V_old * lam0).sum()), float((V_new * lam1).sum()), float((V_path[0] * lam0).sum())
    return dict(
        cev_state=g,
        cev_mean_pct=100.0 * float((g * lam0).sum()),
        cev_utilitarian_pct=100.0 * float(cev(W_path, W_old, p)),
        share_worse_off_pct=100.0 * float(lam0[g < 0.0].sum()),
        cev_steady_states_own_distribution_pct=100.0 * float(cev(W_new, W_old, p)),
        cev_steady_states_same_state_mean_pct=100.0 * float((g_same_state * lam0).sum()),
        cev_by_wealth_decile_pct=[100.0 * x for x in _decile_means(g, lam0)],
        cev_by_productivity_pct=[100.0 * float((g[:, q] * lam0[:, q]).sum() / lam0[:, q].sum())
                                 for q in range(lam0.shape[1])],
        cev_min_pct=100.0 * float(g[lam0 > 1e-12].min()), cev_max_pct=100.0 * float(g[lam0 > 1e-12].max()),
        gini_old=gini(np.repeat(grid[:, None], lam0.shape[1], 1), lam0),
        gini_new=gini(np.repeat(grid[:, None], lam1.shape[1], 1), lam1),
        V_old=V_old, V_new=V_new, V_path0=V_path[0])


def cev_sanity(ss, p: Params, T: int = 25) -> dict:
    """Two checks of the welfare arithmetic. (1) No change, no gain: along a path that IS the steady state
    the CEV is zero. (2) Scale the old plan by (1 + g0): the formula returns g0."""
    grid, Pi = ss["grid"], ss["Pi"]
    V = steady_state_value(ss["c"], ss["ap"], grid, Pi, p)
    V_path = values_along([ss["c"]] * (T + 1), [ss["ap"]] * (T + 1), grid, Pi, p, V)
    no_change = float(np.abs(cev(V_path[0], V, p)).max())
    g0 = 0.0123
    scaled = (1.0 + g0) ** (1.0 - p.sigma) * V if p.sigma != 1.0 else V + np.log(1.0 + g0) / (1.0 - p.beta)
    recovered = float(np.abs(cev(scaled, V, p) - g0).max())
    assert no_change < 1e-8, f"CEV of no change is {no_change:.2e}, not zero"
    assert recovered < 1e-12, f"CEV formula misses a known scaling by {recovered:.2e}"
    return dict(no_change_max_abs_cev=no_change, known_scaling_error=recovered)


# --------------------------------------------------------------------------------------
# The stationary life-cycle economy with a pay-as-you-go pension financed by a payroll tax
# --------------------------------------------------------------------------------------
def olg_setup(p: Params = Params(), n_cohorts: int = 60, n_work: int = 45, a_max: float = 80.0, n_a: int = 250):
    """Sixty equal cohorts (ages 21-80), forty-five working years, a hump-shaped efficiency profile."""
    p_olg = replace(p, tau_k=0.0, a_min=0.0, a_max=a_max, n_a=n_a)
    grid, e, Pi, pi = economy(p_olg)
    i = np.arange(n_cohorts, dtype=float)
    eff = np.exp(0.09 * i - 0.0013 * i ** 2)
    eff[n_work:] = 0.0
    eff[:n_work] /= eff[:n_work].mean()                   # mean one over the working years
    L = float(eff.sum() / n_cohorts)                      # effective labour per person (E[e] = 1)
    return dict(p=p_olg, grid=grid, e=e, Pi=Pi, pi=pi, eff=eff, L=L, I=n_cohorts, I_R=n_work,
                ages=21 + np.arange(n_cohorts))


def payroll_tax(replacement: float, I: int, I_R: int) -> float:
    """Pension b = replacement x the average NET wage income of workers, paid from a payroll tax tau_p.
    Budget: tau_p w (I_R / I) = b (I - I_R) / I with b = replacement (1 - tau_p) w. Solve for tau_p."""
    return replacement * (I - I_R) / (I_R + replacement * (I - I_R))


def olg_at_r(r: float, replacement: float, s: dict) -> dict:
    """Everything at a candidate interest rate: households backward over ages, cohorts forward, aggregates."""
    p, grid, e, Pi, pi, eff, I, I_R = s["p"], s["grid"], s["e"], s["Pi"], s["pi"], s["eff"], s["I"], s["I_R"]
    kl, w = firm(r, p)
    tau_p = payroll_tax(replacement, I, I_R)
    b = replacement * (1.0 - tau_p) * w                   # average gross wage income of workers is w
    na, ne = grid.size, e.size
    c_age, ap_age = [None] * I, [None] * I
    c_age[I - 1] = np.repeat(np.maximum((1.0 + r) * grid + b, 1e-10)[:, None], ne, axis=1)   # last year: eat everything
    ap_age[I - 1] = np.zeros((na, ne))
    for i in range(I - 2, -1, -1):                        # no inner fixed point: one pass over ages
        wage_i, pension_i = ((1.0 - tau_p) * w * eff[i], 0.0) if i < I_R else (0.0, b)
        c_age[i], ap_age[i] = egm_step(c_age[i + 1], r, r, wage_i, pension_i, grid, e, Pi, p)
        c_age[i] = np.maximum(c_age[i], 1e-10)           # a retiree with no pension and no assets eats nothing
    lam = np.zeros((I, na, ne))
    lam[0, 0, :] = pi / I                                 # newborns: no assets, productivity from pi
    for i in range(I - 1):
        j, w_hi = lottery(ap_age[i], grid)
        lam[i + 1] = push_forward(lam[i], j, w_hi, Pi)
    K_supply = float((lam.sum(axis=2) * grid[None, :]).sum())
    K_demand = kl * s["L"]
    gross_wages = float(sum((lam[i].sum(axis=0) * e).sum() * w * eff[i] for i in range(I_R)))
    pensions = float(sum(lam[i].sum() * b for i in range(I_R, I)))
    return dict(r=r, w=w, kl=kl, tau_p=tau_p, pension=b, replacement=replacement, K_supply=K_supply,
                K_demand=K_demand, c_age=c_age, ap_age=ap_age, lam=lam,
                budget_residual=abs(tau_p * gross_wages - pensions),
                bequests=float((lam[I - 1] * ap_age[I - 1]).sum()))


def solve_olg(p: Params = Params(), replacement: float = 0.40, r_lo: float = None, r_hi: float = 0.30,
              tol: float = 1e-9, setup: dict = None, verbose: bool = False) -> dict:
    """Bisection on r over a WIDE bracket: in finite lives r may exceed 1/beta - 1.
    Refuses to run without a sign change of excess capital supply at the two ends."""
    t0 = time.time()
    s = olg_setup(p) if setup is None else setup
    p_olg = s["p"]
    r_lo = -p_olg.delta + 0.01 if r_lo is None else r_lo
    lo, hi = olg_at_r(r_lo, replacement, s), olg_at_r(r_hi, replacement, s)
    d_lo, d_hi = lo["K_supply"] - lo["K_demand"], hi["K_supply"] - hi["K_demand"]
    if not (d_lo < 0.0 < d_hi):
        raise RuntimeError(f"no sign change on [{r_lo:.4f}, {r_hi:.4f}]: excess supply {d_lo:.4f}, {d_hi:.4f}")
    a, b_, n = r_lo, r_hi, 0
    sol = hi
    while b_ - a > tol:
        n += 1
        sol = olg_at_r(0.5 * (a + b_), replacement, s)
        d = sol["K_supply"] - sol["K_demand"]
        if verbose:
            print(f"   {n:2d}  r = {sol['r']:.7f}   excess supply = {d:+.7f}")
        if d > 0:
            b_ = sol["r"]
        else:
            a = sol["r"]
    lam, grid, e, I = sol["lam"], s["grid"], s["e"], s["I"]
    mass_age = lam.sum(axis=(1, 2))
    assets = (lam.sum(axis=2) * grid[None, :]).sum(axis=1) / mass_age
    cons = np.array([(lam[i] * sol["c_age"][i]).sum() for i in range(I)]) / mass_age
    gross_lab = np.array([(lam[i].sum(axis=0) * e).sum() * sol["w"] * s["eff"][i] for i in range(I)]) / mass_age
    net_income = (1.0 - sol["tau_p"]) * gross_lab + np.where(np.arange(I) >= s["I_R"], sol["pension"], 0.0)
    K, L = sol["K_supply"], s["L"]
    Y = K ** p_olg.alpha * L ** (1.0 - p_olg.alpha)
    wealth = np.repeat(np.repeat(grid[None, :, None], I, axis=0), e.size, axis=2)
    sol.update(setup=s, bisections=n, bracket=(r_lo, r_hi), sign_change=(d_lo, d_hi),
               clearing_residual=abs(sol["K_supply"] - sol["K_demand"]) / sol["K_demand"],
               mass_error=float(abs(lam.sum() - 1.0)), mass_at_top=float(lam[:, -1, :].sum()),
               K=K, L=L, Y=Y, K_over_Y=K / Y, C=float((cons * mass_age).sum()),
               resource_residual=float(abs(Y - p_olg.delta * K - (cons * mass_age).sum()) / Y),
               r_vs_patience=sol["r"] - (1.0 / p_olg.beta - 1.0),
               profile_age=s["ages"], profile_assets=assets, profile_consumption=cons,
               profile_gross_labour_income=gross_lab, profile_net_income=net_income,
               peak_asset_age=int(s["ages"][int(np.argmax(assets))]),
               gini_wealth=gini(wealth, lam), seconds=time.time() - t0)
    assert sol["budget_residual"] < 1e-10, f"pension budget does not balance: {sol['budget_residual']:.2e}"
    assert sol["bequests"] == 0.0, "assets are left at the last age"
    return sol


def olg_comparison(p: Params = Params(), rates=(0.40, 0.0), setup: dict = None) -> list:
    """The same economy under different replacement rates: how a pension moves K and r."""
    s = olg_setup(p) if setup is None else setup
    rows = []
    for rep in rates:
        o = solve_olg(p, replacement=rep, setup=s)
        rows.append(dict(replacement=rep, tau_p=o["tau_p"], r_pct=100.0 * o["r"], K=o["K"], Y=o["Y"],
                         K_over_Y=o["K_over_Y"], w=o["w"], pension=o["pension"],
                         peak_asset_age=o["peak_asset_age"], gini_wealth=o["gini_wealth"],
                         clearing_residual=o["clearing_residual"], solution=o))
    return rows


def olg_scan(replacement: float, setup: dict, rates=None) -> dict:
    """Capital supplied and demanded on a grid of interest rates -- the picture behind the wide bracket."""
    rates = np.linspace(0.0, 0.12, 13) if rates is None else np.asarray(rates)
    pts = [olg_at_r(float(r), replacement, setup) for r in rates]
    return dict(r=rates, K_supply=np.array([x["K_supply"] for x in pts]),
                K_demand=np.array([x["K_demand"] for x in pts]))


# --------------------------------------------------------------------------------------
# Self-test and the recorded runs
# --------------------------------------------------------------------------------------
def _thin(x, n=200):
    x = np.asarray(x, dtype=float)
    idx = np.unique(np.linspace(0, x.size - 1, min(n, x.size)).round().astype(int))
    return idx, x[idx]


def _series(tr, n=200):
    idx, K = _thin(tr["K"], n)
    return dict(t=[int(i) for i in idx], K=[round(float(v), 6) for v in K],
                r_pre_pct=[round(100.0 * float(v), 6) for v in tr["r"][idx]],
                r_net_pct=[round(100.0 * float(v), 6) for v in tr["r_net"][idx]],
                w=[round(float(v), 6) for v in tr["w"][idx]],
                log10_residual=[round(float(np.log10(max(v, 1e-16))), 4) for v in tr["residual"][idx]])


def _transition_record(tr, wf) -> dict:
    p = tr["params"]
    return dict(
        tau_k=tr["tau_k"], T=tr["T"], damping=tr["damping"], tol=tr["tol"], n_a=p.n_a, n_e=p.n_e,
        outer_iterations=tr["outer_iterations"], converged=tr["converged"],
        max_clearing_residual=tr["max_residual"], max_resource_residual=tr["max_resource_residual"],
        terminal_L1_to_new_invariant=tr["terminal_L1"],
        C_old=float(tr["K_old"] ** p.alpha - p.delta * tr["K_old"]), C_new=float(tr["K_new"] ** p.alpha - p.delta * tr["K_new"]),
        C_impact=float(tr["C"][0]),
        terminal_K_gap=tr["terminal_K_gap"],
        K_old=tr["K_old"], K_new=tr["K_new"], K_change_pct=tr["K_change_pct"],
        r_old_pct=100 * tr["r_old"], r_new_pct=100 * tr["r_new"],
        r_net_old_pct=100 * tr["r_net_old"], r_net_new_pct=100 * tr["r_net_new"],
        r_net_impact_pct=100 * float(tr["r_net"][0]), w_old=tr["w_old"], w_new=tr["w_new"],
        w_change_pct=100 * (tr["w_new"] / tr["w_old"] - 1.0), rebate_impact=float(tr["transfer"][0]),
        rebate_new=float(tr["transfer"][-1]), half_life_K_years=tr["half_life_K"],
        steady_state_residual_old=tr["ss_old"]["clearing_residual"],
        steady_state_residual_new=tr["ss_new"]["clearing_residual"],
        steady_state_sign_change_old=list(tr["ss_old"]["sign_change"]),
        steady_state_sign_change_new=list(tr["ss_new"]["sign_change"]),
        after_tax_bracket_top_new_pct=100 * r_upper(p),
        seconds_steady_states=tr["seconds_steady_states"], seconds_path=tr["seconds_path"],
        welfare={k: v for k, v in wf.items() if not isinstance(v, np.ndarray)},
        series=_series(tr))


def _olg_record(rows, scan) -> dict:
    out = dict(comparison=[{k: v for k, v in r.items() if k != "solution"} for r in rows])
    o = rows[0]["solution"]
    s = o["setup"]
    out.update(
        cohorts=s["I"], working_years=s["I_R"], ages="21-80", n_a=s["p"].n_a, n_e=s["p"].n_e, a_max=s["p"].a_max,
        replacement=o["replacement"], tau_p=o["tau_p"], pension=o["pension"], r_pct=100 * o["r"],
        patience_rate_pct=100 * (1.0 / s["p"].beta - 1.0), r_minus_patience_pct=100 * o["r_vs_patience"],
        w=o["w"], K=o["K"], L=o["L"], Y=o["Y"], K_over_Y=o["K_over_Y"], clearing_residual=o["clearing_residual"],
        bracket=list(o["bracket"]), sign_change=list(o["sign_change"]), bisections=o["bisections"],
        budget_residual=o["budget_residual"], mass_error=o["mass_error"], mass_at_top=o["mass_at_top"],
        resource_residual=o["resource_residual"], peak_asset_age=o["peak_asset_age"],
        gini_wealth=o["gini_wealth"], seconds=o["seconds"],
        profiles=dict(age=[int(a) for a in o["profile_age"]],
                      assets=[round(float(v), 5) for v in o["profile_assets"]],
                      consumption=[round(float(v), 5) for v in o["profile_consumption"]],
                      gross_labour_income=[round(float(v), 5) for v in o["profile_gross_labour_income"]],
                      net_income=[round(float(v), 5) for v in o["profile_net_income"]],
                      efficiency=[round(float(v), 5) for v in s["eff"]]),
        profiles_no_pension=dict(assets=[round(float(v), 5) for v in rows[-1]["solution"]["profile_assets"]],
                                 consumption=[round(float(v), 5) for v in rows[-1]["solution"]["profile_consumption"]]),
        scan=dict(r_pct=[round(100 * float(v), 4) for v in scan["r"]],
                  K_supply=[round(float(v), 5) for v in scan["K_supply"]],
                  K_demand=[round(float(v), 5) for v in scan["K_demand"]]))
    return out


def _print_gates(tr, wf, sane, rows):
    print("\n-- Block G: transition path ---------------------------------------------------------")
    for name, ss in (("old", tr["ss_old"]), ("new", tr["ss_new"])):
        print(f"  {name} steady state: r = {100 * ss['r']:.4f}%   sign change {ss['sign_change'][0]:+.3f} / "
              f"{ss['sign_change'][1]:+.3f}   clearing residual {ss['clearing_residual']:.1e}")
    print(f"  K {tr['K_old']:.4f} -> {tr['K_new']:.4f} ({tr['K_change_pct']:+.2f}%), half-life {tr['half_life_K']:.1f} years; "
          f"after-tax r {100 * tr['r_net_old']:.3f}% -> {100 * tr['r_net'][0]:.3f}% on impact -> {100 * tr['r_net_new']:.3f}%")
    print(f"  path: T = {tr['T']}, {tr['outer_iterations']} outer iterations, converged = {tr['converged']}, "
          f"max clearing residual {tr['max_residual']:.2e}")
    print(f"  goods add up along the path to {tr['max_resource_residual']:.1e} of output; consumption {tr['K_old'] ** tr['params'].alpha - tr['params'].delta * tr['K_old']:.4f} -> {tr['C'][0]:.4f} on impact -> {tr['K_new'] ** tr['params'].alpha - tr['params'].delta * tr['K_new']:.4f}")
    print(f"  terminal: L1(lambda_T, new invariant) = {tr['terminal_L1']:.2e},  |K_T - K_new| / K_new = {tr['terminal_K_gap']:.2e}")
    print(f"  CEV sanity: no change -> {sane['no_change_max_abs_cev']:.1e};  known scaling error {sane['known_scaling_error']:.1e}")
    print(f"  welfare: mean CEV {wf['cev_mean_pct']:+.3f}% (utilitarian {wf['cev_utilitarian_pct']:+.3f}%), "
          f"{wf['share_worse_off_pct']:.1f}% worse off;  two steady states alone would say "
          f"{wf['cev_steady_states_own_distribution_pct']:+.3f}%")
    print("  CEV by wealth decile (%): " + "  ".join(f"{x:+.2f}" for x in wf["cev_by_wealth_decile_pct"]))
    print("  CEV by productivity (%):  " + "  ".join(f"{x:+.2f}" for x in wf["cev_by_productivity_pct"]))
    print(f"  seconds: steady states {tr['seconds_steady_states']:.1f}, path {tr['seconds_path']:.1f}")
    print("-- Life-cycle economy ---------------------------------------------------------------")
    for r in rows:
        o = r["solution"]
        print(f"  replacement {r['replacement']:.2f}: tau_p = {r['tau_p']:.4f}, r = {r['r_pct']:.4f}% "
              f"(1/beta - 1 = {100 * (1 / o['setup']['p'].beta - 1):.4f}%), K = {r['K']:.4f}, K/Y = {r['K_over_Y']:.3f}, "
              f"peak assets at {r['peak_asset_age']}")
        print(f"      sign change {o['sign_change'][0]:+.3f} / {o['sign_change'][1]:+.3f} on [{o['bracket'][0]:.3f}, {o['bracket'][1]:.3f}], "
              f"clearing residual {o['clearing_residual']:.1e}, budget residual {o['budget_residual']:.1e}, "
              f"mass error {o['mass_error']:.1e}, top-of-grid mass {o['mass_at_top']:.1e}, "
              f"resources {o['resource_residual']:.1e}, {o['seconds']:.2f} s")


def run_live(p: Params = Params(), verbose: bool = False):
    tr = transition_path(p, verbose=verbose)
    wf = transition_welfare(tr)
    sane = cev_sanity(tr["ss_old"], tr["params"])
    rows = olg_comparison(p)
    assert tr["converged"], "the path did not converge"
    assert tr["max_resource_residual"] < 1e-4, "goods do not add up along the path"
    assert tr["terminal_L1"] < 5e-2, "the horizon is too short: lambda_T is far from the new invariant distribution"
    _print_gates(tr, wf, sane, rows)
    return tr, wf, sane, rows


def record(p_live: Params = Params()):
    """Full-budget runs behind every number a slide quotes; the live defaults are stored beside them."""
    t0 = time.time()
    tr_l, wf_l, sane, rows_l = run_live(p_live)
    p_full = replace(p_live, n_a=400)
    print("\n== full budget: n_a = 400, T = 300, tol = 1e-7 ==")
    tr_f = transition_path(p_full, T=300, tol=1e-7, K_init=None)
    wf_f = transition_welfare(tr_f)
    tr_g = transition_path(p_full, T=450, tol=1e-7, ss_old=tr_f["ss_old"], ss_new=tr_f["ss_new"], K_init=tr_f["K"])
    wf_g = transition_welfare(tr_g)
    n = tr_f["T"] + 1
    sens = dict(T_short=tr_f["T"], T_long=tr_g["T"],
                max_rel_K_difference=float(np.abs(tr_g["K"][:n] - tr_f["K"]).max() / tr_f["K_old"]),
                cev_mean_pct_short=wf_f["cev_mean_pct"], cev_mean_pct_long=wf_g["cev_mean_pct"],
                terminal_L1_short=tr_f["terminal_L1"], terminal_L1_long=tr_g["terminal_L1"])
    s_full = olg_setup(p_live, n_a=500)
    rows_f = olg_comparison(p_live, rates=(0.40, 0.20, 0.0), setup=s_full)
    scan = olg_scan(0.40, s_full)
    sane_f = cev_sanity(tr_f["ss_old"], tr_f["params"])
    _print_gates(tr_f, wf_f, sane_f, rows_f)
    print(f"  sensitivity to T: max |K_t(T=450) - K_t(T=300)| / K_old = {sens['max_rel_K_difference']:.2e}; "
          f"mean CEV {sens['cev_mean_pct_short']:+.4f}% vs {sens['cev_mean_pct_long']:+.4f}%")
    rec = dict(
        meta=dict(what="Session 4 lab, block G and the life-cycle economy: recorded full-budget runs",
                  numpy=np.__version__, python=sys.version.split()[0],
                  recorded=time.strftime("%Y-%m-%d %H:%M"), seconds_total=None,
                  calibration={k: v for k, v in asdict(p_live).items()},
                  timing_note="K_0 and lambda_0 are the old steady state's; in every period t >= 0 households earn "
                              "(1 - tau_k) r_t and receive the rebate tau_k r_t K_t."),
        transition_full=_transition_record(tr_f, wf_f), transition_live=_transition_record(tr_l, wf_l),
        transition_sensitivity_to_T=sens, cev_sanity=sane_f,
        olg_full=_olg_record(rows_f, scan), olg_live=_olg_record(rows_l, olg_scan(0.40, rows_l[0]["solution"]["setup"])))
    rec["meta"]["seconds_total"] = round(time.time() - t0, 1)
    with open(RECORD_FILE, "w") as fh:
        json.dump(rec, fh, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    print(f"\nwrote {os.path.basename(RECORD_FILE)} in {rec['meta']['seconds_total']} s")
    return rec


if __name__ == "__main__":
    if "--record" in sys.argv:
        record()
    else:
        run_live(verbose="--verbose" in sys.argv)
