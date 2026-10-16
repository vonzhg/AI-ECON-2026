"""Session 4 lab, block F -- Krusell and Smith (1998) in plain numpy.

The economy is the one every comparison in the literature uses: Krusell and Smith's, in the
parameterisation of Den Haan, Judd and Juillard (JEDC 2010, "Computational suite of models with
heterogeneous agents"), where the unemployed also receive a benefit financed by a tax on the employed.

  solve_ks          household by the endogenous grid method on (a, employment, K, z) under a log-linear
                    perceived law of motion; non-stochastic simulation of the histogram with Young's (2010)
                    lottery; regression by aggregate state; damped update of the coefficients
  diagnose_ks       R^2, and Den Haan's (2010) dynamic forecast error on a FRESH shock path: the rule is
                    iterated forward from one initial K and never re-synchronised with the simulation
  neural_forecasting_rule
                    the idea of Fernandez-Villaverde, Hurtado and Nuno (2023): let a small network be the
                    perceived law of motion, with K alone and with the histogram as its input (torch)

Parameter values, the household budget, the tax rule and all sixteen transition probabilities were checked
on 2026-09-20 against the model description the comparison project circulated to its participants
(Den Haan, Judd and Juillard, "Model B", Tables 1 and 2). The two restrictions on job-finding when the
aggregate state switches (1.25 and 0.75) are Krusell and Smith's; they were not read at source, but the
matrix built from them below reproduces the project's Table 2 to six decimals, which is asserted.

Written in the style of session04_ha.py, from the solver lineage of the instructor's quantitative-macro
course; rebuilt at the published calibration, with the published transition matrix, an aggregate-capital
grid that is never allowed to clamp, and an out-of-sample accuracy test.
Imports: numpy; torch only inside neural_forecasting_rule.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, replace

import numpy as np

try:                                   # Young's lottery is shared with blocks B and G
    from session04_ha import lottery
except Exception:                      # pragma: no cover - keeps this file usable by itself
    def lottery(ap, grid):
        a1 = np.clip(ap, grid[0], grid[-1])
        j = np.clip(np.searchsorted(grid, a1, side="right") - 1, 0, grid.size - 2)
        return j, (a1 - grid[j]) / (grid[j + 1] - grid[j])

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDED = os.path.join(HERE, "recorded_ks.json")

# State order used everywhere, as in the published table:
#   0 = (recession, unemployed)  1 = (recession, employed)  2 = (boom, unemployed)  3 = (boom, employed)
Z_OF_S = np.array([0, 0, 1, 1])
E_OF_S = np.array([0, 1, 0, 1])

PUBLISHED_MATRIX = np.array([
    [0.525,    0.35,     0.03125,  0.09375],
    [0.038889, 0.836111, 0.002083, 0.122917],
    [0.09375,  0.03125,  0.291667, 0.583333],
    [0.009115, 0.115885, 0.024306, 0.850694]])

# Maliar, Maliar and Valli (JEDC 2010), non-stochastic simulation ("Algorithm 2"), same economy.
PUBLISHED_BENCHMARK = dict(
    source="Maliar, Maliar & Valli (2010), JEDC 34: law of motion under non-stochastic simulation",
    kappa_bad=[0.122146, 0.965942], kappa_good=[0.136272, 0.963582], K_mean=39.037,
    note="their stochastic-simulation solution: Den Haan error 0.050% mean, 0.156% max over 10,000 periods")


# --------------------------------------------------------------------------------------
# Calibration and numerical settings
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class KSParams:
    """Den Haan, Judd and Juillard (2010), Table 1; quarterly."""
    beta: float = 0.99
    gamma: float = 1.0        # relative risk aversion; 1 is log utility
    alpha: float = 0.36
    delta: float = 0.025
    delta_a: float = 0.01     # productivity is 1 - delta_a in a recession and 1 + delta_a in a boom
    u_b: float = 0.10         # unemployment rate in a recession
    u_g: float = 0.04         # ... and in a boom
    mu: float = 0.15          # unemployment benefit as a fraction of the wage
    dur_z: float = 8.0        # expected duration of a boom or a recession, in quarters
    dur_u_b: float = 2.5      # expected duration of unemployment while the recession lasts
    dur_u_g: float = 1.5      # ... while the boom lasts
    a_min: float = 0.0        # k' >= 0

    @property
    def l_bar(self) -> float:             # time endowment: labour supply in a recession is one
        return 1.0 / (1.0 - self.u_b)

    @property
    def z(self) -> np.ndarray:
        return np.array([1.0 - self.delta_a, 1.0 + self.delta_a])

    @property
    def u(self) -> np.ndarray:
        return np.array([self.u_b, self.u_g])


@dataclass(frozen=True)
class KSSettings:
    n_a: int = 120            # household policy grid: nodes, top, curvature (dense near the constraint)
    a_max: float = 300.0       # nothing may pile up at the top: the right tail is long (wealth is near a random walk)
    curv: float = 2.5
    h_step: float = 0.1       # histogram grid: uniform on [0, a_max], as in the comparison project.
                              # Individual wealth is close to a random walk here, so the lottery's spreading on a
                              # coarse histogram would swamp the true dispersion; the histogram has to be fine.
    n_K: int = 7              # aggregate capital: nodes, equally spaced in logs
    K_lo: float = 30.0
    K_hi: float = 50.0
    T_sim: int = 3000         # fitting sample, including the burn-in
    T_burn: int = 500
    T_oos: int = 2100         # fresh path for the accuracy test, including its burn-in
    T_oos_burn: int = 100
    damping: float = 0.3      # weight on the newly estimated coefficients
    tol_coef: float = 1e-5
    tol_egm: float = 1e-8
    max_outer: int = 80
    max_egm: int = 6000
    n_coarse: int = 20        # nodes of the coarse histogram handed to the network
    K0: float = 40.0          # everyone starts here; the burn-in removes it
    seed: int = 20260920
    seed_oos: int = 20261016


FULL = KSSettings()
SHORT = replace(FULL, T_sim=1100, T_burn=100, T_oos=1100, T_oos_burn=100, max_outer=3)


class KGridError(RuntimeError):
    """Aggregate capital, or its forecast, reached the edge of the K grid."""


def make_grid(n, lo, hi, curv):
    return lo + (hi - lo) * np.linspace(0.0, 1.0, n) ** curv


# --------------------------------------------------------------------------------------
# The exogenous process
# --------------------------------------------------------------------------------------
def ks_transition_matrix(p: KSParams, check=True):
    """Joint chain over (aggregate state, employment), built from the published restrictions.

    Returns (Pz, P): the 2x2 chain for z and the 4x4 joint chain in the state order given above.
    """
    stay = 1.0 - 1.0 / p.dur_z
    Pz = np.array([[stay, 1.0 - stay], [1.0 - stay, stay]])
    u, dur_u = p.u, (p.dur_u_b, p.dur_u_g)
    # probability of staying unemployed, conditional on the aggregate move z -> z'
    uu = np.empty((2, 2))
    uu[0, 0] = 1.0 - 1.0 / dur_u[0]
    uu[1, 1] = 1.0 - 1.0 / dur_u[1]
    uu[1, 0] = 1.25 * uu[0, 0]            # boom -> recession: harder to leave unemployment
    uu[0, 1] = 0.75 * uu[1, 1]            # recession -> boom: easier
    P = np.zeros((4, 4))
    for zi in range(2):
        for zj in range(2):
            # the employed lose jobs at the rate that makes next period's unemployment equal u(z')
            eu = (u[zj] - u[zi] * uu[zi, zj]) / (1.0 - u[zi])
            P[2 * zi + 0, 2 * zj + 0] = Pz[zi, zj] * uu[zi, zj]
            P[2 * zi + 0, 2 * zj + 1] = Pz[zi, zj] * (1.0 - uu[zi, zj])
            P[2 * zi + 1, 2 * zj + 0] = Pz[zi, zj] * eu
            P[2 * zi + 1, 2 * zj + 1] = Pz[zi, zj] * (1.0 - eu)
    if check:
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12), "rows must sum to one"
        if p == KSParams():
            dev = float(np.abs(P - PUBLISHED_MATRIX).max())
            assert dev < 1e-6, f"transition matrix differs from the published table by {dev:.2e}"
    return Pz, P


def prices(K, zi, p: KSParams):
    """Rental rate, wage and tax rate at aggregate capital K in aggregate state zi (0 recession, 1 boom)."""
    L = 1.0 - p.u[zi]
    kl = K / (p.l_bar * L)
    r = p.alpha * p.z[zi] * kl ** (p.alpha - 1.0)
    w = (1.0 - p.alpha) * p.z[zi] * kl ** p.alpha
    tau = p.mu * p.u[zi] / (p.l_bar * L)          # balances the benefit budget every period
    return r, w, tau


def draw_z(T, Pz, rng, z0):
    z = np.empty(T, dtype=int)
    z[0] = z0
    draws = rng.random(T)
    for t in range(1, T):
        z[t] = z[t - 1] if draws[t] < Pz[z[t - 1], z[t - 1]] else 1 - z[t - 1]
    return z


# --------------------------------------------------------------------------------------
# The household, given a perceived law of motion  log K' = kappa[z,0] + kappa[z,1] log K
# --------------------------------------------------------------------------------------
def _resources(p: KSParams, K):
    """Gross return and non-asset income for each of the four states at capital K (scalar or array)."""
    K = np.asarray(K, dtype=float)
    R = np.empty((4,) + K.shape)
    inc = np.empty((4,) + K.shape)
    for s in range(4):
        r, w, tau = prices(K, Z_OF_S[s], p)
        R[s] = 1.0 - p.delta + r
        inc[s] = w * ((1.0 - tau) * p.l_bar if E_OF_S[s] else p.mu)
    return R, inc


def check_rule_inside(kappa, logK):
    """The forecast must map the K grid into itself, or the household would be reading a clamped policy."""
    logKp = kappa[:, 0][:, None] + kappa[:, 1][:, None] * logK[None, :]
    if logKp.min() <= logK[0] or logKp.max() >= logK[-1]:
        raise KGridError(f"forecast K' in [{np.exp(logKp.min()):.2f}, {np.exp(logKp.max()):.2f}] "
                         f"reaches the K grid [{np.exp(logK[0]):.2f}, {np.exp(logK[-1]):.2f}]: widen it")
    return logKp


def _back_to_grid(a, x, y, R, inc, a_min):
    """Interpolate every column of the endogenous-grid solution (x, y) back to the fixed grid, in one call.

    Columns are laid end to end with an offset that keeps the abscissae increasing. Queries below a column's
    first node are where the constraint binds; queries above its last node continue linearly.
    """
    nC = x.shape[1]
    off = np.arange(nC) * (2.0 * (max(a[-1], x.max()) - min(a[0], x.min())) + 1.0)
    cc = np.interp((a[:, None] + off).T.ravel(), (x + off).T.ravel(), y.T.ravel()).reshape(nC, a.size).T
    slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
    cc = np.where(a[:, None] > x[-1], y[-1] + slope * (a[:, None] - x[-1]), cc)
    return np.where(a[:, None] < x[0], R * a[:, None] + inc - a_min, cc)


def solve_household(kappa, p: KSParams, a, logK, P, c_init=None, tol=1e-8, max_iter=6000):
    """Endogenous grid method. Returns (consumption policy c[a, s, K-node], sweeps, last change)."""
    nA, nK = a.size, logK.size
    R, inc = _resources(p, np.exp(logK))                         # (4, nK)
    logKp = check_rule_inside(kappa, logK)                       # (2, nK): tomorrow's K by today's z
    pos = (logKp - logK[0]) / (logK[1] - logK[0])
    m0 = np.clip(np.floor(pos).astype(int), 0, nK - 2)
    th = pos - m0
    Rp, _ = _resources(p, np.exp(logKp))                         # (4, 2, nK): return tomorrow, by s'
    c = (R - 1.0)[None] * a[:, None, None] + inc[None] if c_init is None else c_init.copy()
    c = np.maximum(c, 1e-10)
    Rf, incf = R.reshape(-1), inc.reshape(-1)
    diff, it = np.inf, 0
    for it in range(1, max_iter + 1):
        cn = (1.0 - th) * c[:, :, m0] + th * c[:, :, m0 + 1]     # c'(a', s', K'(z, K)): (nA, 4, 2, nK)
        MU = Rp[None] * cn ** (-p.gamma)
        Euc = np.einsum("sp,apsm->asm", P, MU[:, :, Z_OF_S, :])  # E[(1 - delta + r') u'(c') | s] at a' = grid
        c_end = (p.beta * Euc) ** (-1.0 / p.gamma)               # invert the Euler equation
        a_end = (c_end + a[:, None, None] - inc[None]) / R[None]  # the endogenous grid
        c_new = _back_to_grid(a, a_end.reshape(nA, -1), c_end.reshape(nA, -1), Rf, incf, p.a_min)
        c_new = np.maximum(c_new.reshape(nA, 4, nK), 1e-10)
        diff = float(np.abs(c_new - c).max())
        c = c_new
        if diff < tol:
            break
    return c, it, diff


# --------------------------------------------------------------------------------------
# Non-stochastic simulation of the cross-section
# --------------------------------------------------------------------------------------
def hist_grid(s: KSSettings, a_min=0.0):
    n = int(round((s.a_max - a_min) / s.h_step)) + 1
    return np.linspace(a_min, s.a_max, n)


def coarse_projection(h, n_coarse, curv=2.0):
    """Mean-preserving projection of the fine histogram onto a few nodes: K = nodes @ x exactly."""
    nodes = make_grid(n_coarse, h[0], h[-1], curv)
    j, w_hi = lottery(h, nodes)
    W = np.zeros((h.size, n_coarse))
    W[np.arange(h.size), j] += 1.0 - w_hi
    W[np.arange(h.size), j + 1] += w_hi
    return nodes, W


def initial_distribution(h, K0, u0):
    j, w_hi = lottery(np.array([K0]), h)
    col = np.zeros(h.size)
    col[j[0]], col[j[0] + 1] = 1.0 - w_hi[0], w_hi[0]
    return np.column_stack([u0 * col, (1.0 - u0) * col])


def restart_distribution(lam, u0):
    """Re-use a distribution as a starting point in a period whose unemployment rate is u0."""
    out = lam.copy()
    out[:, 0] *= u0 / out[:, 0].sum()
    out[:, 1] *= (1.0 - u0) / out[:, 1].sum()
    return out


def simulate(c, p: KSParams, a, logK, Pz, P, z, lam0, h, W=None, burn=0):
    """Push the histogram over (wealth, employment) along the path z. No individuals are drawn: no sampling error.

    The policy lives on the grid `a`; the histogram lives on the fine uniform grid `h`, and Young's lottery
    puts each household's choice back on it without changing the mean. Returns K (one entry per period), the
    last distribution, the time-averaged cross-section after `burn`, the gate statistics and, if W is given,
    the coarse histogram each period. Raises KGridError if K touches the edge of the K grid.
    """
    T, nH, nK = z.size, h.size, logK.size
    dlog, K_lo, K_hi = logK[1] - logK[0], np.exp(logK[0]), np.exp(logK[-1])
    dh = h[1] - h[0]
    l_bar, mu, delta, alpha, u, zv = p.l_bar, p.mu, p.delta, p.alpha, p.u, p.z
    K = np.empty(T)
    X = np.empty((T, W.shape[1])) if W is not None else None
    mass_err = unemp_err = top = 0.0
    cross = np.zeros(nH)
    lam = lam0.copy()
    for t in range(T):
        marg = lam.sum(axis=1)
        Kt = float(h @ marg)
        if not (K_lo < Kt < K_hi):
            raise KGridError(f"K = {Kt:.3f} at t = {t} is outside the K grid ({K_lo:.1f}, {K_hi:.1f}): widen it")
        K[t] = Kt
        if X is not None:
            X[t] = marg @ W
        zi = z[t]
        mass_err = max(mass_err, abs(marg.sum() - 1.0))
        unemp_err = max(unemp_err, abs(lam[:, 0].sum() - u[zi]))
        top = max(top, marg[-1])
        if t >= burn:
            cross += marg
        if t == T - 1:
            break
        pos = (np.log(Kt) - logK[0]) / dlog
        m0 = min(int(pos), nK - 2)
        th = pos - m0
        s0 = 2 * zi
        ct = (1.0 - th) * c[:, s0:s0 + 2, m0] + th * c[:, s0:s0 + 2, m0 + 1]      # (nA, 2), policy grid
        L = 1.0 - u[zi]
        kl = Kt / (l_bar * L)
        r = alpha * zv[zi] * kl ** (alpha - 1.0)
        w = (1.0 - alpha) * zv[zi] * kl ** alpha
        tau = mu * u[zi] / (l_bar * L)
        income = (mu * w, (1.0 - tau) * l_bar * w)
        moved = np.empty((nH, 2))
        for e in range(2):
            ap = np.clip((1.0 - delta + r) * h + income[e] - np.interp(h, a, ct[:, e]), p.a_min, h[-1])
            q = (ap - h[0]) / dh                                   # Young's lottery on a uniform grid
            j = np.minimum(q.astype(np.int64), nH - 2)
            w_hi = q - j
            moved[:, e] = (np.bincount(j, weights=lam[:, e] * (1.0 - w_hi), minlength=nH)
                           + np.bincount(j + 1, weights=lam[:, e] * w_hi, minlength=nH))
        zj = z[t + 1]
        lam = moved @ (P[s0:s0 + 2, 2 * zj:2 * zj + 2] / Pz[zi, zj])   # employment moves, given z -> z'
    return dict(K=K, z=z, X=X, lam_end=lam, cross_section=cross / max(1, T - burn), mass_error_max=mass_err,
                unemployment_error_max=unemp_err, mass_at_top_max=top)


def wealth_statistics(h, mass) -> dict:
    m = mass / mass.sum()
    cdf = np.cumsum(m)
    S = np.cumsum(m * h)
    Lz = np.concatenate([[0.0], S / S[-1]])
    F = np.concatenate([[0.0], cdf])
    pct = {f"p{q}": float(h[np.searchsorted(cdf, q / 100.0)]) for q in (10, 50, 90, 99)}
    return dict(gini=float(1.0 - np.sum((F[1:] - F[:-1]) * (Lz[1:] + Lz[:-1]))), mean=float(S[-1]),
                share_at_constraint=float(m[0]), share_below_one=float(m[h < 1.0].sum()),
                top_1pct_share=float(1.0 - np.interp(0.99, F, Lz)), maximum_with_mass=float(h[m > 1e-12].max()), **pct)


def fit_rule(K, z, start):
    """Least squares of log K' on log K, separately by today's aggregate state. Returns (kappa, R^2)."""
    x, y, zz = np.log(K[start:-1]), np.log(K[start + 1:]), z[start:-1]
    kappa, r2 = np.empty((2, 2)), np.empty(2)
    for zi in range(2):
        xi, yi = x[zz == zi], y[zz == zi]
        A = np.column_stack([np.ones_like(xi), xi])
        b, *_ = np.linalg.lstsq(A, yi, rcond=None)
        kappa[zi] = b
        r2[zi] = 1.0 - ((yi - A @ b) ** 2).sum() / ((yi - yi.mean()) ** 2).sum()
    return kappa, r2


# --------------------------------------------------------------------------------------
# The algorithm: solve, simulate, regress, update
# --------------------------------------------------------------------------------------
def rule_gap_pct(kappa_a, kappa_b, K_lo, K_hi):
    """Largest difference between two rules AS FORECASTS, in per cent of K', over the range K actually visits.

    Intercept and slope move against each other from one sample to the next, so coefficients can differ in the
    third decimal while the forecasts agree to a few parts in a hundred thousand.
    """
    d = np.asarray(kappa_a) - np.asarray(kappa_b)
    lk = np.log(np.linspace(K_lo, K_hi, 50))
    return float(100.0 * np.abs(d[:, :1] + d[:, 1:] * lk[None, :]).max())


def default_kappa():
    """A flat guess whose resting point sits in the middle of the K grid."""
    return np.array([[0.04 * np.log(38.5), 0.96], [0.04 * np.log(39.5), 0.96]])


def solve_ks(p: KSParams = KSParams(), s: KSSettings = FULL, kappa_init=None, lam_init=None, z0=1,
             verbose=True) -> dict:
    """Krusell-Smith fixed point in the coefficients of the perceived law of motion.

    kappa_init, lam_init, z0: a warm start (coefficients, a cross-section, and the aggregate state it belongs to).
    Without one, everybody starts with K0 and the distribution found by each pass starts the next one.
    """
    t0 = time.time()
    Pz, P = ks_transition_matrix(p)
    a = make_grid(s.n_a, p.a_min, s.a_max, s.curv)
    h = hist_grid(s, p.a_min)
    logK = np.linspace(np.log(s.K_lo), np.log(s.K_hi), s.n_K)
    nodes, W = coarse_projection(h, s.n_coarse)
    z = draw_z(s.T_sim, Pz, np.random.default_rng(s.seed), z0=z0)
    lam0 = initial_distribution(h, s.K0, p.u[z0]) if lam_init is None else restart_distribution(lam_init, p.u[z0])
    kappa = default_kappa() if kappa_init is None else np.array(kappa_init, dtype=float)
    c, rows, sweeps, widened = None, [], 0, 0
    sim = kappa_hat = r2 = None
    for outer in range(1, s.max_outer + 1):
        t1 = time.time()
        try:
            c, it, _ = solve_household(kappa, p, a, logK, P, c_init=c, tol=s.tol_egm, max_iter=s.max_egm)
            sim = simulate(c, p, a, logK, Pz, P, z, lam0, h)
        except KGridError as err:                        # never clamp: widen the grid and start over
            widened += 1
            if widened > 3:
                raise
            lo, hi = np.exp(logK[0]) * 0.85, np.exp(logK[-1]) * 1.15
            logK, c = np.linspace(np.log(lo), np.log(hi), s.n_K), None
            if verbose:
                print(f"   {err}\n   K grid widened to [{lo:.1f}, {hi:.1f}]")
            continue
        sweeps += it
        kappa_hat, r2 = fit_rule(sim["K"], z, s.T_burn)
        change = float(np.abs(kappa_hat - kappa).max())
        gap = rule_gap_pct(kappa_hat, kappa, sim["K"].min(), sim["K"].max())
        rows.append(dict(iteration=outer, egm_sweeps=it, change=change, forecast_gap_pct=gap,
                         kappa0_bad=float(kappa_hat[0, 0]),
                         kappa1_bad=float(kappa_hat[0, 1]), kappa0_good=float(kappa_hat[1, 0]),
                         kappa1_good=float(kappa_hat[1, 1]), seconds=time.time() - t1))
        if verbose:
            print(f"   pass {outer:2d}: {it:4d} household sweeps, coefficients moved {change:.2e}, forecasts moved "
                  f"{gap:.5f}%, R2 = {r2[0]:.6f} / {r2[1]:.6f}, {time.time() - t1:.1f} s")
        if change < s.tol_coef:
            break
        kappa = (1.0 - s.damping) * kappa + s.damping * kappa_hat
        lam0 = restart_distribution(sim["lam_end"], p.u[z0])    # start the next pass from a lived-in cross-section
    # `kappa` is the rule households used; `kappa_hat` is what their behaviour implies. They agree to `change`.
    sim = simulate(c, p, a, logK, Pz, P, z, lam0, h, W=W, burn=s.T_burn)
    return dict(params=p, settings=s, Pz=Pz, P=P, a=a, h=h, logK=logK, nodes=nodes, W=W, c=c, kappa=kappa,
                kappa_regression=kappa_hat, r2=r2, rows=rows, sim=sim, egm_sweeps=sweeps,
                converged=bool(rows and rows[-1]["change"] < s.tol_coef), K_grid_widened=widened,
                seconds=time.time() - t0)


# --------------------------------------------------------------------------------------
# Is the rule any good?  R^2 is a one-step statistic; Den Haan's test is not.
# --------------------------------------------------------------------------------------
def rule_path(kappa, K_start, z):
    """Iterate the perceived law forward from one initial K, never looking at the simulation again."""
    Kf = np.empty(z.size)
    Kf[0] = K_start
    for t in range(z.size - 1):
        Kf[t + 1] = np.exp(kappa[z[t], 0] + kappa[z[t], 1] * np.log(Kf[t]))
    return Kf


def pct_log_error(K_hat, K):
    return 100.0 * np.abs(np.log(K_hat) - np.log(K))


def _summary(err):
    return dict(mean_pct=float(err.mean()), max_pct=float(err.max()), periods=int(err.size))


def diagnose_ks(sol, long_run=200000, verbose=True) -> dict:
    """In-sample and out-of-sample accuracy of the converged rule."""
    p, s, kappa = sol["params"], sol["settings"], sol["kappa"]
    fit = sol["sim"]
    b = s.T_burn
    err_in = pct_log_error(rule_path(kappa, fit["K"][b], fit["z"][b:]), fit["K"][b:])
    # a fresh shock path, started from where the fitting sample ended
    z_oos = draw_z(s.T_oos, sol["Pz"], np.random.default_rng(s.seed_oos), z0=int(fit["z"][-1]))
    oos = simulate(sol["c"], p, sol["a"], sol["logK"], sol["Pz"], sol["P"], z_oos, fit["lam_end"], sol["h"],
                   W=sol["W"], burn=s.T_oos_burn)
    bo = s.T_oos_burn
    K_rule = rule_path(kappa, oos["K"][bo], z_oos[bo:])
    err_out = pct_log_error(K_rule, oos["K"][bo:])
    one = np.exp(kappa[z_oos[bo:-1], 0] + kappa[z_oos[bo:-1], 1] * np.log(oos["K"][bo:-1]))
    err_one = pct_log_error(one, oos["K"][bo + 1:])
    z_long = draw_z(long_run, sol["Pz"], np.random.default_rng(s.seed_oos + 1), z0=1)
    K_long = rule_path(kappa, fit["K"][b:].mean(), z_long)[1000:]
    out = dict(r2=dict(bad=float(sol["r2"][0]), good=float(sol["r2"][1])),
               den_haan_in_sample=_summary(err_in), den_haan_out_of_sample=_summary(err_out),
               one_step_out_of_sample=_summary(err_one), oos=oos, K_rule=K_rule, err_out=err_out,
               wealth=wealth_statistics(sol["h"], fit["cross_section"]),
               K=dict(mean=float(fit["K"][b:].mean()), sd=float(fit["K"][b:].std()),
                      min=float(min(fit["K"].min(), oos["K"].min())), max=float(max(fit["K"].max(), oos["K"].max())),
                      grid_lo=float(np.exp(sol["logK"][0])), grid_hi=float(np.exp(sol["logK"][-1])),
                      share_of_boom_quarters=float(fit["z"][b:].mean()),
                      rule_long_run_mean=float(K_long.mean()), rule_long_run_sd=float(K_long.std()),
                      rule_rest_bad=float(np.exp(kappa[0, 0] / (1.0 - kappa[0, 1]))),
                      rule_rest_good=float(np.exp(kappa[1, 0] / (1.0 - kappa[1, 1])))))
    if verbose:
        print(f"   R2 by state                     {out['r2']['bad']:.7f} (recession)   {out['r2']['good']:.7f} (boom)")
        for key, label in (("den_haan_in_sample", "Den Haan error, fitting path   "),
                           ("den_haan_out_of_sample", "Den Haan error, FRESH path     "),
                           ("one_step_out_of_sample", "one-step error, fresh path     ")):
            d = out[key]
            print(f"   {label} mean {d['mean_pct']:.4f}%   max {d['max_pct']:.4f}%   ({d['periods']} quarters)")
    return out


def gates(sol, diag) -> dict:
    """The checks block F prints. Every entry is a number; `passed` collects the verdicts."""
    fit, oos, K = sol["sim"], diag["oos"], diag["K"]
    g = dict(matrix_max_abs_dev=float(np.abs(sol["P"] - PUBLISHED_MATRIX).max()),
             K_min=K["min"], K_max=K["max"], K_grid=(K["grid_lo"], K["grid_hi"]),
             K_grid_widened=int(sol["K_grid_widened"]),
             mass_error_max=float(max(fit["mass_error_max"], oos["mass_error_max"])),
             unemployment_error_max=float(max(fit["unemployment_error_max"], oos["unemployment_error_max"])),
             mass_at_top_max=float(max(fit["mass_at_top_max"], oos["mass_at_top_max"])),
             share_at_constraint=float(diag["wealth"]["share_at_constraint"]),
             coefficient_change=float(sol["rows"][-1]["change"]),
             rule_gap_pct=float(sol["rows"][-1]["forecast_gap_pct"]))
    g["passed"] = dict(matrix=g["matrix_max_abs_dev"] < 1e-6,
                       K_strictly_inside_grid=K["grid_lo"] < K["min"] and K["max"] < K["grid_hi"],
                       mass_is_one=g["mass_error_max"] < 1e-10,
                       unemployment_matches=g["unemployment_error_max"] < 1e-10,
                       # the right tail is long and moves with aggregate history; below 1e-6 the ceiling
                       # cannot move K by more than one part in a hundred thousand
                       nothing_at_the_top=g["mass_at_top_max"] < 1e-6,
                       # the rule households use and the rule their behaviour implies agree as forecasts
                       rule_is_a_fixed_point=g["rule_gap_pct"] < 0.005)
    return g


# --------------------------------------------------------------------------------------
# A network as the perceived law of motion
# --------------------------------------------------------------------------------------
def neural_forecasting_rule(sol, diag, seeds=(0,), steps=1200, hidden=16, min_share=1e-3, verbose=True) -> dict:
    """Small networks as forecasting rules, graded exactly as the log-linear rule is graded.

    network K:          log K' = (log-linear rule) + correction(log K, z)
    network histogram:  log K' = (log-linear rule) + correction(coarse histogram, z).  A rule that conditions on
                        the histogram has to forecast the histogram too, or it cannot be iterated; a second head
                        forecasts its change, projected to have mass one and mean K'.
    Corrections start at zero, so training begins at the log-linear rule. Only nodes holding at least
    `min_share` of households are shown to the networks. The last fifth of the fitting sample picks the stopping
    point; every grade comes from the fresh path. Several seeds, because the verdict on the iterated histogram
    rule moves with the seed; paths are returned for the seed with the median Den Haan error.
    """
    import torch
    t0 = time.time()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    dt = torch.float64
    s, fit, oos, kap, nodes = sol["settings"], sol["sim"], diag["oos"], sol["kappa"], sol["nodes"]
    b, bo = s.T_burn, s.T_oos_burn

    def pack(sim, start):
        K, z, X = sim["K"][start:], sim["z"][start:], sim["X"][start:]
        return np.log(K[:-1]), z[:-1], X[:-1], np.log(K[1:]), X[1:] - X[:-1]

    k, z, X, y, dX = pack(fit, b)
    ko, zo, Xo, yo, _ = pack(oos, bo)
    base = lambda kk, zz: kap[zz, 0] + kap[zz, 1] * kk               # the rule households use
    res = y - base(k, z)
    sig = float(np.sqrt((res ** 2).mean()))
    k_m, k_s = k.mean(), k.std()
    X_m = X.mean(axis=0)
    live = X_m > min_share
    X_s = np.where(live, X.std(axis=0), 1.0)
    d_s = np.where(live, dX.std(axis=0), 1.0)

    def feat_K(kk, zz):
        return np.column_stack([(kk - k_m) / k_s, 2.0 * zz - 1.0])

    def feat_X(XX, zz):
        return np.column_stack([np.where(live, (XX - X_m) / X_s, 0.0)[:, live], 2.0 * zz - 1.0])

    def train(F, target, n_out):
        net = torch.nn.Sequential(torch.nn.Linear(F.shape[1], hidden), torch.nn.Tanh(),
                                  torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
                                  torch.nn.Linear(hidden, n_out)).to(dt)
        torch.nn.init.zeros_(net[-1].weight)
        torch.nn.init.zeros_(net[-1].bias)
        n_tr = int(0.8 * F.shape[0])
        Ft, Tt = torch.tensor(F[:n_tr], dtype=dt), torch.tensor(target[:n_tr], dtype=dt)
        Fv, Tv = torch.tensor(F[n_tr:], dtype=dt), torch.tensor(target[n_tr:], dtype=dt)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(1e-4 / 1e-2) ** (1.0 / steps))
        zero = float((Tv ** 2).mean())                              # what the uncorrected rule scores
        best, best_state = zero, {n: v.clone() for n, v in net.state_dict().items()}
        for step in range(1, steps + 1):
            opt.zero_grad()
            loss = ((net(Ft) - Tt) ** 2).mean()
            loss.backward()
            opt.step()
            sched.step()
            if step % 20 == 0 or step == steps:
                with torch.no_grad():
                    val = float(((net(Fv) - Tv) ** 2).mean())
                if val < best:
                    best, best_state = val, {n: v.clone() for n, v in net.state_dict().items()}
        net.load_state_dict(best_state)
        f = lambda A: net(torch.tensor(np.atleast_2d(A), dtype=dt)).detach().numpy()
        return f, best / zero

    T = ko.size + 1
    z_all, K_true = oos["z"][bo:], oos["K"][bo:]
    A = np.vstack([np.ones(nodes.size), nodes])                     # mass and mean of the histogram
    AAt_inv = np.linalg.inv(A @ A.T)
    lin_one = _summary(100.0 * np.abs(base(ko, zo) - yo))
    lin_dyn = _summary(pct_log_error(rule_path(kap, K_true[0], z_all), K_true))
    runs = []
    for seed in seeds:
        torch.manual_seed(seed)
        net_K, val_K = train(feat_K(k, z), (res / sig)[:, None], 1)
        net_X, val_X = train(feat_X(X, z), (res / sig)[:, None], 1)
        net_D, val_D = train(feat_X(X, z), (dX / d_s)[:, live], int(live.sum()))
        one_K = base(ko, zo) + sig * net_K(feat_K(ko, zo))[:, 0]
        one_X = base(ko, zo) + sig * net_X(feat_X(Xo, zo))[:, 0]
        # Den Haan's test: iterate each rule forward from the first period, never re-synchronised
        path_K = np.empty(T)
        path_K[0] = K_true[0]
        for t in range(T - 1):
            lk = np.log(path_K[t])
            path_K[t + 1] = np.exp(base(lk, z_all[t]) + sig * net_K(feat_K(np.array([lk]), z_all[t:t + 1]))[0, 0])
        x = oos["X"][bo].copy()
        path_X = np.full(T, np.nan)
        path_X[0] = nodes @ x
        for t in range(T - 1):
            F = feat_X(x[None, :], z_all[t:t + 1])
            K_next = np.exp(base(np.log(nodes @ x), z_all[t]) + sig * net_X(F)[0, 0])
            step = np.zeros(nodes.size)
            step[live] = net_D(F)[0] * d_s[live]
            for _ in range(2):                                      # mass one, mean K', nothing negative
                step = step - A.T @ (AAt_inv @ (A @ step - np.array([1.0 - x.sum(), K_next - nodes @ x])))
                step = np.maximum(step, -x)
            x = x + step
            if not np.isfinite(nodes @ x) or not (0.5 * K_true.min() < nodes @ x < 2.0 * K_true.max()):
                break
            path_X[t + 1] = nodes @ x
        ok = np.isfinite(path_X)
        runs.append(dict(seed=int(seed),
                         one_step=dict(network_K=_summary(100.0 * np.abs(one_K - yo)),
                                       network_histogram=_summary(100.0 * np.abs(one_X - yo))),
                         den_haan=dict(network_K=_summary(pct_log_error(path_K, K_true)),
                                       network_histogram=dict(_summary(pct_log_error(path_X[ok], K_true[ok])),
                                                              left_the_range=bool(not ok.all()))),
                         validation_mse_ratio=dict(network_K=val_K, network_histogram=val_X, histogram_head=val_D),
                         paths=dict(network_K=path_K, network_histogram=path_X)))
    torch.set_num_threads(threads)
    order = sorted(range(len(runs)), key=lambda i: runs[i]["den_haan"]["network_histogram"]["mean_pct"])
    mid = runs[order[len(order) // 2]]

    def across(group, name, key):
        v = np.array([r[group][name][key] for r in runs])
        return dict(median=float(np.median(v)), min=float(v.min()), max=float(v.max()))

    summary = {name: dict(one_step_mean_pct=across("one_step", name, "mean_pct"),
                          one_step_max_pct=across("one_step", name, "max_pct"),
                          den_haan_mean_pct=across("den_haan", name, "mean_pct"),
                          den_haan_max_pct=across("den_haan", name, "max_pct"),
                          validation_mse_ratio=dict(median=float(np.median([r["validation_mse_ratio"][name] for r in runs])),
                                                    note="held-back fifth of the fitting sample; 1.0 = no gain over the log-linear rule"))
               for name in ("network_K", "network_histogram")}
    out = dict(loglinear=dict(one_step=lin_one, den_haan=lin_dyn), summary=summary,
               runs=[{k_: v for k_, v in r.items() if k_ != "paths"} for r in runs],
               paths=dict(seed=mid["seed"], **mid["paths"]),
               design=dict(hidden=hidden, steps=steps, seeds=[int(v) for v in seeds], min_share=min_share,
                           coarse_nodes=int(nodes.size), nodes_shown=int(live.sum()),
                           residual_rms_of_loglinear_rule=sig),
               seconds=time.time() - t0)
    if verbose:
        print(f"   forecasting rules on the fresh path ({len(runs)} network seed{'s' if len(runs) > 1 else ''}; median [min, max])")
        print(f"   {'log-linear':<18} one-step {lin_one['mean_pct']:.5f}%                       "
              f"Den Haan {lin_dyn['mean_pct']:.4f}%   (max {lin_dyn['max_pct']:.4f}%)")
        for name in ("network_K", "network_histogram"):
            o, d_ = summary[name]["one_step_mean_pct"], summary[name]["den_haan_mean_pct"]
            m_ = summary[name]["den_haan_max_pct"]
            print(f"   {name:<18} one-step {o['median']:.5f}% [{o['min']:.5f}, {o['max']:.5f}]   "
                  f"Den Haan {d_['median']:.4f}% [{d_['min']:.4f}, {d_['max']:.4f}]   (max {m_['median']:.4f}%)")
        print(f"   networks trained and iterated in {out['seconds']:.1f} s")
    return out


# --------------------------------------------------------------------------------------
# Recorded full-budget run, and the short run the lab executes live
# --------------------------------------------------------------------------------------
def load_recorded():
    if not os.path.exists(RECORDED):
        return None
    with open(RECORDED) as f:
        return json.load(f)


def _store_distribution(sol, step=0.5) -> dict:
    """The cross-section at the end of the fitting sample, on a coarser grid that keeps its mean, for warm starts."""
    h, lam = sol["h"], sol["sim"]["lam_end"]
    grid = np.arange(h[0], h[-1] + 0.5 * step, step)
    j, w_hi = lottery(h, grid)
    out = np.zeros((grid.size, 2))
    for e in range(2):
        out[:, e] = (np.bincount(j, weights=lam[:, e] * (1.0 - w_hi), minlength=grid.size)
                     + np.bincount(j + 1, weights=lam[:, e] * w_hi, minlength=grid.size))
    last = int(np.nonzero(out.sum(axis=1) > 1e-13)[0].max()) + 1
    return dict(note="end of the fitting sample; mass at wealth = step * index, by employment status",
                step=step, z=int(sol["sim"]["z"][-1]),
                unemployed=[float(f"{v:.9e}") for v in out[:last, 0]],
                employed=[float(f"{v:.9e}") for v in out[:last, 1]])


def load_warm_start(rec, s: KSSettings, a_min=0.0):
    """Coefficients, cross-section and aggregate state from a recorded run, put on this run's histogram grid."""
    ws = rec["warm_start"]
    h = hist_grid(s, a_min)
    lam = np.zeros((h.size, 2))
    where = np.clip(np.round((np.arange(len(ws["unemployed"])) * ws["step"] - h[0]) / s.h_step).astype(int), 0, h.size - 1)
    np.add.at(lam[:, 0], where, ws["unemployed"])
    np.add.at(lam[:, 1], where, ws["employed"])
    kappa = np.array([rec["solution"]["kappa"]["bad"], rec["solution"]["kappa"]["good"]])
    return kappa, lam, int(ws["z"])


def _thin(x, n=300):
    idx = np.unique(np.linspace(0, len(x) - 1, min(n, len(x))).round().astype(int))
    return idx


def record_full(path=RECORDED, verbose=True) -> dict:
    """Full budget: solve from the flat guess to the coefficient tolerance, grade, fit the networks, write JSON."""
    import platform
    t0 = time.time()
    p, s = KSParams(), FULL
    sol = solve_ks(p, s, verbose=verbose)
    diag = diagnose_ks(sol, verbose=verbose)
    g = gates(sol, diag)
    nn = neural_forecasting_rule(sol, diag, seeds=(0, 1, 2, 3, 4), verbose=verbose)
    try:
        import torch
        torch_version = torch.__version__
    except Exception:                                     # pragma: no cover
        torch_version = None
    bo = s.T_oos_burn
    K_true, K_rule, err = diag["oos"]["K"][bo:], diag["K_rule"], diag["err_out"]
    peak = int(err.argmax())
    lo = int(np.clip(peak - 200, 0, max(0, err.size - 300)))
    win = slice(lo, lo + 300)
    idx = _thin(err)
    rec = dict(
        meta=dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), python=platform.python_version(),
                  numpy=np.__version__, torch=torch_version, seed=s.seed, seed_out_of_sample=s.seed_oos,
                  seconds_total=None,
                  source_check="parameters, budget, tax rule and all 16 transition probabilities read on 2026-09-20 in "
                               "Den Haan, Judd & Juillard's model description (Tables 1-2); the 1.25 / 0.75 "
                               "restrictions are Krusell & Smith's and were confirmed only through the assertion "
                               "that they reproduce Table 2"),
        params={**asdict(p), "l_bar": p.l_bar},
        settings=asdict(s),
        transition_matrix=[[float(v) for v in row] for row in sol["P"]],
        solution=dict(outer_iterations=len(sol["rows"]), converged=sol["converged"],
                      coefficient_change_last=sol["rows"][-1]["change"],
                      kappa=dict(bad=[float(v) for v in sol["kappa"][0]], good=[float(v) for v in sol["kappa"][1]]),
                      kappa_last_regression=dict(bad=[float(v) for v in sol["kappa_regression"][0]],
                                                 good=[float(v) for v in sol["kappa_regression"][1]]),
                      r2=diag["r2"], egm_sweeps_total=int(sol["egm_sweeps"]), seconds=sol["seconds"],
                      K_grid_widened=int(sol["K_grid_widened"]), iterations=sol["rows"]),
        capital=diag["K"],
        wealth=diag["wealth"],
        den_haan=dict(in_sample=diag["den_haan_in_sample"], out_of_sample=diag["den_haan_out_of_sample"]),
        one_step_out_of_sample=diag["one_step_out_of_sample"],
        gates={k: (list(v) if isinstance(v, tuple) else v) for k, v in g.items()},
        neural={k: v for k, v in nn.items() if k != "paths"},
        warm_start=_store_distribution(sol),
        published_benchmark=PUBLISHED_BENCHMARK,
        series=dict(
            window=dict(note="300 consecutive quarters of the fresh path around the largest error",
                        quarter=[int(v) for v in range(win.start, win.stop)],
                        K_simulated=[round(float(v), 5) for v in K_true[win]],
                        K_rule=[round(float(v), 5) for v in K_rule[win]],
                        network_seed=nn["paths"]["seed"],
                        K_network_K=[round(float(v), 5) for v in nn["paths"]["network_K"][win]],
                        K_network_histogram=[None if not np.isfinite(v) else round(float(v), 5)
                                             for v in nn["paths"]["network_histogram"][win]],
                        boom=[int(v) for v in diag["oos"]["z"][bo:][win]]),
            error=dict(note="Den Haan error of the log-linear rule over the whole fresh path, thinned",
                       quarter=[int(v) for v in idx], pct=[round(float(v), 6) for v in err[idx]])))
    rec["meta"]["seconds_total"] = time.time() - t0
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    if verbose:
        print(f"   wrote {os.path.basename(path)} ({rec['meta']['seconds_total']:.1f} s in all)")
    return rec


def run_short(verbose=True, neural=True) -> dict:
    """What the lab runs live: start from the recorded solution, make three passes of the loop, print the gates."""
    t0 = time.time()
    rec = load_recorded()
    kappa0, lam0, z0 = (None, None, 1) if rec is None else load_warm_start(rec, SHORT)
    if verbose:
        print("Block F, short run: " + ("starting from the recorded coefficients and cross-section" if rec
                                         else "no recorded run found; flat guess, everyone at K0"))
    sol = solve_ks(KSParams(), SHORT, kappa_init=kappa0, lam_init=lam0, z0=z0, verbose=verbose)
    diag = diagnose_ks(sol, long_run=50000, verbose=verbose)
    g = gates(sol, diag)
    nn = neural_forecasting_rule(sol, diag, seeds=(0,), steps=600, verbose=verbose) if neural else None
    if verbose:
        print(f"   gates: transition matrix within {g['matrix_max_abs_dev']:.1e} of the published table; "
              f"K in [{g['K_min']:.2f}, {g['K_max']:.2f}], strictly inside the grid ({g['K_grid'][0]:.0f}, {g['K_grid'][1]:.0f}); "
              f"mass error {g['mass_error_max']:.1e}; unemployment-rate error {g['unemployment_error_max']:.1e}; "
              f"mass at the top node {g['mass_at_top_max']:.1e}; perceived and implied rule differ by "
              f"{g['rule_gap_pct']:.5f}% of K as forecasts (coefficients by {g['coefficient_change']:.1e})")
        print("   " + ("ALL GATES PASS" if all(g["passed"].values()) else f"GATE FAILURE: {g['passed']}")
              + f"   ({time.time() - t0:.1f} s)")
    return dict(sol=sol, diag=diag, gates=g, neural=nn, seconds=time.time() - t0)


if __name__ == "__main__":
    if "--full" in sys.argv:
        record_full()
    else:
        out = run_short()
        sys.exit(0 if all(out["gates"]["passed"].values()) else 1)
