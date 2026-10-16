"""Session 4 lab, block H -- the sequence space on an overlapping-generations economy with idiosyncratic income
risk and an optional retirement choice, in PyTorch, at CPU scale.

Written for this lab from section 6 of Azinovic-Yang and Zemlicka, "Deep Learning in the Sequence Space" (2025),
and Deck 4.3, Part C. The paper publishes no code for this economy; everything here follows its text, and every
departure is listed below and said in the notebook.

The economy (the paper's 6.1-6.2): J = 10 cohorts of six years; ages 1-6 work, age 7 chooses to work or retire
(a utility cost of working, a logit taste shock), ages 8-10 are retired on home income; income e in {0.5, 1, 1.5}
with the paper's Markov matrix, dynastic (newborns inherit e and the dying cohort's assets through a warm-glow
bequest); CRRA with gamma 2, beta = 0.95^6; one asset, capital, k >= 0; a representative firm with alpha = 1/3; three
aggregate regimes with the paper's matrix, the third a disaster with depreciation 1.5 times the normal one.

The method (the paper's 6.3-6.4): a network reads the last T_hist regimes (one-hot) and outputs the
savings-choice-specific continuation values W(j, e, k') on the asset grid for ages 1-9, as a first value plus
softplus increments (so W rises in k'), and a guess lambda_hat of the share of age 7's labour supplied.
Within a period, given (z, Gamma, W): guess L; prices from (K, L, z); every cohort's problem; the labour market
cleared by Newton on ED(L) = L - L^hh(L) with ED' by automatic differentiation; then the policy bundle, Young's
lottery age by age, the age-7 split by p_work, newborns from the bequests: n_z = 3 candidate successors. Targets
W_tar(j, e, k') = sum_z' Pi_z sum_e' Pi_e V_{t+1}(j+1, e', k' | z') with t+1 solved under a target network (an
exponential moving average of the online one); the loss is the mean squared relative error on W's first values,
its increments, and lambda. E = B pairs (z, Gamma, history) run side by side; every update uses all of them and
then moves each one period: no pool, no epochs, nothing reused. Three stages: a deterministic steady state (seeds
the pairs), pretraining with the continuation held at the steady state's, the full loop.

Departures from the paper, each said in the notebook:
  1. Inside the period, every cohort's saving problem is solved by DIRECT MAXIMISATION on a fine saving grid
     (N_f points), the continuation value read between asset-grid points by linear interpolation. The paper uses
     generalised EGM with an upper-envelope step. The maximum over the grid is global, so the non-concave
     continuation values around the retirement choice need no envelope; the price is a policy that is exact only
     on the fine grid.
  2. Scale: N_k = 40 asset points (the paper 300), N_f = 200, T_hist = 20 (the paper 100; a lifetime is 10), E = B =
     64 pairs (the paper 1,024), two GELU layers of 128 (the paper three of 1,024), a few hundred updates at a
     learning rate of 3e-4 (the paper 1e-6 over many more), tau = 1e-2 for the target network (the paper 1e-4).
     The asset grid is dense near zero (curvature 1.5); the paper's is evenly spaced.
  3. The starting histories are all-normal (the paper gives none); Stage 2 then runs the pairs forward.
Imports: numpy, torch.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)
DT = torch.float64
HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------------------------- the economy (6.1-6.2)
J = 10                                     # cohorts; code ages 0..9 are the deck's 1..10
AGE_OPT = 6                                # the optional retirement age (the deck's age 7)
AGES_WORK = tuple(range(0, 6))
AGES_RET = (7, 8, 9)
N_E = 3
E_VALS = torch.tensor([0.5, 1.0, 1.5], dtype=DT)
PI_E = torch.tensor([[0.70, 0.20, 0.10], [0.15, 0.70, 0.15], [0.10, 0.20, 0.70]], dtype=DT)
N_PROFILE = torch.tensor([0.53, 0.83, 1.04, 1.17, 1.21, 1.17, 1.04, 0.0, 0.0, 0.0], dtype=DT)
GAMMA = 2.0
BETA = 0.95 ** 6
ALPHA = 1.0 / 3.0
N_Z = 3
PI_Z = torch.tensor([[0.90, 0.07, 0.03], [0.70, 0.20, 0.10], [0.50, 0.25, 0.25]], dtype=DT)
DELTA_N = 1.0 - 0.95 ** 6
DELTA = torch.tensor([DELTA_N, DELTA_N, 1.5 * DELTA_N], dtype=DT)
WORK_COST = 1.68
LOGIT_SCALE = 0.08
Y_HOME = 0.1
PSI_B, NU_B, B_BAR = 0.1, 2.0, 0.1
K_MAX = 5.0


def utility(c):
    return (c ** (1.0 - GAMMA) - 1.0) / (1.0 - GAMMA)


def bequest(k):
    return PSI_B * ((k + B_BAR) ** (1.0 - NU_B) - 1.0) / (1.0 - NU_B)


def prices(K, L, z):
    """Rental rate, wage and gross return R = 1 - delta(z) + r. K, L: (B,); z: (B,) long."""
    r = ALPHA * K ** (ALPHA - 1.0) * L ** (1.0 - ALPHA)
    w = (1.0 - ALPHA) * K ** ALPHA * L ** (-ALPHA)
    return r, w, 1.0 - DELTA[z] + r


def stationary(Pi, tol=1e-14):
    p = torch.full((Pi.shape[0],), 1.0 / Pi.shape[0], dtype=DT)
    for _ in range(100000):
        q = p @ Pi
        if (q - p).abs().max() < tol:
            return q
        p = q
    return p


# ---------------------------------------------------------------------------------------------- settings
@dataclass(frozen=True)
class Config:
    N_k: int = 40             # asset-grid points (the paper: 300)
    N_f: int = 120            # fine saving grid for the maximisation (the lab's; the paper: EGM)
    curv: float = 1.5         # grid curvature, dense near zero (the paper: evenly spaced)
    T_hist: int = 20          # regimes in the window (the paper: 100)
    E: int = 64               # pairs run side by side = the batch (the paper: 1,024)
    hidden: int = 128         # width of the two GELU layers (the paper: 1,024, three layers)
    n_newton: int = 4         # Newton steps on the labour market (the paper: 5); they converge to round-off in 3
    prefit_steps: int = 1000  # the lab's: fit the network to the steady state's W before Stage 2 (the paper: none)
    lr_prefit: float = 1e-3
    U2: int = 60              # Stage 2 updates (the paper gives no count)
    U3: int = 300             # Stage 3 updates (the paper gives no count)
    lr: float = 3e-4          # Adam (the paper: 1e-6)
    tau: float = 1e-2         # the target network's nudge (the paper: 1e-4)
    eps_schedule: tuple = (1.0, 0.1, 0.01)   # eps_loss by thirds of Stage 3 (the paper: 1 -> 0.1 -> 0.01)
    K_bracket: tuple = (0.05, 2.0)           # steady-state bisection on K; a sign change is asserted
    log_every: int = 25
    seed: int = 0


LIVE = Config()
RECORD = replace(LIVE, E=128, U2=120, U3=900)
SMOKE = replace(LIVE, N_k=16, N_f=60, T_hist=6, E=6, hidden=32, prefit_steps=20, U2=3, U3=6, n_newton=3, log_every=2)


class Grids:
    def __init__(self, cfg: Config):
        self.k = K_MAX * torch.linspace(0.0, 1.0, cfg.N_k, dtype=DT) ** cfg.curv       # the state grid
        self.kf = K_MAX * torch.linspace(0.0, 1.0, cfg.N_f, dtype=DT) ** cfg.curv      # the choice grid
        hi = torch.clamp(torch.searchsorted(self.k, self.kf, right=True), 1, cfg.N_k - 1)
        lo = hi - 1
        self.f_lo, self.f_hi = lo, hi
        self.f_w = (self.kf - self.k[lo]) / (self.k[hi] - self.k[lo])                # weight on hi
        self.bequest_f = bequest(self.kf)                                             # (N_f,)
        self.bequest_k = bequest(self.k)

    def to_fine(self, W):
        """W (..., N_k) on the state grid -> (..., N_f) on the choice grid, linear interpolation."""
        return W.index_select(-1, self.f_lo) * (1.0 - self.f_w) + W.index_select(-1, self.f_hi) * self.f_w


# ---------------------------------------------------------------------------------------------- within the period (6.3)
def solve_age(R, y, W_fine, g: Grids, cost=0.0):
    """One cohort's saving problem on the grid. R: (B,); y: (B, N_E) income; W_fine: (B, N_E, N_f) continuation
    of saving. Returns the value V (B, N_E, N_k) and the fine index of the chosen saving."""
    cash = R[:, None, None] * g.k[None, None, :] + y[:, :, None]                      # (B, N_E, N_k)
    c = cash[..., None] - g.kf[None, None, None, :]                                   # (B, N_E, N_k, N_f)
    feas = c > 1e-9
    val = torch.where(feas, utility(c.clamp(min=1e-9)) - cost + BETA * W_fine[:, :, None, :], -1e12)
    return val.max(-1)


def income(w, j, retired: bool):
    y = torch.full((w.shape[0], N_E), Y_HOME, dtype=DT)
    if not retired:
        y = w[:, None] * N_PROFILE[j] * E_VALS[None, :]
    return y


def age_opt(R, w, W6_fine, g: Grids):
    """The optional age: both conditional problems, the logit value, the work probability."""
    Vw, iw = solve_age(R, income(w, AGE_OPT, False), W6_fine, g, cost=WORK_COST)
    Vr, ir = solve_age(R, income(w, AGE_OPT, True), W6_fine, g)
    s = LOGIT_SCALE
    V = s * torch.logsumexp(torch.stack([Vw, Vr], -1) / s, -1)
    p_work = torch.sigmoid((Vw - Vr) / s)
    return V, iw, ir, p_work


def labour_parts(Gamma, g: Grids):
    """Fixed labour of ages 1-6 and the optional cohort's maximum, from the masses. Gamma: (B, J, N_E, N_k)."""
    e_mass = Gamma.sum(-1)                                                            # (B, J, N_E)
    L_fixed = sum(N_PROFILE[j] * (e_mass[:, j] * E_VALS[None, :]).sum(-1) for j in AGES_WORK)
    L6_e = N_PROFILE[AGE_OPT] * e_mass[:, AGE_OPT] * E_VALS[None, :]                  # (B, N_E)
    return L_fixed, L6_e


def capital(Gamma, g: Grids):
    return (Gamma.sum((1, 2)) * g.k[None, :]).sum(-1)


def clear_labour(z, Gamma, W6_fine, lam0, g: Grids, n_newton: int):
    """Newton on ED(L) = L - L^hh(L); only the optional cohort's problem depends on L through its work
    probability, so only that problem is solved inside the loop. ED' by autograd. Returns (L, lambda, |ED|)."""
    K = capital(Gamma, g)
    L_fixed, L6_e = labour_parts(Gamma, g)
    L6_max = L6_e.sum(-1)
    L = (L_fixed + lam0 * L6_max).detach()
    for _ in range(n_newton):
        L = L.detach().requires_grad_(True)
        _, w, R = prices(K, L, z)
        _, _, _, p_work = age_opt(R, w, W6_fine, g)
        lam = (p_work * (Gamma[:, AGE_OPT] * E_VALS[None, :, None] * N_PROFILE[AGE_OPT])).sum((1, 2)) / L6_max
        ED = L - (L_fixed + lam * L6_max)
        dED, = torch.autograd.grad(ED.sum(), L)
        with torch.no_grad():
            L = L - ED / dED
    with torch.no_grad():
        _, w, R = prices(K, L, z)
        _, _, _, p_work = age_opt(R, w, W6_fine, g)
        lam = (p_work * (Gamma[:, AGE_OPT] * E_VALS[None, :, None] * N_PROFILE[AGE_OPT])).sum((1, 2)) / L6_max
        ED = L - (L_fixed + lam * L6_max)
    return L.detach(), lam, ED.abs()


@torch.no_grad()
def solve_period(z, Gamma, W_hat, lam0, g: Grids, n_newton: int):
    """Given (z, Gamma) and the continuation values W_hat (B, J-1, N_E, N_k): clear labour, then every cohort.
    Returns a dict with prices, values V (B, J, N_E, N_k), the saving choices (fine indices), p_work, lambda."""
    B = Gamma.shape[0]
    W_fine = torch.cat([g.to_fine(W_hat), g.bequest_f.expand(B, 1, N_E, -1)], 1)      # (B, J, N_E, N_f)
    with torch.enable_grad():
        L, lam, ED = clear_labour(z, Gamma, W_fine[:, AGE_OPT], lam0, g, n_newton)
    K = capital(Gamma, g)
    r, w, R = prices(K, L, z)
    V = torch.empty(B, J, N_E, g.k.numel(), dtype=DT)
    idx = torch.empty(B, J, N_E, g.k.numel(), dtype=torch.long)
    for j in range(J):
        if j == AGE_OPT:
            V[:, j], idx[:, j], idx_ret6, p_work = age_opt(R, w, W_fine[:, j], g)
        else:
            V[:, j], idx[:, j] = solve_age(R, income(w, j, j in AGES_RET), W_fine[:, j], g)
    return dict(K=K, L=L, r=r, w=w, R=R, V=V, idx=idx, idx_ret6=idx_ret6, p_work=p_work, lam=lam, ED=ED)


@torch.no_grad()
def move(Gamma, sol, g: Grids):
    """One period of the distribution: Young's lottery on the chosen savings, Pi_e, ageing; the optional cohort
    split by p_work; the dying cohort's bequests become the newborns'. Gamma: (B, J, N_E, N_k)."""
    B, _, _, Nk = Gamma.shape
    new = torch.zeros_like(Gamma)
    for j in range(J):
        mass = Gamma[:, j]
        parts = [(mass, sol["idx"][:, j])]
        if j == AGE_OPT:
            parts = [(sol["p_work"] * mass, sol["idx"][:, j]), ((1.0 - sol["p_work"]) * mass, sol["idx_ret6"])]
        dest = 0 if j == J - 1 else j + 1
        for m_, ix in parts:
            lo, hi, wgt = g.f_lo[ix], g.f_hi[ix], g.f_w[ix]
            moved = torch.zeros(B, N_E, Nk, dtype=DT)
            moved.scatter_add_(2, lo, m_ * (1.0 - wgt))
            moved.scatter_add_(2, hi, m_ * wgt)
            new[:, dest] += torch.einsum("ef,bek->bfk", PI_E, moved)
    return new


# ---------------------------------------------------------------------------------------------- the operator network
class Operator(nn.Module):
    """History of regimes -> W(j, e, k') for ages 1-9 (first value + softplus increments) and lambda_hat."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        n_in, h = N_Z * cfg.T_hist, cfg.hidden
        self.n_first = (J - 1) * N_E
        self.n_inc = (J - 1) * N_E * (cfg.N_k - 1)
        self.net = nn.Sequential(nn.Linear(n_in, h), nn.GELU(), nn.Linear(h, h), nn.GELU(),
                                 nn.Linear(h, self.n_first + self.n_inc + 1))
        self.to(DT)

    def forward(self, hist):
        B = hist.shape[0]
        x = F.one_hot(hist, N_Z).to(DT).view(B, -1)
        y = self.net(x)
        W0 = y[:, :self.n_first].view(B, J - 1, N_E, 1)
        inc = F.softplus(y[:, self.n_first:self.n_first + self.n_inc]).view(B, J - 1, N_E, self.cfg.N_k - 1)
        W = torch.cat([W0, W0 + torch.cumsum(inc, -1)], -1)
        return W, torch.sigmoid(y[:, -1])


def shift(hist, z_new):
    return torch.cat([z_new[:, None], hist[:, :-1]], 1)


# ---------------------------------------------------------------------------------------------- Stage 1: the steady state
@torch.no_grad()
def life_cycle(R, w, g: Grids):
    """Backward induction without aggregate risk at prices (R, w), batch of one. Returns W_ss (J-1, N_E, N_k) and a
    solution dict in the batched layout."""
    B = 1
    R, w = torch.tensor([R], dtype=DT), torch.tensor([w], dtype=DT)
    V = torch.empty(B, J, N_E, g.k.numel(), dtype=DT)
    idx = torch.empty(B, J, N_E, g.k.numel(), dtype=torch.long)
    W = torch.empty(B, J - 1, N_E, g.k.numel(), dtype=DT)
    W_next_fine = g.bequest_f.expand(B, N_E, -1)
    idx_ret6 = p_work = None
    for j in range(J - 1, -1, -1):
        if j == AGE_OPT:
            V[:, j], idx[:, j], idx_ret6, p_work = age_opt(R, w, W_next_fine, g)
        else:
            V[:, j], idx[:, j] = solve_age(R, income(w, j, j in AGES_RET), W_next_fine, g)
        if j > 0:
            W[:, j - 1] = torch.einsum("ef,bfk->bek", PI_E, V[:, j])
            W_next_fine = g.to_fine(W[:, j - 1])
    return W, dict(V=V, idx=idx, idx_ret6=idx_ret6, p_work=p_work)


@torch.no_grad()
def stationary_distribution(sol, g: Grids, tol=1e-12, max_iter=500):
    """Iterate the generational map until the newborns' distribution settles. Each cohort has mass 1/J."""
    pi_e = stationary(PI_E)
    Gamma = torch.zeros(1, J, N_E, g.k.numel(), dtype=DT)
    Gamma[0, :, :, 0] = pi_e[None, :] / J
    for it in range(max_iter):
        new = move(Gamma, sol, g)
        gap = float((new - Gamma).abs().max())
        Gamma = new
        if gap < tol:
            break
    return Gamma, it + 1, gap


@torch.no_grad()
def steady_state(cfg: Config, g: Grids | None = None, verbose=False):
    """Bisection on K (sign change asserted); inside, L by a fixed point on the optional cohort's choice."""
    g = Grids(cfg) if g is None else g
    z0 = torch.tensor([0], dtype=torch.long)

    def excess(K, L_guess, iters=40):
        L = L_guess
        for _ in range(iters):
            _, w, R = prices(torch.tensor([K], dtype=DT), torch.tensor([L], dtype=DT), z0)
            W, sol = life_cycle(float(R[0]), float(w[0]), g)
            Gamma, _, _ = stationary_distribution(sol, g)
            L_fixed, L6_e = labour_parts(Gamma, g)
            lam = float((sol["p_work"][0] * Gamma[0, AGE_OPT] * E_VALS[None, :, None] * N_PROFILE[AGE_OPT]).sum() / L6_e.sum())
            L_new = float(L_fixed[0] + lam * L6_e.sum())
            if abs(L_new - L) < 1e-10:
                L = L_new
                break
            L = 0.5 * L + 0.5 * L_new
        Ks = float(capital(Gamma, g)[0])
        return Ks - K, dict(K=K, L=L, lam=lam, Gamma=Gamma, W=W, sol=sol, R=float(R[0]), w=float(w[0]), K_supply=Ks)

    lo, hi = cfg.K_bracket
    L = 0.6
    f_lo, out_lo = excess(lo, L)
    f_hi, out_hi = excess(hi, L)
    assert f_lo * f_hi < 0, f"no sign change on the K bracket: f({lo}) = {f_lo:.3e}, f({hi}) = {f_hi:.3e}"
    out = out_lo
    for it in range(60):
        mid = 0.5 * (lo + hi)
        f_mid, out = excess(mid, out["L"])
        if f_mid * f_lo < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
        if hi - lo < 1e-9:
            break
    out["residual"] = abs(out["K_supply"] - out["K"])
    out["bisections"] = it + 1
    if verbose:
        print(f"steady state: K {out['K']:.5f} (supply {out['K_supply']:.5f}), L {out['L']:.5f}, lambda {out['lam']:.4f}, "
              f"R {out['R']:.4f}, w {out['w']:.4f}; {out['bisections']} bisections")
    return out


# ---------------------------------------------------------------------------------------------- targets and the loss
@torch.no_grad()
def targets(z, Gamma_next, hist, W_next_fn, g: Grids, n_newton: int):
    """W_tar(j, e, k') = sum_z' Pi_z(z, z') sum_e' Pi_e(e, e') V_{t+1}(j+1, e', k' | z'), t+1 solved for each z'
    with the continuation values W_next_fn(history shifted by z')."""
    B = Gamma_next.shape[0]
    V_next = []
    ED_max = 0.0
    for zp in range(N_Z):
        z_new = torch.full((B,), zp, dtype=torch.long)
        W_n, lam_n = W_next_fn(shift(hist, z_new))
        sol = solve_period(z_new, Gamma_next, W_n, lam_n, g, n_newton)
        V_next.append(sol["V"])
        ED_max = max(ED_max, float(sol["ED"].max()))
    V_next = torch.stack(V_next, 1)                                                   # (B, N_Z, J, N_E, N_k)
    EV = torch.einsum("ef,bzjfk->bzjek", PI_E, V_next[:, :, 1:])                       # (B, N_Z, J-1, N_E, N_k)
    W_tar = torch.einsum("bz,bzjek->bjek", PI_Z[z], EV)
    return W_tar, ED_max


def loss_fn(W_hat, lam_hat, W_tar, lam, eps):
    e_lam = (lam - lam_hat) / (eps + lam)
    e_first = (W_tar[..., 0] - W_hat[..., 0]) / (eps + W_tar[..., 0].abs())
    dW_tar = W_tar[..., 1:] - W_tar[..., :-1]
    dW_hat = W_hat[..., 1:] - W_hat[..., :-1]
    e_inc = (dW_tar - dW_hat) / (eps + dW_tar.abs())
    return (e_lam ** 2).mean() + (e_first ** 2).mean() + (e_inc ** 2).mean()


# ---------------------------------------------------------------------------------------------- training
class Trainer:
    def __init__(self, cfg: Config = LIVE):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.gen = torch.Generator().manual_seed(cfg.seed + 1)
        self.g = Grids(cfg)
        self.net = Operator(cfg)
        self.target = copy.deepcopy(self.net).requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.log = []                              # (stage, update, loss, eps, K mean, lambda mean, ED max)
        self.ss = None
        self.stage_times = {}

    def ema(self):
        with torch.no_grad():
            for p_t, p in zip(self.target.parameters(), self.net.parameters()):
                p_t.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p)

    def prefit(self):
        """The lab's warm start: regress the network on the steady state's W and lambda at random histories, so
        that Stage 2 starts from continuation values of the right size and shape."""
        cfg = self.cfg
        W_ss = self.ss["W"].expand(cfg.E, -1, -1, -1)
        lam_ss = torch.full((cfg.E,), float(self.ss["lam"]), dtype=DT)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr_prefit)
        loss = float("nan")
        for _ in range(cfg.prefit_steps):
            hist = torch.randint(0, N_Z, (cfg.E, cfg.T_hist), generator=self.gen)
            W_hat, lam_hat = self.net(hist)
            l = ((W_hat - W_ss) ** 2).mean() + ((lam_hat - lam_ss) ** 2).mean()
            opt.zero_grad(); l.backward(); opt.step()
            loss = l.item()
        self.target.load_state_dict(self.net.state_dict())
        return loss

    def seed_pairs(self):
        """E pairs at the steady state, z drawn from the regimes' stationary distribution, all-normal histories."""
        cfg, ss = self.cfg, self.ss
        pz = stationary(PI_Z)
        z = torch.multinomial(pz.expand(cfg.E, -1), 1, generator=self.gen).squeeze(-1)
        Gamma = ss["Gamma"].expand(cfg.E, -1, -1, -1).clone()
        hist = torch.zeros(cfg.E, cfg.T_hist, dtype=torch.long)
        hist[:, 0] = z
        return z, Gamma, hist

    def update(self, z, Gamma, hist, stage: int, eps: float):
        """One update on all E pairs, then move them one period. Returns the new (z, Gamma, hist)."""
        cfg, g = self.cfg, self.g
        W_hat, lam_hat = self.net(hist)
        if stage == 2:                                               # continuation held at the steady state's
            W_ss = self.ss["W"].expand(cfg.E, -1, -1, -1)
            sol = solve_period(z, Gamma, W_ss, lam_hat.detach(), g, cfg.n_newton)
            W_next_fn = lambda h: (W_ss, torch.full((cfg.E,), float(self.ss["lam"]), dtype=DT))
        else:
            sol = solve_period(z, Gamma, W_hat.detach(), lam_hat.detach(), g, cfg.n_newton)
            W_next_fn = self.target
        Gamma_next = move(Gamma, sol, g)
        W_tar, ED_next = targets(z, Gamma_next, hist, W_next_fn, g, cfg.n_newton)
        loss = loss_fn(W_hat, lam_hat, W_tar, sol["lam"], eps)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        if stage == 3:
            self.ema()
        self.log.append((stage, len(self.log), loss.item(), eps, float(sol["K"].mean()), float(sol["lam"].mean()),
                         max(float(sol["ED"].max()), ED_next)))
        z_new = torch.multinomial(PI_Z[z], 1, generator=self.gen).squeeze(-1)
        return z_new, Gamma_next, shift(hist, z_new)

    def run(self, verbose=True):
        cfg = self.cfg
        t0 = time.time()
        self.ss = steady_state(cfg, self.g, verbose=verbose)
        self.stage_times["steady_state"] = time.time() - t0
        tp = time.time()
        self.prefit_loss = self.prefit()
        self.stage_times["prefit"] = time.time() - tp
        if verbose:
            print(f"prefit to the steady state's W: mse {self.prefit_loss:.2e} after {cfg.prefit_steps} steps")
        z, Gamma, hist = self.seed_pairs()
        t1 = time.time()
        for u in range(cfg.U2):
            z, Gamma, hist = self.update(z, Gamma, hist, 2, cfg.eps_schedule[0])
            if verbose and (u % cfg.log_every == 0 or u == cfg.U2 - 1):
                self._print("stage 2", u, cfg.U2, t0)
        self.target.load_state_dict(self.net.state_dict())
        self.stage_times["stage2"] = time.time() - t1
        t2 = time.time()
        for u in range(cfg.U3):
            eps = cfg.eps_schedule[min(len(cfg.eps_schedule) - 1, (u * len(cfg.eps_schedule)) // cfg.U3)]
            z, Gamma, hist = self.update(z, Gamma, hist, 3, eps)
            if verbose and (u % cfg.log_every == 0 or u == cfg.U3 - 1):
                self._print("stage 3", u, cfg.U3, t0)
        self.stage_times["stage3"] = time.time() - t2
        self.pairs = (z, Gamma, hist)
        self.seconds = time.time() - t0
        return self

    def _print(self, stage, u, U, t0):
        s, n, loss, eps, K, lam, ED = self.log[-1]
        print(f"{stage} update {u:4d}/{U} | loss {loss:.3e} (eps {eps:g}) | K {K:.4f} | lambda {lam:.3f} "
              f"| max |ED| {ED:.1e} | {time.time() - t0:5.0f} s")


# ---------------------------------------------------------------------------------------------- diagnostics
@torch.no_grad()
def simulate(tr: Trainer, T=200, seed=7, pairs=None):
    """Run the trained pairs T more periods under the online network. Returns K, lambda, z paths (E, T)."""
    cfg, g = tr.cfg, tr.g
    gen = torch.Generator().manual_seed(seed)
    z, Gamma, hist = (tr.pairs if pairs is None else pairs)
    z, Gamma, hist = z.clone(), Gamma.clone(), hist.clone()
    K = torch.empty(cfg.E, T, dtype=DT); lam = torch.empty(cfg.E, T, dtype=DT); zs = torch.empty(cfg.E, T, dtype=torch.long)
    ED_max = 0.0
    for t in range(T):
        W_hat, lam_hat = tr.net(hist)
        sol = solve_period(z, Gamma, W_hat, lam_hat, g, cfg.n_newton)
        K[:, t], lam[:, t], zs[:, t] = sol["K"], sol["lam"], z
        ED_max = max(ED_max, float(sol["ED"].max()))
        Gamma = move(Gamma, sol, g)
        z = torch.multinomial(PI_Z[z], 1, generator=gen).squeeze(-1)
        hist = shift(hist, z)
    return dict(K=K.numpy(), lam=lam.numpy(), z=zs.numpy(), ED_max=ED_max, state=(z, Gamma, hist))


@torch.no_grad()
def consistency(tr: Trainer, pairs=None):
    """On the (fresh) pairs: the online network's W against targets built with the online network itself (a
    Bellman-type check), as relative errors on the first values and the increments, and lambda_hat against lambda."""
    cfg, g = tr.cfg, tr.g
    z, Gamma, hist = (tr.pairs if pairs is None else pairs)
    W_hat, lam_hat = tr.net(hist)
    sol = solve_period(z, Gamma, W_hat, lam_hat, g, cfg.n_newton)
    Gamma_next = move(Gamma, sol, g)
    W_tar, ED_next = targets(z, Gamma_next, hist, tr.net, g, cfg.n_newton)
    eps = 1e-3
    e_first = ((W_tar[..., 0] - W_hat[..., 0]) / (eps + W_tar[..., 0].abs())).abs()
    dW_tar = W_tar[..., 1:] - W_tar[..., :-1]; dW_hat = W_hat[..., 1:] - W_hat[..., :-1]
    e_inc = ((dW_tar - dW_hat) / (eps + dW_tar.abs())).abs()
    e_abs = (W_tar - W_hat).abs()
    return dict(rel_first_mean=float(e_first.mean()), rel_inc_mean=float(e_inc.mean()), rel_inc_p90=float(torch.quantile(e_inc, 0.9)),
                abs_W_mean=float(e_abs.mean()), abs_W_max=float(e_abs.max()), lam_abs_err=float((lam_hat - sol["lam"]).abs().mean()),
                ED_max=max(float(sol["ED"].max()), ED_next), lam_mean=float(sol["lam"].mean()))


def loss_drop(log, stage=3) -> dict:
    L = np.array([row[2] for row in log if row[0] == stage])
    n = max(1, len(L) // 10)
    return dict(first_tenth=float(L[:n].mean()), last_tenth=float(L[-n:].mean()), ratio=float(L[-n:].mean() / L[:n].mean()))


@torch.no_grad()
def policy_pictures(tr: Trainer):
    """p_work(e, k) at the optional age and consumption at the age before, under an all-normal history and under
    a history with a disaster today, at the steady-state distribution."""
    cfg, g, ss = tr.cfg, tr.g, tr.ss
    out = {}
    for name, z0 in (("normal", 0), ("disaster", 2)):
        hist = torch.zeros(1, cfg.T_hist, dtype=torch.long); hist[0, 0] = z0
        z = torch.tensor([z0]); Gamma = ss["Gamma"].clone()
        W_hat, lam_hat = tr.net(hist)
        sol = solve_period(z, Gamma, W_hat, lam_hat, g, cfg.n_newton)
        j = AGE_OPT - 1
        cash = sol["R"][0] * g.k[None, :] + income(sol["w"], j, False)[0][:, None]
        c = cash - g.kf[sol["idx"][0, j]]
        out[name] = dict(p_work=sol["p_work"][0].numpy(), c_before=c.numpy(), K=float(sol["K"][0]), L=float(sol["L"][0]))
    return dict(k=g.k.numpy(), **out)


def summary(tr: Trainer) -> dict:
    sim = simulate(tr, T=200, seed=7)
    cons = consistency(tr, pairs=sim["state"])
    drop = loss_drop(tr.log)
    K = sim["K"][:, 50:]
    return dict(seed=tr.cfg.seed, seconds=round(tr.seconds, 1), stage_times={k: round(v, 1) for k, v in tr.stage_times.items()},
                prefit_loss=tr.prefit_loss,
                ss_K=tr.ss["K"], ss_L=tr.ss["L"], ss_lambda=tr.ss["lam"], ss_residual=tr.ss["residual"],
                ss_mass_error=float(abs(tr.ss["Gamma"].sum() - 1.0)),
                K_mean=float(K.mean()), K_min=float(K.min()), K_max=float(K.max()), lam_sim_mean=float(sim["lam"][:, 50:].mean()),
                disaster_share=float((sim["z"] == 2).mean()), ED_max_sim=sim["ED_max"], **cons, loss_drop=drop)


def save(path, tr: Trainer):
    torch.save(dict(state_dict=tr.net.state_dict(), target=tr.target.state_dict(), cfg=asdict(tr.cfg), log=tr.log,
                    pairs=tr.pairs, stage_times=tr.stage_times, seconds=tr.seconds, prefit_loss=tr.prefit_loss,
                    ss={k: v for k, v in tr.ss.items() if k != "sol"}), path)


def load(path) -> Trainer:
    rec = torch.load(path, weights_only=False)
    cfg = Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in rec["cfg"].items()})
    tr = Trainer(cfg)
    tr.net.load_state_dict(rec["state_dict"]); tr.target.load_state_dict(rec["target"])
    tr.log, tr.pairs, tr.stage_times, tr.seconds = rec["log"], rec["pairs"], rec["stage_times"], rec["seconds"]
    tr.prefit_loss = rec["prefit_loss"]
    tr.ss = rec["ss"]
    tr.ss["sol"] = life_cycle(tr.ss["R"], tr.ss["w"], tr.g)[1]
    return tr


if __name__ == "__main__":
    import sys
    cfg = SMOKE if "smoke" in sys.argv else LIVE
    tr = Trainer(cfg).run(verbose=True)
    print(json.dumps(summary(tr), indent=1, default=float))
