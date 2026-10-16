"""Session 4 lab, block D -- Krusell and Smith (1998) by Deep Equilibrium Nets, in PyTorch.

Ported from the course notebook `Lab11A_KS1998_Pytorch.ipynb` (itself a PyTorch port of Jan Zemlicka's JAX
code for the Deep Equilibrium Net method of Azinovic, Gaegauf and Scheidegger, IER 2022). Same calibration,
same network, same "cloud" training loop: a batch of economies, each carrying its whole histogram, moves one
period per gradient step, and the loss is the Euler equation of households drawn from those economies.

Deviations from the source (each is said in the notebook as well):
  1. No `sequence_jacobian`. The steady state is solved here: Rouwenhorst + EGM + Young + bisection on K,
     with a sign-change assertion (session04_ha.py). Grids follow SSJ's conventions exactly (agrid,
     markov_rouwenhorst, markov_tauchen), re-implemented in numpy.
  2. amax = 200, not 50. At 50 the top node binds: 14.7% of the steady-state mass sits on it and K is
     27.96 instead of 33.25. (SSJ 1.0 itself, run on the source's code, fails to converge from K = 30; from
     K = 28 it returns K = 32.66 with an invalid histogram, top-node mass 1.76. At amax = 200, SSJ gives
     K = 33.2458, as here.)
  3. TFP levels. SSJ's `markov_tauchen` already returns levels with E[Z] = 1; the source applied `np.exp`
     to them a second time, so its Z was about e = 2.72. Here Z has mean one.
  4. The source defines a Fischer--Burmeister function but never calls it: its loss is (1 - RHS/LHS)^2,
     which forces the Euler equation to hold with equality even where the borrowing limit binds. Here the
     loss is FB(Euler gap, saving slack)^2, which is zero exactly when the KKT conditions hold.
  5. Saving above the top grid node is clamped before the lottery; the source's weights left [0, 1] there and
     could create negative mass. (Capping saving inside the policy instead was tried and rejected: it
     creates an upper constraint the FB residual does not know about, and the loss blew up at the top node.)
  6. The transport is one vectorised scatter_add (the source looped over e and e'); float64 throughout;
     the network sees K/K_ss and log Z / sd instead of raw K and Z (scale only).
  7. Warm start (TrainCfg.init = "egm"): before the cloud loop the network is fitted to the steady-state
     EGM policy (relative consumption error), it takes TrainCfg.steps_per_period gradient steps on each
     cloud state before the cloud moves (the source: one; Lab12 and the DEQN paper train several epochs on
     each simulated cloud), and Adam runs at lr 3e-4 rather than 1e-3. init = "random" is the source's start:
     from it the cloud collapses before the network learns anything (see pilot_deqn.json, "random_start").
  8. Tomorrow's consumption c' is read from a frozen copy of the network, re-synced every 20 gradient steps
     (TrainCfg.target_sync; the source differentiates through c' as well). Without it the optimiser lowers
     the Euler gap by moving c' and today's c together, and the cloud leaves the steady state: from the warm
     start K fell to about 15 (no copy) or rose to the top of the grid (other settings), although the loss
     evaluated at the exact EGM policy is 5e-7. With the copy the cloud stays at K_ss.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

import session04_ha as ha

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)
DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------------------
# Calibration (Lab11A) and SSJ-convention grids
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class KSCal:
    eis: float = 0.5
    delta: float = 0.025
    alpha: float = 0.36
    rho_e: float = 0.966
    sd_e: float = 0.5           # cross-sectional s.d. of log e (SSJ's sigma)
    nE: int = 3
    nA: int = 100
    amin: float = 0.0
    amax: float = 200.0         # the source has 50, which binds: see deviation 2
    beta: float = 0.98
    L: float = 1.0
    rho_tfp: float = 0.9
    sd_tfp: float = 0.007       # unconditional s.d. of log Z (SSJ's sigma)
    n_tfp: int = 5

    @property
    def gamma(self) -> float:
        return 1.0 / self.eis


CAL = KSCal()


def agrid(amax, n, amin=0.0):
    """SSJ's asset grid: equidistant in log(a + pivot)."""
    pivot = abs(amin) + 0.25
    g = np.geomspace(amin + pivot, amax + pivot, n) - pivot
    g[0] = amin
    return g


def _stationary(Pi, tol=1e-14, maxit=100000):
    pi = np.ones(Pi.shape[0]) / Pi.shape[0]
    for _ in range(maxit):
        new = pi @ Pi
        if np.max(np.abs(new - pi)) < tol:
            return new
        pi = new
    return pi


def markov_tauchen(rho, sigma, N=7, m=3):
    """SSJ's Tauchen: returns LEVELS y proportional to exp(s), E[y] = 1, sd(s) = sigma."""
    s = np.linspace(-m, m, N); ds = s[1] - s[0]; sd_innov = np.sqrt(1 - rho ** 2)
    Pi = np.empty((N, N))
    Pi[:, 0] = norm.cdf(s[0] - rho * s + ds / 2, scale=sd_innov)
    Pi[:, -1] = 1 - norm.cdf(s[-1] - rho * s - ds / 2, scale=sd_innov)
    for j in range(1, N - 1):
        Pi[:, j] = (norm.cdf(s[j] - rho * s + ds / 2, scale=sd_innov)
                    - norm.cdf(s[j] - rho * s - ds / 2, scale=sd_innov))
    pi = _stationary(Pi)
    s = s * sigma / np.sqrt(pi @ s ** 2 - (pi @ s) ** 2)
    y = np.exp(s) / np.sum(pi * np.exp(s))
    return y, pi, Pi


def grids(cal: KSCal = CAL) -> dict:
    e, Pi_e, pi_e = ha.rouwenhorst(cal.nE, cal.rho_e, cal.sd_e)   # same as SSJ markov_rouwenhorst
    z, pi_z, Pi_z = markov_tauchen(cal.rho_tfp, cal.sd_tfp, cal.n_tfp)
    return dict(a=agrid(cal.amax, cal.nA, cal.amin), e=e, Pi_e=Pi_e, pi_e=pi_e, z=z, Pi_z=Pi_z, pi_z=pi_z)


# --------------------------------------------------------------------------------------
# Steady state without aggregate risk (replaces SSJ's solve_steady_state)
# --------------------------------------------------------------------------------------
def prices_np(K, Z, cal: KSCal = CAL):
    r = cal.alpha * Z * (K / cal.L) ** (cal.alpha - 1) - cal.delta
    w = (1 - cal.alpha) * Z * (K / cal.L) ** cal.alpha
    return r, w


def steady_state(cal: KSCal = CAL, K_lo=15.0, K_hi=80.0, tol=1e-9, verbose=False) -> dict:
    """Bisection on K: excess supply A(K) - K must change sign on [K_lo, K_hi]."""
    t0 = time.time()
    g = grids(cal)
    p = replace(ha.Params(), alpha=cal.alpha, beta=cal.beta, delta=cal.delta, sigma=cal.gamma,
                n_e=cal.nE, a_min=cal.amin, a_max=cal.amax, n_a=cal.nA)
    cache = {}

    def excess(K):
        r, w = prices_np(K, 1.0, cal)
        ap, c, _, _ = ha.solve_household(r, w, 0.0, g["a"], g["e"], g["Pi_e"], p, c_init=cache.get("c"))
        lam, _, _ = ha.stationary_distribution(ap, g["a"], g["Pi_e"], g["pi_e"], method="solve")
        cache.update(c=c, ap=ap, lam=lam, r=r, w=w)
        return float(lam.sum(1) @ g["a"]) - K

    f_lo, f_hi = excess(K_lo), excess(K_hi)
    assert f_lo * f_hi < 0, f"no sign change on [{K_lo}, {K_hi}]: {f_lo:.3g}, {f_hi:.3g}"
    it = 0
    while K_hi - K_lo > tol and it < 200:
        it += 1
        K = 0.5 * (K_lo + K_hi)
        f = excess(K)
        if f * f_lo > 0:
            K_lo, f_lo = K, f
        else:
            K_hi = K
    K = 0.5 * (K_lo + K_hi)
    excess(K)
    p_ss = dict(cal=cal, K=K, r=cache["r"], w=cache["w"], ap=cache["ap"], c=cache["c"], D=cache["lam"],
                bisections=it, seconds=time.time() - t0, **g)
    a_off, res, slack = ha.euler_residuals(cache["ap"], cache["c"], cache["r"], cache["w"], 0.0,
                                           g["a"], g["e"], g["Pi_e"], p)
    p_ss["egm_euler_log10_max"] = float(np.nanmax(np.where(slack, res, np.nan)))
    if verbose:
        print(f"steady state K = {K:.4f}, r = {p_ss['r']:.5f}, {it} bisections, {p_ss['seconds']:.1f} s")
    return p_ss


# --------------------------------------------------------------------------------------
# The network and the model's "physics" (as in Lab11A)
# --------------------------------------------------------------------------------------
class KS_Network(nn.Module):
    """Input (a, e, Z, K), output the logit of the saving rate. Three Mish layers, Xavier init."""

    def __init__(self, n_input=4, n_hidden=64, n_out=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_input, n_hidden), nn.Mish(),
                                 nn.Linear(n_hidden, n_hidden), nn.Mish(),
                                 nn.Linear(n_hidden, n_hidden), nn.Mish(),
                                 nn.Linear(n_hidden, n_out))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class KrusellSmithModel:
    def __init__(self, ss: dict, cal: KSCal = CAL):
        self.cal = cal
        t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64, device=DEVICE)
        self.a_grid, self.e_grid, self.Pi_e = t(ss["a"]), t(ss["e"]), t(ss["Pi_e"])
        self.tfp_grid, self.Pi_tfp = t(ss["z"]), t(ss["Pi_z"])
        self.D_ss, self.K_ss = t(ss["D"]), float(ss["K"])
        self.nA, self.nE, self.nTFP = len(ss["a"]), len(ss["e"]), len(ss["z"])

    # prices: analytic from (K, Z) -- there is no price network in this method's KS application
    def get_prices(self, K, Z):
        c = self.cal
        r = c.alpha * Z * (K / c.L) ** (c.alpha - 1) - c.delta
        w = (1 - c.alpha) * Z * (K / c.L) ** c.alpha
        return r, w

    def u_prime(self, c):
        return torch.clamp(c, min=1e-10) ** (-self.cal.gamma)

    @staticmethod
    def fischer_burmeister(a, b):
        return a + b - torch.sqrt(a ** 2 + b ** 2 + 1e-14)

    def features(self, a, e, Z, K):
        c = self.cal
        a, e, Z, K = torch.broadcast_tensors(a, torch.as_tensor(e), torch.as_tensor(Z), torch.as_tensor(K))
        piv = abs(c.amin) + 0.25                                          # the grid's own log spacing
        xa = torch.log((a - c.amin + piv) / piv) / np.log((c.amax - c.amin + piv) / piv)
        return torch.stack([xa, e, torch.log(Z) / c.sd_tfp / 3.0,
                            K / self.K_ss - 1.0], dim=-1)

    def policy(self, net, a, e, Z, K):
        """Saving and consumption at continuous (a, e) given aggregate (Z, K). All tensors broadcast."""
        r, w = self.get_prices(K, Z)
        coh = (1 + r) * a + w * e
        s = torch.sigmoid(net(self.features(a, e, Z, K)).squeeze(-1))
        sav = self.cal.amin + s * (coh - self.cal.amin)
        return sav, coh - sav, coh

    def transport(self, sav, D):
        """Young's lottery then e -> e'. sav, D: (B, nA, nE). One scatter_add, no loops."""
        sav = torch.clamp(sav, self.a_grid[0], self.a_grid[-1])
        hi = torch.clamp(torch.searchsorted(self.a_grid, sav.contiguous(), right=True), 1, self.nA - 1)
        lo = hi - 1
        w_hi = (sav - self.a_grid[lo]) / (self.a_grid[hi] - self.a_grid[lo])
        moved = torch.zeros_like(D)
        moved.scatter_add_(1, lo, D * (1 - w_hi))
        moved.scatter_add_(1, hi, D * w_hi)
        return moved @ self.Pi_e                                        # (B, nA, nE') ; rows of Pi_e sum to 1

    def aggregate_K(self, D):
        return (D.sum(-1) * self.a_grid).sum(-1)

    def step_cloud(self, net, D, z_idx):
        """Next histogram of every economy in the cloud under the current policy (no gradient)."""
        with torch.no_grad():
            B = D.shape[0]
            K = self.aggregate_K(D)
            Z = self.tfp_grid[z_idx]
            sav, _, _ = self.policy(net, self.a_grid.view(1, -1, 1).expand(B, self.nA, self.nE),
                                    self.e_grid.view(1, 1, -1).expand(B, self.nA, self.nE),
                                    Z.view(B, 1, 1), K.view(B, 1, 1))
            return self.transport(sav, D)

    def residuals(self, net, a, e_idx, Z, K, z_idx, K_next, net_next=None):
        """Euler gap, slack and FB residual for households at (a, e_idx) in economies (Z, K, z_idx).

        All of shape (B, M). Expectation over Z' (Pi_tfp row) and e' (Pi_e row), exactly as the source.
        """
        e = self.e_grid[e_idx]
        sav, con, coh = self.policy(net, a, e, Z, K)
        mu = self.u_prime(con)
        a_p = sav.detach()                                               # as in the source
        rhs = torch.zeros_like(mu)
        for jz in range(self.nTFP):
            Zn = self.tfp_grid[jz]
            r_n, _ = self.get_prices(K_next, Zn)
            Ev = torch.zeros_like(mu)
            for je in range(self.nE):
                _, c_n, _ = self.policy(net if net_next is None else net_next, a_p, self.e_grid[je], Zn, K_next)
                Ev = Ev + self.Pi_e[e_idx, je] * (1 + r_n) * self.u_prime(c_n)
            rhs = rhs + self.Pi_tfp[z_idx, jz] * Ev
        gap = 1.0 - self.cal.beta * rhs / mu                             # >= 0 when the limit binds
        slack = (sav - self.cal.amin) / coh                              # >= 0 always, = 0 at the limit
        return gap, slack, self.fischer_burmeister(gap, slack), sav, con


def pretrain_to_egm(net, model: KrusellSmithModel, ss: dict, steps: int = 500, lr: float = 3e-3,
                    K_band: float = 0.05):
    """Warm start: least squares on the relative consumption error against the EGM steady-state policy, at
    every TFP node and at K drawn within +-K_band of K_ss (same target: a start, not a solution).
    Returns the final mean squared relative error."""
    c = model.cal
    r, w = prices_np(ss["K"], 1.0, c)
    coh = (1 + r) * ss["a"][:, None] + w * ss["e"][None, :]
    c_egm = torch.as_tensor(coh - ss["ap"])                                # (nA, nE)
    a = model.a_grid.view(1, -1, 1); e = model.e_grid.view(1, 1, -1)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    mse = float("nan")
    for _ in range(steps):
        Z = model.tfp_grid.view(-1, 1, 1)
        K = model.K_ss * (1 + K_band * (2 * torch.rand(model.nTFP, 1, 1) - 1))
        s_rate = torch.sigmoid(net(model.features(a, e, Z, K)).squeeze(-1))
        c_net = (1 - s_rate) * (torch.as_tensor(coh) - c.amin)
        loss = torch.mean((c_net / c_egm - 1) ** 2)                       # relative consumption error
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        mse = loss.item()
    return mse


# --------------------------------------------------------------------------------------
# Training (the cloud loop of Lab11A)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainCfg:
    n_epochs: int = 2000
    batch_size: int = 256        # economies in the cloud
    n_agents: int = 64           # households drawn per economy
    n_hidden: int = 64
    lr: float = 3e-4             # the source: 1e-3 (from a random start)
    step_size: int = 500         # StepLR, halve every step_size epochs
    lr_gamma: float = 0.5
    log_every: int = 100
    target_sync: int = 20        # tomorrow's policy from a frozen copy, re-synced every 20 steps (source: 0, none)
    steps_per_period: int = 1    # gradient steps on each cloud state before it moves (the source: 1)
    init: str = "egm"            # "egm": fit the steady-state EGM policy first; "random": the source's start
    pretrain_steps: int = 4000


SHORT = TrainCfg(n_epochs=600, batch_size=64, n_agents=32, step_size=200)   # the live run in class
FULL = TrainCfg()                                                             # the source's sizes (2000 x 256 x 64)
RECORD = replace(FULL, batch_size=128)                                        # the recorded run (CPU budget)
RANDOM = replace(SHORT, init="random", lr=1e-3)                               # the source's start and lr


def train(seed: int = 0, n_epochs: int | None = None, ss: dict | None = None, cfg: TrainCfg = FULL,
          verbose: bool = True):
    """Returns (net, model, history). history: dict of per-epoch loss, mean K, plus wall seconds."""
    if ss is None:
        ss = steady_state()
    if n_epochs is not None:
        cfg = replace(cfg, n_epochs=n_epochs)
    torch.manual_seed(seed)
    model = KrusellSmithModel(ss, ss.get("cal", CAL))
    net = KS_Network(n_hidden=cfg.n_hidden).to(DEVICE)
    t_pre = time.time()
    pre_mse = pretrain_to_egm(net, model, ss, cfg.pretrain_steps) if cfg.init == "egm" else None
    t_pre = time.time() - t_pre
    target = None
    if cfg.target_sync > 0:
        import copy
        target = copy.deepcopy(net).requires_grad_(False)
    n_step = 0
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.lr_gamma)
    B, M = cfg.batch_size, cfg.n_agents
    D = model.D_ss.unsqueeze(0).expand(B, -1, -1).clone()
    z_idx = torch.randint(0, model.nTFP, (B,))
    hist = dict(loss=[], K_mean=[], euler_gap_rms=[])
    t0 = time.time()
    for ep in range(cfg.n_epochs):                                       # one epoch = one simulated period
        K = model.aggregate_K(D)
        Z = model.tfp_grid[z_idx]
        for _ in range(cfg.steps_per_period):
            opt.zero_grad()
            D_next = model.step_cloud(net, D, z_idx)                     # K' under the current policy
            K_next = model.aggregate_K(D_next)
            ia = torch.randint(0, model.nA, (B, M)); ie = torch.randint(0, model.nE, (B, M))
            gap, slack, fb, _, _ = model.residuals(net, model.a_grid[ia], ie, Z[:, None], K[:, None],
                                                   z_idx[:, None], K_next[:, None], net_next=target)
            loss = torch.mean(fb ** 2)
            loss.backward(); opt.step()
            n_step += 1
            if target is not None and n_step % cfg.target_sync == 0:
                target.load_state_dict(net.state_dict())
        sched.step()
        hist["loss"].append(loss.item()); hist["K_mean"].append(K.mean().item())
        hist["euler_gap_rms"].append(float(torch.sqrt(torch.mean(gap.detach() ** 2))))
        D_next = model.step_cloud(net, D, z_idx)
        D = D_next
        z_idx = torch.multinomial(model.Pi_tfp[z_idx], 1).squeeze(-1)
        if verbose and (ep % cfg.log_every == 0 or ep == cfg.n_epochs - 1):
            print(f"epoch {ep:5d} | FB^2 {loss.item():.3e} | mean K {K.mean().item():.3f} "
                  f"| {time.time() - t0:5.0f} s")
    hist["seconds"] = time.time() - t0 + t_pre
    hist["pretrain_seconds"], hist["pretrain_mse"] = t_pre, pre_mse
    hist["cfg"] = cfg.__dict__
    hist["seed"] = seed
    hist["D_final"] = D
    hist["z_final"] = z_idx
    return net, model, hist


# --------------------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------------------
def simulate_K_path(net, model: KrusellSmithModel, T: int = 500, seed: int = 0, n_econ: int = 1,
                    z0: int | None = None):
    """Economies started at the steady-state histogram, TFP drawn from its chain. Returns K (n_econ, T+1),
    z index (n_econ, T+1) and the final histograms."""
    g = torch.Generator().manual_seed(seed)
    D = model.D_ss.unsqueeze(0).expand(n_econ, -1, -1).clone()
    z = torch.full((n_econ,), model.nTFP // 2 if z0 is None else z0, dtype=torch.long)
    Ks, zs = [model.aggregate_K(D)], [z.clone()]
    for _ in range(T):
        D = model.step_cloud(net, D, z)
        z = torch.multinomial(model.Pi_tfp[z], 1, generator=g).squeeze(-1)
        Ks.append(model.aggregate_K(D)); zs.append(z.clone())
    return torch.stack(Ks, 1).numpy(), torch.stack(zs, 1).numpy(), D


def euler_errors(net, model: KrusellSmithModel, D=None, z_idx=None, T_burn: int = 200, n_econ: int = 32,
                 seed: int = 1):
    """Errors at every (a, e) node of simulated economies (after T_burn periods from the steady state).

    Returns the |FB| residual, the unit-free Euler error |1 - c_Euler/c| on slack nodes, and both weighted by
    each economy's own histogram (what a household actually faces) and unweighted (the whole grid).
    """
    if D is None:
        _, zs, D = simulate_K_path(net, model, T_burn, seed=seed, n_econ=n_econ)
        z_idx = torch.as_tensor(zs[:, -1])
    B = D.shape[0]
    with torch.no_grad():
        K = model.aggregate_K(D); Z = model.tfp_grid[z_idx]
        K_next = model.aggregate_K(model.step_cloud(net, D, z_idx))
        a = model.a_grid.view(1, -1, 1).expand(B, model.nA, model.nE).reshape(B, -1)
        ie = torch.arange(model.nE).view(1, 1, -1).expand(B, model.nA, model.nE).reshape(B, -1)
        gap, slack, fb, sav, con = model.residuals(net, a, ie, Z[:, None], K[:, None], z_idx[:, None],
                                                   K_next[:, None])
        c_euler = con * (1 - gap) ** (-1.0 / model.cal.gamma)            # (beta E[R'u'(c')])^(-1/gamma)
        ee = torch.abs(1 - c_euler / con)
        is_slack = slack > 1e-6
    W = D.reshape(B, -1)
    fb_abs = fb.abs()
    out = {
        "fb_abs_weighted_mean": float((W * fb_abs).sum() / W.sum()),
        "fb_abs_p50": float(fb_abs.quantile(0.5)), "fb_abs_p90": float(fb_abs.quantile(0.9)),
        "fb_abs_p99": float(fb_abs.quantile(0.99)), "fb_abs_max": float(fb_abs.max()),
        "euler_log10_weighted_mean": float(torch.log10((W * ee * is_slack).sum() / (W * is_slack).sum())),
        "euler_log10_p50": float(torch.log10(ee[is_slack].quantile(0.5))),
        "euler_log10_p90": float(torch.log10(ee[is_slack].quantile(0.9))),
        "share_at_limit": float((W * (~is_slack)).sum() / W.sum()),
        "K_mean": float(K.mean()), "K_min": float(K.min()), "K_max": float(K.max()),
    }
    return out


def policy_gap_vs_egm(net, model: KrusellSmithModel, ss: dict, iz: int | None = None):
    """Network saving at (K_ss, Z) against the EGM steady-state policy at the same prices (Z = middle node,
    close to one). Max and distribution-weighted mean absolute gap in units of assets."""
    iz = model.nTFP // 2 if iz is None else iz
    with torch.no_grad():
        a = model.a_grid.view(-1, 1).expand(model.nA, model.nE)
        e = model.e_grid.view(1, -1).expand(model.nA, model.nE)
        sav, _, _ = model.policy(net, a, e, model.tfp_grid[iz], torch.tensor(model.K_ss))
    gap = np.abs(sav.numpy() - ss["ap"])
    return {"max": float(gap.max()), "weighted_mean": float((gap * ss["D"]).sum()),
            "Z": float(model.tfp_grid[iz])}


def policy_plot_data(net, model: KrusellSmithModel, ss: dict | None = None, iz: int | None = None,
                     K: float | None = None):
    """Saving a'(a, e) from the network at (K, Z), and the EGM steady-state policy if ss is given."""
    iz = model.nTFP // 2 if iz is None else iz
    K = model.K_ss if K is None else K
    with torch.no_grad():
        a = model.a_grid.view(-1, 1).expand(model.nA, model.nE)
        e = model.e_grid.view(1, -1).expand(model.nA, model.nE)
        sav, _, _ = model.policy(net, a, e, model.tfp_grid[iz], torch.tensor(K))
    out = {"a": model.a_grid.numpy(), "e": model.e_grid.numpy(), "sav_net": sav.numpy(), "K": K,
           "Z": float(model.tfp_grid[iz])}
    if ss is not None:
        out["sav_egm"] = ss["ap"]
    return out
