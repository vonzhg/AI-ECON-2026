"""Session 4 lab, block D -- DeepHAM on Krusell and Smith (1998), in PyTorch, at CPU scale.

Written for this lab from Deck 4.2's description of Han, Yang and E, "DeepHAM: A Global Solution Method for
Heterogeneous Agent Models with Aggregate Shocks" (QE 2026; arXiv 2112.14377), and from the settings of the
public repository's Krusell-Smith configuration as the deck reports them. It is not a port of that code.

The economy is block B's: session04_ks, Krusell and Smith (1998) in Den Haan, Judd and Juillard's (2010)
parameterisation (log utility, beta 0.99, alpha 0.36, delta 0.025, z in {0.99, 1.01}, unemployment 10% / 4%, a
benefit of 15% of the wage financed by a tax on the employed). A "distribution" here is a PANEL: E economies, each
a sample of N households (a_i, e_i) with continuous wealth and their own employment draws from the published
chain P(e' | e, z, z'); prices are marginal products at the panel's mean wealth K.

Three networks, as the deck draws them (pp. 7-8):
  Q      the generalised moment, 1 -> 12 -> 12 -> 1 (tanh); Qbar = mean_i Q(a_i) is the one moment the others read;
  actor  (a_i, e_i, z, Qbar) -> a consumption share in (0, 1) of cash on hand, so saving a' >= 0 by construction;
  critic (a_i, e_i, z, Qbar) -> lifetime utility.
Three clocks (pp. 10-16, 20):
  actor update   one Adam step per mini-batch of B pool rows, each unrolled T_unroll periods with fresh shocks;
                 the objective is household 0's discounted utility plus beta^T times the critic at the end; the
                 other N-1 households use the same network with the gradient stopped (fictitious play);
  pool rebuild   every U_pool = P / B updates (the pool is exhausted): every economy runs T_pool periods from its
                 stored state under the current actor, and every snap_every-th cross-section from snap_from on is
                 kept: P = E x n_snap rows, households shuffled within a row so that household 0 is a random member;
  critic refit   every F updates: every economy runs T_V periods from its stored state; the cross-sections at D
                 dates Delta_V apart are the rows, each household's target its realised discounted utility over the
                 next T_target periods; n_ep passes of mini-batches of B_V rows, one net.
  Where a run starts: the stored state of its economy, the burned-in cross-section until the first refit; from the
  first refit on, every run (pool or critic) replaces the stored state with its own last period.

Departures from the paper, each said in the notebook:
  1. scale: E, N, T_burn, T_pool, B, T_unroll, U, T_V, D, T_target, B_V, n_ep, F are a CPU-minute's worth; every
     field of Config carries the paper's value in a comment;
  2. one critic net, not three averaged;
  3. the actor is warm-started by fitting its consumption share to block B's classical Krusell-Smith policy on the
     burned-in panel (the paper starts from a constant share 0.4); the burn-in itself runs under that policy, as in
     the paper (6,000 periods there);
  4. the generalised-moment net is trained with the actor only; the critic reads Qbar as a fixed input;
  5. prices use the model's unemployment rate u(z) for labour, not the panel's realised employment;
  6. the actor's learning rate is 1e-3 (the paper 4e-4), the critic's 1e-3 (the paper 1e-4).
Imports: numpy, torch, session04_ks.
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

import session04_ks as ks

torch.set_num_threads(4)
DT = torch.float64
HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------------------------- the economy
P = ks.KSParams()
PZ_NP, PJ_NP = ks.ks_transition_matrix(P)             # 2x2 chain for z; 4x4 joint chain, state 2z + e
EMP_NP = np.zeros((2, 2, 2, 2))                       # P(e' | e, z, z') = joint / Pz
for _z in range(2):
    for _zn in range(2):
        for _e in range(2):
            for _en in range(2):
                EMP_NP[_z, _zn, _e, _en] = PJ_NP[2 * _z + _e, 2 * _zn + _en] / PZ_NP[_z, _zn]
PZ = torch.as_tensor(PZ_NP, dtype=DT)
EMP_STAY_U = torch.as_tensor(EMP_NP[:, :, :, 0], dtype=DT)     # P(e' = unemployed | e, z, z'), (2, 2, 2)
Z_LEVEL = torch.as_tensor(P.z, dtype=DT)
U_RATE = torch.as_tensor(P.u, dtype=DT)
A_SCALE = 50.0                                        # wealth is read by the networks as a / A_SCALE


def prices_t(K, zi):
    """Gross return, wage and tax at panel mean K (E,) in aggregate state zi (E,) long. Block B's formulas."""
    L = 1.0 - U_RATE[zi]
    kl = K / (P.l_bar * L)
    zl = Z_LEVEL[zi]
    r = P.alpha * zl * kl ** (P.alpha - 1.0)
    w = (1.0 - P.alpha) * zl * kl ** P.alpha
    tau = P.mu * U_RATE[zi] / (P.l_bar * L)
    return 1.0 - P.delta + r, w, tau


def cash_on_hand(a, e, zi):
    """a, e: (E, N); zi: (E,). Returns cash (E, N), K (E,), R (E,)."""
    K = a.mean(1)
    R, w, tau = prices_t(K, zi)
    inc = w[:, None] * torch.where(e == 1, ((1.0 - tau) * P.l_bar)[:, None].expand_as(a),
                                   torch.full_like(a, P.mu))
    return R[:, None] * a + inc, K, R


def draw_next(zi, e, gen):
    """One period of the exogenous chain: z' from Pz, then e' from P(e' | e, z, z')."""
    zn = torch.multinomial(PZ[zi], 1, generator=gen).squeeze(-1)
    u = torch.rand(e.shape, generator=gen, dtype=DT)
    p_unemp = EMP_STAY_U[zi[:, None], zn[:, None], e]            # (E, N)
    en = (u >= p_unemp).long()
    return zn, en


# ---------------------------------------------------------------------------------------------- the networks
class MLP(nn.Module):
    def __init__(self, n_in, width, n_out=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
                                 nn.Linear(width, n_out))

    def forward(self, x):
        return self.net(x)


class DeepHAM(nn.Module):
    """The three networks and the panel arithmetic. Everything is float64 and runs on the CPU."""

    def __init__(self, hidden=24, q_hidden=12):
        super().__init__()
        self.Q = MLP(1, q_hidden)
        self.actor = MLP(4, hidden)
        self.critic = MLP(4, hidden)
        self.register_buffer("v_mu", torch.zeros((), dtype=DT))     # the critic is fitted on standardised targets
        self.register_buffer("v_sd", torch.ones((), dtype=DT))
        self.to(DT)

    def qbar(self, a):
        return self.Q((a / A_SCALE).unsqueeze(-1)).squeeze(-1).mean(1, keepdim=True)        # (E, 1)

    @staticmethod
    def features(a, e, zi, qbar):
        z = (2.0 * zi.to(DT) - 1.0)[:, None].expand_as(a)
        return torch.stack([a / A_SCALE, e.to(DT), z, qbar.expand_as(a)], dim=-1)

    def share(self, a, e, zi, qbar=None):
        if qbar is None:
            qbar = self.qbar(a)
        return torch.sigmoid(self.actor(self.features(a, e, zi, qbar)).squeeze(-1))

    def value(self, a, e, zi, qbar):
        return self.critic(self.features(a, e, zi, qbar)).squeeze(-1) * self.v_sd + self.v_mu

    def step(self, a, e, zi, gen, grad_hh0=False, share_fn=None):
        """One period of a panel: consumption today, the move to tomorrow. Returns (a', e', z', c, K)."""
        cash, K, _ = cash_on_hand(a, e, zi)
        if share_fn is None:
            sh = self.share(a, e, zi)
            if grad_hh0:                                  # fictitious play: only household 0 keeps its gradient
                sh = torch.cat([sh[:, :1], sh[:, 1:].detach()], dim=1)
        else:
            sh = share_fn(a, e, zi, cash, K)
        c = sh * cash
        a_next = cash - c
        zn, en = draw_next(zi, e, gen)
        return a_next, en, zn, c, K


# ---------------------------------------------------------------------------------------------- settings
@dataclass(frozen=True)
class Config:
    E: int = 32               # economies side by side (the paper: 384)
    N: int = 50               # households per economy (the paper: 50)
    T_burn: int = 300         # burn-in under the classical rule, from everyone at K_ss (the paper: 6,000)
    T_pool: int = 120         # periods of a pool run (the paper: 450)
    snap_from: int = 60       # first cross-section kept in a pool run (the paper: 250)
    snap_every: int = 4       # every 4th kept (the paper: 4)
    B: int = 64               # actor mini-batch, rows of the pool (the paper: 384)
    T_unroll: int = 150       # periods unrolled inside the actor's loss (the paper: 150; beta^149 = 0.22 to the critic)
    U: int = 250              # actor updates in all (the paper: 10,000)
    lr_actor: float = 1e-4    # (the paper: 4e-4; the lab starts from the classical policy, so it steps gently)
    T_V: int = 450            # periods of a critic run (the paper: 2,000)
    D: int = 4                # start dates of the critic's rows (the paper: 12)
    Delta_V: int = 100        # spacing of the dates (the paper: 100)
    T_target: int = 150       # horizon of a target, beta^150 = 0.22 (the paper: 800, beta^800 < 3.3e-4)
    B_V: int = 32             # critic mini-batch, rows (the paper: 128)
    n_ep: int = 120           # passes over the rows per fit (the paper: 201)
    F: int = 100              # actor updates between refits (the paper: 2,000)
    lr_critic: float = 3e-3   # on standardised targets, cosine-annealed over a fit (the paper: 1e-4)
    valid_share: float = 0.125  # rows set aside to check the fit (the paper: 512 of 4,608)
    hidden: int = 32          # actor and critic width (the lab's)
    q_hidden: int = 12        # moment-net width (the paper: 12)
    warm_steps: int = 3000    # warm start: fit the share to the classical policy (the lab's)
    log_every: int = 50
    seed: int = 0

    @property
    def n_snap(self) -> int:
        return len(range(self.snap_from, self.T_pool, self.snap_every))

    @property
    def P(self) -> int:
        return self.E * self.n_snap

    @property
    def U_pool(self) -> int:
        return max(1, self.P // self.B)


LIVE = Config()
RECORD = replace(LIVE, E=64, U=600, T_V=700, D=5, T_target=300, n_ep=150, F=200)
SMOKE = replace(LIVE, E=8, T_burn=40, T_pool=40, snap_from=20, B=8, T_unroll=10, U=12, T_V=60, D=2, Delta_V=20,
                T_target=30, n_ep=3, F=6, warm_steps=20, log_every=4)


# ---------------------------------------------------------------------------------------------- the classical policy
class ClassicalPolicy:
    """Block B's Krusell-Smith household policy c(a, s, K) under the recorded rule kappa, read on a panel."""

    def __init__(self, rec=None, settings: ks.KSSettings = ks.FULL):
        rec = ks.load_recorded() if rec is None else rec
        if rec is None:
            raise RuntimeError("recorded_ks.json is needed for the burn-in policy")
        self.kappa = np.array([rec["solution"]["kappa"]["bad"], rec["solution"]["kappa"]["good"]])
        s = settings
        self.a = ks.make_grid(s.n_a, P.a_min, s.a_max, s.curv)
        self.logK = np.linspace(np.log(s.K_lo), np.log(s.K_hi), s.n_K)
        self.c, self.sweeps, _ = ks.solve_household(self.kappa, P, self.a, self.logK, PJ_NP)
        self.K_ss = float(rec["capital"]["mean"])

    def consumption(self, a, e, zi, K):
        """a, e: (E, N) numpy; zi, K: (E,). Linear in log K between nodes, linear in a; K clipped to the grid."""
        logK = np.clip(np.log(K), self.logK[0], self.logK[-1])
        pos = (logK - self.logK[0]) / (self.logK[1] - self.logK[0])
        m0 = np.clip(np.floor(pos).astype(int), 0, self.logK.size - 2)
        th = pos - m0
        out = np.empty_like(a)
        for i in range(a.shape[0]):
            for ee in range(2):
                sel = e[i] == ee
                if not sel.any():
                    continue
                s = 2 * int(zi[i]) + ee
                lo = np.interp(a[i, sel], self.a, self.c[:, s, m0[i]])
                hi = np.interp(a[i, sel], self.a, self.c[:, s, m0[i] + 1])
                out[i, sel] = (1.0 - th[i]) * lo + th[i] * hi
        return out

    def share_fn(self, a, e, zi, cash, K):
        c = self.consumption(a.numpy(), e.numpy(), zi.numpy(), K.numpy())
        c = np.minimum(c, 0.999 * cash.numpy())
        return torch.as_tensor(c / cash.numpy(), dtype=DT)


# ---------------------------------------------------------------------------------------------- simulation helpers
@torch.no_grad()
def simulate(model: DeepHAM, a, e, zi, T, gen, share_fn=None, keep=(), record_c=False):
    """Run a panel T periods. keep: periods whose cross-sections to return. Returns dict with the final state,
    the kept cross-sections, K path (E, T) and, if asked, every household's log consumption (E, T, N)."""
    E, N = a.shape
    K_path = torch.empty(E, T, dtype=DT)
    logc = torch.empty(E, T, N, dtype=DT) if record_c else None
    kept = {}
    for t in range(T):
        if t in keep:
            kept[t] = (a.clone(), e.clone(), zi.clone())
        a_next, en, zn, c, K = model.step(a, e, zi, gen, share_fn=share_fn)
        K_path[:, t] = K
        if record_c:
            logc[:, t] = torch.log(c)
        a, e, zi = a_next, en, zn
    return dict(a=a, e=e, z=zi, K=K_path, logc=logc, kept=kept)


def initial_panel(cfg: Config, K0, gen):
    """Everyone at K0; the aggregate state drawn; employment drawn at the state's unemployment rate."""
    zi = torch.multinomial(torch.full((cfg.E, 2), 0.5, dtype=DT), 1, generator=gen).squeeze(-1)
    u = torch.rand(cfg.E, cfg.N, generator=gen, dtype=DT)
    e = (u >= U_RATE[zi][:, None]).long()
    a = torch.full((cfg.E, cfg.N), float(K0), dtype=DT)
    return a, e, zi


def shuffle_households(a, e, gen):
    """Households permuted within each row, so that household 0 is a random member (the paper's reshuffle)."""
    perm = torch.argsort(torch.rand(a.shape, generator=gen, dtype=DT), dim=1)
    return torch.gather(a, 1, perm), torch.gather(e, 1, perm)


# ---------------------------------------------------------------------------------------------- training
def warm_start(model: DeepHAM, pol: ClassicalPolicy, a, e, zi, cfg: Config, gen, lr=5e-3):
    """The lab's warm start: the moment net is first fitted to Q(a) = a / A_SCALE, so that Qbar starts as the
    panel's mean wealth (Krusell and Smith's state); then the actor's share (and Q with it) is fitted to the
    classical policy's share on the burned-in panel, by relative consumption error."""
    with torch.no_grad():
        cash, K, _ = cash_on_hand(a, e, zi)
        target = pol.share_fn(a, e, zi, cash, K)
    opt_q = torch.optim.Adam(model.Q.parameters(), lr=3e-3)
    x = (a / A_SCALE).unsqueeze(-1)
    for _ in range(300):
        opt_q.zero_grad()
        lq = torch.mean((model.Q(x).squeeze(-1) - a / A_SCALE) ** 2)
        lq.backward(); opt_q.step()
    opt = torch.optim.Adam(list(model.actor.parameters()) + list(model.Q.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.warm_steps)
    mse = float("nan")
    for _ in range(cfg.warm_steps):
        opt.zero_grad()
        loss = torch.mean((model.share(a, e, zi) / target - 1.0) ** 2)      # relative consumption error
        loss.backward(); opt.step(); sched.step()
        mse = loss.item()
    return mse


def build_pool(model, stored, cfg: Config, gen, move_stored: bool):
    """Every economy runs T_pool periods from its stored state; every snap_every-th cross-section from snap_from
    on is a row. If move_stored, the run's last period replaces the stored state."""
    a, e, zi = stored
    keep = tuple(range(cfg.snap_from, cfg.T_pool, cfg.snap_every))
    out = simulate(model, a, e, zi, cfg.T_pool, gen, keep=keep)
    rows_a = torch.cat([out["kept"][t][0] for t in keep], 0)
    rows_e = torch.cat([out["kept"][t][1] for t in keep], 0)
    rows_z = torch.cat([out["kept"][t][2] for t in keep], 0)
    rows_a, rows_e = shuffle_households(rows_a, rows_e, gen)
    perm = torch.randperm(rows_a.shape[0], generator=gen)
    pool = (rows_a[perm], rows_e[perm], rows_z[perm])
    if move_stored:
        stored = (out["a"], out["e"], out["z"])
    return pool, stored


def actor_objective(model: DeepHAM, a, e, zi, cfg: Config, gen):
    """Household 0's discounted utility over T_unroll fresh periods plus beta^T times the critic at the end."""
    J = torch.zeros(a.shape[0], dtype=DT)
    disc = 1.0
    for _ in range(cfg.T_unroll):
        a, e, zi, c, _ = model.step(a, e, zi, gen, grad_hh0=True)
        J = J + disc * torch.log(c[:, 0])
        disc *= P.beta
    qbar = model.qbar(a)
    J = J + disc * model.value(a[:, :1], e[:, :1], zi, qbar)[:, 0]
    return J.mean()


def critic_rows(model, stored, cfg: Config, gen, move_stored: bool):
    """Runs of T_V periods from the stored states; rows at D dates; targets: realised discounted utility over
    T_target periods, every household its own."""
    a, e, zi = stored
    dates = [d * cfg.Delta_V for d in range(cfg.D)]
    assert dates[-1] + cfg.T_target <= cfg.T_V, "a date needs T_target future periods"
    out = simulate(model, a, e, zi, cfg.T_V, gen, keep=tuple(dates), record_c=True)
    disc = torch.as_tensor(P.beta ** np.arange(cfg.T_target), dtype=DT)
    rows = []
    for s in dates:
        ra, re, rz = out["kept"][s]
        target = (out["logc"][:, s:s + cfg.T_target] * disc[None, :, None]).sum(1)      # (E, N)
        rows.append((ra, re, rz, target))
    rows_a = torch.cat([r[0] for r in rows]); rows_e = torch.cat([r[1] for r in rows])
    rows_z = torch.cat([r[2] for r in rows]); rows_v = torch.cat([r[3] for r in rows])
    if move_stored:
        stored = (out["a"], out["e"], out["z"])
    return (rows_a, rows_e, rows_z, rows_v), stored


def fit_critic(model: DeepHAM, rows, cfg: Config, gen):
    """n_ep passes of mini-batches of B_V rows; a share of the rows set aside; returns (mse, held-out R^2)."""
    ra, re, rz, rv = rows
    R = ra.shape[0]
    perm = torch.randperm(R, generator=gen)
    n_valid = max(1, int(round(cfg.valid_share * R)))
    valid, train = perm[:n_valid], perm[n_valid:]
    with torch.no_grad():
        qbar = model.qbar(ra)                                          # Qbar of each row, a fixed input
        model.v_mu.fill_(rv[train].mean()); model.v_sd.fill_(rv[train].std() + 1e-12)
    opt = torch.optim.Adam(model.critic.parameters(), lr=cfg.lr_critic)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_ep)
    mse = float("nan")
    for _ in range(cfg.n_ep):
        order = train[torch.randperm(train.numel(), generator=gen)]
        for k in range(0, order.numel(), cfg.B_V):
            idx = order[k:k + cfg.B_V]
            opt.zero_grad()
            pred = model.value(ra[idx], re[idx], rz[idx], qbar[idx])
            loss = torch.mean(((pred - rv[idx]) / model.v_sd) ** 2)
            loss.backward(); opt.step()
            mse = loss.item()
        sched.step()
    with torch.no_grad():
        pred = model.value(ra[valid], re[valid], rz[valid], qbar[valid])
        resid = torch.mean((pred - rv[valid]) ** 2)
        r2 = 1.0 - resid / torch.var(rv[valid])
    return mse, float(r2)


def train(cfg: Config = LIVE, verbose: bool = True, pol: ClassicalPolicy | None = None):
    """Burn in, warm-start, fit the critic, then U actor updates with the three clocks. Returns (model, history)."""
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    pol = ClassicalPolicy() if pol is None else pol
    model = DeepHAM(cfg.hidden, cfg.q_hidden)
    # 1. burn in under the classical rule, from everyone at K_ss, employment drawn at u(z)
    a, e, zi = initial_panel(cfg, pol.K_ss, gen)
    burn = simulate(model, a, e, zi, cfg.T_burn, gen, share_fn=pol.share_fn)
    stored = (burn["a"], burn["e"], burn["z"])
    # 2. warm start of the actor on the burned-in panel
    warm_mse = warm_start(model, pol, *stored, cfg, gen)
    hist = dict(objective=[], critic_mse=[], critic_r2=[], refit_at=[], pool_at=[], K_pool=[], warm_mse=warm_mse,
                K_burn=burn["K"][:, -1].mean().item())
    # 3. the first critic fit, before any actor update (the stored states do not move yet)
    rows, stored = critic_rows(model, stored, cfg, gen, move_stored=False)
    mse, r2 = fit_critic(model, rows, cfg, gen)
    hist["critic_mse"].append(mse); hist["critic_r2"].append(r2); hist["refit_at"].append(0)
    opt = torch.optim.Adam(list(model.actor.parameters()) + list(model.Q.parameters()), lr=cfg.lr_actor)
    update_init = False
    pool, stored = build_pool(model, stored, cfg, gen, move_stored=update_init)
    hist["pool_at"].append(0); hist["K_pool"].append(pool[0].mean().item())
    cursor = 0
    for u in range(cfg.U):
        if u > 0 and u % cfg.F == 0:                                   # critic refit; stored states move from now on
            update_init = True
            rows, stored = critic_rows(model, stored, cfg, gen, move_stored=True)
            mse, r2 = fit_critic(model, rows, cfg, gen)
            hist["critic_mse"].append(mse); hist["critic_r2"].append(r2); hist["refit_at"].append(u)
        if cursor + cfg.B > pool[0].shape[0]:                          # exhausted: rebuild the pool
            pool, stored = build_pool(model, stored, cfg, gen, move_stored=update_init)
            hist["pool_at"].append(u); hist["K_pool"].append(pool[0].mean().item())
            cursor = 0
        ra, re, rz = (x[cursor:cursor + cfg.B] for x in pool)
        cursor += cfg.B
        opt.zero_grad()
        J = actor_objective(model, ra, re, rz, cfg, gen)
        (-J).backward(); opt.step()
        hist["objective"].append(J.item())
        if verbose and (u % cfg.log_every == 0 or u == cfg.U - 1):
            print(f"update {u:5d} | objective {J.item():8.4f} | pools {len(hist['pool_at']):3d} "
                  f"| critic fits {len(hist['refit_at'])} (held-out R2 {hist['critic_r2'][-1]:.3f}) | {time.time() - t0:5.0f} s")
    hist["seconds"] = time.time() - t0
    hist["cfg"] = asdict(cfg)
    hist["stored"] = tuple(x.clone() for x in stored)
    return model, hist


# ---------------------------------------------------------------------------------------------- diagnostics
@torch.no_grad()
def fresh_panels(model, cfg: Config, pol: ClassicalPolicy, n_econ=16, T=300, seed=7, share_fn=None):
    """n_econ fresh economies from everyone at K_ss, burned in 100 periods under the policy to be judged."""
    gen = torch.Generator().manual_seed(seed)
    c2 = replace(cfg, E=n_econ)
    a, e, zi = initial_panel(c2, pol.K_ss, gen)
    out = simulate(model, a, e, zi, T, gen, share_fn=share_fn, record_c=True)
    return out, gen


@torch.no_grad()
def euler_errors(model: DeepHAM, cfg: Config, pol: ClassicalPolicy, n_econ=16, T=300, n_mc=16, seed=7):
    """Relative Euler error of the actor on a fresh panel: beta E[R' c / c'] - 1 by Monte Carlo over (z', e'),
    on households with slack saving. Log utility, so u'(c) = 1 / c."""
    out, gen = fresh_panels(model, cfg, pol, n_econ, T, seed)
    a, e, zi = out["a"], out["e"], out["z"]
    cash, K, _ = cash_on_hand(a, e, zi)
    sh = model.share(a, e, zi)
    c = sh * cash
    a_next = cash - c
    rhs = torch.zeros_like(c)
    for _ in range(n_mc):
        zn, en = draw_next(zi, e, gen)
        cash_n, _, Rn = cash_on_hand(a_next, en, zn)
        c_n = model.share(a_next, en, zn) * cash_n
        rhs += Rn[:, None] / c_n
    rhs /= n_mc
    err = P.beta * rhs * c - 1.0
    slack = a_next / cash > 1e-3
    ae = err[slack].abs()
    return dict(mean_abs=float(ae.mean()), p90_abs=float(torch.quantile(ae, 0.9)), share_slack=float(slack.to(DT).mean()),
                K_mean=float(out["K"][:, 100:].mean()), K_min=float(out["K"][:, 100:].min()),
                K_max=float(out["K"][:, 100:].max()), K_path=out["K"].numpy())


@torch.no_grad()
def welfare_vs_classical(model: DeepHAM, cfg: Config, pol: ClassicalPolicy, n_econ=16, T=300, seed=11):
    """Realised discounted utility of every household under the actor and under the classical policy, from the
    same initial panel with the same shocks; the gap as a consumption equivalent (log utility)."""
    disc = torch.as_tensor(P.beta ** np.arange(T), dtype=DT)
    outs = {}
    for name, fn in (("actor", None), ("classical", pol.share_fn)):
        out, _ = fresh_panels(model, cfg, pol, n_econ, T, seed, share_fn=fn)
        outs[name] = (out["logc"] * disc[None, :, None]).sum(1)         # (E, N)
    gap = outs["actor"].mean() - outs["classical"].mean()
    cev = float(torch.exp((1.0 - P.beta) * gap) - 1.0)
    return dict(utility_actor=float(outs["actor"].mean()), utility_classical=float(outs["classical"].mean()),
                cev_vs_classical=cev)


@torch.no_grad()
def share_plot_data(model: DeepHAM, cfg: Config, pol: ClassicalPolicy, seed=3):
    """Consumption share against wealth, at the panel of one fresh economy at K near K_ss, actor vs classical."""
    out, _ = fresh_panels(model, cfg, pol, n_econ=1, T=200, seed=seed)
    a, e, zi = out["a"], out["e"], out["z"]
    grid = torch.linspace(0.0, 120.0, 121, dtype=DT)
    rows = {}
    for ee in (0, 1):
        a_q = torch.cat([grid[None, :], a[:, :cfg.N - grid.numel()]], 1) if cfg.N > grid.numel() else grid[None, :]
        e_q = torch.full_like(a_q, ee, dtype=torch.long)
        cash, K, _ = cash_on_hand(a_q, e_q, zi)
        sh_a = model.share(a_q, e_q, zi)[0, :grid.numel()]
        sh_c = pol.share_fn(a_q, e_q, zi, cash, K)[0, :grid.numel()]
        rows[ee] = (sh_a.numpy(), sh_c.numpy())
    return dict(a=grid.numpy(), share=rows, z=int(zi[0]), K=float(a.mean()))


def objective_change(hist) -> dict:
    J = np.asarray(hist["objective"])
    n = max(1, len(J) // 10)
    return dict(first_tenth=float(J[:n].mean()), last_tenth=float(J[-n:].mean()),
                change=float(J[-n:].mean() - J[:n].mean()))


def summary(model, hist, cfg: Config, pol: ClassicalPolicy) -> dict:
    ee = euler_errors(model, cfg, pol)
    wf = welfare_vs_classical(model, cfg, pol)
    return dict(seed=cfg.seed, seconds=round(hist["seconds"], 1), warm_mse=hist["warm_mse"],
                euler_mean_abs=ee["mean_abs"], euler_p90_abs=ee["p90_abs"], share_slack=ee["share_slack"],
                K_mean=ee["K_mean"], K_min=ee["K_min"], K_max=ee["K_max"], K_ss_classical=pol.K_ss,
                critic_r2_last=hist["critic_r2"][-1], critic_r2_all=hist["critic_r2"],
                objective=objective_change(hist), **wf)


def save(path, model, hist):
    h = {k: v for k, v in hist.items() if k != "stored"}
    torch.save(dict(state_dict=model.state_dict(), history=h, stored=hist["stored"]), path)


def load(path):
    rec = torch.load(path, weights_only=False)
    cfg = Config(**rec["history"]["cfg"])
    model = DeepHAM(cfg.hidden, cfg.q_hidden)
    model.load_state_dict(rec["state_dict"])
    hist = dict(rec["history"]); hist["stored"] = rec["stored"]
    return model, hist, cfg


if __name__ == "__main__":
    import sys
    cfg = SMOKE if "smoke" in sys.argv else LIVE
    pol = ClassicalPolicy()
    print(f"classical policy: {pol.sweeps} EGM sweeps; K_ss {pol.K_ss:.3f}")
    model, hist = train(cfg, verbose=True, pol=pol)
    print(json.dumps(summary(model, hist, cfg, pol), indent=1, default=float))
