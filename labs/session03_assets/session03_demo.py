"""Deck 3.4's example (the simplest actor-critic; Deck 3.1 reads two of its figures): the growth model without a grid, its answer key, and an eight-number learner.

Course-owned.  Nothing here imports the book's code: the book's running example is a 7x2 grid, and
this module is the same model with no grid on the state and none on the action.

    model        k > 0 real, z in {0.9, 1.1} with switching probability 0.2,
                 y(k,z) = z k^alpha + (1 - delta) k,   action a in (0,1),
                 k' = a y,  c = (1 - a) y,  r = ln c.
                 Calibration: alpha = 0.36, beta = 0.95, delta = 0.10  (the book's).

    answer key   value function iteration on 400 log-spaced k points x 2 z, the maximum taken over a
                 fine grid of shares and then refined by golden section, with V interpolated off the
                 grid.

    learner      eight numbers.  Actor: a deterministic saving rule a = sigmoid(mu_theta(k,z)),
                 mu_theta = theta0 + theta1 ktilde + theta2 1{z = 1.1}: three numbers.
                 Critic: V_psi = psi . phi(k,z) with phi = (1, ktilde, ktilde^2, 1{z=1.1},
                 ktilde 1{z=1.1}): five numbers.  ktilde = ln k rescaled to the band.

                 Two trainers share the eight numbers.
                 train_known  (Deck 3.4): the model is KNOWN.  States are drawn uniformly over the
                              band; the critic is fitted by least squares to the one-period backup
                              with the expectation over z' summed exactly; the actor climbs the
                              Bellman right-hand side by a gradient through the model.  Nothing is
                              simulated unless the backup is lengthened to T periods (the tree,
                              enumerated exactly or sampled) or the expectation is replaced by one
                              draw (the bridge to Deck 3.3).
                 train        (Decks 3.4-3.5): model-free, online, one transition at a time, a fixed
                              probe around the rule and the score-function gradient; reads r and s'
                              from the simulator and never reads P_z.

    grade        consumption-equivalent welfare loss of the learned rule against the key, weighted by
                 where that rule's own economy lives.

Run `python3 session03_demo.py` for a self-test of the known-model trainer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- the model

ALPHA, BETA, DELTA = 0.36, 0.95, 0.10
GAMMA = BETA                     # the session writes the discount gamma; it is the book's beta
ZVALS = np.array([0.9, 1.1])
PZ = np.array([[0.8, 0.2], [0.2, 0.8]])    # the simulator's; no learner reads it
BAND = (0.2, 2.0)                # the key's grid, as multiples of the deterministic steady state
NK, NA, RESET_EVERY = 400, 1000, 300
A_MIN = 1e-3                     # keeps ln c finite when the actor draws an extreme share


def k_star() -> float:
    """Deterministic steady state at z = 1: beta (alpha k^(alpha-1) + 1 - delta) = 1."""
    return (ALPHA / (1.0 / BETA - 1.0 + DELTA)) ** (1.0 / (1.0 - ALPHA))


def steady_value() -> float:
    """u(c*)/(1-gamma): consuming the steady state's consumption for ever.  The critic's start."""
    ks = k_star()
    return math.log(ks ** ALPHA + (1.0 - DELTA) * ks - ks) / (1.0 - GAMMA)


def sigmoid(x):
    out = np.empty_like(np.asarray(x, dtype=float))
    pos = np.asarray(x) >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-np.asarray(x)[pos]))
    e = np.exp(np.asarray(x)[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _lookup(Vk, idx):
    """Vk: (NK,) or (R, NK) values on the k grid for one z.  idx: k-indices."""
    if Vk.ndim == 1:
        return Vk[idx]
    return np.take_along_axis(Vk, idx, axis=-1)


class Grid:
    """The answer key's state grid: NK log-spaced k on the band, times two z.  States are flattened
    as s = i * 2 + j, so every array below has a trailing axis of length S = 2 NK."""

    def __init__(self, nk: int = NK, band=BAND):
        ks = k_star()
        self.k_star = ks
        self.lk_lo, self.lk_hi = math.log(band[0] * ks), math.log(band[1] * ks)
        self.nk, self.S = nk, 2 * nk
        self.lk = np.linspace(self.lk_lo, self.lk_hi, nk)
        self.k = np.exp(self.lk)
        self.h = self.lk[1] - self.lk[0]
        self.center = 0.5 * (self.lk_lo + self.lk_hi)
        self.scale = 0.5 * (self.lk_hi - self.lk_lo)
        self.k_s = np.repeat(self.k, 2)
        self.z_s = np.tile(ZVALS, nk)
        self.j_s = np.tile(np.array([0, 1]), nk)
        self.y_s = self.z_s * self.k_s ** ALPHA + (1.0 - DELTA) * self.k_s
        self.ktilde_s = (np.log(self.k_s) - self.center) / self.scale
        self.p0_s, self.p1_s = PZ[self.j_s, 0], PZ[self.j_s, 1]

    def tilde(self, k):
        return (np.log(k) - self.center) / self.scale

    def weights(self, kp):
        """Linear interpolation on the log-k grid.  Returns (i0, w, outside)."""
        lk = np.log(kp)
        outside = (lk < self.lk_lo) | (lk > self.lk_hi)
        pos = (np.clip(lk, self.lk_lo, self.lk_hi) - self.lk_lo) / self.h
        i0 = np.clip(np.floor(pos).astype(np.int64), 0, self.nk - 2)
        return i0, pos - i0, outside

    def cont(self, V, i0, w, act_axis: bool = False):
        """E[V(k', z') | z].  V is (S,) or (R, S); i0, w are (S,), (R, S) or (S, NA)."""
        V = np.asarray(V)
        V0, V1 = V[..., 0::2], V[..., 1::2]
        e0, f0 = _lookup(V0, i0), _lookup(V0, i0 + 1)
        e1, f1 = _lookup(V1, i0), _lookup(V1, i0 + 1)
        p0, p1 = self.p0_s, self.p1_s
        if act_axis:
            p0, p1 = p0[:, None], p1[:, None]
        return p0 * (e0 + w * (f0 - e0)) + p1 * (e1 + w * (f1 - e1))


# ------------------------------------------------------------ the answer key

@dataclass
class Key:
    grid: Grid
    V: np.ndarray            # (S,)  the optimal value
    share: np.ndarray        # (S,)  the optimal saving share
    sweeps: int
    change: np.ndarray       # sup-norm change per sweep
    na: int
    gain: float              # what the golden-section refinement bought, in value units
    slack: float             # smallest distance from the optimal share to its feasible bounds


def _rhs(g: Grid, V, a):
    i0, w, _ = g.weights(a * g.y_s)
    return np.log((1.0 - a) * g.y_s) + GAMMA * g.cont(V, i0, w)


def action_grid(g: Grid, na: int = NA):
    """Shares that keep k' on the band.  A dynamic programme on a grid must land on that grid: the
    feasible k' is [k_lo, min(k_hi, (1 - A_MIN) y)], which the optimum never touches (checked in
    answer_key).  Returns (shares (S, na), a_lo (S,), a_hi (S,))."""
    a_lo = np.exp(g.lk_lo) / g.y_s
    a_hi = np.minimum(np.exp(g.lk_hi) / g.y_s, 1.0 - A_MIN)
    u = (np.arange(na) + 0.5) / na
    return a_lo[:, None] + u[None, :] * (a_hi - a_lo)[:, None], a_lo, a_hi


def answer_key(grid: Grid | None = None, na: int = NA, tol: float = 1e-6) -> Key:
    g = grid or Grid()
    a, a_lo, a_hi = action_grid(g, na)
    r = np.log((1.0 - a) * g.y_s[:, None])
    i0, w, out = g.weights(a * g.y_s[:, None])
    assert not out.any(), "the action grid left the band"
    V, change = np.zeros(g.S), []
    while True:
        Vn = (r + GAMMA * g.cont(V, i0, w, act_axis=True)).max(axis=1)
        d = float(np.max(np.abs(Vn - V)))
        V = Vn
        change.append(d)
        if d < tol:
            break
    m = (r + GAMMA * g.cont(V, i0, w, act_axis=True)).argmax(axis=1)
    step = (a_hi - a_lo) / na
    lo = np.maximum(np.take_along_axis(a, m[:, None], 1)[:, 0] - step, a_lo)
    hi = np.minimum(np.take_along_axis(a, m[:, None], 1)[:, 0] + step, a_hi)
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(80):                                   # golden section on the converged V
        c1, c2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
        take = _rhs(g, V, c1) > _rhs(g, V, c2)
        hi, lo = np.where(take, c2, hi), np.where(take, lo, c1)
    share = 0.5 * (lo + hi)
    slack = float(min(np.min(share - a_lo), np.min(a_hi - share)))
    V_ref = policy_evaluate(g, share)
    return Key(g, V_ref, share, len(change), np.array(change), na,
               float(np.max(V_ref - V)), slack)


def cached_key(path="session03_key.npz", na: int = NA) -> Key:
    """The key takes ~90 s; every caller wants the same one."""
    from pathlib import Path as _P
    f = _P(__file__).resolve().parent / path
    g = Grid()
    if f.exists():
        z = np.load(f)
        if int(z["nk"]) == g.nk and int(z["na"]) == na:
            return Key(g, z["V"], z["share"], int(z["sweeps"]), z["change"], na,
                       float(z["gain"]), float(z["slack"]))
    k = answer_key(g, na=na)
    np.savez_compressed(f, V=k.V, share=k.share, sweeps=k.sweeps, change=k.change,
                        na=k.na, nk=g.nk, gain=k.gain, slack=k.slack)
    return k


def feasible(g: Grid, share):
    """Every rule is graded on the same model the key solves: k' stays on the band.  The learner
    meets the same wall -- its k' is clipped there too -- so the comparison is like for like."""
    _, a_lo, a_hi = action_grid(g, 2)
    return np.clip(np.asarray(share, dtype=float), a_lo, a_hi)


def policy_evaluate(g: Grid, share, tol: float = 1e-12, itmax: int = 40000):
    """V^g on the grid for one rule (S,) or a batch of rules (R, S)."""
    share = feasible(g, share)
    r = np.log((1.0 - share) * g.y_s)
    i0, w, _ = g.weights(share * g.y_s)
    V = np.zeros_like(r)
    for _ in range(itmax):
        Vn = r + GAMMA * g.cont(V, i0, w)
        d = np.max(np.abs(Vn - V))
        V = Vn
        if d < tol:
            break
    return V


def ce_loss(Vstar, Vg):
    """Consumption-equivalent welfare loss.  With u = ln c, scaling consumption by s for ever adds
    ln s/(1-gamma) to lifetime utility, so the fraction given up is 1 - exp(-(1-gamma) dV)."""
    return 1.0 - np.exp(-(1.0 - GAMMA) * (Vstar - Vg))


def ergodic(g: Grid, share, tol: float = 1e-12, itmax: int = 20000):
    """Where the rule's economy lives.  Young's (2010) two-point split: send the mass leaving a grid
    point to its two neighbours by the interpolation weights, then apply P_z.  Deterministic, so the
    grade carries no simulation noise.  One rule (S,) or a batch (R, S)."""
    share = feasible(g, share)
    i0, w, _ = g.weights(share * g.y_s)
    batched = share.ndim == 2
    R = share.shape[0] if batched else 1
    i0 = i0.reshape(R, g.S)
    w = w.reshape(R, g.S)
    off = (np.arange(R) * g.S)[:, None]
    tgt = np.concatenate([off + 2 * i0 + jp for jp in (0, 1)]
                         + [off + 2 * (i0 + 1) + jp for jp in (0, 1)], axis=1).ravel()
    p0, p1 = g.p0_s[None, :], g.p1_s[None, :]
    d = np.full((R, g.S), 1.0 / g.S)
    for _ in range(itmax):
        val = np.concatenate([(1.0 - w) * p0 * d, (1.0 - w) * p1 * d,
                              w * p0 * d, w * p1 * d], axis=1).ravel()
        nxt = np.bincount(tgt, weights=val, minlength=R * g.S).reshape(R, g.S)
        diff = np.max(np.abs(nxt - d))
        d = nxt
        if diff < tol:
            break
    d = d / d.sum(axis=1, keepdims=True)
    return d if batched else d[0]


def share_on_grid(g: Grid, theta):
    """The centre rule a = sigmoid(mu_theta) read off on the key's grid.  theta (3,) or (R, 3);
    returns (S,) or (R, S)."""
    th = np.asarray(theta, dtype=float)
    one = th.ndim == 1
    th = np.atleast_2d(th)
    zf = g.j_s.astype(float)
    mu = th[:, 0:1] + th[:, 1:2] * g.ktilde_s[None, :] + th[:, 2:3] * zf[None, :]
    out = sigmoid(mu)
    return out[0] if one else out


def grade(g: Grid, key: Key, share):
    """(loss where the economy lives, loss over the whole band), in per cent."""
    Vg = policy_evaluate(g, share)
    loss = ce_loss(key.V, Vg)
    wgt = ergodic(g, share)
    return 100.0 * np.sum(wgt * loss, axis=-1), 100.0 * np.mean(loss, axis=-1)


# --------------------------------------------------------------- the learner

@dataclass
class Run:
    theta: np.ndarray        # (R, 3)   level, slope in ktilde, shift for z = 1.1
    lnsig: np.ndarray        # (R,)
    psi: np.ndarray          # (R, 5)
    steps: np.ndarray        # checkpoint step counts
    snap_theta: np.ndarray   # (C, R, 3)
    snap_lnsig: np.ndarray   # (C, R)
    snap_psi: np.ndarray     # (C, R, 5)
    clip_a: np.ndarray       # (R,) fraction of draws whose share hit A_MIN
    clip_k: np.ndarray       # (R,) fraction of transitions whose k' hit the band
    finite: np.ndarray       # (R,) bool


def train(key: Key, seed_of_run, *, eta_theta, eta_psi, eta_lnsig=None, learn_sigma=True,
          resets=True, nstep: int = 1, n_steps: int = 10 ** 6, checkpoints=(),
                    lnsig0: float = math.log(0.5), lnsig_lo=math.log(0.05),
          lnsig_hi=math.log(1.5), psi0=None, master: int = 20260922, chunk: int = 20000) -> Run:
    """The nine-number actor-critic, run for every setting at once.

    Every per-run argument is broadcast to (R,).  The random stream is keyed by (seed, step), not by
    the batch layout, so the same seed gives the same shocks whatever else is in the batch -- which
    is what makes n = 1 here reproduce the one-step run to the digit.

    One transition: draw a, take r and s' from the simulator, form delta on the model's s', move the
    critic, move the actor, and only then replace the state.  The reset never enters a target."""
    g = key.grid
    R = len(seed_of_run)
    B = int(nstep)
    one = np.ones(R)
    eta_theta = np.broadcast_to(np.asarray(eta_theta, float) * one, (R,)).copy()
    eta_psi = np.broadcast_to(np.asarray(eta_psi, float) * one, (R,)).copy()
    eta_lnsig = eta_theta.copy() if eta_lnsig is None else \
        np.broadcast_to(np.asarray(eta_lnsig, float) * one, (R,)).copy()
    learn_sigma = np.broadcast_to(np.asarray(learn_sigma, bool) & np.ones(R, bool), (R,)).copy()
    resets = np.broadcast_to(np.asarray(resets, bool) & np.ones(R, bool), (R,)).copy()
    lnsig_lo = np.broadcast_to(np.asarray(lnsig_lo, float) * one, (R,)).copy()
    lnsig_hi = np.broadcast_to(np.asarray(lnsig_hi, float) * one, (R,)).copy()
    if B > 1 and not resets.all():
        raise ValueError("the n-step buffer truncates at a reset; run n > 1 with resets on")

    useeds = np.unique(seed_of_run)
    pos = np.searchsorted(useeds, seed_of_run)
    rngs = [np.random.default_rng([master, int(s)]) for s in useeds]

    theta = np.zeros((R, 3))
    lnsig = np.full(R, float(lnsig0))
    psi = np.zeros((R, 5))
    psi[:, 0] = steady_value()
    if psi0 is not None:
        psi[:] = np.asarray(psi0, float)
    k = np.full(R, g.k_star)
    zi = np.zeros(R, dtype=np.int64)
    klo, khi = math.exp(g.lk_lo), math.exp(g.lk_hi)
    lspan = g.lk_hi - g.lk_lo
    gpow = GAMMA ** np.arange(B + 1)
    nclip_a = np.zeros(R)
    nclip_k = np.zeros(R)

    b_phi = np.zeros((B, R, 5))
    b_sc = np.zeros((B, R, 3))
    b_sl = np.zeros((B, R))
    b_r = np.zeros((B, R))
    held = 0

    cps = sorted({int(c) for c in checkpoints} | {n_steps})
    snap_t = np.zeros((len(cps), R, 3))
    snap_l = np.zeros((len(cps), R))
    snap_p = np.zeros((len(cps), R, 5))
    ci = 0
    eps_c = uni_c = None
    t0 = 0

    def apply(o, G):
        nonlocal psi, theta, lnsig
        delta = G - np.einsum("ij,ij->i", psi, b_phi[o])
        psi += (eta_psi * delta)[:, None] * b_phi[o]
        theta += (eta_theta * delta)[:, None] * b_sc[o]
        lnsig[:] = np.clip(lnsig + np.where(learn_sigma, eta_lnsig * delta * b_sl[o], 0.0),
                           lnsig_lo, lnsig_hi)

    for t in range(n_steps):
        if t - t0 >= (0 if eps_c is None else eps_c.shape[1]) or eps_c is None:
            nb = min(chunk, n_steps - t)
            eps_c = np.stack([rr.standard_normal(nb) for rr in rngs])
            uni_c = np.stack([rr.random((nb, 3)) for rr in rngs])
            t0 = t
        tl = t - t0
        eps = eps_c[pos, tl]
        u = uni_c[pos, tl]

        zf = zi.astype(float)
        kt = (np.log(k) - g.center) / g.scale
        phi = np.stack([one, kt, kt * kt, zf, kt * zf], axis=1)
        mu = theta[:, 0] + theta[:, 1] * kt + theta[:, 2] * zf
        sig = np.exp(lnsig)
        x = mu + sig * eps
        a_raw = sigmoid(x)
        a = np.clip(a_raw, A_MIN, 1.0 - A_MIN)
        nclip_a += a != a_raw
        y = ZVALS[zi] * k ** ALPHA + (1.0 - DELTA) * k
        kp_raw = a * y
        kp = np.clip(kp_raw, klo, khi)
        nclip_k += kp != kp_raw
        r = np.log((1.0 - a) * y)
        zi_n = np.where(u[:, 0] < PZ[zi, 0], 0, 1)

        ktn = (np.log(kp) - g.center) / g.scale
        zfn = zi_n.astype(float)
        phin = np.stack([one, ktn, ktn * ktn, zfn, ktn * zfn], axis=1)
        Vnext = np.einsum("ij,ij->i", psi, phin)

        j = t % B
        b_phi[j] = phi
        b_sc[j] = (eps / sig)[:, None] * np.stack([one, kt, zf], axis=1)
        b_sl[j] = eps * eps - 1.0
        b_r[j] = r
        held += 1

        if held == B:
            o = (t - B + 1) % B
            G = np.zeros(R)
            for h in range(B):
                G += gpow[h] * b_r[(o + h) % B]
            apply(o, G + gpow[B] * Vnext)
            held -= 1

        boundary = resets.any() and (t + 1) % RESET_EVERY == 0
        if boundary and B > 1 and held:
            for h in range(held, 0, -1):
                o = (t - h + 1) % B
                G = np.zeros(R)
                for q in range(h):
                    G += gpow[q] * b_r[(o + q) % B]
                apply(o, G + gpow[h] * Vnext)
            held = 0

        if boundary:
            k = np.where(resets, np.exp(g.lk_lo + u[:, 1] * lspan), kp)
            zi = np.where(resets, (u[:, 2] < 0.5).astype(np.int64), zi_n)
        else:
            k, zi = kp, zi_n

        bad = ~(np.isfinite(theta).all(1) & np.isfinite(psi).all(1) & np.isfinite(lnsig))
        if bad.any():
            theta[bad] = 0.0
            psi[bad] = 0.0
            lnsig[bad] = lnsig0
            eta_theta[bad] = eta_psi[bad] = eta_lnsig[bad] = 0.0

        if ci < len(cps) and t + 1 == cps[ci]:
            snap_t[ci], snap_l[ci], snap_p[ci] = theta, lnsig, psi
            ci += 1

    return Run(theta, lnsig, psi, np.array(cps), snap_t, snap_l, snap_p,
               nclip_a / n_steps, nclip_k / n_steps, eta_theta > 0)


def best_of_shape(key: Key, x0=(1.1, 0.0, 0.0)):
    """The best rule this actor could represent: the (theta0, theta1, theta2) minimising the same
    weighted grade.  The lab shows it so approximation error and learning error are seen
    apart -- the learner cannot beat this, and the gap to the key is the shape's own cost."""
    from scipy.optimize import minimize
    g = key.grid

    def obj(th):
        return float(grade(g, key, share_on_grid(g, np.asarray(th)))[0])

    res = minimize(obj, np.asarray(x0, float), method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": 1e-11, "maxiter": 3000})
    return res.x, float(res.fun)


# ------------------------------------------------------ the recorded protocol

# The rule is DETERMINISTIC: a = sigmoid(mu_theta(k,z)), three numbers, exactly the saving rule the
# growth model writes.  Model-free, no deterministic rule can be improved without trying something
# else, so while training we draw around it with a FIXED width -- the probe is machinery, it is not
# learned, and grade() switches it off.  Three for the rule plus five for the critic: eight numbers.
BASE = dict(eta_psi=0.02, eta_theta=2e-4, lnsig_ratio=0.2, sigma0=0.5, sigma_floor=0.05,
            learn_sigma=False)
ACTOR_STEPS = (2e-4, 1e-3, 5e-3)
SEEDS = 20
STEPS = 10 ** 6
CHECKPOINTS = (1000, 2000, 5000, 10 ** 4, 2 * 10 ** 4, 5 * 10 ** 4,
               10 ** 5, 2 * 10 ** 5, 5 * 10 ** 5, 10 ** 6)
NSTEPS_TRIED = (1, 5, 20)


def _settings():
    """The knob grid (an old Deck 3.2 frame, no longer on a slide): actor step x (learned sigma / fixed) x (resets / none).
    Row 0 is the baseline, and it is the row Deck 3.4 quotes."""
    out = []
    for step in ACTOR_STEPS:
        for learn in (True, False):
            for res in (True, False):
                out.append(dict(eta_theta=step, learn_sigma=learn, resets=res))
    out.sort(key=lambda d: (d["eta_theta"] != ACTOR_STEPS[0], not d["learn_sigma"],
                            not d["resets"]))
    return out


def experiments(path="session03_runs.npz", steps: int = STEPS, force: bool = False):
    """G2b (the knob grid) and G2c (n-step targets) in two batched runs, graded at every
    checkpoint.  n = 1 of G2c is the baseline row of G2b, to the digit, by construction."""
    from pathlib import Path as _P
    f = _P(__file__).resolve().parent / path
    if f.exists() and not force:
        return dict(np.load(f, allow_pickle=False))
    key = cached_key()
    g = key.grid
    sets = _settings()
    cps = [c for c in CHECKPOINTS if c <= steps]

    seed_of = np.tile(np.arange(SEEDS), len(sets))
    rep = lambda fn: np.repeat([fn(s) for s in sets], SEEDS)
    knob = train(key, seed_of,
                 eta_theta=rep(lambda s: s["eta_theta"]),
                 eta_psi=BASE["eta_psi"],
                 eta_lnsig=rep(lambda s: s["eta_theta"] * BASE["lnsig_ratio"]),
                 learn_sigma=rep(lambda s: s["learn_sigma"]),
                 resets=rep(lambda s: s["resets"]),
                 lnsig0=math.log(BASE["sigma0"]), lnsig_lo=math.log(BASE["sigma_floor"]),
                 n_steps=steps, checkpoints=cps)

    seed_n = np.tile(np.arange(SEEDS), 1)
    nruns = {}
    for n in NSTEPS_TRIED:
        nruns[n] = train(key, seed_n, eta_theta=BASE["eta_theta"], eta_psi=BASE["eta_psi"],
                         eta_lnsig=BASE["eta_theta"] * BASE["lnsig_ratio"],
                         learn_sigma=True, resets=True, nstep=n,
                         lnsig0=math.log(BASE["sigma0"]), lnsig_lo=math.log(BASE["sigma_floor"]),
                         n_steps=steps, checkpoints=cps)

    def graded(run):
        lived = np.zeros(run.snap_theta.shape[:2])
        flat = np.zeros_like(lived)
        cerr = np.zeros_like(lived)
        for c in range(lived.shape[0]):
            sh = share_on_grid(g, run.snap_theta[c])
            lived[c], flat[c] = grade(g, key, sh)
            phi = np.stack([np.ones(g.S), g.ktilde_s, g.ktilde_s ** 2, g.j_s.astype(float),
                            g.ktilde_s * g.j_s.astype(float)], axis=1)
            Vhat = run.snap_psi[c] @ phi.T
            w = ergodic(g, sh)
            cerr[c] = np.sqrt(np.sum(w * (Vhat - key.V) ** 2, axis=1))
        return lived, flat, cerr

    kl, kf, kc = graded(knob)
    out = dict(steps=np.array(knob.steps), key_share=key.share, key_V=key.V,
               k_s=g.k_s, j_s=g.j_s, y_s=g.y_s,
               set_eta=np.array([s["eta_theta"] for s in sets]),
               set_learn=np.array([s["learn_sigma"] for s in sets]),
               set_resets=np.array([s["resets"] for s in sets]),
               knob_lived=kl, knob_flat=kf, knob_cerr=kc,
               knob_theta=knob.snap_theta, knob_lnsig=knob.snap_lnsig,
               knob_psi=knob.snap_psi, knob_clip_a=knob.clip_a, knob_clip_k=knob.clip_k)
    for n in NSTEPS_TRIED:
        nl, nf, nc = graded(nruns[n])
        out[f"n{n}_lived"], out[f"n{n}_flat"], out[f"n{n}_cerr"] = nl, nf, nc
        out[f"n{n}_theta"] = nruns[n].snap_theta
        out[f"n{n}_lnsig"] = nruns[n].snap_lnsig
    bstar, floor = best_of_shape(key)
    out["best_shape_theta"], out["best_shape_grade"] = bstar, np.array(floor)
    np.savez_compressed(f, **out)
    return dict(np.load(f, allow_pickle=False))


NSTEP_ACTOR_STEPS = (2e-5, 5e-5, 1e-4, 2e-4)


def nstep_experiments(path="session03_nstep.npz", steps: int = STEPS, force: bool = False):
    """G2c.  An n-step target is a bigger number than a one-step target -- for n = 5 the return is
    five real rewards plus gamma^5 V, so delta is several times larger and the same actor step is no
    longer the same step.  Each n therefore gets the same grid of actor steps, and the deck reports
    each n at its own best, with the step named.  (Deck 3.3: target and step size cannot be judged
    apart.)  n = 1 at 2e-4 is the baseline row of the knob grid, to the digit."""
    from pathlib import Path as _P
    f = _P(__file__).resolve().parent / path
    if f.exists() and not force:
        return dict(np.load(f, allow_pickle=False))
    key = cached_key()
    g = key.grid
    cps = [c for c in CHECKPOINTS if c <= steps]
    seed_of = np.tile(np.arange(SEEDS), len(NSTEP_ACTOR_STEPS))
    et = np.repeat(NSTEP_ACTOR_STEPS, SEEDS)
    out = dict(steps=np.array(cps), actor_steps=np.array(NSTEP_ACTOR_STEPS),
               nvals=np.array(NSTEPS_TRIED))
    for n in NSTEPS_TRIED:
        run = train(key, seed_of, eta_theta=et, eta_psi=BASE["eta_psi"],
                                        eta_lnsig=et * BASE["lnsig_ratio"], learn_sigma=BASE["learn_sigma"],
                    resets=True, nstep=n,
                    lnsig0=math.log(BASE["sigma0"]), lnsig_lo=math.log(BASE["sigma_floor"]),
                    n_steps=steps, checkpoints=cps)
        C, R = run.snap_theta.shape[:2]
        lived = np.zeros((C, R))
        cerr = np.zeros((C, R))
        phi = np.stack([np.ones(g.S), g.ktilde_s, g.ktilde_s ** 2, g.j_s.astype(float),
                        g.ktilde_s * g.j_s.astype(float)], axis=1)
        for c in range(C):
            sh = share_on_grid(g, run.snap_theta[c])
            lived[c], _ = grade(g, key, sh)
            cerr[c] = np.sqrt(np.sum(ergodic(g, sh) * (run.snap_psi[c] @ phi.T - key.V) ** 2, axis=1))
        out[f"n{n}_lived"], out[f"n{n}_cerr"] = lived, cerr
        out[f"n{n}_lnsig"] = run.snap_lnsig
    np.savez_compressed(f, **out)
    return dict(np.load(f, allow_pickle=False))


def worked_period(key: Key | None = None, k0: float = 3.75, z0: int = 0, seed: int = 0):
    """One transition with every number on it, from the untrained start (an old Deck 3.2 frame, no longer on a slide).  Written
    to a .txt so the slide and the run cannot drift."""
    key = key or cached_key()
    g = key.grid
    rng = np.random.default_rng(seed)
    theta = np.zeros(3)
    lnsig = math.log(BASE["sigma0"])
    psi = np.array([steady_value(), 0.0, 0.0, 0.0, 0.0])
    kt = (math.log(k0) - g.center) / g.scale
    zf = float(z0)
    phi = np.array([1.0, kt, kt * kt, zf, kt * zf])
    mu = theta[0] + theta[1] * kt + theta[2] * zf
    sig = math.exp(lnsig)
    eps = float(rng.standard_normal())
    x = mu + sig * eps
    a = float(sigmoid(np.array(x)))
    y = ZVALS[z0] * k0 ** ALPHA + (1 - DELTA) * k0
    kp, c = a * y, (1 - a) * y
    r = math.log(c)
    z1 = 1 if rng.random() >= PZ[z0, 0] else 0
    ktn = (math.log(kp) - g.center) / g.scale
    zfn = float(z1)
    phin = np.array([1.0, ktn, ktn * ktn, zfn, ktn * zfn])
    V, Vn = float(psi @ phi), float(psi @ phin)
    delta = r + GAMMA * Vn - V
    eta_psi, eta_th = BASE["eta_psi"], BASE["eta_theta"]
    psi_new = psi + eta_psi * delta * phi
    sc = (eps / sig) * np.array([1.0, kt, zf])
    th_new = theta + eta_th * delta * sc
    ln_new = lnsig if not BASE["learn_sigma"] else \
        lnsig + eta_th * BASE["lnsig_ratio"] * delta * (eps * eps - 1.0)
    return dict(k0=k0, z0=ZVALS[z0], ktilde=kt, phi=phi, mu=mu, sigma=sig, eps=eps, x=x, a=a, y=y,
                kp=kp, c=c, r=r, z1=ZVALS[z1], V=V, Vn=Vn, delta=delta, psi=psi, psi_new=psi_new,
                theta=theta, theta_new=th_new, score=sc, lnsig=lnsig, lnsig_new=ln_new,
                score_lnsig=eps * eps - 1.0, eta_psi=eta_psi, eta_theta=eta_th)


# ------------------------------------------- the known-model learner (Deck 3.4)

# The critic's features, and their derivative in ktilde (the actor's gradient needs the slope).
FEATURES = {"quad5": 5, "cubic8": 8}


def phi_of(kt, zf, features: str = "quad5"):
    """(phi, dphi/dktilde), each with a trailing axis of length F, for arrays kt and zf of one shape."""
    kt = np.asarray(kt, float)
    zf = np.asarray(zf, float)
    one, zero = np.ones_like(kt), np.zeros_like(kt)
    if features == "quad5":
        phi = np.stack([one, kt, kt * kt, zf, kt * zf], axis=-1)
        dphi = np.stack([zero, one, 2.0 * kt, zero, zf], axis=-1)
    elif features == "cubic8":
        phi = np.stack([one, kt, kt ** 2, kt ** 3, zf, kt * zf, kt ** 2 * zf, kt ** 3 * zf], axis=-1)
        dphi = np.stack([zero, one, 2.0 * kt, 3.0 * kt ** 2, zero, zf, 2.0 * kt * zf,
                         3.0 * kt ** 2 * zf], axis=-1)
    else:
        raise KeyError(features)
    return phi, dphi


def _bc(v, ndim):
    """A per-run vector (R,) broadcast against arrays of shape (R, ...)."""
    return np.asarray(v, float).reshape((-1,) + (1,) * (ndim - 1))


def rule_share(theta, kt, zf):
    """The deterministic rule a = sigmoid(theta0 + theta1 ktilde + theta2 1{z=1.1}); theta is (R, 3)
    against kt, zf of shape (R, ...)."""
    th = np.asarray(theta, float)
    nd = np.asarray(kt).ndim
    mu = _bc(th[:, 0], nd) + _bc(th[:, 1], nd) * kt + _bc(th[:, 2], nd) * zf
    return np.clip(sigmoid(mu), A_MIN, 1.0 - A_MIN)


@dataclass
class KnownRun:
    theta: np.ndarray        # (R, 3)
    psi: np.ndarray          # (R, F)
    iters: np.ndarray        # checkpoint iteration counts
    snap_theta: np.ndarray   # (C, R, 3)
    snap_psi: np.ndarray     # (C, R, F)
    prefit_err: np.ndarray   # (P, R)  rms of V_psi - V^{g_0} on the band, during the prefit
    prefit_res: np.ndarray   # (P, R)  rms Bellman residual on the batch, during the prefit
    residual: np.ndarray     # (C, R)  rms Bellman residual on the batch at each checkpoint
    grad_norm: np.ndarray    # (C, R)  |grad_theta J| at each checkpoint
    finite: np.ndarray       # (R,) bool
    features: str
    seconds: float


def train_known(key: Key, seeds, *, eta_theta: float = 0.5, n_iter: int = 1000, prefit: int = 200,
                batch: int = 256, box=None, features: str = "quad5", horizon: int = 1,
                exact: bool = True, paths: int = 1, expectation: str = "sum", checkpoints=(),
                master: int = 20261008) -> KnownRun:
    """The known-model actor-critic of Deck 3.4, run for every seed at once.

    Each iteration draws `batch` states uniformly in ln k over the box (the band by default) and
    both z, forms the backup B(s) at every state, refits the critic by least squares to B, and
    moves the actor one gradient step up the one-period Bellman right-hand side with the critic
    held fixed.  Before the first actor step the critic is prefitted to the frozen starting rule
    for `prefit` iterations.

    The backup:  horizon = 1, expectation = "sum"  -- u(c) + gamma sum_{z'} P_z(z'|z) V_psi(k', z'),
                 both branches weighted, nothing drawn (the deck's main line).
                 horizon = T, exact = True           -- the 2^T shock histories enumerated and
                 weighted (the tree).
                 horizon = T, exact = False, paths=N -- N simulated runs of T periods from each state,
                 averaged (the sampled tree; T = 1, N = 1 is one drawn branch).
                 expectation = "draw"                -- one drawn z' in the backup AND in the actor's
                 gradient (the bridge to Deck 3.3; horizon must be 1).
    Only the critic's target lengthens with the horizon; the actor always climbs one period."""
    import time
    t_start = time.time()
    g = key.grid
    R = len(seeds)
    F = FEATURES[features]
    T = int(horizon)
    draw = expectation == "draw"
    if draw and (T != 1 or paths != 1):
        raise ValueError("expectation='draw' is one drawn branch: horizon = 1, paths = 1")
    if expectation not in ("sum", "draw"):
        raise KeyError(expectation)
    rngs = [np.random.default_rng([master, int(s)]) for s in seeds]
    if box is None:
        lo, hi = g.lk_lo, g.lk_hi
    else:
        lo, hi = math.log(box[0] * g.k_star), math.log(box[1] * g.k_star)
    hw = g.scale
    gpow = GAMMA ** np.arange(T + 1)
    theta = np.zeros((R, 3))
    psi = np.zeros((R, F))
    psi[:, 0] = steady_value()
    cps = sorted({int(c) for c in checkpoints} | {n_iter})
    snap_t = np.zeros((len(cps), R, 3))
    snap_p = np.zeros((len(cps), R, F))
    resid = np.zeros((len(cps), R))
    gnorm = np.zeros((len(cps), R))
    pre_err = np.zeros((prefit, R))
    pre_res = np.zeros((prefit, R))
    on_band = (g.lk >= lo - 1e-12) & (g.lk <= hi + 1e-12)
    on_band_s = np.repeat(on_band, 2)
    V_g0 = policy_evaluate(g, share_on_grid(g, theta))            # (R, S): the frozen rule's value
    phi_grid, _ = phi_of(g.ktilde_s, g.j_s.astype(float), features)   # (S, F)

    def uni(shape):
        return np.stack([rr.random(shape) for rr in rngs])          # (R,) + shape

    def draw_states():
        u = uni((batch, 2))
        return np.exp(lo + u[..., 0] * (hi - lo)), (u[..., 1] < 0.5).astype(np.int64)

    def backup(k, zi):
        """B(s) for k, zi of shape (R, N); also the first-step z' draw when one is made."""
        kc, zc = k[..., None], zi[..., None]                         # (R, N, 1)
        w = np.ones_like(kc)
        G = np.zeros_like(kc)
        z_first = None
        for t in range(T):
            a = rule_share(theta, g.tilde(kc), zc.astype(float))
            y = ZVALS[zc] * kc ** ALPHA + (1.0 - DELTA) * kc
            G = G + gpow[t] * np.log((1.0 - a) * y)
            kp = a * y
            if exact and not draw:
                M = kp.shape[-1]
                p = PZ[zc]                                            # (R, N, M, 2)
                w = (w[..., None] * p).reshape(R, batch, 2 * M)
                kc = np.repeat(kp, 2, axis=-1)
                zc = np.tile(np.array([0, 1]), M)[None, None, :] * np.ones_like(kc, dtype=np.int64)
                G = np.repeat(G, 2, axis=-1)
            else:
                if t == 0 and paths > 1:
                    kp = np.repeat(kp, paths, axis=-1)
                    zc = np.repeat(zc, paths, axis=-1)
                    G = np.repeat(G, paths, axis=-1)
                    w = np.full_like(kp, 1.0 / paths)
                u = uni(kp.shape[1:])
                zc = np.where(u < PZ[zc, 0], 0, 1)
                kc = kp
                if t == 0:
                    z_first = zc[..., 0]
        phiT, _ = phi_of(g.tilde(kc), zc.astype(float), features)
        VT = np.einsum("rf,rnmf->rnm", psi, phiT)
        return np.sum(w * (G + gpow[T] * VT), axis=-1), z_first

    def refit(k, zi, B):
        """psi <- argmin sum_n (phi(s_n) psi - B_n)^2, one small least-squares problem per run."""
        phi, _ = phi_of(g.tilde(k), zi.astype(float), features)
        new = np.empty_like(psi)
        for r in range(R):
            new[r] = np.linalg.lstsq(phi[r], B[r], rcond=None)[0]
        res = np.sqrt(np.mean((np.einsum("rf,rnf->rn", new, phi) - B) ** 2, axis=1))
        return new, res

    def actor_grad(k, zi, z_first):
        """d/dtheta of the one-period Bellman right-hand side, critic fixed, through the model."""
        kt, zf = g.tilde(k), zi.astype(float)
        a = rule_share(theta, kt, zf)
        y = ZVALS[zi] * k ** ALPHA + (1.0 - DELTA) * k
        ktp = g.tilde(a * y)
        if draw:
            _, dphi = phi_of(ktp, z_first.astype(float), features)
            slope = np.einsum("rf,rnf->rn", psi, dphi)
        else:
            slope = np.zeros_like(a)
            for zn in (0, 1):
                _, dphi = phi_of(ktp, np.full_like(ktp, float(zn)), features)
                slope += PZ[zi, zn] * np.einsum("rf,rnf->rn", psi, dphi)
        dJda = -1.0 / (1.0 - a) + GAMMA * slope / (a * hw)
        dadth = (a * (1.0 - a))[..., None] * np.stack([np.ones_like(kt), kt, zf], axis=-1)
        return np.mean(dJda[..., None] * dadth, axis=1)              # (R, 3)

    for p in range(prefit):
        k, zi = draw_states()
        B, _ = backup(k, zi)
        psi, res = refit(k, zi, B)
        pre_res[p] = res
        err = (psi @ phi_grid.T - V_g0)[:, on_band_s]
        pre_err[p] = np.sqrt(np.mean(err ** 2, axis=1))

    ci = 0
    for it in range(1, n_iter + 1):
        k, zi = draw_states()
        B, z_first = backup(k, zi)
        psi, res = refit(k, zi, B)
        grad = actor_grad(k, zi, z_first)
        theta = theta + eta_theta * grad
        bad = ~(np.isfinite(theta).all(1) & np.isfinite(psi).all(1))
        if bad.any():
            theta[bad] = 0.0
            psi[bad] = 0.0
        if ci < len(cps) and it == cps[ci]:
            snap_t[ci], snap_p[ci] = theta, psi
            resid[ci] = res
            gnorm[ci] = np.sqrt(np.sum(grad ** 2, axis=1))
            ci += 1
    finite = np.isfinite(theta).all(1) & np.isfinite(psi).all(1) & ~(theta == 0).all(1)
    return KnownRun(theta, psi, np.array(cps), snap_t, snap_p, pre_err, pre_res, resid, gnorm,
                    finite, features, time.time() - t_start)


def critic_rms(key: Key, psi, share, features: str = "quad5"):
    """rms of V_psi - V* on the key's grid, weighted by where the rule's economy lives, and
    unweighted over the band.  psi (R, F), share (R, S)."""
    g = key.grid
    phi_grid, _ = phi_of(g.ktilde_s, g.j_s.astype(float), features)
    err = psi @ phi_grid.T - key.V[None, :]
    w = ergodic(g, share)
    return np.sqrt(np.sum(w * err ** 2, axis=1)), np.sqrt(np.mean(err ** 2, axis=1))


KNOWN = dict(eta_theta=0.5, n_iter=1000, prefit=200, batch=256, features="quad5")
KNOWN_SEEDS = 20
KNOWN_CHECKPOINTS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
KNOWN_ACTOR_STEPS = (0.1, 0.5, 2.0)
KNOWN_BATCHES = (16, 64, 256)
KNOWN_BOXES = {"band": None, "half": (0.5, 1.5)}
KNOWN_EXACT_T = (1, 2, 5, 10)
KNOWN_HEAVY = dict(seeds=5, batch=64)      # the 2^10-leaf tree: 1,024 branches per state, so fewer of both
KNOWN_SAMPLED = ((5, 1), (5, 8), (5, 32), (20, 1), (20, 8), (20, 32))


def known_experiments(path="session03_known.npz", force: bool = False):
    """Every run the old Deck 3.2 quoted, graded at every checkpoint: the baseline (its grade is on
    Deck 3.4's algorithm page), the knobs, the tree enumerated and sampled, and one drawn branch."""
    from pathlib import Path as _P
    f = _P(__file__).resolve().parent / path
    if f.exists() and not force:
        return dict(np.load(f, allow_pickle=False))
    key = cached_key()
    g = key.grid
    seeds = np.arange(KNOWN_SEEDS)
    cps = KNOWN_CHECKPOINTS

    def graded(run):
        C, R = run.snap_theta.shape[:2]
        lived, flat, cerr_w, cerr_u = (np.zeros((C, R)) for _ in range(4))
        for c in range(C):
            sh = share_on_grid(g, run.snap_theta[c])
            lived[c], flat[c] = grade(g, key, sh)
            cerr_w[c], cerr_u[c] = critic_rms(key, run.snap_psi[c], sh, run.features)
        return dict(lived=lived, flat=flat, cerr=cerr_w, cerr_band=cerr_u, theta=run.snap_theta,
                    psi=run.snap_psi, residual=run.residual, grad=run.grad_norm,
                    prefit_err=run.prefit_err, prefit_res=run.prefit_res, seconds=run.seconds,
                    finite=run.finite)

    out = dict(iters=np.array(cps), key_share=key.share, key_V=key.V, k_s=g.k_s, j_s=g.j_s)

    def store(tag, run):
        for name, val in graded(run).items():
            out[f"{tag}_{name}"] = np.asarray(val)
        print(f"{tag:14s} {run.seconds:6.1f} s  lived median {np.median(out[tag + '_lived'][-1]):.5f} %"
              f"  band {np.median(out[tag + '_flat'][-1]):.5f} %  critic rms {np.median(out[tag + '_cerr'][-1]):.4f}")

    store("base", train_known(key, seeds, checkpoints=cps, **KNOWN))
    for e in KNOWN_ACTOR_STEPS:
        if e != KNOWN["eta_theta"]:
            store(f"eta{e:g}".replace(".", "p"), train_known(key, seeds, checkpoints=cps, **{**KNOWN, "eta_theta": e}))
    for b in KNOWN_BATCHES:
        if b != KNOWN["batch"]:
            store(f"batch{b}", train_known(key, seeds, checkpoints=cps, **{**KNOWN, "batch": b}))
    store("cubic", train_known(key, seeds, checkpoints=cps, **{**KNOWN, "features": "cubic8"}))
    for name, box in KNOWN_BOXES.items():
        if box is not None:
            store(f"box_{name}", train_known(key, seeds, checkpoints=cps, box=box, **KNOWN))
    for T in KNOWN_EXACT_T:
        if T != 1:
            heavy = T >= 10
            store(f"exactT{T}", train_known(key, seeds[:KNOWN_HEAVY["seeds"]] if heavy else seeds,
                                            checkpoints=cps, horizon=T, exact=True,
                                            **({**KNOWN, "batch": KNOWN_HEAVY["batch"]} if heavy else KNOWN)))
    for T, N in KNOWN_SAMPLED:
        store(f"sampT{T}N{N}", train_known(key, seeds, checkpoints=cps, horizon=T, exact=False,
                                           paths=N, **KNOWN))
    store("draw", train_known(key, seeds, checkpoints=cps, expectation="draw", **KNOWN))
    bstar, floor = best_of_shape(key)
    out["best_shape_theta"], out["best_shape_grade"] = bstar, np.array(floor)
    np.savez_compressed(f, **out)
    return dict(np.load(f, allow_pickle=False))


def worked_iteration(key: Key | None = None, k0: float = 3.75, z0: int = 0, seed: int = 0,
                     features: str = "quad5"):
    """The worked iteration (an old Deck 3.2 frame, no longer on a slide): one state, every number, after the critic's prefit and before the
    first actor step -- so the critic is the prefitted one and the rule the untrained one."""
    key = key or cached_key()
    g = key.grid
    run = train_known(key, [seed], n_iter=0, prefit=KNOWN["prefit"], batch=KNOWN["batch"],
                      eta_theta=KNOWN["eta_theta"], features=features, checkpoints=())
    psi = run.psi[0]
    theta = np.zeros(3)
    hw = g.scale
    kt = (math.log(k0) - g.center) / g.scale
    zf = float(z0)
    a = float(np.clip(sigmoid(np.array(theta[0] + theta[1] * kt + theta[2] * zf)), A_MIN, 1 - A_MIN))
    y = ZVALS[z0] * k0 ** ALPHA + (1 - DELTA) * k0
    kp, c = a * y, (1 - a) * y
    r = math.log(c)
    ktp = (math.log(kp) - g.center) / g.scale
    phi_s, _ = phi_of(np.array(kt), np.array(zf), features)
    V_s = float(psi @ phi_s)
    branch = {}
    for zn in (0, 1):
        ph, dph = phi_of(np.array(ktp), np.array(float(zn)), features)
        branch[zn] = dict(p=float(PZ[z0, zn]), z=float(ZVALS[zn]), V=float(psi @ ph),
                          slope=float(psi @ dph))
    EV = sum(branch[zn]["p"] * branch[zn]["V"] for zn in (0, 1))
    B = r + GAMMA * EV
    Eslope = sum(branch[zn]["p"] * branch[zn]["slope"] for zn in (0, 1))
    term_c = -1.0 / (1.0 - a)
    term_v = GAMMA * Eslope / (a * hw)
    dJda = term_c + term_v
    dadth = a * (1 - a) * np.array([1.0, kt, zf])
    gradient = dJda * dadth
    dp = float(key.share[2 * int(np.argmin(np.abs(g.k - k0))) + z0])
    return dict(k0=k0, z0=float(ZVALS[z0]), ktilde=kt, psi=psi, theta=theta, a=a, y=y, kp=kp, c=c,
                r=r, ktp=ktp, branch=branch, EV=EV, B=B, V_s=V_s, residual=B - V_s, Eslope=Eslope,
                term_c=term_c, term_v=term_v, dJda=dJda, dadth=dadth, gradient=gradient,
                eta_theta=KNOWN["eta_theta"], dp_share=dp, prefit=KNOWN["prefit"],
                batch=KNOWN["batch"], hw=hw)


if __name__ == "__main__":
    import time
    key = cached_key()
    g = key.grid
    t0 = time.time()
    run = train_known(key, np.arange(3), checkpoints=(10, 100), **KNOWN)
    lived, band = grade(g, key, share_on_grid(g, run.theta))
    print(f"known-model actor-critic, 3 seeds, {KNOWN['n_iter']} iterations: {time.time() - t0:.1f} s")
    print("  theta (median):", np.round(np.median(run.theta, axis=0), 4))
    print(f"  loss where the economy lives: median {np.median(lived):.5f} %, worst {lived.max():.5f} %")
    print(f"  loss over the band          : median {np.median(band):.5f} %")
    print("  critic rms (lived, band)    :", np.round(critic_rms(key, run.psi, share_on_grid(g, run.theta))[0], 4),
          np.round(critic_rms(key, run.psi, share_on_grid(g, run.theta))[1], 4))
