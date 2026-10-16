"""Session 4 lab, block E: Krusell-Smith in the sequence space, in PyTorch.

A port of the teaching notebook that accompanies Azinovic-Yang and Zemlicka, "Deep Learning in the Sequence
Space" (2025), originally written in JAX (`01_KrusellSmith_Tutorial_CPU.ipynb`). Same calibration, grid, I-spline
MPC actor, shock-history encoding, Young (2010) histogram step, Fischer-Burmeister loss, replay buffer and
training loop. What changes is only the machinery:

* `jax.vmap` over replay states becomes a leading batch dimension on every tensor;
* `jax.lax.stop_gradient` becomes `.detach()` (and the simulation runs under `torch.no_grad()`);
* `optax.chain(clip_by_global_norm(1.0), adam(lr))` becomes `clip_grad_norm_` followed by `torch.optim.Adam`;
* JAX's counter-based random keys become one `numpy` generator (buffer sampling, as in the source) and one
  `torch.Generator` (network initialisation and aggregate shocks). Runs are reproducible per seed, but not
  draw-for-draw identical to the JAX notebook.

The arithmetic is float32, as in the source: JAX runs in single precision unless x64 is switched on, and the
reference numbers (FB^2 ~ 3e-7) sit four orders of magnitude above float32 round-off.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
from scipy.interpolate import BSpline

torch.set_num_threads(4)
DTYPE = torch.float32

# ---------------------------------------------------------------------------------------------------------------
# Part 1: the economy (source cell 5)
ALPHA = 0.36        # capital share
DELTA = 0.10        # depreciation
BETA = 0.96         # discount factor
EPS_VALS_NP = np.array([0.5, 1.5])                      # idiosyncratic efficiency: low, high
PI_EPS_NP = np.array([[0.9, 0.1], [0.1, 0.9]])
Z_VALS_NP = np.array([0.93, 1.07])                      # aggregate productivity: low, high
PI_Z_NP = np.array([[0.7, 0.3], [0.3, 0.7]])
N_EPS, N_Z = 2, 2
K_SS = ((1.0 / BETA - 1.0 + DELTA) / ALPHA) ** (1.0 / (ALPHA - 1.0))   # representative-agent reference, ~4.29

# Part 2a: the log-spaced capital grid (source cell 8)
N = 100                 # intervals; N+1 grid points
K_MAX = 100.0
K_GRID_SHIFT = 0.5
BASIS_SHIFT = 0.5
_x_grid = np.linspace(np.log(K_GRID_SHIFT), np.log(K_MAX + K_GRID_SHIFT), N + 1)
K_GRID_NP = np.exp(_x_grid) - K_GRID_SHIFT
X0 = float(_x_grid[0])
DELTA_X = float((_x_grid[-1] - _x_grid[0]) / N)

# Part 2b: the I-spline basis (source cell 11)
N_INTERIOR = 56
ORDER = 4               # cubic I-splines


def build_ispline_basis(k_grid, n_interior=N_INTERIOR, order=ORDER, basis_shift=BASIS_SHIFT) -> np.ndarray:
    """B[j, n] = I_j(log(basis_shift + k_n)): each row is an increasing function of k running from 0 to 1."""
    k_arr = np.asarray(k_grid, dtype=float)
    x = np.log(basis_shift + k_arr)
    x_min, x_max = float(x[0]), float(x[-1])
    knots_int = np.linspace(x_min, x_max, n_interior + 2)[1:-1]
    knots = np.concatenate([np.repeat(x_min, order), knots_int, np.repeat(x_max, order)])
    n_basis = len(knots) - order
    basis = np.zeros((n_basis, len(k_arr)))
    for idx in range(n_basis):
        c = np.zeros(n_basis)
        c[idx] = 1.0
        bspl = BSpline(knots, c, order - 1, extrapolate=False)
        t_span = knots[idx + order] - knots[idx]
        if t_span < 1e-12:
            continue
        anti = bspl.antiderivative()
        basis[idx] = (order / t_span) * np.nan_to_num(anti(x) - anti(x_min), nan=0.0)
    return np.clip(basis, 0.0, 1.0)


ISPLINE_NP = build_ispline_basis(K_GRID_NP)
J = ISPLINE_NP.shape[0]  # 60

# Part 2c: the actor's input (source cell 13)
H = 50                   # history length
HIDDEN_DIM = 64
INPUT_DIM = H * (N_Z + 1)

# tensors
_t = lambda a: torch.tensor(np.asarray(a), dtype=DTYPE)
EPS_VALS, PI_EPS, Z_VALS, PI_Z = _t(EPS_VALS_NP), _t(PI_EPS_NP), _t(Z_VALS_NP), _t(PI_Z_NP)
K_GRID = _t(K_GRID_NP)
DK = K_GRID[1:] - K_GRID[:-1]
ISPLINE = _t(ISPLINE_NP)


def compute_prices(K, L, Z):
    """Cobb-Douglas gross return R = 1 - delta + MPK and wage w. Works on tensors or floats."""
    KL = K / L
    R = 1.0 - DELTA + ALPHA * Z * KL ** (ALPHA - 1.0)
    w = (1.0 - ALPHA) * Z * KL ** ALPHA
    return R, w


# ---------------------------------------------------------------------------------------------------------------
# The actor: aggregate shock history -> MPC schedule on the grid -> consumption on the grid
class Actor(torch.nn.Module):
    """Two tanh hidden layers. Output, per income state: one boundary-MPC logit and J basis-weight logits.

    Initialisation as in the source: He-normal for the hidden layers, N(0, 0.01^2) for the output layer, zero biases.
    """

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, generator: torch.Generator | None = None):
        super().__init__()
        out = N_EPS * (1 + J)
        g = generator
        self.W_0 = torch.nn.Parameter(np.sqrt(2.0 / input_dim) * torch.randn(input_dim, hidden_dim, generator=g, dtype=DTYPE))
        self.b_0 = torch.nn.Parameter(torch.zeros(hidden_dim, dtype=DTYPE))
        self.W_1 = torch.nn.Parameter(np.sqrt(2.0 / hidden_dim) * torch.randn(hidden_dim, hidden_dim, generator=g, dtype=DTYPE))
        self.b_1 = torch.nn.Parameter(torch.zeros(hidden_dim, dtype=DTYPE))
        self.W_2 = torch.nn.Parameter(0.01 * torch.randn(hidden_dim, out, generator=g, dtype=DTYPE))
        self.b_2 = torch.nn.Parameter(torch.zeros(out, dtype=DTYPE))

    def mpc(self, agg_input: torch.Tensor) -> torch.Tensor:
        """(batch, INPUT_DIM) -> MPC on the grid, (batch, N_EPS, N+1). Does not depend on prices."""
        h = torch.tanh(agg_input @ self.W_0 + self.b_0)
        h = torch.tanh(h @ self.W_1 + self.b_1)
        raw = (h @ self.W_2 + self.b_2).reshape(-1, N_EPS, 1 + J)
        alpha = torch.sigmoid(raw[..., 0])                                     # boundary MPC in (0, 1)
        logits = raw[..., 1:]
        augmented = torch.cat([torch.zeros_like(logits[..., :1]), logits], dim=-1)
        w_tilde = torch.softmax(augmented, dim=-1)[..., 1:]                    # phantom zero: w >= 0, sum < 1
        return alpha[..., None] * (1.0 - w_tilde @ ISPLINE)                    # decreasing in k

    def forward(self, agg_input, R, w):
        return c_from_mpc(self.mpc(agg_input), R, w)


def c_from_mpc(mpc: torch.Tensor, R, w) -> torch.Tensor:
    """Cumulate consumption: c(k_0) = MPC_0 * m(k_0), c(k_n) = c(k_{n-1}) + MPC_n * R * dk_n. R, w: (batch,)."""
    R = torch.as_tensor(R, dtype=DTYPE).reshape(-1, 1)
    w = torch.as_tensor(w, dtype=DTYPE).reshape(-1, 1)
    c0 = mpc[..., 0] * (w * EPS_VALS[None, :] + R * K_GRID[0])                 # (batch, N_EPS)
    dc = mpc[..., 1:] * (R[..., None] * DK[None, None, :])
    return torch.cat([c0[..., None], c0[..., None] + torch.cumsum(dc, dim=-1)], dim=-1)


def actor_c_grid(net: Actor, agg_input, R, w):
    """Source name kept: consumption on the grid, (batch, N_EPS, N+1)."""
    return net(agg_input, R, w)


def encode_Z_history(z_hist: torch.Tensor) -> torch.Tensor:
    """(batch, H) shock indices -> (batch, 3H): all one-hot blocks (lag-major) first, then the realised Z values."""
    z_hist = torch.as_tensor(z_hist, dtype=torch.long)
    if z_hist.dim() == 1:
        z_hist = z_hist[None]
    onehot = torch.nn.functional.one_hot(z_hist, N_Z).to(DTYPE).reshape(z_hist.shape[0], -1)
    return torch.cat([onehot, Z_VALS[z_hist]], dim=-1)


# ---------------------------------------------------------------------------------------------------------------
# Part 3: Young's histogram step (source cell 17)
def _bracket(k: torch.Tensor):
    """Closed-form bracketing interval on the log-equidistant grid, and the lower weight in k-space."""
    x = torch.log(K_GRID_SHIFT + k)
    j = torch.clamp(torch.floor((x - X0) / DELTA_X).long(), 0, N - 1)
    k_lo, k_hi = K_GRID[j], K_GRID[j + 1]
    return j, (k_hi - k) / torch.clamp(k_hi - k_lo, min=1e-12)


def interpolate_rows(vals: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation in k-space. vals (batch, N+1) on the grid; k (batch, P) -> (batch, P)."""
    x = torch.log(K_GRID_SHIFT + k)
    j = torch.clamp(torch.floor((x - X0) / DELTA_X).long(), 0, N - 1)
    w_lo = (K_GRID[j + 1] - k) / (K_GRID[j + 1] - K_GRID[j])
    return w_lo * torch.gather(vals, 1, j) + (1.0 - w_lo) * torch.gather(vals, 1, j + 1)


def distribution_aggregates(mu: torch.Tensor):
    """mu (batch, N_EPS, N+1) -> K, L (batch,), floored at 0.01 as in the source."""
    K = (mu * K_GRID).sum(dim=(-2, -1))
    L = (mu * EPS_VALS[:, None]).sum(dim=(-2, -1))
    return torch.clamp(K, min=0.01), torch.clamp(L, min=0.01)


def next_capital(c_grid, R, w):
    R = torch.as_tensor(R, dtype=DTYPE).reshape(-1, 1, 1)
    w = torch.as_tensor(w, dtype=DTYPE).reshape(-1, 1, 1)
    m_grid = w * EPS_VALS[None, :, None] + R * K_GRID[None, None, :]
    return torch.clamp(m_grid - c_grid, 0.0, float(K_GRID_NP[-1]))


def distribution_step(mu, c_grid, R, w):
    """One period of Young (2010): k' on every node, split each node's mass between the two grid points around k',
    then move income states with Pi_eps. Batched over the leading dimension."""
    b = mu.shape[0]
    k_next = next_capital(c_grid, R, w)
    j_lo, w_lo = _bracket(k_next)
    flat = lambda t: t.reshape(b * N_EPS, N + 1)
    mass = flat(mu)
    after = torch.zeros_like(mass)
    after.scatter_add_(1, flat(j_lo), mass * flat(w_lo))
    after.scatter_add_(1, flat(j_lo) + 1, mass * (1.0 - flat(w_lo)))
    return torch.einsum("ij,bik->bjk", PI_EPS, after.reshape(b, N_EPS, N + 1))


def initial_distribution(K_target=K_SS) -> torch.Tensor:
    """Uniform over income states; a Gaussian in k around K_target with sd 0.3 K_target. Shape (N_EPS, N+1)."""
    density = torch.exp(-0.5 * ((K_GRID - K_target) / (0.3 * K_target)) ** 2)
    density = density / density.sum()
    mu = torch.full((N_EPS, 1), 1.0 / N_EPS, dtype=DTYPE) * density[None, :]
    return mu / mu.sum()


def draw_Z_paths(gen: torch.Generator, T: int, z_init: torch.Tensor):
    """For each economy, the emitted path [z_t, ..., z_{t+T-1}] (starting with z_init) and the terminal z_{t+T}."""
    z = torch.as_tensor(z_init, dtype=torch.long).clone()
    stay_low = torch.tensor(PI_Z_NP[:, 0])                                   # P(z' = low | z)
    u = torch.rand(z.shape[0], T, generator=gen, dtype=torch.float64)
    out = torch.empty(z.shape[0], T, dtype=torch.long)
    for t in range(T):
        out[:, t] = z
        z = (u[:, t] >= stay_low[z]).long()
    return out, z


def make_agg_inputs(full_Z: torch.Tensor, T: int) -> torch.Tensor:
    """full_Z (batch, H+T-1) -> encoded windows for every step, (batch, T, INPUT_DIM). Window t ends at step t's shock."""
    windows = full_Z.unfold(1, H, 1)[:, :T, :]                                 # (batch, T, H)
    b = windows.shape[0]
    return encode_Z_history(windows.reshape(b * T, H)).reshape(b, T, INPUT_DIM)


@torch.no_grad()
def simulate_young(net: Actor, mu_init, Z_path, agg_inputs, return_path=False):
    """Simulate T periods with the histogram. mu_init (batch, N_EPS, N+1); Z_path (batch, T); agg_inputs (batch, T, D).

    The MPC schedule depends only on the shock history, so it is computed for all T dates in one pass; only the
    cumulation into consumption needs the date's prices. Returns the terminal histogram (and the K, L path)."""
    b, T = Z_path.shape
    mpc_all = net.mpc(agg_inputs.reshape(b * T, INPUT_DIM)).reshape(b, T, N_EPS, N + 1)
    mu = mu_init.clone()
    Ks, Ls = [], []
    for t in range(T):
        K_t, L_t = distribution_aggregates(mu)
        R_t, w_t = compute_prices(K_t, L_t, Z_VALS[Z_path[:, t]])
        c_t = c_from_mpc(mpc_all[:, t], R_t, w_t)
        if return_path:
            Ks.append(K_t)
            Ls.append(L_t)
        mu = distribution_step(mu, c_t, R_t, w_t)
    if return_path:
        return mu, torch.stack(Ks, 1), torch.stack(Ls, 1)
    return mu


# ---------------------------------------------------------------------------------------------------------------
# Part 4: the Fischer-Burmeister loss (source cell 19)
def _one_step(net: Actor, z_hist: torch.Tensor, mu: torch.Tensor, detach_prices=True):
    """Everything the loss and the diagnostics share, batched over replay states.

    z_hist (b, H) long, mu (b, N_EPS, N+1). Returns c_t, k', q (= beta E[R' u'(c')]), mu', prices."""
    b = z_hist.shape[0]
    z_t = z_hist[:, -1]
    K_t, L_t = distribution_aggregates(mu)
    K_t, L_t = K_t.detach(), L_t.detach()
    R_t, w_t = compute_prices(K_t, L_t, Z_VALS[z_t])
    c_t = net(encode_Z_history(z_hist), R_t, w_t)
    k_next = next_capital(c_t, R_t, w_t)
    mu_next = distribution_step(mu, c_t.detach(), R_t, w_t)
    K_n, L_n = distribution_aggregates(mu_next)
    K_n, L_n = K_n.detach(), L_n.detach()
    q = torch.zeros_like(c_t)
    k_flat = k_next.reshape(b, N_EPS * (N + 1))
    for z_next in range(N_Z):
        z_hist_next = torch.cat([z_hist[:, 1:], torch.full((b, 1), z_next, dtype=torch.long)], dim=1)
        R_n, w_n = compute_prices(K_n, L_n, Z_VALS[z_next])
        c_next = net(encode_Z_history(z_hist_next), R_n, w_n)                 # (b, N_EPS', N+1)
        # c'(eps', k'(eps, k)) for every next income state: (b, N_EPS', N_EPS, N+1)
        c_next_all = torch.stack([interpolate_rows(c_next[:, i], k_flat) for i in range(N_EPS)], dim=1)
        mu_u = 1.0 / torch.clamp(c_next_all.reshape(b, N_EPS, N_EPS, N + 1), min=1e-10)   # u'(c) = 1/c
        exp_uc = torch.einsum("ij,bjik->bik", PI_EPS, mu_u)
        q = q + (PI_Z[z_t, z_next] * BETA * R_n)[:, None, None] * exp_uc
    return dict(c=c_t, k_next=k_next, q=q, mu_next=mu_next, K=K_t, L=L_t, R=R_t, w=w_t, K_next=K_n)


def fb_residual(c, k_next, q):
    g = 1.0 / (q * c) - 1.0          # relative Euler gap, (c_Euler - c)/c with c_Euler = 1/q
    s = k_next / c                   # relative saving slack
    return torch.sqrt(g ** 2 + s ** 2 + 1e-12) - g - s, g


def fb_loss(net: Actor, buf_z: torch.Tensor, buf_mu: torch.Tensor) -> torch.Tensor:
    """Mean FB^2 over every (eps, k) node of every replay state in the mini-batch."""
    o = _one_step(net, buf_z, buf_mu)
    fb, _ = fb_residual(o["c"], o["k_next"], o["q"])
    return (fb ** 2).mean()


# ---------------------------------------------------------------------------------------------------------------
# Part 5: training (source cells 21 and 23)
@dataclass
class Config:
    T: int = 100                # simulation length per epoch
    N_AGG: int = 8              # economies simulated per epoch
    BUFFER_CAP: int = 128
    LR: float = 5e-5
    FB_STEPS: int = 10          # gradient steps per epoch
    FB_BATCH: int = 16
    N_EPOCHS: int = 10000
    LOG_EVERY: int = 250
    SEED: int = 0


REFERENCE = Config()                                        # the source notebook's settings
LIVE = Config(N_EPOCHS=2000, LOG_EVERY=250)                 # the lab's live default: about 4 CPU-minutes


def train(seed: int = 0, n_epochs: int | None = None, cfg: Config = REFERENCE, verbose: bool = True,
          net: Actor | None = None, buffer=None):
    """Train the actor by the source's loop: each epoch, simulate N_AGG buffer states forward T dates, push the
    terminal (history, histogram) pairs into the FIFO buffer, then take FB_STEPS Adam steps on mini-batches.

    Returns (net, history, buffer): history rows are (epoch, FB^2 of the last step, mean K in the buffer, seconds);
    buffer is (list of z-histories as int arrays, list of histograms as float arrays)."""
    n_epochs = cfg.N_EPOCHS if n_epochs is None else n_epochs
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    if net is None:
        net = Actor(generator=gen)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.LR)
    if buffer is None:
        mu0 = initial_distribution(K_SS).numpy()
        buffer_z = [np.zeros(H, dtype=np.int64) for _ in range(min(cfg.BUFFER_CAP, 16))]
        buffer_mu = [mu0.copy() for _ in range(len(buffer_z))]
    else:
        buffer_z, buffer_mu = [np.asarray(z) for z in buffer[0]], [np.asarray(m) for m in buffer[1]]
    history = []
    t0 = time.time()
    fb2 = float("nan")
    for epoch in range(n_epochs):
        idx = rng.integers(0, len(buffer_z), size=cfg.N_AGG)
        z_hists = torch.as_tensor(np.stack([buffer_z[i] for i in idx]), dtype=torch.long)
        mu_starts = torch.as_tensor(np.stack([buffer_mu[i] for i in idx]), dtype=DTYPE)
        Z_paths, Z_terms = draw_Z_paths(gen, cfg.T, z_hists[:, -1])
        full_Z = torch.cat([z_hists, Z_paths[:, 1:]], dim=1)
        mu_f = simulate_young(net, mu_starts, Z_paths, make_agg_inputs(full_Z, cfg.T)).numpy()
        Zp, Zt = Z_paths.numpy(), Z_terms.numpy()
        for i in range(cfg.N_AGG):
            full = np.concatenate([buffer_z[idx[i]], Zp[i, 1:], [Zt[i]]])
            buffer_z.append(full[-H:])
            buffer_mu.append(mu_f[i])
        while len(buffer_z) > cfg.BUFFER_CAP:
            buffer_z.pop(0)
            buffer_mu.pop(0)
        all_z = torch.as_tensor(np.stack(buffer_z), dtype=torch.long)
        all_mu = torch.as_tensor(np.stack(buffer_mu), dtype=DTYPE)
        for _ in range(cfg.FB_STEPS):
            mb = torch.as_tensor(rng.integers(0, all_z.shape[0], size=min(cfg.FB_BATCH, all_z.shape[0])))
            opt.zero_grad()
            loss = fb_loss(net, all_z[mb], all_mu[mb])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            fb2 = float(loss.detach())
        if (epoch + 1) % cfg.LOG_EVERY == 0 or epoch + 1 == n_epochs:
            buf_K = float(np.mean([np.sum(m * K_GRID_NP[None, :]) for m in buffer_mu]))
            wall = time.time() - t0
            history.append((epoch + 1, fb2, buf_K, wall))
            if verbose:
                print(f"Epoch {epoch + 1:5d} | FB² = {fb2:.2e} | K̄ = {buf_K:.2f} | {wall:.0f}s", flush=True)
    return net, history, (buffer_z, buffer_mu)


# ---------------------------------------------------------------------------------------------------------------
# Part 7: Euler diagnostics (source cell 28), on every replay state at once
@torch.no_grad()
def euler_diagnostics(net: Actor, buffer) -> dict:
    """Per-node Euler gap g and FB residual on every replay state, plus consumption, MPC and the market-clearing
    check (capital the policy chooses vs capital in next period's histogram)."""
    z = torch.as_tensor(np.stack(buffer[0]), dtype=torch.long)
    mu = torch.as_tensor(np.stack(buffer[1]), dtype=DTYPE)
    o = _one_step(net, z, mu)
    c, k_next, q = o["c"], o["k_next"], o["q"]
    c_scale = torch.clamp(c, min=1e-10)
    g = (1.0 / torch.clamp(q, min=1e-10) - c) / c_scale
    s = k_next / c_scale
    fb = torch.sqrt(g ** 2 + s ** 2 + 1e-12) - g - s
    K_policy = (mu * k_next).sum(dim=(-2, -1))
    K_dist = (o["mu_next"] * K_GRID).sum(dim=(-2, -1))
    m_grid = o["w"][:, None, None] * EPS_VALS[None, :, None] + o["R"][:, None, None] * K_GRID[None, None, :]
    c_np, m_np, R_np = c.numpy(), m_grid.numpy(), o["R"].numpy()
    mpc = np.zeros_like(c_np)
    mpc[..., 0] = c_np[..., 0] / np.maximum(m_np[..., 0], 1e-10)
    mpc[..., 1:] = np.diff(c_np, axis=-1) / (R_np[:, None, None] * DK.numpy()[None, None, :])
    abs_fb = np.abs(fb.numpy())
    return dict(K=o["K"].numpy(), c_grid=c_np, mpc_grid=mpc, g_grid=g.numpy(), fb_grid=fb.numpy(),
                abs_fb=abs_fb, market_clearing_gap=np.abs((K_policy - K_dist).numpy()))


def summarise(diag: dict) -> dict:
    """The numbers the source prints, plus the percentiles the lab gates on (|FB| over all nodes and states)."""
    a = diag["abs_fb"].ravel()
    return dict(fb2_buffer=float((diag["fb_grid"] ** 2).mean()),
                K_min=float(diag["K"].min()), K_max=float(diag["K"].max()), K_mean=float(diag["K"].mean()),
                mean_abs_fb=float(a.mean()), p50_abs_fb=float(np.percentile(a, 50)),
                p90_abs_fb=float(np.percentile(a, 90)), p99_abs_fb=float(np.percentile(a, 99)),
                max_abs_fb=float(a.max()),
                mean_mc_gap=float(diag["market_clearing_gap"].mean()),
                max_mc_gap=float(diag["market_clearing_gap"].max()))


# ---------------------------------------------------------------------------------------------------------------
# A long simulation, and plot data
@torch.no_grad()
def simulate(net: Actor, T: int = 1000, seed: int = 0, mu0=None, z0: int = 0, z_hist0=None) -> dict:
    """One economy simulated for T dates from mu0 (default: the initial Gaussian at K_ss) and a history of z0's.
    Returns the shock path and the K, L, R, w paths."""
    gen = torch.Generator().manual_seed(seed)
    mu0 = initial_distribution(K_SS) if mu0 is None else torch.as_tensor(mu0, dtype=DTYPE)
    hist = (torch.full((1, H), z0, dtype=torch.long) if z_hist0 is None
            else torch.as_tensor(z_hist0, dtype=torch.long).reshape(1, H))
    Z_path, _ = draw_Z_paths(gen, T, hist[:, -1])
    full_Z = torch.cat([hist, Z_path[:, 1:]], dim=1)
    mu_T, K, L = simulate_young(net, mu0[None], Z_path, make_agg_inputs(full_Z, T), return_path=True)
    R, w = compute_prices(K, L, Z_VALS[Z_path])
    return dict(Z=Z_path[0].numpy(), K=K[0].numpy(), L=L[0].numpy(), R=R[0].numpy(), w=w[0].numpy(),
                mu_T=mu_T[0].numpy())


@torch.no_grad()
def policy_on_reference_histories(net: Actor, buffer) -> dict:
    """Source cell 26: the learned c and MPC under an all-low and an all-high history, prices at the buffer's mean
    K and L. Returns arrays indexed [z, eps, k]."""
    buf_K = float(np.mean([np.sum(m * K_GRID_NP[None, :]) for m in buffer[1]]))
    buf_L = float(np.mean([np.sum(m * EPS_VALS_NP[:, None]) for m in buffer[1]]))
    c_out, mpc_out = np.zeros((N_Z, N_EPS, N + 1)), np.zeros((N_Z, N_EPS, N + 1))
    for zi in range(N_Z):
        R, w = compute_prices(max(buf_K, 0.01), max(buf_L, 0.01), float(Z_VALS_NP[zi]))
        c = net(encode_Z_history(torch.full((1, H), zi, dtype=torch.long)), R, w)[0].numpy()
        m = w * EPS_VALS_NP[:, None] + R * K_GRID_NP[None, :]
        c_out[zi] = c
        mpc_out[zi, :, 0] = c[:, 0] / np.maximum(m[:, 0], 1e-10)
        mpc_out[zi, :, 1:] = np.diff(c, axis=1) / (R * np.diff(K_GRID_NP))[None, :]
    return dict(k=K_GRID_NP.copy(), c=c_out, mpc=mpc_out, K=buf_K, L=buf_L)


def fb_bands(diag: dict) -> dict:
    """Source cell 30: median and 10th-90th percentile of |FB| across replay states, per income state and node."""
    a = diag["abs_fb"]
    return dict(k=K_GRID_NP.copy(), median=np.median(a, axis=0), p10=np.percentile(a, 10, axis=0),
                p90=np.percentile(a, 90, axis=0))


# ---------------------------------------------------------------------------------------------------------------
# Save / load
def save(path, net: Actor, history, buffer, cfg: Config, extra: dict | None = None) -> None:
    torch.save(dict(state_dict=net.state_dict(), history=history, buffer_z=np.stack(buffer[0]),
                    buffer_mu=np.stack(buffer[1]), config=asdict(cfg), extra=extra or {}), path)


def load(path):
    """Returns (net, history, buffer, record)."""
    rec = torch.load(path, weights_only=False)
    net = Actor()
    net.load_state_dict(rec["state_dict"])
    buffer = (list(rec["buffer_z"]), list(rec["buffer_mu"]))
    return net, rec["history"], buffer, rec
