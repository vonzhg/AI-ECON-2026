"""Session 4 lab, block F: a life-cycle economy with aggregate risk, two assets and a bond market that clears
by construction, solved by a Deep Equilibrium Net with a stabilising homotopy.

A PyTorch port of the JAX/haiku notebook that accompanies Azinovic and Zemlicka, "Economics-Inspired Neural
Networks with Stabilizing Homotopies" (arXiv:2303.14802). The economy, the network, the residuals, the cloud
method and the homotopy are the notebook's. What differs is listed in DEVIATIONS below and in the lab.

Economy: N_COHORT = 7 cohorts, each living one period of 72/7 years. Cohort j wakes up with capital k_j and bonds
b_j, earns l_j * w, and chooses k'_j >= 0 (with a quadratic adjustment cost) and b'_j >= b_min. The last cohort
consumes everything. TFP follows an AR(1) in logs; depreciation falls with TFP. Bonds are in zero net supply.

State x = [tfp, k_1..k_7, b_1..b_7] (d = 15). The network maps x to 6 capital choices, 6 bond choices and one
bond price. The bond choices pass through a market-clearing layer: the mean excess demand is subtracted from
each, so that sum_j b'_j = B_supply holds exactly for any weights.

Homotopy (Homotopy vector, as in the notebook):
  b_sup  bond supply (0 throughout)
  b_min  borrowing limit on bonds
  b_msk  multiplies the network's bond output (0 = no bond trade)
  b_wgh  weight on the bond Euler residual in the loss
  b_pre  weight on the bond-price pretraining residual
Stages: capital only -> pretrain the bond price -> first bond step (b_min = target/6) -> relax b_min to its
target step by step -> final training -> fine-tuning at a tenth of the learning rate.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field, replace

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
DTYPE = torch.float32   # the notebook runs in JAX's default float32; diagnostics are re-run in float64

# ----------------------------------------------------------------------------------------------- calibration
N_COHORT = 7
N_YEARLY = 72
PLOT_AGE = 20 + np.arange(N_COHORT) * (N_YEARLY / N_COHORT)
BETA = 0.96 ** (N_YEARLY / N_COHORT)
GAMMA = 2.0
PSI_K = 0.1
B_MIN_TARGET = -0.05
ALPHA = 1 / 3
DELTA_CAP = 1 - (1.0 - 0.10) ** (N_YEARLY / N_COHORT)


def L_profile(n_cohort: int = N_COHORT) -> np.ndarray:
    """Labour endowment over the life cycle (Brumm, Kubler and Scheidegger 2017), normalised to mean one."""
    endow = np.zeros(n_cohort)
    a_is = int(n_cohort * 0.6)
    temp_l = 0.6 + (0.6 * 1.27 / (n_cohort / 2.0) ** 2) * ((n_cohort / 2.0) ** 2 - (a_is - n_cohort / 2.0) ** 2)
    for a in range(n_cohort):
        if a < a_is:
            endow[a] = 0.6 + (0.6 * 1.27 / (n_cohort / 2.0) ** 2) * ((n_cohort / 2.0) ** 2 - (a - n_cohort / 2.0) ** 2)
        else:
            endow[a] = 0.5 * temp_l
    return endow / endow.sum() * n_cohort


def Sigma_normalize(sigma_e_yearly, rho_z_yearly, n_cohort=N_COHORT, n_yearly=N_YEARLY):
    """Convert a yearly AR(1) to one model period (the notebook's rule)."""
    rho = rho_z_yearly ** (n_yearly / n_cohort)
    factor = 0.0
    if n_cohort <= n_yearly:
        for i in range(int(n_yearly / n_cohort)):
            factor += rho_z_yearly ** (2 * i)
        sigma = factor ** 0.5 * sigma_e_yearly
    else:
        for i in range(int(n_cohort / n_yearly)):
            factor += rho ** (2 * i)
        sigma = sigma_e_yearly / factor ** 0.5
    return rho, sigma


RHO_TFP, SIGMA_TFP = Sigma_normalize(0.03, 0.85)
LABOR = L_profile()

NUM_GH = 8
_eta, _omega = np.polynomial.hermite.hermgauss(NUM_GH)
GH_NODES = _eta * np.sqrt(2.0)              # standard-normal nodes
GH_WEIGHTS = _omega / np.sqrt(np.pi)        # sum to one

D_STATE = 1 + 2 * N_COHORT
N_OUT = 2 * (N_COHORT - 1) + 1
WIDTH = N_COHORT * 10

# Homotopy vector indices
H_SUP, H_MIN, H_MSK, H_WGH, H_PRE = range(5)

DEVIATIONS = [
    "PyTorch instead of JAX/haiku/optax; the network, residuals and loss are line-for-line the notebook's.",
    "Weights are drawn by PyTorch from haiku's rule (truncated normal, s.d. 1/sqrt(fan_in), zero bias), "
    "so the draws differ.",
    "Shocks are drawn from a standard normal truncated at +-3 by rejection, as jax.random.truncated_normal.",
    "Scaled for a CPU: a smaller cloud, fewer episodes and stages, and a larger learning rate (see LIVE vs SOURCE).",
    "Diagnostics are evaluated in float64 with a float64 copy of the trained network.",
]


@dataclass(frozen=True)
class Config:
    n_cloud: int = 2 ** 13
    n_minib: int = 2 ** 6
    epochs_epis: int = 3
    lr: float = 1e-5
    shock_mult: float = 2.0
    steps_cap: int = 1024          # episodes, capital only
    steps_pre: int = 1024          # episodes, bond-price pretraining
    steps_first: int = 2048        # episodes, first bond step
    homotopy_steps: int = 4096     # b_min grid points, one episode each by default
    steps_per_homotopy: int = 1
    steps_final: int = 2024
    steps_fine: int = 124
    fine_factor: float = 0.1
    mask_first: float = 0.1        # the notebook's code sets 0.1 (its comment says 0.01)
    seed: int = 0

    @property
    def steps_epoch(self) -> int:
        return int(self.epochs_epis * self.n_cloud / self.n_minib)


SOURCE = Config()
# CPU version: an eighth of the cloud and a few hundred episodes, with a larger step. The homotopy keeps its shape.
LIVE = Config(n_cloud=2 ** 10, n_minib=2 ** 6, epochs_epis=3, lr=1e-3,
              steps_cap=150, steps_pre=60, steps_first=150, homotopy_steps=32, steps_per_homotopy=3,
              steps_final=150, steps_fine=40)
# Longer recorded run (about 25 minutes on four threads): twice the cloud, about three times the episodes.
RECORD = replace(LIVE, n_cloud=2 ** 11, lr=5e-4, steps_cap=400, steps_pre=150, steps_first=400,
                 homotopy_steps=128, steps_per_homotopy=3, steps_final=600, steps_fine=150)


# ----------------------------------------------------------------------------------------------- network
class PolicyNet(nn.Module):
    """hk.Sequential([Linear(N), selu, Linear(N), selu, Linear(2*(N_cohort-1)+1)])."""

    def __init__(self, width: int = WIDTH):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D_STATE, width), nn.SELU(), nn.Linear(width, width), nn.SELU(),
                                 nn.Linear(width, N_OUT))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=1.0 / np.sqrt(m.in_features),
                                      a=-2.0 / np.sqrt(m.in_features), b=2.0 / np.sqrt(m.in_features))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def homotopy_init() -> torch.Tensor:
    return torch.zeros(5, dtype=torch.float64)


class Model:
    """Constants as tensors in one dtype; all functions are batched over the leading axis."""

    def __init__(self, dtype=DTYPE):
        self.dtype = dtype
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype)
        self.labor = t(LABOR)
        self.gh_nodes = t(GH_NODES)
        self.gh_w = t(GH_WEIGHTS)

    # --- primitives
    @staticmethod
    def fb(a, b):
        return a + b - torch.sqrt(torch.abs(a ** 2 + b ** 2) + 1e-10)

    @staticmethod
    def u_c(c):
        return torch.clamp(torch.abs(c), min=1e-10) ** (-GAMMA)

    @staticmethod
    def u_c_inv(v):
        return torch.clamp(torch.abs(v), min=1e-10) ** (-1.0 / GAMMA)

    def firm(self, tfp, K):
        L = self.labor.sum()
        Y = torch.exp(tfp) * (torch.abs(K) + 1e-10) ** ALPHA * L ** (1 - ALPHA)
        rK = ALPHA * Y / (torch.abs(K) + 1e-10)
        w = (1.0 - ALPHA) * Y / L
        return Y, rK, w

    @staticmethod
    def delta(tfp, clip=True):
        d = DELTA_CAP * 2 / (1 + torch.exp(tfp))
        return torch.clamp(d, max=1.0) if clip else d

    # --- policy with the market-clearing layer
    def S(self, x, net, H):
        pred = net(x)
        k = 0.1 * torch.nn.functional.softplus(pred[..., : N_COHORT - 1])
        b_pre = float(H[H_MSK]) * pred[..., N_COHORT - 1: 2 * (N_COHORT - 1)]
        excess = (b_pre.sum(-1, keepdim=True) - float(H[H_SUP])) / (N_COHORT - 1)
        b = b_pre - excess
        p = torch.nn.functional.softplus(pred[..., -1:])
        return k, b, p

    @staticmethod
    def split(x):
        return x[..., :1], x[..., 1:1 + N_COHORT], x[..., 1 + N_COHORT:]

    # --- residuals (E and ExtractSolution of the notebook, in one batched function)
    def residuals(self, x, net, H, extract=False):
        B = x.shape[0]
        b_min = float(H[H_MIN])
        tfp, kw, bw = self.split(x)
        K = kw.sum(-1, keepdim=True)
        _, r, w = self.firm(tfp, K)
        k_n, b_n, p_b = self.S(x, net, H)
        delta = self.delta(tfp, clip=not extract)
        zero = torch.zeros(B, 1, dtype=x.dtype)
        k_next = torch.cat([k_n, zero], -1)
        b_next = torch.cat([b_n, zero], -1)
        adj = PSI_K * (k_next - kw) ** 2
        marg_adj = 2 * PSI_K * (k_next - kw)
        c = self.labor * w + bw + (1 - delta + r) * kw - p_b * b_next - k_next - adj
        c_adj = torch.clamp(c, min=1e-4)
        mu_c = self.u_c(c_adj[:, : N_COHORT - 1])

        # t+1 at the Gauss-Hermite nodes: (B, G, .)
        tfp_n = RHO_TFP * tfp + SIGMA_TFP * self.gh_nodes.view(1, -1)             # (B, G)
        kw_n = torch.cat([zero, k_next[:, : N_COHORT - 1]], -1)                     # (B, 7)
        bw_n = torch.cat([zero, b_next[:, : N_COHORT - 1]], -1)
        G = NUM_GH
        x_n = torch.cat([tfp_n.unsqueeze(-1), kw_n.unsqueeze(1).expand(B, G, N_COHORT),
                         bw_n.unsqueeze(1).expand(B, G, N_COHORT)], -1)              # (B, G, 15)
        K_n = kw_n.sum(-1, keepdim=True)                                            # (B, 1)
        _, r_n, w_n = self.firm(tfp_n.unsqueeze(-1), K_n.unsqueeze(1))              # (B, G, 1)
        k_nn, b_nn, p_b_n = self.S(x_n.reshape(B * G, -1), net, H)
        k_nn = torch.cat([k_nn.view(B, G, -1), torch.zeros(B, G, 1, dtype=x.dtype)], -1)
        b_nn = torch.cat([b_nn.view(B, G, -1), torch.zeros(B, G, 1, dtype=x.dtype)], -1)
        p_b_n = p_b_n.view(B, G, 1)
        delta_n = self.delta(tfp_n, clip=not extract).unsqueeze(-1)
        kw_nb = kw_n.unsqueeze(1)
        adj_n = PSI_K * (k_nn - kw_nb) ** 2
        marg_adj_n = 2 * PSI_K * (k_nn - kw_nb)
        c_n = self.labor * w_n + bw_n.unsqueeze(1) + (1 - delta_n + r_n) * kw_nb - p_b_n * b_nn - k_nn - adj_n
        c_n_adj = torch.clamp(c_n, min=1e-4)
        mu_c_n = self.u_c(c_n_adj[..., 1:N_COHORT])                                # (B, G, 6)
        gw = self.gh_w.view(1, G, 1)
        expect_b = (gw * mu_c_n).sum(1)                                             # (B, 6)
        gross = torch.clamp(1 - delta_n + r_n + marg_adj_n[..., 1:N_COHORT], min=1e-4)
        expect_k = (gw * mu_c_n * gross).sum(1)
        p_b_implied = (BETA * expect_b / mu_c).max(-1, keepdim=True).values

        denom = 1 + marg_adj[:, : N_COHORT - 1]
        if not extract:
            denom = torch.clamp(denom, min=1e-4)
        c6 = c_adj[:, : N_COHORT - 1]
        fb_k_a = self.u_c_inv(BETA * expect_k / denom) / c6 - 1
        fb_k_b = k_next[:, : N_COHORT - 1] / c6
        eps_kee = self.fb(fb_k_a, fb_k_b)
        fb_b_a = self.u_c_inv(BETA * expect_b / p_b) / c6 - 1
        fb_b_b = (b_next[:, : N_COHORT - 1] - b_min) / c6
        eps_bee = self.fb(fb_b_a, fb_b_b)
        eps_cnn = c_adj - c
        eps_bpr = p_b - p_b_implied.detach()
        if extract:
            return dict(c=c, p_b=p_b, p_b_implied=p_b_implied, fb_k_a=fb_k_a, fb_b_a=fb_b_a,
                        eps_bee=eps_bee, eps_kee=eps_kee, eps_cnn=eps_cnn, k_next=k_next, b_next=b_next)
        return eps_kee, eps_bee, eps_cnn, eps_bpr

    def loss(self, x, net, H):
        eps_kee, eps_bee, eps_cnn, eps_bpr = self.residuals(x, net, H)
        return (0.5 * (eps_kee ** 2).mean() + 0.5 * float(H[H_WGH]) * (eps_bee ** 2).mean()
                + float(H[H_PRE]) * (eps_bpr ** 2).mean() + 100 * (eps_cnn ** 2).mean())

    @torch.no_grad()
    def forward_step(self, x, eps, net, H, shock_mult):
        B = x.shape[0]
        tfp, _, _ = self.split(x)
        k_n, b_n, _ = self.S(x, net, H)
        zero = torch.zeros(B, 1, dtype=x.dtype)
        tfp_n = RHO_TFP * tfp + shock_mult * SIGMA_TFP * eps.view(-1, 1)
        return torch.cat([tfp_n, zero, k_n, zero, b_n], -1)


def truncated_normal(n, gen, lo=-3.0, hi=3.0, dtype=DTYPE):
    out = torch.empty(0, dtype=dtype)
    while out.numel() < n:
        z = torch.randn(2 * n, generator=gen, dtype=dtype)
        out = torch.cat([out, z[(z > lo) & (z < hi)]])
    return out[:n]


def representative_state(n: int, dtype=DTYPE) -> torch.Tensor:
    """tfp = 0, cohorts 2..7 hold 1.5 of capital, no bonds (the notebook's representative_input)."""
    x = torch.zeros(n, D_STATE, dtype=dtype)
    x[:, 2:1 + N_COHORT] = 1.5
    return x


# ----------------------------------------------------------------------------------------------- the cloud method
@dataclass
class Solver:
    cfg: Config = LIVE
    model: Model = field(default_factory=Model)
    net: PolicyNet = None
    opt: torch.optim.Optimizer = None
    x: torch.Tensor = None
    gen: torch.Generator = None
    log: list = field(default_factory=list)        # (stage, episode, mean loss of the episode)
    stage_times: dict = field(default_factory=dict)

    def __post_init__(self):
        torch.manual_seed(self.cfg.seed)
        self.gen = torch.Generator().manual_seed(self.cfg.seed + 1)
        if self.net is None:
            self.net = PolicyNet().to(DTYPE)
        if self.opt is None:
            self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        if self.x is None:
            self.x = representative_state(self.cfg.n_cloud)

    def set_lr(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr

    def episode(self, H, stage, k):
        """Train epochs_epis passes over the cloud in minibatches, then move the cloud one period."""
        c = self.cfg
        data = self.x.repeat(c.epochs_epis, 1)          # as the notebook: copies, not reshuffled
        losses = []
        for s in range(c.steps_epoch):
            xb = data[s * c.n_minib:(s + 1) * c.n_minib]
            self.opt.zero_grad(set_to_none=True)
            loss = self.model.loss(xb, self.net, H)
            loss.backward()
            self.opt.step()
            losses.append(loss.item())
        eps = truncated_normal(c.n_cloud, self.gen)
        self.x = self.model.forward_step(self.x, eps, self.net, H, c.shock_mult)
        self.log.append((stage, k, float(np.mean(losses)), losses[0], losses[-1]))

    def cloud(self, n_episodes, H, stage, verbose=False):
        t0 = time.time()
        for k in range(n_episodes):
            self.episode(H, stage, k)
            if verbose and (k + 1) % max(1, n_episodes // 5) == 0:
                print(f"  {stage:<10s} episode {k + 1:5d}/{n_episodes}  loss {self.log[-1][2]:.3e}")
        self.stage_times[stage] = self.stage_times.get(stage, 0.0) + time.time() - t0

    # --- the homotopy, stage by stage
    def run(self, verbose=True):
        c = self.cfg
        H = homotopy_init()
        if verbose:
            print("Stage 1: capital only")
        self.cloud(c.steps_cap, H, "capital", verbose)
        self.H_cap = H.clone()

        H = H.clone(); H[H_PRE] = 1.0
        if verbose:
            print("Stage 2: pretrain the bond price")
        self.cloud(c.steps_pre, H, "pretrain", verbose)

        H = H.clone(); H[H_WGH] = 1.0; H[H_PRE] = 0.0; H[H_MSK] = c.mask_first
        b_first = B_MIN_TARGET / (N_COHORT - 1)
        H[H_MIN] = b_first
        if verbose:
            print(f"Stage 3: first bond step, b_min = {b_first:.4f}")
        self.cloud(c.steps_first, H, "first", verbose)

        if verbose:
            print(f"Stage 4: relax b_min to {B_MIN_TARGET} in {c.homotopy_steps} steps")
        for i, bm in enumerate(np.linspace(b_first, B_MIN_TARGET, c.homotopy_steps)):
            H = H.clone(); H[H_MIN] = float(bm)
            self.cloud(c.steps_per_homotopy, H, "relax", False)
        if verbose:
            print(f"  relax      done                loss {self.log[-1][2]:.3e}")

        if verbose:
            print("Stage 5: final training")
        self.cloud(c.steps_final, H, "final", verbose)
        self.set_lr(c.lr * c.fine_factor)
        if verbose:
            print("Stage 6: fine-tuning at a tenth of the step")
        self.cloud(c.steps_fine, H, "fine", verbose)
        self.H = H
        return self

    # --- saving
    def state(self):
        return dict(cfg=self.cfg.__dict__, net=self.net.state_dict(), x=self.x, H=self.H, log=self.log,
                    stage_times=self.stage_times)


def load(path) -> Solver:
    st = torch.load(path, weights_only=False)
    s = Solver(cfg=Config(**st["cfg"]))
    s.net.load_state_dict(st["net"])
    s.x, s.H, s.log, s.stage_times = st["x"], st["H"], st["log"], st["stage_times"]
    return s


# ----------------------------------------------------------------------------------------------- diagnostics
def _net64(net):
    n = copy.deepcopy(net).to(torch.float64)
    n.eval()
    return n


@torch.no_grad()
def simulate(net, H, n_paths=2 ** 10, T=200, burn=100, shock_mult=1.0, seed=7, dtype=torch.float64):
    """Fresh simulation at the true shock size (the training cloud uses shock_mult = 2)."""
    m = Model(dtype)
    gen = torch.Generator().manual_seed(seed)
    x = representative_state(n_paths, dtype)
    for _ in range(burn + T):
        x = m.forward_step(x, truncated_normal(n_paths, gen, dtype=dtype), net, H, shock_mult)
    return x


@torch.no_grad()
def diagnostics(solver_or_net, H=None, states=None, fresh=True) -> dict:
    """Euler (KKT) errors per cohort, in percent of consumption, and the bond market's excess demand."""
    if isinstance(solver_or_net, Solver):
        net, H = solver_or_net.net, solver_or_net.H
        cloud = solver_or_net.x
    else:
        net, cloud = solver_or_net, None
    n64 = _net64(net)
    m = Model(torch.float64)
    if states is None:
        states = simulate(n64, H) if fresh or cloud is None else cloud.to(torch.float64)
    out = m.residuals(states.to(torch.float64), n64, H, extract=True)
    ek = out["eps_kee"].abs().numpy() * 100
    eb = out["eps_bee"].abs().numpy() * 100
    excess = out["b_next"].sum(-1).abs().numpy()
    k = states[:, 1:1 + N_COHORT].numpy()
    b = states[:, 1 + N_COHORT:].numpy()
    return dict(
        kee_mean=ek.mean(0), kee_p99=np.percentile(ek, 99, axis=0),
        bee_mean=eb.mean(0), bee_p999=np.percentile(eb, 99.9, axis=0),
        kee_mean_all=float(ek.mean()), kee_p99_all=float(np.percentile(ek, 99)),
        bee_mean_all=float(eb.mean()), bee_p99_all=float(np.percentile(eb, 99)),
        bond_excess_max=float(excess.max()),
        cnn_max=float(out["eps_cnn"].abs().max()),
        p_b_mean=float(out["p_b"].mean()), p_b_implied_mean=float(out["p_b_implied"].mean()),
        K_mean=float(k.sum(1).mean()), k_profile=k.mean(0), b_profile=b.mean(0), c_profile=out["c"].numpy().mean(0),
        share_at_bmin=float((np.abs(out["b_next"][:, :-1].numpy() - float(H[H_MIN])) < 1e-3).mean()),
        states=states,
    )


def stage_drops(log) -> dict:
    """Mean loss over the first and last tenth of each stage's episodes (relax is one stage)."""
    out = {}
    for stage in ["capital", "pretrain", "first", "relax", "final", "fine"]:
        L = np.array([r[2] for r in log if r[0] == stage])
        if len(L) == 0:
            continue
        n = max(1, len(L) // 10)
        out[stage] = dict(first=float(L[:n].mean()), last=float(L[-n:].mean()), ratio=float(L[-n:].mean() / L[:n].mean()))
    return out


def plot(diag, log=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    ax[0].plot(PLOT_AGE, diag["k_profile"], label="capital")
    ax[0].plot(PLOT_AGE, diag["b_profile"], label="bonds")
    ax[0].axhline(0, color="grey", lw=0.5); ax[0].set_xlabel("age"); ax[0].legend(); ax[0].set_title("holdings at wake-up, mean")
    ax[1].plot(PLOT_AGE, diag["c_profile"]); ax[1].set_xlabel("age"); ax[1].set_title("consumption, mean")
    ax[2].plot(PLOT_AGE[:-1], diag["kee_mean"], label="capital, mean")
    ax[2].plot(PLOT_AGE[:-1], diag["kee_p99"], "--", label="capital, 99th pct")
    ax[2].plot(PLOT_AGE[:-1], diag["bee_mean"], label="bond, mean")
    ax[2].plot(PLOT_AGE[:-1], diag["bee_p999"], "--", label="bond, 99.9th pct")
    ax[2].set_xlabel("age"); ax[2].set_ylabel("KKT error, %"); ax[2].legend(fontsize=7); ax[2].set_title("Euler (KKT) errors")
    if log is not None:
        stages = [r[0] for r in log]
        ax[3].semilogy([r[2] for r in log], lw=0.8)
        for i in range(1, len(stages)):
            if stages[i] != stages[i - 1]:
                ax[3].axvline(i, color="grey", lw=0.5)
        ax[3].set_xlabel("episode"); ax[3].set_title("loss; lines mark homotopy stages")
    fig.tight_layout()
    return fig
