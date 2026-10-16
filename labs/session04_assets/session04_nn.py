"""Session 4 lab -- the two torch blocks.

Block D  A neural household policy for the stationary Aiyagari economy, trained on the Euler
         equation alone. The borrowing constraint and the Euler equation are one equation through
         the Fischer-Burmeister function (Maliar, Maliar & Winant 2021); training a network on the
         residuals of equilibrium conditions is also the approach of Azinovic, Gaegauf & Scheidegger
         (2022). The trained policy is graded FIRST by the capital stock it implies,
         through Young's (2010) lottery and the invariant distribution, and only THEN by its residual.
Block E  Young's lottery written in torch, so that a derivative can pass through T steps of the
         distribution's law of motion. Autograd is checked against a central finite difference of
         the same computation and set beside what a simulated panel of households gives.

Prices are held at the classical equilibrium of session04_ha.py; no classical policy enters a loss.
Imports: numpy, torch and session04_ha.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time

if __name__ == "__main__":      # instructor runs only: keep the linear-algebra library from taking every core
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "2")
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from session04_ha import (Params, economy, euler_residuals, firm, gini, lottery, solve_equilibrium,
                          solve_household, stationary_distribution)

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE / "policy_reference.pt"
RECORDED = HERE / "recorded_nn.json"
DTYPE = torch.float64
NEAR_LIMIT = 0.05            # "at or near the constraint": assets within this distance of the limit


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------
# The household's problem at fixed prices
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Setting:
    """What the network has to know about the economy: prices, preferences, bounds."""
    r_net: float
    w: float
    transfer: float
    beta: float
    sigma: float
    a_min: float
    a_max: float
    log_e_bound: float


@dataclass
class HouseholdProblem:
    """The setting, the discretised economy, and the classical answer key used only for grading."""
    setting: Setting
    params: Params
    grid: np.ndarray
    e: np.ndarray
    Pi: np.ndarray
    pi: np.ndarray
    ap_classical: np.ndarray
    c_classical: np.ndarray
    lam_classical: np.ndarray
    K_classical: float
    r: float


def make_problem(sol: dict) -> HouseholdProblem:
    """From a solved stationary equilibrium (the dict returned by solve_equilibrium)."""
    p = sol["params"]
    setting = Setting(r_net=float(sol["r_net"]), w=float(sol["w"]), transfer=float(sol["transfer"]),
                      beta=p.beta, sigma=p.sigma, a_min=p.a_min, a_max=float(sol["grid"][-1]),
                      log_e_bound=float(np.abs(np.log(sol["e"])).max()))
    return HouseholdProblem(setting, p, sol["grid"], sol["e"], sol["Pi"], sol["pi"], sol["ap"], sol["c"],
                            sol["lam"], float(sol["K_supply"]), float(sol["r"]))


def problem_at_return(r: float, p: Params = Params()) -> HouseholdProblem:
    """The same object at a given pre-tax return, without the bisection (household and distribution only)."""
    grid, e, Pi, pi = economy(p)
    kl, w = firm(r, p)
    r_net, transfer = (1.0 - p.tau_k) * r, p.tau_k * r * kl
    ap, c, _, _ = solve_household(r_net, w, transfer, grid, e, Pi, p)
    lam, _, _ = stationary_distribution(ap, grid, Pi, pi)
    sol = dict(params=p, r=r, r_net=r_net, w=w, transfer=transfer, grid=grid, e=e, Pi=Pi, pi=pi, ap=ap, c=c,
               lam=lam, K_supply=float((lam.sum(axis=1) * grid).sum()))
    return make_problem(sol)


# --------------------------------------------------------------------------------------
# Block D -- the network, the loss, the training loop
# --------------------------------------------------------------------------------------
class SavingShare(nn.Module):
    """Feasible by construction: a' = a_min + (cash on hand - a_min) * sigmoid(g_theta(a, e))."""

    def __init__(self, setting: Setting, hidden: int = 32) -> None:
        super().__init__()
        self.setting, self.hidden = setting, hidden
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1)).to(DTYPE)
        self._log_span = math.log1p(setting.a_max - setting.a_min)

    def cash(self, a: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        s = self.setting
        return (1.0 + s.r_net) * a + s.w * e + s.transfer

    def share(self, a: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        s = self.setting
        xa = 2.0 * torch.log1p(torch.clamp(a - s.a_min, min=0.0)) / self._log_span - 1.0
        xe = torch.log(e) / s.log_e_bound
        return torch.sigmoid(self.net(torch.stack([xa, xe], dim=-1)).squeeze(-1))

    def forward(self, a: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        s = self.setting
        return s.a_min + (self.cash(a, e) - s.a_min) * self.share(a, e)

    @torch.no_grad()
    def on_grid(self, grid: np.ndarray, e: np.ndarray) -> np.ndarray:
        """Savings a'(a, e) on the classical grid, clipped to it, as a numpy array (n_a, n_e)."""
        A = torch.as_tensor(grid, dtype=DTYPE)[:, None].expand(-1, e.size)
        E = torch.as_tensor(e, dtype=DTYPE)[None, :].expand(grid.size, -1)
        return np.clip(self(A, E).numpy(), self.setting.a_min, grid[-1])


def fischer_burmeister(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Zero exactly when x1 >= 0, x2 >= 0 and x1 * x2 = 0: the two Kuhn-Tucker cases in one equation."""
    return x1 + x2 - torch.sqrt(x1 * x1 + x2 * x2)


def euler_terms(model: SavingShare, a: torch.Tensor, e: torch.Tensor, row: torch.Tensor,
                e_all: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The saving share s and the Euler ratio h = beta R E[u'(c')] / u'(c), the expectation exact.

    a, e: states (N,); row: the row of the transition matrix for each state (N, n_e); e_all: (n_e,).
    """
    s = model.setting
    share = model.share(a, e)
    coh = model.cash(a, e)
    ap = s.a_min + (coh - s.a_min) * share
    c = coh - ap
    ap_next = ap[:, None].expand(-1, e_all.numel())
    e_next = e_all[None, :].expand(a.numel(), -1)
    c_next = model.cash(ap_next, e_next) - model(ap_next, e_next)
    h = s.beta * (1.0 + s.r_net) * (row * (c[:, None] / c_next) ** s.sigma).sum(dim=1)
    return share, h


@dataclass(frozen=True)
class TrainConfig:
    start: str = "informed"        # "informed": pre-fit to the rule a' = a ; "random": torch's default start
    steps: int = 5000
    prefit_steps: int = 300
    batch_assets: int = 96         # assets per step; every productivity state is evaluated at each
    curvature: float = 3.0         # a = a_min + (a_max - a_min) U^curvature: more draws at low assets
    hidden: int = 32
    learning_rate: float = 3e-3
    seed: int = 0


LIVE = TrainConfig()
FULL = replace(LIVE, steps=30000)
LADDER = (500, 1000, 2000, 10000)      # further budgets recorded for the residual-against-capital picture


def _draw_assets(setting: Setting, n: int, curvature: float, gen: torch.Generator) -> torch.Tensor:
    return setting.a_min + (setting.a_max - setting.a_min) * torch.rand(n, generator=gen, dtype=DTYPE) ** curvature


def train_policy(prob: HouseholdProblem, cfg: TrainConfig = LIVE, threads: int = 1):
    """Minimise the mean squared Fischer-Burmeister residual. Returns (model, history, seconds)."""
    torch.set_num_threads(threads)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(10_000 + cfg.seed)
    model = SavingShare(prob.setting, cfg.hidden)
    e_all = torch.as_tensor(prob.e, dtype=DTYPE)
    Pi = torch.as_tensor(prob.Pi, dtype=DTYPE)
    n_e = e_all.numel()
    started = time.perf_counter()

    def states():
        a = _draw_assets(prob.setting, cfg.batch_assets, cfg.curvature, gen)
        a = a[:, None].expand(-1, n_e).reshape(-1)
        q = torch.arange(n_e).repeat(cfg.batch_assets)
        return a, e_all[q], Pi[q]

    if cfg.start == "informed":
        # The one rule that needs no answer key: consume your income, keep your assets.
        opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        for _ in range(cfg.prefit_steps):
            a, e, _ = states()
            target = (a - prob.setting.a_min) / (model.cash(a, e) - prob.setting.a_min)
            loss = (model.share(a, e) - target).square().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    elif cfg.start != "random":
        raise ValueError("start must be 'informed' or 'random'")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps + 1, eta_min=cfg.learning_rate / 30)
    history: list[tuple[int, float]] = []
    every = max(cfg.steps // 150, 1)
    for step in range(cfg.steps + 1):
        a, e, row = states()
        share, h = euler_terms(model, a, e, row, e_all)
        loss = fischer_burmeister(share, 1.0 - h).square().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % every == 0 or step == cfg.steps:
            history.append((step, float(loss.detach())))
    model.eval()
    return model, history, time.perf_counter() - started


# --------------------------------------------------------------------------------------
# Block D -- grading: the capital stock first, the residual second
# --------------------------------------------------------------------------------------
def implied_distribution(prob: HouseholdProblem, ap: np.ndarray) -> np.ndarray:
    """The invariant distribution a savings policy implies at the same prices (Young's lottery)."""
    try:
        lam, _, _ = stationary_distribution(ap, prob.grid, prob.Pi, prob.pi, method="solve")
        if not np.isfinite(lam).all():
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        lam, _, _ = stationary_distribution(ap, prob.grid, prob.Pi, prob.pi, method="iterate")
    return lam


def _near_limit(prob: HouseholdProblem, lam: np.ndarray) -> float:
    return float(lam[prob.grid <= prob.params.a_min + NEAR_LIMIT + 1e-15, :].sum())


def grade_policy(prob: HouseholdProblem, model: SavingShare) -> dict:
    """Aggregates first, then residuals on held-out points between the grid nodes, then the policy itself."""
    s, grid, e = prob.setting, prob.grid, prob.e
    A2 = np.repeat(grid[:, None], e.size, 1)
    ap = model.on_grid(grid, e)
    lam = implied_distribution(prob, ap)
    K = float((lam.sum(axis=1) * grid).sum())
    out = dict(
        K_classical=prob.K_classical, K_network=K, K_relative_gap=(K - prob.K_classical) / prob.K_classical,
        share_at_limit_classical=float(prob.lam_classical[0, :].sum()), share_at_limit_network=float(lam[0, :].sum()),
        share_near_limit_classical=_near_limit(prob, prob.lam_classical), share_near_limit_network=_near_limit(prob, lam),
        gini_classical=gini(A2, prob.lam_classical), gini_network=gini(A2, lam),
        mass_at_top_network=float(lam[-1, :].sum()),
    )
    # held-out points: three between each pair of neighbouring nodes, every productivity state
    a_off, res_cl, slack_cl = euler_residuals(prob.ap_classical, prob.c_classical, s.r_net, s.w, s.transfer,
                                              grid, e, prob.Pi, prob.params)
    with torch.no_grad():
        n_e = e.size
        a = torch.as_tensor(a_off, dtype=DTYPE)[:, None].expand(-1, n_e).reshape(-1)
        q = torch.arange(n_e).repeat(a_off.size)
        e_all = torch.as_tensor(e, dtype=DTYPE)
        share, h = euler_terms(model, a, e_all[q], torch.as_tensor(prob.Pi, dtype=DTYPE)[q], e_all)
        fb = fischer_burmeister(share, 1.0 - h).abs().numpy().reshape(a_off.size, n_e)
        err = np.abs(1.0 - h.numpy() ** (-1.0 / s.sigma)).reshape(a_off.size, n_e)   # |1 - c_Euler / c|
        c = (model.cash(a, e_all[q]) - model(a, e_all[q])).numpy()
    wts = np.repeat(prob.lam_classical[:-1, :] / 3.0, 3, axis=0)
    log_err = np.log10(err + 1e-16)
    out.update(
        feasible=bool((c > 0).all() and (ap >= s.a_min).all()),
        fb_log10_mean=float(np.log10(fb.mean())), fb_log10_max=float(np.log10(fb.max())),
        euler_log10_mean=float(np.log10(err[slack_cl].mean())), euler_log10_max=float(np.log10(err[slack_cl].max())),
        euler_log10_mean_weighted=float((np.where(slack_cl, log_err, 0.0) * wts).sum() / (wts * slack_cl).sum()),
        euler_log10_mean_weighted_classical=float((np.where(slack_cl, res_cl, 0.0) * wts).sum() / (wts * slack_cl).sum()),
        euler_log10_max_classical=float(np.nanmax(np.where(slack_cl, res_cl, np.nan))),
    )
    d = ap - prob.ap_classical
    m = prob.lam_classical
    signed = float((d * m).sum())
    out.update(
        policy_max_abs_error=float(np.abs(d).max()),
        policy_max_abs_error_where_mass=float(np.abs(d)[m > 1e-9].max()),
        policy_mean_abs_error_weighted=float((np.abs(d) * m).sum()),
        policy_mean_signed_error_weighted=signed,
        K_gap_per_unit_of_mean_policy_error=float((K - prob.K_classical) / signed) if abs(signed) > 1e-12 else None,
    )
    return out


def describe(g: dict) -> list[str]:
    """The grade in the order it should be read: the capital stock, then the residual, then the policy."""
    return [
        f"1  capital implied by the policy   {g['K_network']:.4f}  against the classical {g['K_classical']:.4f}  "
        f"({100 * g['K_relative_gap']:+.2f}%)",
        f"   households within {NEAR_LIMIT} of the limit  {100 * g['share_near_limit_network']:.1f}%  against "
        f"{100 * g['share_near_limit_classical']:.1f}%;  wealth Gini {g['gini_network']:.3f} against {g['gini_classical']:.3f}",
        f"2  Euler error, held-out points     log10 mean {g['euler_log10_mean']:.2f}, log10 max {g['euler_log10_max']:.2f}  "
        f"(Fischer-Burmeister residual: log10 mean {g['fb_log10_mean']:.2f})",
        f"3  policy against the classical one  largest gap {g['policy_max_abs_error']:.3f}, mass-weighted mean gap "
        f"{g['policy_mean_abs_error_weighted']:.4f}, mass-weighted signed gap {g['policy_mean_signed_error_weighted']:+.4f}",
    ]


# Gates for the informed start at the live budget: the worst of the five-seed pilot plus a margin
# (lab_gate_pilot_nn.json: capital within 2.3%, log10 mean Euler error -2.91, weighted policy gap 0.0092).
# The capital stock is gated first; a band, not a digit.
GATES_D = {"K_relative_gap_abs": 0.05, "euler_log10_mean": -2.5, "policy_mean_abs_error_weighted": 0.02,
           "mass_at_top_network": 1e-6}


def check_gates_d(g: dict, gates: dict = GATES_D) -> dict[str, bool]:
    return {
        "feasible by construction": bool(g["feasible"]),
        f"1  implied capital within {100 * gates['K_relative_gap_abs']:.0f}% of the classical K":
            abs(g["K_relative_gap"]) < gates["K_relative_gap_abs"],
        f"2  no mass piled at the top of the grid (< {gates['mass_at_top_network']:.0e})":
            g["mass_at_top_network"] < gates["mass_at_top_network"],
        f"3  mass-weighted policy error < {gates['policy_mean_abs_error_weighted']}":
            g["policy_mean_abs_error_weighted"] < gates["policy_mean_abs_error_weighted"],
        f"4  log10 mean Euler error < {gates['euler_log10_mean']}": g["euler_log10_mean"] < gates["euler_log10_mean"],
    }


# --------------------------------------------------------------------------------------
# Block D -- the fallback checkpoint
# --------------------------------------------------------------------------------------
def save_reference(model: SavingShare, extra: dict) -> str:
    meta = {"setting": asdict(model.setting), "hidden": model.hidden, **extra}
    torch.save({"state_dict": model.state_dict(), "meta": json.dumps(meta)}, CHECKPOINT)
    return file_sha256(CHECKPOINT)


def load_reference(expected_sha256: str | None = None) -> tuple[SavingShare, dict]:
    """The full-budget, informed-start policy. The hash is checked against recorded_nn.json."""
    if expected_sha256 is None and RECORDED.exists():
        expected_sha256 = json.loads(RECORDED.read_text(encoding="utf-8"))["block_D"]["checkpoint_sha256"]
    if expected_sha256 is not None and file_sha256(CHECKPOINT) != expected_sha256:
        raise RuntimeError("policy checkpoint hash does not match its record")
    blob = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    meta = json.loads(blob["meta"])
    model = SavingShare(Setting(**meta["setting"]), meta["hidden"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, meta


# --------------------------------------------------------------------------------------
# Block E -- a derivative through Young's lottery
# --------------------------------------------------------------------------------------
def lottery_torch(ap: torch.Tensor, grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Young (2010) in torch. The node index is piecewise constant; the weight is linear in a' and carries the gradient."""
    a1 = torch.clamp(ap, grid[0], grid[-1])
    j = torch.clamp(torch.searchsorted(grid, a1.detach().contiguous(), right=True) - 1, 0, grid.numel() - 2)
    return j, (a1 - grid[j]) / (grid[j + 1] - grid[j])


def push_forward_torch(lam: torch.Tensor, j: torch.Tensor, w_hi: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    """One step of the law of motion, lambda_t -> lambda_{t+1}, differentiable in the weights."""
    n_a, n_e = lam.shape
    out = torch.zeros_like(lam)
    for q in range(n_e):
        m = lam[:, q]
        tmp = torch.zeros(n_a, dtype=lam.dtype).index_add(0, j[:, q], m * (1.0 - w_hi[:, q]))
        tmp = tmp.index_add(0, j[:, q] + 1, m * w_hi[:, q])
        out = out + tmp[:, None] * Pi[q, :][None, :]
    return out


def fitted_linear_rule(prob: HouseholdProblem) -> tuple[float, float]:
    """The mass-weighted least-squares line a' = theta0 + theta1 * cash on hand through the classical policy."""
    s = prob.setting
    coh = (1.0 + s.r_net) * prob.grid[:, None] + s.w * prob.e[None, :] + s.transfer
    wt, x, y = prob.lam_classical.ravel(), coh.ravel(), prob.ap_classical.ravel()
    xm, ym = (wt * x).sum(), (wt * y).sum()
    t1 = (wt * (x - xm) * (y - ym)).sum() / (wt * (x - xm) ** 2).sum()
    return float(ym - t1 * xm), float(t1)


def aggregates_after(theta1, prob: HouseholdProblem, horizons, theta0: float, lam0=None) -> dict:
    """(K, E[a^2], mass at the borrowing limit) after T steps of the rule a' = clamp(theta0 + theta1 * cash on hand)."""
    s = prob.setting
    grid = torch.as_tensor(prob.grid, dtype=DTYPE)
    e = torch.as_tensor(prob.e, dtype=DTYPE)
    Pi = torch.as_tensor(prob.Pi, dtype=DTYPE)
    lam = torch.as_tensor(prob.lam_classical if lam0 is None else lam0, dtype=DTYPE)
    coh = (1.0 + s.r_net) * grid[:, None] + s.w * e[None, :] + s.transfer
    ap = torch.clamp(theta0 + theta1 * coh, s.a_min, s.a_max)
    j, w_hi = lottery_torch(ap, grid)
    out = {}
    for t in range(1, max(horizons) + 1):
        lam = push_forward_torch(lam, j, w_hi, Pi)
        if t in horizons:
            wealth = lam.sum(dim=1)
            out[t] = ((wealth * grid).sum(), (wealth * grid * grid).sum(), lam[0, :].sum())
    return out


def _panel(prob: HouseholdProblem, theta0: float, theta1: float, horizons, n: int, seed: int) -> dict:
    """The same experiment with n simulated households: continuous assets, drawn productivity paths."""
    s, rng = prob.setting, np.random.default_rng(seed)
    flat = rng.choice(prob.lam_classical.size, size=n, p=prob.lam_classical.ravel() / prob.lam_classical.sum())
    a, q = prob.grid[flat // prob.e.size].copy(), flat % prob.e.size
    cum = np.cumsum(prob.Pi, axis=1)
    out = {}
    for t in range(1, max(horizons) + 1):
        a = np.clip(theta0 + theta1 * ((1.0 + s.r_net) * a + s.w * prob.e[q] + s.transfer), s.a_min, s.a_max)
        q = np.minimum((rng.random(n)[:, None] > cum[q]).sum(axis=1), prob.e.size - 1)
        if t in horizons:
            out[t] = (float(a.mean()), float((a * a).mean()), float((a <= s.a_min + 1e-12).mean()))
    return out


def lottery_gradients(prob: HouseholdProblem, horizons=(1, 2, 5, 20), h: float = 1e-6, n_panel: int = 10_000,
                      replications: int = 20, h_panel: float = 1e-3, seed: int = 0) -> dict:
    """d/d theta1 of K, of E[a^2] and of the mass at the limit after T steps, three ways.

    Autograd through the T lottery steps; a central finite difference of the same histogram computation;
    and a finite difference on a simulated panel, with common random numbers and with independent draws.
    """
    started = time.perf_counter()
    names = ("K", "second_moment", "mass_at_limit")
    theta0, t1 = fitted_linear_rule(prob)
    theta1 = torch.tensor(t1, dtype=DTYPE, requires_grad=True)
    agg = aggregates_after(theta1, prob, horizons, theta0)
    with torch.no_grad():
        up = aggregates_after(torch.tensor(t1 + h, dtype=DTYPE), prob, horizons, theta0)
        dn = aggregates_after(torch.tensor(t1 - h, dtype=DTYPE), prob, horizons, theta0)
    s = prob.setting
    coh = (1.0 + s.r_net) * prob.grid[:, None] + s.w * prob.e[None, :] + s.transfer
    inside = (theta0 + t1 * coh > s.a_min) & (theta0 + t1 * coh < s.a_max)
    sims = {mode: [] for mode in ("common", "independent")}
    for rep in range(replications):
        base = 1000 * (seed + 1) + rep
        for mode in sims:
            hi = _panel(prob, theta0, t1 + h_panel, horizons, n_panel, base)
            lo = _panel(prob, theta0, t1 - h_panel, horizons, n_panel, base if mode == "common" else base + 500_000)
            sims[mode].append({T: [(hi[T][k] - lo[T][k]) / (2 * h_panel) for k in range(3)] for T in horizons})
    rows, worst = [], 0.0
    for T in horizons:
        row = dict(T=T)
        for k, name in enumerate(names):
            grad, = torch.autograd.grad(agg[T][k], theta1, retain_graph=True)
            ad, fd = float(grad), float((up[T][k] - dn[T][k]) / (2 * h))
            rel = abs(ad - fd) / max(abs(ad), abs(fd), 1e-300)
            row.update({name: float(agg[T][k].detach()), f"d_{name}_autograd": ad, f"d_{name}_finite_difference": fd,
                        f"d_{name}_relative_error": rel})
            worst = max(worst, rel)
            for mode, runs in sims.items():
                x = np.array([r[T][k] for r in runs])
                row.update({f"d_{name}_panel_{mode}_mean": float(x.mean()), f"d_{name}_panel_{mode}_sd": float(x.std(ddof=1))})
        rows.append(row)
    return dict(theta0=theta0, theta1=t1, h=h, h_panel=h_panel, n_panel=n_panel, replications=replications,
                objects=list(names), dK_T1_closed_form=float((prob.lam_classical * coh * inside).sum()), rows=rows,
                max_relative_error=worst, seconds=time.perf_counter() - started)


GATE_E = 1e-6     # autograd against the central finite difference, relative, float64 (measured: 2e-10 at h = 1e-6)


def check_gate_e(table: dict, tol: float = GATE_E) -> dict[str, bool]:
    return {f"autograd = finite difference to {tol:.0e} (relative): K, E[a^2] and the mass at the limit, every horizon":
            table["max_relative_error"] < tol,
            "T = 1: dK/dtheta is the mass-weighted mean of cash on hand (the lottery keeps the mean)":
            abs(table["rows"][0]["d_K_autograd"] - table["dK_T1_closed_form"]) < 1e-9 * abs(table["dK_T1_closed_form"])}


# --------------------------------------------------------------------------------------
# Instructor tools: runs in parallel processes, the recorded results, the checkpoint
# --------------------------------------------------------------------------------------
def _worker(job: dict) -> dict:
    """One training run in its own process. job: label, r, params, config, and what to send back."""
    prob = problem_at_return(job["r"], Params(**job["params"]))
    model, hist, secs = train_policy(prob, TrainConfig(**job["config"]))
    out = dict(label=job["label"], config=job["config"], seconds=secs, final_loss=hist[-1][1],
               grade=grade_policy(prob, model), history=hist)
    if job.get("series"):
        ap = model.on_grid(prob.grid, prob.e)
        out["ap"], out["wealth"] = ap, implied_distribution(prob, ap).sum(axis=1)
    if job.get("state"):
        out["state_dict"] = {k: v.clone() for k, v in model.state_dict().items()}
    return out


def run_jobs(jobs: list[dict], workers: int) -> list[dict]:
    import concurrent.futures as cf
    import multiprocessing as mp
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        return list(pool.map(_worker, jobs))


def _coarsen(grid: np.ndarray, wealth: np.ndarray, top: float = 40.0, bins: int = 80):
    edges = np.linspace(0.0, top, bins + 1)
    mass, _ = np.histogram(grid, bins=edges, weights=wealth)
    return edges, mass, float(wealth[grid > top].sum())


RECORD_NOTES = [
    "share_at_limit_network is near zero by construction: a sigmoid share never reaches exactly zero, so compare share_near_limit (assets within near_limit_distance of the limit)",
    "K_gap_per_unit_of_mean_policy_error is informative only when the mass-weighted signed gap is not close to zero (about 30 for the random start at the live budget)",
    "held-out points: three between each pair of neighbouring grid nodes, every productivity state; Euler statistics use the points where the classical constraint is slack",
    "euler_log10_mean is log10 of the mean error; euler_log10_mean_weighted is the mass-weighted mean of log10 errors, the statistic the classical checks report",
]


def build_records(workers: int = 12) -> dict:
    """Both starts at the live and the full budget, block E's table, the checkpoint. Run once by the instructor."""
    from dataclasses import asdict as _asdict
    sol = solve_equilibrium(Params())
    prob = make_problem(sol)
    jobs = [dict(label=f"{start}_{name}", r=prob.r, params=_asdict(prob.params), series=True,
                 state=(start == "informed" and name == "full"), config=_asdict(replace(cfg, start=start)))
            for name, cfg in (("live", LIVE), ("full", FULL)) for start in ("random", "informed")]
    # shorter budgets, grades only: how the residual and the implied capital move together as training lengthens
    jobs += [dict(label=f"{start}_{n}", r=prob.r, params=_asdict(prob.params), config=_asdict(replace(LIVE, start=start, steps=n)))
             for n in LADDER for start in ("random", "informed")]
    _, _, alone = train_policy(prob, LIVE)          # one live run with nothing of ours beside it: the timing to quote
    done = run_jobs(jobs, workers)
    runs = {r["label"]: r for r in done if "ap" in r}
    ladder = [dict(start=r["config"]["start"], steps=r["config"]["steps"], seconds=r["seconds"],
                   K_relative_gap=r["grade"]["K_relative_gap"], euler_log10_mean=r["grade"]["euler_log10_mean"],
                   euler_log10_max=r["grade"]["euler_log10_max"], fb_log10_mean=r["grade"]["fb_log10_mean"],
                   policy_mean_abs_error_weighted=r["grade"]["policy_mean_abs_error_weighted"])
              for r in sorted(done, key=lambda r: (r["config"]["start"], r["config"]["steps"]))]
    table = lottery_gradients(prob)

    best = runs["informed_full"]
    model = SavingShare(prob.setting, FULL.hidden)
    model.load_state_dict(best.pop("state_dict"))
    sha = save_reference(model, dict(config=best["config"], r=prob.r, K_classical=prob.K_classical,
                                     K_network=best["grade"]["K_network"]))
    slope = np.gradient(prob.ap_classical, prob.grid, axis=0)
    edges, mass_cl, beyond_cl = _coarsen(prob.grid, prob.lam_classical.sum(axis=1))
    states = [0, prob.e.size // 2, prob.e.size - 1]
    rnd = lambda x, k=6: np.round(np.asarray(x, dtype=float), k).tolist()
    record = dict(
        helper=Path(__file__).name, written=time.strftime("%Y-%m-%d"),
        versions=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__),
        calibration=_asdict(prob.params),
        classical=dict(r=prob.r, r_net=prob.setting.r_net, w=prob.setting.w, K=prob.K_classical,
                       clearing_residual=float(sol["clearing_residual"]), beta_R=prob.setting.beta * (1 + prob.setting.r_net),
                       share_at_limit=float(prob.lam_classical[0, :].sum()), share_near_limit=_near_limit(prob, prob.lam_classical),
                       near_limit_distance=NEAR_LIMIT, mass_weighted_slope_of_savings=float((slope * prob.lam_classical).sum())),
        block_D=dict(
            loss="mean squared Fischer-Burmeister residual of (saving share, 1 - beta R E[u'(c')]/u'(c)); expectation exact",
            network="2 inputs (scaled log(1+a), scaled log e), two tanh layers of 32, sigmoid share of cash on hand",
            informed_start="300 steps of least squares on the rule a' = a (consume income, keep assets); no classical policy",
            budgets=dict(live=_asdict(LIVE), full=_asdict(FULL)), gates=GATES_D, checkpoint_sha256=sha,
            live_run_seconds_alone=alone, notes=RECORD_NOTES,
            timing_note="the seconds stored with each run were measured with the other runs in parallel on a shared machine",
            runs={k: dict(config=v["config"], seconds=v["seconds"], final_loss=v["final_loss"], grade=v["grade"])
                  for k, v in runs.items()},
            residual_against_capital=ladder),
        block_E=dict(gate_relative_error=GATE_E, table=table),
        series=dict(
            loss={k: [[int(a), float("%.4g" % b)] for a, b in v["history"]] for k, v in runs.items()},
            policy=dict(a=rnd(prob.grid), states=states, e=rnd(prob.e[states]),
                        classical=[rnd(prob.ap_classical[:, q]) for q in states],
                        **{k: [rnd(v["ap"][:, q]) for q in states] for k, v in runs.items()}),
            wealth_density=dict(edges=rnd(edges, 3), classical=rnd(mass_cl, 7), classical_beyond=beyond_cl,
                                **{k: rnd(_coarsen(prob.grid, v["wealth"])[1], 7) for k, v in runs.items()},
                                **{k + "_beyond": _coarsen(prob.grid, v["wealth"])[2] for k, v in runs.items()})),
    )
    RECORDED.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def self_test() -> None:
    t0 = time.perf_counter()
    problem = make_problem(solve_equilibrium(Params()))
    print(f"classical equilibrium: r = {100 * problem.r:.3f}%, K = {problem.K_classical:.4f}  ({time.perf_counter() - t0:.1f} s)")
    for start in ("random", "informed"):
        net, hist, secs = train_policy(problem, replace(LIVE, start=start))
        g = grade_policy(problem, net)
        print(f"\n{start} start, {LIVE.steps} steps, {secs:.1f} s, final loss {hist[-1][1]:.2e}")
        print("\n".join("   " + line for line in describe(g)))
        if start == "informed":
            for name, ok in check_gates_d(g).items():
                print(f"   {'PASS' if ok else 'FAIL'}  {name}")
    ref, meta = load_reference()
    g = grade_policy(problem, ref)
    print(f"\nfallback checkpoint ({meta['config']['steps']} steps, informed start): capital {100 * g['K_relative_gap']:+.2f}%, "
          f"log10 mean Euler error {g['euler_log10_mean']:.2f}")
    tab = lottery_gradients(problem)
    print(f"\nblock E ({tab['seconds']:.1f} s): largest relative gap, autograd against finite difference: {tab['max_relative_error']:.1e}")
    for name, ok in check_gate_e(tab).items():
        print(f"   {'PASS' if ok else 'FAIL'}  {name}")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["record"]:
        rec = build_records()
        for label, run in rec["block_D"]["runs"].items():
            print(f"{label:<14} {run['seconds']:7.1f} s   capital {100 * run['grade']['K_relative_gap']:+8.2f}%   "
                  f"log10 mean Euler error {run['grade']['euler_log10_mean']:.2f}")
        print("checkpoint sha256", rec["block_D"]["checkpoint_sha256"])
    else:
        self_test()
