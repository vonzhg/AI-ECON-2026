"""Session 2 growth ladder: the Euler equation as the source of labels.

Three rungs, one code path.

    rung 0  log utility, delta = 1, deterministic      k' = alpha*beta*A*k^alpha
    rung 1  CRRA gamma = 2, delta = 0.025, deterministic   benchmark: time iteration
    rung 2  Brock-Mirman: log, delta = 1, AR(1) log TFP   k' = alpha*beta*z*A*k^alpha

The network is the policy, sampled states are the inputs, the log Euler residual is the
prediction error, and zero is the target.  No benchmark policy ever enters a loss.
Session 1's assets are read for the rung 0 regression test and never written.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
SESSION01_REPORT = HERE.parent / "session01_assets" / "growth_report.json"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Rung:
    name: str
    alpha: float
    beta: float
    delta: float
    gamma: float
    productivity: float = 1.0
    rho: float = 0.0
    sigma: float = 0.0  # st. dev. of the log-TFP innovation; 0 means deterministic
    steps: int = 500
    learning_rate: float = 3e-3
    batch_size: int = 256
    hidden: int = 32
    seed: int = 2026
    informed_start: bool = False  # start the saving share at its steady-state value

    @property
    def stochastic(self) -> bool:
        return self.sigma > 0.0

    @property
    def k_ss(self) -> float:
        """Deterministic steady state (z = 1)."""
        rental = 1.0 / self.beta - 1.0 + self.delta
        return (self.alpha * self.productivity / rental) ** (1.0 / (1.0 - self.alpha))

    @property
    def log_z_bound(self) -> float:
        """Three unconditional standard deviations of log z."""
        if not self.stochastic:
            return 0.0
        return 3.0 * self.sigma / math.sqrt(1.0 - self.rho**2)


RUNGS = {
    0: Rung("rung 0: log, delta=1, deterministic", 0.33, 0.96, 1.0, 1.0),
    1: Rung("rung 1: CRRA gamma=2, delta=0.025, deterministic", 0.33, 0.99, 0.025, 2.0,
            steps=3000, learning_rate=2e-3, informed_start=True),
    2: Rung("rung 2: Brock-Mirman, log, delta=1, AR(1) TFP", 0.33, 0.96, 1.0, 1.0,
            rho=0.9, sigma=0.05, steps=1500),
}

DOMAIN = (0.2, 2.0)       # training band, as multiples of k_ss (Session 1's band)
OFF_DOMAIN = (0.1, 3.0)   # reporting band outside the training support
GH_NODES = 7


def gauss_hermite(n: int = GH_NODES) -> tuple[torch.Tensor, torch.Tensor]:
    """Nodes and weights for E[h(eps)], eps ~ N(0,1): sum_j w_j h(x_j)."""
    x, w = np.polynomial.hermite.hermgauss(n)
    nodes = torch.tensor(math.sqrt(2.0) * x, dtype=torch.float32)
    weights = torch.tensor(w / math.sqrt(math.pi), dtype=torch.float32)
    return nodes, weights


class SavingPolicy(nn.Module):
    """Feasible by construction: k' = resources(k, z) * sigmoid(g_theta(normalized state))."""

    def __init__(self, rung: Rung) -> None:
        super().__init__()
        self.rung = rung
        lo, hi = (math.log(m * rung.k_ss) for m in DOMAIN)
        self.log_center, self.log_scale = (lo + hi) / 2.0, (hi - lo) / 2.0
        n_in = 2 if rung.stochastic else 1
        self.net = nn.Sequential(
            nn.Linear(n_in, rung.hidden), nn.Tanh(),
            nn.Linear(rung.hidden, rung.hidden), nn.Tanh(),
            nn.Linear(rung.hidden, 1),
        )
        if rung.informed_start:
            # Version 0 is the one state where the answer is known: at k_ss the household saves
            # k_ss out of resources(k_ss).  Start the share there, nearly flat, and let training bend it.
            share = rung.k_ss / (rung.productivity * rung.k_ss**rung.alpha + (1.0 - rung.delta) * rung.k_ss)
            with torch.no_grad():
                self.net[-1].bias.fill_(math.log(share / (1.0 - share)))
                self.net[-1].weight.mul_(0.1)

    def output(self, k: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return z * self.rung.productivity * k.pow(self.rung.alpha)

    def resources(self, k: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.output(k, z) + (1.0 - self.rung.delta) * k

    def saving_share(self, k: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = (torch.log(k) - self.log_center) / self.log_scale
        if self.rung.stochastic:
            x = torch.cat([x, torch.log(z) / self.rung.log_z_bound], dim=1)
        return torch.sigmoid(self.net(x))

    def forward(self, k: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.resources(k, z) * self.saving_share(k, z)


def expected_rhs(model: SavingPolicy, k: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return c today and beta * E[ u'(c') * (1 + f'(k') - delta) ] under the model's policy."""
    r = model.rung
    k_next = model(k, z)
    c = model.resources(k, z) - k_next
    if r.stochastic:
        nodes, weights = gauss_hermite()
        z_next = torch.exp(r.rho * torch.log(z) + r.sigma * nodes.reshape(1, -1))  # (B, J)
    else:
        weights = torch.ones(1)
        z_next = torch.ones_like(k)
    k_rep = k_next.expand(-1, z_next.shape[1])
    flat_k, flat_z = k_rep.reshape(-1, 1), z_next.reshape(-1, 1)
    c_next = model.resources(flat_k, flat_z) - model(flat_k, flat_z)
    gross = r.alpha * flat_z * r.productivity * flat_k.pow(r.alpha - 1.0) + 1.0 - r.delta
    integrand = (c_next.pow(-r.gamma) * gross).reshape(z_next.shape)
    return c, r.beta * (integrand * weights.reshape(1, -1)).sum(dim=1, keepdim=True)


def log_residual(model: SavingPolicy, k: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """log( beta E[u'(c') R'] / u'(c) ).  Unit-free; zero at the solution."""
    c, rhs = expected_rhs(model, k, z)
    return torch.log(rhs) + model.rung.gamma * torch.log(c)


def euler_error(model: SavingPolicy, k: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Judd's consumption-equivalent error |c_implied / c - 1|; report its log10."""
    c, rhs = expected_rhs(model, k, z)
    return (rhs.pow(-1.0 / model.rung.gamma) / c - 1.0).abs()


def sample_states(rung: Rung, n: int, band: tuple[float, float] = DOMAIN,
                  generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    lo, hi = (math.log(m * rung.k_ss) for m in band)
    k = torch.exp(lo + (hi - lo) * torch.rand(n, 1, generator=generator))
    if rung.stochastic:
        b = rung.log_z_bound * (band[1] / DOMAIN[1] if band != DOMAIN else 1.0)
        z = torch.exp(-b + 2.0 * b * torch.rand(n, 1, generator=generator))
    else:
        z = torch.ones(n, 1)
    return k, z


def train(rung_id: int, steps: int | None = None, informed_start: bool | None = None) -> tuple[SavingPolicy, list[tuple[int, float]], float]:
    """Train on Euler residuals only."""
    rung = RUNGS[rung_id]
    if informed_start is not None:
        rung = replace(rung, informed_start=informed_start)
    torch.manual_seed(rung.seed)
    torch.set_num_threads(1)
    model = SavingPolicy(rung)
    optimizer = torch.optim.Adam(model.parameters(), lr=rung.learning_rate)
    steps = rung.steps if steps is None else steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps + 1, eta_min=rung.learning_rate / 30)
    history: list[tuple[int, float]] = []
    started = time.perf_counter()
    for step in range(steps + 1):
        k, z = sample_states(rung, rung.batch_size)
        loss = log_residual(model, k, z).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % max(steps // 20, 1) == 0 or step == steps:
            history.append((step, float(loss.detach())))
    model.eval()
    return model, history, time.perf_counter() - started


# ---------------------------------------------------------------- benchmarks

def closed_form(rung: Rung, k: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Exact policy when utility is log and delta = 1."""
    if not (rung.gamma == 1.0 and rung.delta == 1.0):
        raise ValueError("closed form needs log utility and full depreciation")
    return rung.alpha * rung.beta * z * rung.productivity * k**rung.alpha


def time_iteration(rung: Rung, n_grid: int = 400, tol: float = 1e-10, max_iter: int = 2000) -> dict[str, object]:
    """Policy-function iteration on the deterministic Euler equation.  No value function.

    Given next period's policy g, today's k' solves u'(c) = beta u'(c'(k')) R(k') at each grid
    point; the root is bracketed and found by bisection, and g is interpolated in log k.
    """
    if rung.stochastic:
        raise ValueError("time iteration here is deterministic")
    a, b, d, gam, A = rung.alpha, rung.beta, rung.delta, rung.gamma, rung.productivity
    grid = np.exp(np.linspace(math.log(OFF_DOMAIN[0] * 0.8 * rung.k_ss),
                              math.log(OFF_DOMAIN[1] * 1.2 * rung.k_ss), n_grid))
    res = A * grid**a + (1 - d) * grid
    g = 0.5 * res if d == 1.0 else grid * 0.0 + np.minimum(grid, 0.95 * res)

    def gap(kp: np.ndarray, g_now: np.ndarray) -> np.ndarray:
        c = res - kp
        kpp = np.interp(np.log(kp), np.log(grid), g_now)
        c_next = np.maximum(A * kp**a + (1 - d) * kp - kpp, 1e-12)
        return c**(-gam) - b * c_next**(-gam) * (a * A * kp**(a - 1) + 1 - d)

    for it in range(max_iter):
        lo, hi = np.full(n_grid, 1e-8), res - 1e-10
        for _ in range(80):  # gap is increasing in k'
            mid = 0.5 * (lo + hi)
            positive = gap(mid, g) > 0
            hi, lo = np.where(positive, mid, hi), np.where(positive, lo, mid)
        g_new = 0.5 * (lo + hi)
        change = float(np.max(np.abs(g_new - g) / g))
        g = g_new
        if change < tol:
            break
    return {"grid": grid, "policy": g, "iterations": it + 1, "sup_change": change}


def benchmark_policy(rung_id: int, k: np.ndarray, z: np.ndarray, cache: dict | None = None) -> np.ndarray:
    rung = RUNGS[rung_id]
    if rung_id == 1:
        ti = cache if cache is not None else time_iteration(rung)
        return np.interp(np.log(k), np.log(ti["grid"]), ti["policy"])
    return closed_form(rung, k, z)


# ---------------------------------------------------------------- validation

def simulate_to_rest(model: SavingPolicy, periods: int = 3000) -> float:
    """Deterministic path (z = 1) from 0.5 k_ss; returns the terminal capital stock."""
    k = torch.tensor([[0.5 * model.rung.k_ss]])
    z = torch.ones(1, 1)
    with torch.no_grad():
        for _ in range(periods):
            k = model(k, z)
    return float(k)


def validate(rung_id: int, model: SavingPolicy, n: int = 4000, ti_cache: dict | None = None) -> dict[str, float]:
    """Held-out and off-domain diagnostics; states come from a generator training never used."""
    rung = model.rung
    gen = torch.Generator().manual_seed(777)
    out: dict[str, float] = {}
    with torch.no_grad():
        for label, band in (("held_out", DOMAIN), ("off_domain", OFF_DOMAIN)):
            k, z = sample_states(rung, n, band, gen)
            if label == "off_domain":  # keep only the states outside the training band
                keep = ((k < DOMAIN[0] * rung.k_ss) | (k > DOMAIN[1] * rung.k_ss)).squeeze()
                k, z = k[keep], z[keep]
            k_next = model(k, z)
            c = model.resources(k, z) - k_next
            err = euler_error(model, k, z)
            exact = benchmark_policy(rung_id, k.numpy(), z.numpy(), ti_cache)
            rel = np.abs(k_next.numpy() - exact) / exact
            out[f"{label}_feasible"] = bool(((c > 0) & (k_next > 0)).all())
            out[f"{label}_log10_max_euler_error"] = float(torch.log10(err.max()))
            out[f"{label}_log10_mean_euler_error"] = float(torch.log10(err.mean()))
            out[f"{label}_max_relative_policy_error"] = float(rel.max())
            out[f"{label}_median_relative_policy_error"] = float(np.median(rel))
    k_end = simulate_to_rest(model)
    out["k_ss_analytical"] = rung.k_ss
    out["k_ss_simulated"] = k_end
    out["k_ss_relative_gap"] = abs(k_end - rung.k_ss) / rung.k_ss
    return out


GATES = {
    "feasible": lambda v: v["held_out_feasible"] and v["off_domain_feasible"],
    "max policy error < 1% (held-out)": lambda v: v["held_out_max_relative_policy_error"] < 0.01,
    "steady state within 1%": lambda v: v["k_ss_relative_gap"] < 0.01,
}


def check_gates(v: dict[str, float]) -> dict[str, bool]:
    return {name: bool(test(v)) for name, test in GATES.items()}


def session01_regression(v: dict[str, float]) -> dict[str, object]:
    """Rung 0 is Session 1's model: same calibration, band, network, optimizer settings."""
    ref = json.loads(SESSION01_REPORT.read_text(encoding="utf-8"))
    return {
        "session01_max_relative_error": ref["max_relative_error"],
        "session02_max_relative_error": v["held_out_max_relative_policy_error"],
        "both_below_one_percent": ref["max_relative_error"] < 0.01
        and v["held_out_max_relative_policy_error"] < 0.01,
        "calibration_matches": (ref["model"]["alpha"], ref["model"]["beta"]) == (RUNGS[0].alpha, RUNGS[0].beta),
    }


# ---------------------------------------------------------------- fallbacks

def checkpoint_path(rung_id: int) -> Path:
    return HERE / f"growth_rung{rung_id}_reference.pt"


def load_reference(rung_id: int) -> tuple[SavingPolicy, dict[str, object]]:
    report = json.loads((HERE / "growth_ladder_report.json").read_text(encoding="utf-8"))
    entry = report["rungs"][str(rung_id)]
    if file_sha256(checkpoint_path(rung_id)) != entry["checkpoint_sha256"]:
        raise RuntimeError("growth checkpoint hash does not match its report")
    model = SavingPolicy(RUNGS[rung_id])
    model.load_state_dict(torch.load(checkpoint_path(rung_id), map_location="cpu", weights_only=True))
    model.eval()
    return model, entry


def build_references() -> dict[str, object]:
    """Train all rungs, validate, write checkpoints and the report.  Run once by the instructor."""
    report: dict[str, object] = {"helper": Path(__file__).name, "domain": DOMAIN,
                                 "off_domain": OFF_DOMAIN, "gauss_hermite_nodes": GH_NODES, "rungs": {}}
    ti = time_iteration(RUNGS[1])
    for rung_id, rung in RUNGS.items():
        model, history, seconds = train(rung_id)
        v = validate(rung_id, model, ti_cache=ti)
        torch.save(model.state_dict(), checkpoint_path(rung_id))
        entry = {"calibration": asdict(rung), "elapsed_seconds": seconds,
                 "final_training_loss": history[-1][1], "validation": v, "gates": check_gates(v),
                 "checkpoint_sha256": file_sha256(checkpoint_path(rung_id))}
        if rung_id == 0:
            entry["session01_regression"] = session01_regression(v)
        if rung_id == 1:
            cold, _, _ = train(1, informed_start=False)
            entry["uninformed_start"] = validate(1, cold, ti_cache=ti)
            entry["time_iteration"] = {"iterations": ti["iterations"], "sup_change": ti["sup_change"],
                                       "grid_points": len(ti["grid"])}
        report["rungs"][str(rung_id)] = entry
    (HERE / "growth_ladder_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    rep = build_references()
    for rid, e in rep["rungs"].items():
        print(rid, f"{e['elapsed_seconds']:.1f}s", e["gates"])
        for key, val in e["validation"].items():
            print("   ", key, val)
