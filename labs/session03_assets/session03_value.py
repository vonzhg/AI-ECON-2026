"""Session 3, block C: deep value iteration on Session 2's continuous growth model.

Session 2 solved this model on the Euler road: one network (the policy), trained on Euler residuals.
This file takes the Bellman road on the same rungs, with the same policy class and the same validators:

    rung 0  log utility, delta = 1          benchmark: closed forms  k' = alpha*beta*A*k^alpha,  V = a + b ln k
    rung 1  CRRA gamma = 2, delta = 0.025   benchmark: Session 2's time iteration, and its value by policy evaluation

Two networks.  The actor is Session 2's SavingPolicy (feasible by construction).  The critic is a value
network V_phi(k).  Each step draws fresh states and rolls the actor forward n periods through the *known*
law of motion:

    J_n(k) = sum_{t<n} beta^t u(c_t) + beta^n V_phi(k_n)

    actor step   raise  mean J_n(k)               with the critic held fixed
    critic step  lower  mean (V_phi(k) - J_n(k))^2  with the target detached

n = 1 is one-step fitted value iteration; larger n leans less on the critic (the tail weighs beta^n).
This is actor-critic with a known model: no sampling noise, no exploration, the same two moving targets.
No benchmark ever enters a loss.  Session 2's assets are imported and never written.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
S2 = HERE.parent / "session02_assets"
if str(S2) not in sys.path:
    sys.path.insert(0, str(S2))
import session02_growth as g2  # noqa: E402

DOMAIN, OFF_DOMAIN = g2.DOMAIN, g2.OFF_DOMAIN


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class ValueConfig:
    rung_id: int
    n_step: int = 10
    steps: int = 1500
    lr_actor: float = 3e-3
    lr_critic: float = 1e-2
    batch_size: int = 256
    hidden: int = 32
    seed: int = 2026
    informed_start: bool = True   # actor: Session 2's steady-state share; critic: the steady-state value


CONFIGS = {
    0: ValueConfig(0, n_step=10, steps=1500),
    1: ValueConfig(1, n_step=80, steps=3000, lr_actor=2e-3, lr_critic=5e-3),
}


# ---------------------------------------------------------------- model pieces

def utility(c: torch.Tensor, crra: float) -> torch.Tensor:
    """Session 2's Rung calls the curvature `gamma`; Session 3 writes gamma for the discount, so the
    curvature is `crra` everywhere in this file and on the slide that shows it."""
    return torch.log(c) if crra == 1.0 else c.pow(1.0 - crra) / (1.0 - crra)


def steady_state_value(rung: g2.Rung) -> float:
    """u(c*) / (1 - beta): the value of staying at the steady state for ever."""
    c_ss = rung.productivity * rung.k_ss**rung.alpha - rung.delta * rung.k_ss
    u = math.log(c_ss) if rung.gamma == 1.0 else c_ss ** (1.0 - rung.gamma) / (1.0 - rung.gamma)
    return u / (1.0 - rung.beta)


def value_scale(rung: g2.Rung) -> float:
    """dV/dlog k at the steady state is k* u'(c*) / beta (envelope condition); times the half-width of the
    log band it gives the size of V's variation over the training domain."""
    c_ss = rung.productivity * rung.k_ss**rung.alpha - rung.delta * rung.k_ss
    half = 0.5 * (math.log(DOMAIN[1]) - math.log(DOMAIN[0]))
    return rung.k_ss * c_ss ** (-rung.gamma) / rung.beta * half


class ValueNet(nn.Module):
    """V_phi(k) = V_ss + scale * net(normalised log k).  The network learns an O(1) shape; the level and
    the scale come from the one state where the answer is known."""

    def __init__(self, rung: g2.Rung, hidden: int = 32, informed_start: bool = True) -> None:
        super().__init__()
        lo, hi = (math.log(m * rung.k_ss) for m in DOMAIN)
        self.log_center, self.log_scale = (lo + hi) / 2.0, (hi - lo) / 2.0
        self.level = steady_state_value(rung) if informed_start else 0.0
        self.scale = value_scale(rung)
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        x = (torch.log(k) - self.log_center) / self.log_scale
        return self.level + self.scale * self.net(x)


def rollout_objective(actor: g2.SavingPolicy, critic: ValueNet, k: torch.Tensor, n_step: int) -> torch.Tensor:
    """J_n(k): n periods of utility under the actor, then the critic's value of where it lands."""
    rung, z = actor.rung, torch.ones_like(k)
    crra, beta = rung.gamma, rung.beta          # curvature, discount -- see utility() on the names
    total, disc = torch.zeros_like(k), 1.0
    for _ in range(n_step):
        k_next = actor(k, z)
        total = total + disc * utility(actor.resources(k, z) - k_next, crra)
        disc *= beta
        k = k_next
    return total + disc * critic(k)


def train(cfg: ValueConfig) -> tuple[g2.SavingPolicy, ValueNet, list[tuple[int, float, float]], float]:
    rung = g2.RUNGS[cfg.rung_id]
    if rung.stochastic:
        raise ValueError("block C runs the deterministic rungs 0 and 1")
    rung = g2.replace(rung, informed_start=cfg.informed_start, hidden=cfg.hidden)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(1)
    actor, critic = g2.SavingPolicy(rung), ValueNet(rung, cfg.hidden, cfg.informed_start)
    opt_a = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    opt_c = torch.optim.Adam(critic.parameters(), lr=cfg.lr_critic)
    sched = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=cfg.steps + 1, eta_min=lr / 30)
             for o, lr in ((opt_a, cfg.lr_actor), (opt_c, cfg.lr_critic))]
    scale = critic.scale
    history: list[tuple[int, float, float]] = []
    started = time.perf_counter()
    for step in range(cfg.steps + 1):
        k, _ = g2.sample_states(rung, cfg.batch_size)
        # actor step: climb J with the critic held fixed (its gradients are discarded)
        loss_actor = -rollout_objective(actor, critic, k, cfg.n_step).mean() / scale
        opt_a.zero_grad(set_to_none=True)
        loss_actor.backward()
        opt_a.step()
        # critic step: regress V(k) on the detached n-step target under the updated actor
        with torch.no_grad():
            target = rollout_objective(actor, critic, k, cfg.n_step)
        loss_critic = ((critic(k) - target) / scale).square().mean()
        opt_c.zero_grad(set_to_none=True)
        loss_critic.backward()
        opt_c.step()
        for s in sched:
            s.step()
        if step % max(cfg.steps // 20, 1) == 0 or step == cfg.steps:
            history.append((step, float(loss_actor.detach()), float(loss_critic.detach())))
    actor.eval(), critic.eval()
    return actor, critic, history, time.perf_counter() - started


# ---------------------------------------------------------------- benchmarks

def closed_form_value(rung: g2.Rung, k: np.ndarray) -> np.ndarray:
    """Exact V(k) = a + b ln k for log utility, delta = 1, z = 1 (Brock and Mirman 1972)."""
    if not (rung.gamma == 1.0 and rung.delta == 1.0):
        raise ValueError("closed form needs log utility and full depreciation")
    ab, A = rung.alpha * rung.beta, rung.productivity
    b = rung.alpha / (1.0 - ab)
    a = (math.log((1.0 - ab) * A) + ab / (1.0 - ab) * math.log(ab * A)) / (1.0 - rung.beta)
    return a + b * np.log(k)


def value_of_policy_on_grid(rung: g2.Rung, grid: np.ndarray, policy: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """V(k) = u(c) + beta V(g(k)) on a grid, by iteration with interpolation in log k."""
    c = rung.productivity * grid**rung.alpha + (1.0 - rung.delta) * grid - policy
    u = np.log(c) if rung.gamma == 1.0 else c ** (1.0 - rung.gamma) / (1.0 - rung.gamma)
    V = u / (1.0 - rung.beta)
    for _ in range(20000):
        V_new = u + rung.beta * np.interp(np.log(policy), np.log(grid), V)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new
    return V


def benchmarks(rung_id: int) -> dict[str, object]:
    """Exact (rung 0) or classical (rung 1) policy and value, as callables of k."""
    rung = g2.RUNGS[rung_id]
    if rung_id == 0:
        return {"policy": lambda k: g2.closed_form(rung, k, np.ones_like(k)),
                "value": lambda k: closed_form_value(rung, k), "kind": "closed form"}
    ti = g2.time_iteration(rung)
    grid, pol = ti["grid"], ti["policy"]
    V = value_of_policy_on_grid(rung, grid, pol)
    return {"policy": lambda k: np.interp(np.log(k), np.log(grid), pol),
            "value": lambda k: np.interp(np.log(k), np.log(grid), V), "kind": "time iteration"}


def consumption_equivalent(rung: g2.Rung, V_hat: np.ndarray, V_true: np.ndarray) -> np.ndarray:
    """Permanent consumption change that would turn V_true into V_hat, as a fraction."""
    if rung.gamma == 1.0:
        return np.exp((1.0 - rung.beta) * (V_hat - V_true)) - 1.0
    return (V_hat / V_true) ** (1.0 / (1.0 - rung.gamma)) - 1.0


# ---------------------------------------------------------------- validation

def validate(rung_id: int, actor: g2.SavingPolicy, critic: ValueNet, n: int = 4000,
             bench: dict | None = None) -> dict[str, float]:
    rung = actor.rung
    bench = benchmarks(rung_id) if bench is None else bench
    gen = torch.Generator().manual_seed(777)
    out: dict[str, float] = {}
    with torch.no_grad():
        for label, band in (("held_out", DOMAIN), ("off_domain", OFF_DOMAIN)):
            k, z = g2.sample_states(rung, n, band, gen)
            if label == "off_domain":
                keep = ((k < DOMAIN[0] * rung.k_ss) | (k > DOMAIN[1] * rung.k_ss)).squeeze()
                k, z = k[keep], z[keep]
            k_next = actor(k, z)
            c = actor.resources(k, z) - k_next
            kn = k.numpy().ravel()
            rel = np.abs(k_next.numpy().ravel() - bench["policy"](kn)) / bench["policy"](kn)
            ce = np.abs(consumption_equivalent(rung, critic(k).numpy().ravel(), bench["value"](kn)))
            # Bellman residual under the learned actor, in consumption units: divide by u'(c) c
            resid = (critic(k) - utility(c, rung.gamma) - rung.beta * critic(k_next)).abs() / c.pow(1.0 - rung.gamma)
            out[f"{label}_feasible"] = bool(((c > 0) & (k_next > 0)).all())
            out[f"{label}_max_relative_policy_error"] = float(rel.max())
            out[f"{label}_max_value_error_ce"] = float(ce.max())
            out[f"{label}_log10_max_bellman_residual"] = float(torch.log10(resid.max()))
            out[f"{label}_log10_max_euler_error"] = float(torch.log10(g2.euler_error(actor, k, z).max()))
    k_end = g2.simulate_to_rest(actor)
    out["k_ss_analytical"], out["k_ss_simulated"] = rung.k_ss, k_end
    out["k_ss_relative_gap"] = abs(k_end - rung.k_ss) / rung.k_ss
    return out


# Order matters: answers first, residuals last.  A small residual is evidence, not a right answer.
# Bounds were fixed from a pilot over seeds and step counts (lab_gate_pilot.json), before the reference run.
GATES = {
    "feasible": lambda v: v["held_out_feasible"] and v["off_domain_feasible"],
    "max policy error < 1% (held-out)": lambda v: v["held_out_max_relative_policy_error"] < 0.01,
    "max value error < 0.1% of consumption (held-out)": lambda v: v["held_out_max_value_error_ce"] < 0.001,
    "steady state within 1%": lambda v: v["k_ss_relative_gap"] < 0.01,
    "log10 max Bellman residual < -1.5 (held-out)": lambda v: v["held_out_log10_max_bellman_residual"] < -1.5,
}


def check_gates(v: dict[str, float]) -> dict[str, bool]:
    return {name: bool(test(v)) for name, test in GATES.items()}


# ---------------------------------------------------------------- fallbacks

def checkpoint_path(rung_id: int) -> Path:
    return HERE / f"value_rung{rung_id}_reference.pt"


REPORT = HERE / "value_ladder_report.json"


def load_reference(rung_id: int) -> tuple[g2.SavingPolicy, ValueNet, dict[str, object]]:
    entry = json.loads(REPORT.read_text(encoding="utf-8"))["rungs"][str(rung_id)]
    if file_sha256(checkpoint_path(rung_id)) != entry["checkpoint_sha256"]:
        raise RuntimeError("value checkpoint hash does not match its report")
    cfg = CONFIGS[rung_id]
    rung = g2.replace(g2.RUNGS[rung_id], informed_start=cfg.informed_start, hidden=cfg.hidden)
    actor, critic = g2.SavingPolicy(rung), ValueNet(rung, cfg.hidden, cfg.informed_start)
    state = torch.load(checkpoint_path(rung_id), map_location="cpu", weights_only=True)
    actor.load_state_dict(state["actor"]), critic.load_state_dict(state["critic"])
    actor.eval(), critic.eval()
    return actor, critic, entry


def build_references() -> dict[str, object]:
    """Train both rungs, validate, write checkpoints and the report.  Run once by the instructor."""
    report: dict[str, object] = {"helper": Path(__file__).name, "domain": DOMAIN, "off_domain": OFF_DOMAIN,
                                 "session02_helper_sha256": file_sha256(S2 / "session02_growth.py"), "rungs": {}}
    for rung_id, cfg in CONFIGS.items():
        bench = benchmarks(rung_id)
        actor, critic, history, seconds = train(cfg)
        v = validate(rung_id, actor, critic, bench=bench)
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, checkpoint_path(rung_id))
        entry = {"config": asdict(cfg), "calibration": asdict(g2.RUNGS[rung_id]), "benchmark": bench["kind"],
                 "elapsed_seconds": seconds, "validation": v, "gates": check_gates(v),
                 "checkpoint_sha256": file_sha256(checkpoint_path(rung_id))}
        # the comparisons the deck quotes: one step against many, and an uninformed start
        one = train(ValueConfig(**{**asdict(cfg), "n_step": 1}))
        entry["one_step"] = validate(rung_id, one[0], one[1], bench=bench) | {"elapsed_seconds": one[3]}
        cold = train(ValueConfig(**{**asdict(cfg), "informed_start": False}))
        entry["uninformed_start"] = validate(rung_id, cold[0], cold[1], bench=bench) | {"elapsed_seconds": cold[3]}
        report["rungs"][str(rung_id)] = entry
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    rep = build_references()
    for rid, e in rep["rungs"].items():
        print(rid, f"{e['elapsed_seconds']:.1f}s", e["gates"])
        for label in ("validation", "one_step", "uninformed_start"):
            print("  ", label, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in e[label].items()})
