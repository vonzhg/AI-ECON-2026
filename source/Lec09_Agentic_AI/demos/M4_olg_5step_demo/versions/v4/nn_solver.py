"""V4 — neural-network policy via imitation learning of V3's grid policy.

Phase 5a (this file as shipped): supervised learning. The NN learns to map
``(age, a, b, z_idio, z_tfp)`` → ``(a', b')`` by imitating the policy that
V3's grid solver produces at a fixed equilibrium ``r``. Validates that an
MLP can encode the household policy; sets up V5 where the NN is trained
from residual losses directly with a homotopy schedule.

Why imitation first: pure Euler/KKT residual losses on a randomly-initialized
policy network are notoriously unstable (saturated outputs at the corners of
the action space, sparse gradient signal). Imitation gives a clean, verifiable
"NN replicates the grid" demonstration; V5's homotopy-stabilized residual
training is what actually scales beyond V3's grid.

Architecture:

    PolicyNet(state_dim → 2):
        Linear(in → hidden) → SELU
        Linear(hidden → hidden) → SELU
        Linear(hidden → 2)
        sigmoid-rescale outputs to [0, a_max] × [b_min, b_max]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from two_asset_spec import (
    TwoAssetParams,
    asset_grid,
    bond_grid,
    labor_profile,
    stationary_income_distribution,
)
from tfp import tauchen, tfp_stationary_distribution


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainingConfig:
    """Imitation-learning config for V4."""
    n_steps: int = 800
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    hidden_dim: int = 64
    capital_only: bool = True              # Phase 5a: ignore bond targets
    seed: int = 0
    log_every: int = 100
    device: str = ""


@dataclass
class TrainingHistory:
    step: List[int] = field(default_factory=list)
    loss: List[float] = field(default_factory=list)
    a_loss: List[float] = field(default_factory=list)
    b_loss: List[float] = field(default_factory=list)


class PolicyNet(nn.Module):
    """MLP mapping individual state → (a_next, b_next), bounded to the grid."""

    def __init__(self, state_dim: int, hidden_dim: int,
                 a_max: float, b_min: float, b_max: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, 2),
        )
        # Bias the final layer so the initial sigmoid output is ~0.05 for both
        # actions — the empirical "near-zero savings" prior for OLG households
        # whose typical V3 policy is well below a_max / 2.
        with torch.no_grad():
            self.net[-1].bias.fill_(-3.0)
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(x)
        a_next = torch.sigmoid(raw[:, 0]) * self.a_max
        b_next = self.b_min + torch.sigmoid(raw[:, 1]) * (self.b_max - self.b_min)
        return a_next, b_next


def state_dim(n_z: int, n_tfp: int) -> int:
    return 3 + n_z + n_tfp


def encode_state_batch(
    age_idx: np.ndarray, a: np.ndarray, b: np.ndarray,
    iz: np.ndarray, itfp: np.ndarray,
    n_age: int, a_max: float, b_min: float, b_max: float,
    n_z: int, n_tfp: int,
    device: torch.device,
) -> torch.Tensor:
    bs = age_idx.shape[0]
    age_f = (age_idx.astype(np.float32) / max(n_age - 1, 1))[:, None]
    a_f = (a.astype(np.float32) / a_max)[:, None]
    span = max(b_max - b_min, 1.0e-6)
    b_f = ((b - b_min).astype(np.float32) / span)[:, None]
    iz_oh = np.zeros((bs, n_z), dtype=np.float32)
    iz_oh[np.arange(bs), iz] = 1.0
    itfp_oh = np.zeros((bs, n_tfp), dtype=np.float32)
    itfp_oh[np.arange(bs), itfp] = 1.0
    feats = np.concatenate([age_f, a_f, b_f, iz_oh, itfp_oh], axis=1)
    return torch.from_numpy(feats).to(device)


def collect_v3_targets(result_v3) -> dict:
    """Flatten V3's 5D grid policy into a (state, action) dataset.

    Returns numpy arrays with one row per (age, ia, ib, iz, itfp) cell
    that has non-trivial mass under the V3 stationary distribution.
    """
    n_age = result_v3.params.n_cohorts
    n_a = len(result_v3.household.asset_grid)
    n_b = len(result_v3.household.bond_grid)
    n_z = len(result_v3.params.income_states)
    n_tfp = len(result_v3.tfp_grid)

    a_grid_arr = np.array(result_v3.household.asset_grid)
    b_grid_arr = np.array(result_v3.household.bond_grid)

    # Build the full state-action table — only ages that have a continuation
    # (age < n_age - 1) carry a meaningful policy.
    states = []
    targets = []
    weights = []
    distribution = result_v3.household.distribution  # numpy ndarray
    for age in range(n_age - 1):
        for ia in range(n_a):
            for ib in range(n_b):
                for iz in range(n_z):
                    for itfp in range(n_tfp):
                        ia_next = int(result_v3.household.policy_a_idx[age, ia, ib, iz, itfp])
                        ib_next = int(result_v3.household.policy_b_idx[age, ia, ib, iz, itfp])
                        states.append((age, a_grid_arr[ia], b_grid_arr[ib], iz, itfp))
                        targets.append((a_grid_arr[ia_next], b_grid_arr[ib_next]))
                        # Use stationary mass + small floor so under-visited states
                        # still contribute (otherwise the NN can ignore them).
                        weights.append(distribution[age, ia, ib, iz, itfp] + 1.0e-4)

    states = np.array(states, dtype=object)
    age_idx = states[:, 0].astype(np.int64)
    a = states[:, 1].astype(np.float64)
    b = states[:, 2].astype(np.float64)
    iz = states[:, 3].astype(np.int64)
    itfp = states[:, 4].astype(np.int64)
    targets_arr = np.array(targets, dtype=np.float64)
    weights_arr = np.array(weights, dtype=np.float64)
    weights_arr = weights_arr / weights_arr.sum()

    return {
        "age_idx": age_idx, "a": a, "b": b, "iz": iz, "itfp": itfp,
        "a_target": targets_arr[:, 0], "b_target": targets_arr[:, 1],
        "weight": weights_arr,
        "n_age": n_age, "n_z": n_z, "n_tfp": n_tfp,
        "a_max": result_v3.params.asset_max,
        "b_min": result_v3.params.bond_min, "b_max": result_v3.params.bond_max,
    }


def train_imitation(
    dataset: dict,
    params: TwoAssetParams,
    config: TrainingConfig | None = None,
) -> Tuple[PolicyNet, TrainingHistory, dict]:
    """Supervised training of PolicyNet on V3's grid policy."""
    config = config or TrainingConfig()
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device) if config.device else _device()

    n_age = dataset["n_age"]
    n_z = dataset["n_z"]
    n_tfp = dataset["n_tfp"]
    a_max = dataset["a_max"]
    b_min = dataset["b_min"]
    b_max = dataset["b_max"]

    net = PolicyNet(state_dim(n_z, n_tfp), config.hidden_dim, a_max, b_min, b_max).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)

    n_total = dataset["age_idx"].shape[0]
    weights = dataset["weight"]
    history = TrainingHistory()

    for step in range(config.n_steps):
        idx = rng.choice(n_total, size=config.batch_size, p=weights)
        feats = encode_state_batch(
            dataset["age_idx"][idx], dataset["a"][idx], dataset["b"][idx],
            dataset["iz"][idx], dataset["itfp"][idx],
            n_age, a_max, b_min, b_max, n_z, n_tfp, device,
        )
        a_t = torch.tensor(dataset["a_target"][idx], dtype=torch.float32, device=device)
        b_t = torch.tensor(dataset["b_target"][idx], dtype=torch.float32, device=device)

        a_pred, b_pred = net(feats)
        a_loss = (a_pred - a_t).pow(2).mean()
        b_loss = (b_pred - b_t).pow(2).mean()
        if config.capital_only:
            loss = a_loss
        else:
            loss = a_loss + b_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % config.log_every == 0 or step == config.n_steps - 1:
            history.step.append(step)
            history.loss.append(float(loss.item()))
            history.a_loss.append(float(a_loss.item()))
            history.b_loss.append(float(b_loss.item()))

    metadata = {
        "device": str(device),
        "config": config.__dict__.copy(),
        "n_age": n_age, "n_z": n_z, "n_tfp": n_tfp,
        "a_max": a_max, "b_min": b_min, "b_max": b_max,
    }
    return net, history, metadata


@torch.no_grad()
def simulate_lifecycle(
    net: PolicyNet, params: TwoAssetParams, r: float, metadata: dict,
    tfp_grid: List[float], tfp_transition: List[List[float]],
    n_paths: int = 4000, seed: int = 7,
    capital_only: bool = True,
) -> dict:
    """Forward-simulate ``n_paths`` lifetimes under the trained NN policy."""
    from two_asset_spec import (
        aggregate_labor_supply,
        firm_prices_from_r_tfp,
    )
    rng = np.random.default_rng(seed)
    device = next(net.parameters()).device
    net.eval()

    n_age = params.n_cohorts
    n_z = len(params.income_states)
    n_tfp = len(tfp_grid)
    a_max = metadata["a_max"]
    b_min = metadata["b_min"]
    b_max = metadata["b_max"]

    income_states = np.array(params.income_states)
    Pi_z = np.array(params.income_transition)
    Pi_tfp = np.array(tfp_transition)
    z_stat = stationary_income_distribution(params)
    tfp_stat = tfp_stationary_distribution(tfp_transition)
    labor = labor_profile(n_age)
    L_agg = aggregate_labor_supply(params)
    pension_factor = (
        params.pension_replacement
        * sum(labor[: params.retire_age_index]) / params.retire_age_index
    )

    iz_path = rng.choice(n_z, size=n_paths, p=z_stat)
    itfp_path = rng.choice(n_tfp, size=n_paths, p=tfp_stat)
    a_path = np.zeros(n_paths)
    b_path = np.zeros(n_paths)

    mean_a, mean_b, mean_c = [], [], []
    for age in range(n_age):
        z_vals = income_states[iz_path]
        tfp_vals = np.array([tfp_grid[i] for i in itfp_path])
        K_d = L_agg * ((r + params.delta) / (params.alpha * tfp_vals)) ** (1.0 / (params.alpha - 1.0))
        wages = (1.0 - params.alpha) * tfp_vals * (K_d / L_agg) ** params.alpha
        income = (np.where(age < params.retire_age_index,
                           wages * labor[age] * z_vals,
                           pension_factor * wages))

        feats = encode_state_batch(
            np.full(n_paths, age, dtype=np.int64),
            a_path, b_path, iz_path, itfp_path,
            n_age, a_max, b_min, b_max, n_z, n_tfp, device,
        )
        a_pred, b_pred = net(feats)
        a_next_np = a_pred.cpu().numpy()
        b_next_np = b_pred.cpu().numpy() if not capital_only else np.zeros(n_paths)

        c_path = (1.0 + r) * a_path + b_path + income - a_next_np - params.bond_price * b_next_np

        mean_a.append(float(a_path.mean()))
        mean_b.append(float(b_path.mean()))
        mean_c.append(float(np.maximum(c_path, params.min_consumption).mean()))

        new_iz = np.array([rng.choice(n_z, p=Pi_z[iz_path[i]]) for i in range(n_paths)])
        new_itfp = np.array([rng.choice(n_tfp, p=Pi_tfp[itfp_path[i]]) for i in range(n_paths)])
        iz_path, itfp_path = new_iz, new_itfp
        a_path, b_path = a_next_np, b_next_np

    return {
        "mean_assets_by_age": mean_a,
        "mean_bonds_by_age": mean_b,
        "mean_consumption_by_age": mean_c,
        "aggregate_assets": float(np.mean(mean_a)),
    }
