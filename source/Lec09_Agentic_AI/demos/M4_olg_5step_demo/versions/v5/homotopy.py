"""V5 — homotopy training schedule for the NN policy.

Wraps V4's imitation-trained PolicyNet and progressively shifts the loss
from imitation MSE toward Euler/KKT residuals over several phases. The
intuition mirrors Lab12's homotopy: start where the network already
performs well (imitation loss is small) and walk gradually toward the
target objective (residual loss), keeping training stable.

Five-phase default schedule:

    Phase 1: pure imitation refresh (small a/b MSE)         — sanity check
    Phase 2: imitation + light Euler-k                      — start residual signal
    Phase 3: imitation + Euler-k + KKT                      — feasibility pressure
    Phase 4: small imitation + Euler-k + Euler-b + KKT      — bonds active
    Phase 5: residual only (no imitation regularizer)       — final polish

Each phase records a residual snapshot at start and end so the demo can
show the monotone (or near-monotone) decrease that justifies the
homotopy machinery.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from nn_solver import (  # type: ignore[import-not-found]
    PolicyNet, encode_state_batch, state_dim,
    _device,
)
from two_asset_spec import (
    TwoAssetParams,
    aggregate_labor_supply,
    labor_profile,
    stationary_income_distribution,
)
from tfp import tfp_stationary_distribution


@dataclass
class PhaseConfig:
    name: str
    n_steps: int
    w_imitation: float
    w_euler_k: float
    w_euler_b: float
    w_kkt: float
    learning_rate: float = 5.0e-4


@dataclass
class HomotopySchedule:
    phases: List[PhaseConfig]

    @classmethod
    def default(cls) -> "HomotopySchedule":
        return cls(phases=[
            PhaseConfig("1_imitation_refresh", 200, 1.0, 0.0,  0.0,  0.0,  lr := 5.0e-4),
            PhaseConfig("2_light_euler_k",     200, 1.0, 0.05, 0.0,  10.0, 5.0e-4),
            PhaseConfig("3_kkt_pressure",      200, 0.5, 0.1,  0.0,  50.0, 5.0e-4),
            PhaseConfig("4_bonds_on",          200, 0.2, 0.1,  0.05, 50.0, 3.0e-4),
            PhaseConfig("5_residual_only",     200, 0.0, 0.1,  0.05, 50.0, 1.0e-4),
        ])


@dataclass
class PhaseSnapshot:
    phase: str
    step: int
    total_loss: float
    imitation_loss: float
    euler_k: float
    euler_b: float
    kkt: float


@dataclass
class HomotopyHistory:
    snapshots: List[PhaseSnapshot] = field(default_factory=list)

    def add(self, snapshot: PhaseSnapshot) -> None:
        self.snapshots.append(snapshot)


def _residuals_at_states(
    *, net: PolicyNet, params: TwoAssetParams, r: float,
    states: dict,
    tfp_grid_t: torch.Tensor, tfp_transition_t: torch.Tensor,
    income_states_t: torch.Tensor, income_transition_t: torch.Tensor,
    labor_profile_t: torch.Tensor, retire_age_index: int,
    pension_factor_per_wage: float,
    a_max: float, b_min: float, b_max: float,
    n_age: int, n_z: int, n_tfp: int,
    capital_only: bool,
    L_agg: float, alpha: float, delta: float,
    bond_price: float, beta_period: float, gamma: float, min_consumption: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (Euler_k, Euler_b, KKT) losses on a sampled batch of states.

    All tensors carry gradient; backward through this returns gradients
    of the residual loss with respect to the network's parameters.
    """
    age_idx = torch.tensor(states["age_idx"], dtype=torch.long, device=device)
    a = torch.tensor(states["a"], dtype=torch.float32, device=device)
    b = torch.tensor(states["b"], dtype=torch.float32, device=device)
    iz = torch.tensor(states["iz"], dtype=torch.long, device=device)
    itfp = torch.tensor(states["itfp"], dtype=torch.long, device=device)
    bs = age_idx.size(0)

    # current state encoding + policy
    feats = encode_state_batch(
        states["age_idx"], states["a"], states["b"], states["iz"], states["itfp"],
        n_age, a_max, b_min, b_max, n_z, n_tfp, device,
    )
    a_next, b_next = net(feats)
    if capital_only:
        b_next = torch.zeros_like(b_next)

    z_val = income_states_t[iz]
    tfp_val = tfp_grid_t[itfp]
    K_d = L_agg * ((r + delta) / (alpha * tfp_val)) ** (1.0 / (alpha - 1.0))
    wage = (1.0 - alpha) * tfp_val * (K_d / L_agg) ** alpha
    pension = pension_factor_per_wage * wage
    labor_age = labor_profile_t[age_idx]
    is_working = (age_idx < retire_age_index).float()
    income = is_working * wage * labor_age * z_val + (1.0 - is_working) * pension

    c = (1.0 + r) * a + b + income - a_next - bond_price * b_next
    kkt = torch.relu(min_consumption - c).pow(2).mean()
    c_pos = torch.clamp(c, min=min_consumption)
    lhs = c_pos.pow(-gamma)

    # next-state grid: for each sample evaluate the NN at all (jz, jtfp)
    n_next = n_z * n_tfp
    age_next = age_idx + 1
    age_next_rep = age_next.repeat_interleave(n_next).cpu().numpy()
    a_rep = a_next.repeat_interleave(n_next)
    b_rep = b_next.repeat_interleave(n_next)
    jz_idx_t = torch.arange(n_z, device=device).repeat(n_tfp).unsqueeze(0).expand(bs, -1).reshape(-1)
    jtfp_idx_t = torch.arange(n_tfp, device=device).repeat_interleave(n_z).unsqueeze(0).expand(bs, -1).reshape(-1)

    # Encode next state — re-use encode_state_batch, but we need numpy a' and b'
    # values for the encoding (one-hot is index-based; continuous fields use raw values).
    feats_next = torch.cat([
        (torch.tensor(age_next_rep, dtype=torch.float32, device=device) / max(n_age - 1, 1)).unsqueeze(1),
        (a_rep / a_max).unsqueeze(1),
        ((b_rep - b_min) / max(b_max - b_min, 1.0e-6)).unsqueeze(1),
        torch.nn.functional.one_hot(jz_idx_t, num_classes=n_z).float(),
        torch.nn.functional.one_hot(jtfp_idx_t, num_classes=n_tfp).float(),
    ], dim=1)

    a_nn, b_nn = net(feats_next)
    if capital_only:
        b_nn = torch.zeros_like(b_nn)

    z_next_val = income_states_t[jz_idx_t]
    tfp_next_val = tfp_grid_t[jtfp_idx_t]
    K_d_next = L_agg * ((r + delta) / (alpha * tfp_next_val)) ** (1.0 / (alpha - 1.0))
    wage_next = (1.0 - alpha) * tfp_next_val * (K_d_next / L_agg) ** alpha
    pension_next = pension_factor_per_wage * wage_next
    labor_next = labor_profile_t[torch.tensor(age_next_rep, dtype=torch.long, device=device)]
    working_next = (torch.tensor(age_next_rep, dtype=torch.long, device=device) < retire_age_index).float()
    income_next = working_next * wage_next * labor_next * z_next_val + (1.0 - working_next) * pension_next
    c_next = (1.0 + r) * a_rep + b_rep + income_next - a_nn - bond_price * b_nn
    c_next_pos = torch.clamp(c_next, min=min_consumption)
    rhs_terms = c_next_pos.pow(-gamma)

    iz_rep = iz.repeat_interleave(n_next)
    itfp_rep = itfp.repeat_interleave(n_next)
    pz = income_transition_t[iz_rep, jz_idx_t]
    pt = tfp_transition_t[itfp_rep, jtfp_idx_t]
    weights = pz * pt
    rhs_per_sample = (weights * rhs_terms).reshape(bs, n_next).sum(dim=1)

    # Capital Euler residual
    rhs_k = beta_period * (1.0 + r) * rhs_per_sample
    rhs_k_pos = torch.clamp(rhs_k, min=1.0e-30)
    euler_k = (torch.log(rhs_k_pos) - torch.log(lhs)).pow(2).mean()

    # Bond Euler residual: 1/p_b ≈ beta * E[c'^-γ] / c^-γ → log diff squared
    rhs_b = beta_period * rhs_per_sample
    rhs_b_pos = torch.clamp(rhs_b, min=1.0e-30)
    bond_lhs = lhs * bond_price
    euler_b = (torch.log(rhs_b_pos) - torch.log(torch.clamp(bond_lhs, min=1.0e-30))).pow(2).mean()

    return euler_k, euler_b, kkt


def run_homotopy(
    net: PolicyNet,
    schedule: HomotopySchedule,
    dataset: dict,
    params: TwoAssetParams,
    r: float,
    tfp_grid: List[float],
    tfp_transition: List[List[float]],
    *,
    batch_size: int = 256,
    seed: int = 1,
    capital_only: bool = False,
) -> Tuple[PolicyNet, HomotopyHistory]:
    """Run V5's homotopy schedule on a pre-trained PolicyNet."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = next(net.parameters()).device

    n_age = params.n_cohorts
    n_z = len(params.income_states)
    n_tfp = len(tfp_grid)
    a_max = params.asset_max
    b_min = params.bond_min
    b_max = params.bond_max
    L_agg = aggregate_labor_supply(params)
    pension_factor_per_wage = (
        params.pension_replacement
        * sum(labor_profile(n_age)[: params.retire_age_index]) / params.retire_age_index
    )

    income_states_t = torch.tensor(params.income_states, dtype=torch.float32, device=device)
    income_transition_t = torch.tensor(params.income_transition, dtype=torch.float32, device=device)
    tfp_grid_t = torch.tensor(tfp_grid, dtype=torch.float32, device=device)
    tfp_transition_t = torch.tensor(tfp_transition, dtype=torch.float32, device=device)
    labor_profile_t = torch.tensor(labor_profile(n_age), dtype=torch.float32, device=device)

    n_total = dataset["age_idx"].shape[0]
    weights = dataset["weight"]
    history = HomotopyHistory()

    def sample_batch():
        idx = rng.choice(n_total, size=batch_size, p=weights)
        return {
            "age_idx": dataset["age_idx"][idx],
            "a": dataset["a"][idx],
            "b": dataset["b"][idx],
            "iz": dataset["iz"][idx],
            "itfp": dataset["itfp"][idx],
            "a_target": dataset["a_target"][idx],
            "b_target": dataset["b_target"][idx],
        }

    @torch.no_grad()
    def snapshot(phase_name: str, step: int) -> PhaseSnapshot:
        net.eval()
        states = sample_batch()
        ek, eb, kkt = _residuals_at_states(
            net=net, params=params, r=r, states=states,
            tfp_grid_t=tfp_grid_t, tfp_transition_t=tfp_transition_t,
            income_states_t=income_states_t, income_transition_t=income_transition_t,
            labor_profile_t=labor_profile_t, retire_age_index=params.retire_age_index,
            pension_factor_per_wage=pension_factor_per_wage,
            a_max=a_max, b_min=b_min, b_max=b_max,
            n_age=n_age, n_z=n_z, n_tfp=n_tfp,
            capital_only=capital_only,
            L_agg=L_agg, alpha=params.alpha, delta=params.delta,
            bond_price=params.bond_price, beta_period=params.beta,
            gamma=params.gamma, min_consumption=params.min_consumption,
            device=device,
        )
        # Imitation MSE for reporting
        feats = encode_state_batch(
            states["age_idx"], states["a"], states["b"], states["iz"], states["itfp"],
            n_age, a_max, b_min, b_max, n_z, n_tfp, device,
        )
        a_pred, b_pred = net(feats)
        a_t = torch.tensor(states["a_target"], dtype=torch.float32, device=device)
        b_t = torch.tensor(states["b_target"], dtype=torch.float32, device=device)
        imitation = (a_pred - a_t).pow(2).mean() + (b_pred - b_t).pow(2).mean()
        net.train()
        return PhaseSnapshot(
            phase=phase_name, step=step,
            total_loss=float(ek.item() + eb.item() + kkt.item() + imitation.item()),
            imitation_loss=float(imitation.item()),
            euler_k=float(ek.item()), euler_b=float(eb.item()), kkt=float(kkt.item()),
        )

    for phase in schedule.phases:
        history.add(snapshot(phase.name, step=0))
        optimizer = torch.optim.Adam(net.parameters(), lr=phase.learning_rate)
        for step in range(phase.n_steps):
            net.train()
            states = sample_batch()
            feats = encode_state_batch(
                states["age_idx"], states["a"], states["b"], states["iz"], states["itfp"],
                n_age, a_max, b_min, b_max, n_z, n_tfp, device,
            )
            a_pred, b_pred = net(feats)
            a_t = torch.tensor(states["a_target"], dtype=torch.float32, device=device)
            b_t = torch.tensor(states["b_target"], dtype=torch.float32, device=device)
            imitation_loss = (a_pred - a_t).pow(2).mean() + (b_pred - b_t).pow(2).mean()

            ek, eb, kkt = _residuals_at_states(
                net=net, params=params, r=r, states=states,
                tfp_grid_t=tfp_grid_t, tfp_transition_t=tfp_transition_t,
                income_states_t=income_states_t, income_transition_t=income_transition_t,
                labor_profile_t=labor_profile_t, retire_age_index=params.retire_age_index,
                pension_factor_per_wage=pension_factor_per_wage,
                a_max=a_max, b_min=b_min, b_max=b_max,
                n_age=n_age, n_z=n_z, n_tfp=n_tfp,
                capital_only=capital_only,
                L_agg=L_agg, alpha=params.alpha, delta=params.delta,
                bond_price=params.bond_price, beta_period=params.beta,
                gamma=params.gamma, min_consumption=params.min_consumption,
                device=device,
            )

            loss = (
                phase.w_imitation * imitation_loss
                + phase.w_euler_k * ek
                + phase.w_euler_b * eb
                + phase.w_kkt * kkt
            )
            if torch.isnan(loss) or torch.isinf(loss):
                # Skip the step; keeps training robust to occasional blow-ups
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()

        history.add(snapshot(phase.name, step=phase.n_steps))

    return net, history
