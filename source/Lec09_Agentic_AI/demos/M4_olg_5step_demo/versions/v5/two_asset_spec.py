"""V3 — model contract: two-asset OLG with aggregate TFP.

Extends V2's contract with TFP state. With ``n_tfp = 1`` (or ``sigma_tfp = 0``)
the TFP process collapses to ``z = 1`` and V3 reduces numerically to V2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class TwoAssetParams:
    """V3 parameters. New fields: ``rho_tfp``, ``sigma_tfp``, ``n_tfp``."""

    n_cohorts: int = 7
    model_years: int = 72
    beta_yearly: float = 0.96
    gamma: float = 2.0

    alpha: float = 1.0 / 3.0
    delta_yearly: float = 0.10

    asset_grid_size: int = 10
    asset_max: float = 8.0

    retire_age_index: int = 5
    pension_replacement: float = 0.25
    income_states: Tuple[float, ...] = (0.70, 1.00, 1.35)
    income_transition: Tuple[Tuple[float, ...], ...] = (
        (0.86, 0.13, 0.01),
        (0.08, 0.84, 0.08),
        (0.01, 0.13, 0.86),
    )

    bond_grid_size: int = 4
    bond_min: float = -0.05
    bond_max: float = 0.4
    bond_price: float = 0.98
    psi_k: float = 0.0

    # NEW IN V3: aggregate TFP
    rho_tfp: float = 0.85
    sigma_tfp: float = 0.03
    n_tfp: int = 3
    tauchen_m: float = 3.0

    min_consumption: float = 1.0e-10
    r_min: float = 0.005
    r_max: float = 0.6
    r_grid_search_points: int = 11
    ge_tolerance: float = 5.0e-3
    ge_max_iter: int = 12

    @property
    def period_length(self) -> float:
        return self.model_years / self.n_cohorts

    @property
    def beta(self) -> float:
        return self.beta_yearly ** self.period_length

    @property
    def delta(self) -> float:
        return 1.0 - (1.0 - self.delta_yearly) ** self.period_length


def asset_grid(params: TwoAssetParams) -> List[float]:
    if params.asset_grid_size < 3:
        raise ValueError("asset_grid_size must be at least 3")
    return [
        params.asset_max * (i / (params.asset_grid_size - 1)) ** 2.2
        for i in range(params.asset_grid_size)
    ]


def bond_grid(params: TwoAssetParams) -> List[float]:
    n = params.bond_grid_size
    if n < 1:
        raise ValueError("bond_grid_size must be >= 1")
    if n == 1:
        return [0.5 * (params.bond_min + params.bond_max)]
    return [
        params.bond_min + (params.bond_max - params.bond_min) * i / (n - 1)
        for i in range(n)
    ]


def labor_profile(n_cohorts: int) -> List[float]:
    a_is = int(n_cohorts * 0.6)
    temp_l = 0.6 + (0.6 * 1.27 / (n_cohorts / 2.0) ** 2) * (
        (n_cohorts / 2.0) ** 2 - (a_is - (n_cohorts / 2.0)) ** 2
    )
    values: List[float] = []
    for age in range(n_cohorts):
        if age < a_is:
            value = 0.6 + (0.6 * 1.27 / (n_cohorts / 2.0) ** 2) * (
                (n_cohorts / 2.0) ** 2 - (age - (n_cohorts / 2.0)) ** 2
            )
        else:
            value = 0.5 * temp_l
        values.append(max(value, 0.0))
    scale = n_cohorts / sum(values)
    return [value * scale for value in values]


def stationary_income_distribution(params: TwoAssetParams) -> List[float]:
    n = len(params.income_states)
    dist = [1.0 / n for _ in range(n)]
    for _ in range(10_000):
        new = [0.0 for _ in range(n)]
        for i in range(n):
            for j in range(n):
                new[j] += dist[i] * params.income_transition[i][j]
        err = max(abs(new[i] - dist[i]) for i in range(n))
        dist = new
        if err < 1.0e-14:
            break
    total = sum(dist)
    return [x / total for x in dist]


def aggregate_labor_supply(params: TwoAssetParams) -> float:
    labor = labor_profile(params.n_cohorts)
    z_dist = stationary_income_distribution(params)
    mean_z = sum(p * z for p, z in zip(z_dist, params.income_states))
    return sum(labor[j] for j in range(params.retire_age_index)) * mean_z / params.n_cohorts


def firm_prices_from_r_tfp(
    r: float, z_tfp: float, params: TwoAssetParams,
) -> Tuple[float, float, float]:
    """Firm side, conditioned on aggregate TFP shock ``z_tfp``.

    Solves α * z_tfp * (K/L)^(α-1) = r + δ for K_demand, then computes wage.
    """
    labor = aggregate_labor_supply(params)
    if r + params.delta <= 0.0:
        raise ValueError("r + delta must be positive")
    K_demand = labor * ((r + params.delta) / (params.alpha * z_tfp)) ** (1.0 / (params.alpha - 1.0))
    wage = (1.0 - params.alpha) * z_tfp * (K_demand / labor) ** params.alpha
    return K_demand, labor, wage
