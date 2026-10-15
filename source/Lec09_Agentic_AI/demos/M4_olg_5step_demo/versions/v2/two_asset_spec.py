"""V2 — model contract for the two-asset OLG.

Pure-stdlib spec module: dataclasses, grid construction, and the new budget
equation. No solver code lives here. The solver in ``solver.py`` imports from
this module so the contract is inspectable in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class TwoAssetParams:
    """Parameters for V2's two-asset OLG model.

    Inherits the V0/V1 baseline (1-asset settings preserved) and adds bond
    fields. Setting ``bond_grid_size = 1`` with ``bond_min = bond_max = 0``
    reduces V2 to V1.
    """

    # Demographics & preferences (matches V0 defaults)
    n_cohorts: int = 7
    model_years: int = 72
    beta_yearly: float = 0.96
    gamma: float = 2.0

    # Production
    alpha: float = 1.0 / 3.0
    delta_yearly: float = 0.10

    # Capital grid (asset = capital)
    asset_grid_size: int = 15
    asset_max: float = 8.0

    # Retirement & labor
    retire_age_index: int = 5
    pension_replacement: float = 0.25
    income_states: Tuple[float, ...] = (0.70, 1.00, 1.35)
    income_transition: Tuple[Tuple[float, ...], ...] = (
        (0.86, 0.13, 0.01),
        (0.08, 0.84, 0.08),
        (0.01, 0.13, 0.86),
    )

    # NEW IN V2: bond grid
    bond_grid_size: int = 5
    bond_min: float = -0.05
    bond_max: float = 0.5
    bond_price: float = 0.98          # exogenous in V2 (no bond market clearing)
    psi_k: float = 0.0                # capital adjustment cost; 0 reduces to V1

    # Numerical knobs
    min_consumption: float = 1.0e-10
    r_min: float = 0.005
    r_max: float = 0.45
    r_grid_search_points: int = 17
    ge_tolerance: float = 5.0e-3
    ge_max_iter: int = 20

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
    """Linear grid over [bond_min, bond_max] with ``bond_grid_size`` points.

    When ``bond_grid_size == 1`` the only point is the average of the two
    bounds (which equals 0 if ``bond_min = -bond_max``); the V2 → V1 reduction
    sets ``bond_min = bond_max = 0`` to force b = 0.
    """
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


def firm_prices_from_r(r: float, params: TwoAssetParams) -> Tuple[float, float, float]:
    labor = aggregate_labor_supply(params)
    if r + params.delta <= 0.0:
        raise ValueError("r + delta must be positive for the firm's first-order condition")
    capital_demand = labor * ((r + params.delta) / params.alpha) ** (1.0 / (params.alpha - 1.0))
    wage = (1.0 - params.alpha) * (capital_demand / labor) ** params.alpha
    return capital_demand, labor, wage


def budget_two_asset(
    *,
    age: int,
    a_now: float,
    b_now: float,
    a_next: float,
    b_next: float,
    z: float,
    r: float,
    wage: float,
    pension: float,
    params: TwoAssetParams,
) -> float:
    """Period budget for V2.

    Working ages: c + a_next + p_b * b_next + adjustment_cost = (1+r) * a_now + b_now + w * labor_age * z
    Retirement:   c + a_next + p_b * b_next + adjustment_cost = (1+r) * a_now + b_now + pension

    The bond carried into this period (``b_now``) returns face value 1; the
    bond bought today (``b_next``) costs ``p_b`` per unit. With ``psi_k > 0``
    capital faces a quadratic adjustment cost ``psi_k * (a_next - a_now)^2``.
    """
    labor_age = labor_profile(params.n_cohorts)[age]
    if age < params.retire_age_index:
        income = wage * labor_age * z
    else:
        income = pension
    resources = (1.0 + r) * a_now + b_now + income
    adj_cost = params.psi_k * (a_next - a_now) ** 2 if params.psi_k > 0.0 else 0.0
    return resources - a_next - params.bond_price * b_next - adj_cost
