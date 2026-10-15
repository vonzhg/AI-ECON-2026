"""V2 — two-asset OLG solver (capital + bonds, exogenous bond price).

Structure mirrors V1's solver: backward induction over the household state
space, distribution propagation forward, bisection on ``r`` to clear the
capital market. The state space is now 4D ``(age, ia, ib, iz)`` and the
control is the pair ``(ia_next, ib_next)``.

Bonds are in elastic supply at the exogenous price ``params.bond_price`` —
no bond market clearing in V2. V4 (NN policy) introduces endogenous bond
pricing.

Reduces to V1 numerically when ``bond_grid_size = 1``, ``bond_min = bond_max
= 0``, and ``psi_k = 0``: the bond dimension collapses to a single point at
b = 0, capital adjustment cost vanishes, and the budget reduces to V1's.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

from two_asset_spec import (
    TwoAssetParams,
    aggregate_labor_supply,
    asset_grid,
    bond_grid,
    budget_two_asset,
    firm_prices_from_r,
    labor_profile,
    stationary_income_distribution,
)

NEG_INF = -1.0e30

# Re-export so the notebook can write `from solver import OLGParams`
OLGParams = TwoAssetParams


@dataclass
class HouseholdSolution:
    asset_grid: List[float]
    bond_grid: List[float]
    value: List[List[List[List[float]]]]            # [age][ia][ib][iz]
    policy_a_idx: List[List[List[List[int]]]]       # next-period capital index
    policy_b_idx: List[List[List[List[int]]]]       # next-period bond index
    consumption: List[List[List[List[float]]]]
    distribution: List[List[List[List[float]]]]
    mean_assets_by_age: List[float]
    mean_bonds_by_age: List[float]
    mean_consumption_by_age: List[float]
    aggregate_assets: float
    aggregate_bonds: float
    aggregate_labor: float
    euler_max: float
    euler_p95: float
    borrowing_constraint_mass: float                  # mass with a = a_min (capital)


@dataclass
class EquilibriumResult:
    params: TwoAssetParams
    interest_rate: float
    wage: float
    bond_price: float
    capital_demand: float
    capital_supply: float
    labor: float
    excess_assets: float
    iterations: int
    bracketed: bool
    search_history: List[dict]
    household: HouseholdSolution


def utility(consumption: float, gamma: float) -> float:
    if consumption <= 0.0:
        return NEG_INF
    if abs(gamma - 1.0) < 1.0e-12:
        return math.log(consumption)
    return (consumption ** (1.0 - gamma) - 1.0) / (1.0 - gamma)


def solve_household(r: float, params: TwoAssetParams) -> HouseholdSolution:
    a_grid = asset_grid(params)
    b_grid = bond_grid(params)
    n_a = len(a_grid)
    n_b = len(b_grid)
    n_z = len(params.income_states)
    n_age = params.n_cohorts

    _, _, wage = firm_prices_from_r(r, params)
    labor = labor_profile(n_age)
    mean_work_labor = sum(labor[: params.retire_age_index]) / params.retire_age_index
    pension = params.pension_replacement * wage * mean_work_labor

    # 4D structures: [age][ia][ib][iz]
    def zeros4(): return [
        [[[0.0 for _ in range(n_z)] for _ in range(n_b)] for _ in range(n_a)]
        for _ in range(n_age)
    ]
    def izeros4(): return [
        [[[0 for _ in range(n_z)] for _ in range(n_b)] for _ in range(n_a)]
        for _ in range(n_age)
    ]

    value = zeros4()
    policy_a_idx = izeros4()
    policy_b_idx = izeros4()
    consumption = zeros4()

    for age in reversed(range(n_age)):
        for ia, a_now in enumerate(a_grid):
            for ib, b_now in enumerate(b_grid):
                for iz, z in enumerate(params.income_states):
                    best_value = NEG_INF
                    best_ia_next = 0
                    best_ib_next = 0
                    best_c = params.min_consumption

                    for ia_next, a_next in enumerate(a_grid):
                        for ib_next, b_next in enumerate(b_grid):
                            c = budget_two_asset(
                                age=age,
                                a_now=a_now, b_now=b_now,
                                a_next=a_next, b_next=b_next,
                                z=z, r=r, wage=wage, pension=pension,
                                params=params,
                            )
                            cur = utility(c, params.gamma)
                            if cur <= NEG_INF / 2:
                                continue
                            cont = 0.0
                            if age < n_age - 1:
                                v_next_row = value[age + 1][ia_next][ib_next]
                                cont = sum(
                                    params.income_transition[iz][jz] * v_next_row[jz]
                                    for jz in range(n_z)
                                )
                            cand = cur + params.beta * cont
                            if cand > best_value:
                                best_value = cand
                                best_ia_next = ia_next
                                best_ib_next = ib_next
                                best_c = c

                    value[age][ia][ib][iz] = best_value
                    policy_a_idx[age][ia][ib][iz] = best_ia_next
                    policy_b_idx[age][ia][ib][iz] = best_ib_next
                    consumption[age][ia][ib][iz] = best_c

    # Distribution propagation: cohort 0 starts at (a=0, b=0) with z drawn from stationary
    distribution = [
        [[[0.0 for _ in range(n_z)] for _ in range(n_b)] for _ in range(n_a)]
        for _ in range(n_age)
    ]
    z0 = stationary_income_distribution(params)

    # Find ib0 closest to 0 in the bond grid (initial bond holdings = 0)
    ib0 = min(range(n_b), key=lambda i: abs(b_grid[i]))
    for iz, prob in enumerate(z0):
        distribution[0][0][ib0][iz] = prob / n_age

    for age in range(n_age - 1):
        for ia in range(n_a):
            for ib in range(n_b):
                for iz in range(n_z):
                    mass = distribution[age][ia][ib][iz]
                    if mass == 0.0:
                        continue
                    ia_next = policy_a_idx[age][ia][ib][iz]
                    ib_next = policy_b_idx[age][ia][ib][iz]
                    for jz in range(n_z):
                        distribution[age + 1][ia_next][ib_next][jz] += (
                            mass * params.income_transition[iz][jz]
                        )

    aggregate_assets = 0.0
    aggregate_bonds = 0.0
    aggregate_labor = 0.0
    mean_assets_by_age: List[float] = []
    mean_bonds_by_age: List[float] = []
    mean_consumption_by_age: List[float] = []
    borrowing_mass = 0.0
    total_mass = 0.0
    for age in range(n_age):
        age_mass = 0.0
        age_assets = 0.0
        age_bonds = 0.0
        age_consumption = 0.0
        for ia, a_now in enumerate(a_grid):
            for ib, b_now in enumerate(b_grid):
                for iz, z in enumerate(params.income_states):
                    mass = distribution[age][ia][ib][iz]
                    total_mass += mass
                    age_mass += mass
                    age_assets += mass * a_now
                    age_bonds += mass * b_now
                    age_consumption += mass * consumption[age][ia][ib][iz]
                    aggregate_assets += mass * a_now
                    aggregate_bonds += mass * b_now
                    if ia == 0:
                        borrowing_mass += mass
                    if age < params.retire_age_index:
                        aggregate_labor += mass * labor[age] * z
        mean_assets_by_age.append(age_assets / age_mass if age_mass else 0.0)
        mean_bonds_by_age.append(age_bonds / age_mass if age_mass else 0.0)
        mean_consumption_by_age.append(age_consumption / age_mass if age_mass else 0.0)

    euler_errors = euler_residuals(
        r=r, params=params,
        a_grid=a_grid, b_grid=b_grid,
        consumption=consumption,
        policy_a_idx=policy_a_idx, policy_b_idx=policy_b_idx,
        distribution=distribution,
    )
    if euler_errors:
        euler_max = max(euler_errors)
        s = sorted(euler_errors)
        p95 = s[min(len(s) - 1, int(0.95 * len(s)))]
    else:
        euler_max = 0.0
        p95 = 0.0

    return HouseholdSolution(
        asset_grid=a_grid,
        bond_grid=b_grid,
        value=value,
        policy_a_idx=policy_a_idx,
        policy_b_idx=policy_b_idx,
        consumption=consumption,
        distribution=distribution,
        mean_assets_by_age=mean_assets_by_age,
        mean_bonds_by_age=mean_bonds_by_age,
        mean_consumption_by_age=mean_consumption_by_age,
        aggregate_assets=aggregate_assets,
        aggregate_bonds=aggregate_bonds,
        aggregate_labor=aggregate_labor,
        euler_max=euler_max,
        euler_p95=p95,
        borrowing_constraint_mass=borrowing_mass / total_mass if total_mass else 0.0,
    )


def euler_residuals(
    *,
    r: float,
    params: TwoAssetParams,
    a_grid: Sequence[float],
    b_grid: Sequence[float],
    consumption: List[List[List[List[float]]]],
    policy_a_idx: List[List[List[List[int]]]],
    policy_b_idx: List[List[List[List[int]]]],
    distribution: List[List[List[List[float]]]],
) -> List[float]:
    """Capital Euler residual (bonds Euler tracked separately in V4)."""
    errors: List[float] = []
    n_z = len(params.income_states)
    n_a = len(a_grid)
    n_b = len(b_grid)
    for age in range(params.n_cohorts - 1):
        for ia in range(n_a):
            for ib in range(n_b):
                for iz in range(n_z):
                    mass = distribution[age][ia][ib][iz]
                    if mass <= 1.0e-12:
                        continue
                    ia_next = policy_a_idx[age][ia][ib][iz]
                    ib_next = policy_b_idx[age][ia][ib][iz]
                    if ia_next in (0, n_a - 1):
                        continue
                    c = max(consumption[age][ia][ib][iz], params.min_consumption)
                    lhs = c ** (-params.gamma)
                    rhs = 0.0
                    for jz in range(n_z):
                        c_next = max(
                            consumption[age + 1][ia_next][ib_next][jz],
                            params.min_consumption,
                        )
                        rhs += params.income_transition[iz][jz] * c_next ** (-params.gamma)
                    rhs *= params.beta * (1.0 + r)
                    if lhs > 0.0 and rhs > 0.0:
                        errors.append(abs(math.log(rhs / lhs)))
    return errors


def excess_assets_at_r(
    r: float, params: TwoAssetParams,
) -> tuple[float, HouseholdSolution, float, float, float]:
    capital_demand, labor, wage = firm_prices_from_r(r, params)
    household = solve_household(r, params)
    return (
        household.aggregate_assets - capital_demand,
        household,
        capital_demand,
        labor,
        wage,
    )


def solve_equilibrium(params: TwoAssetParams | None = None) -> EquilibriumResult:
    params = params or TwoAssetParams()
    history: List[dict] = []
    grid_rs = [
        params.r_min + (params.r_max - params.r_min) * i / (params.r_grid_search_points - 1)
        for i in range(params.r_grid_search_points)
    ]

    grid_results = []
    for i, r in enumerate(grid_rs):
        gap, hh, k_demand, labor, wage = excess_assets_at_r(r, params)
        grid_results.append((r, gap, hh, k_demand, labor, wage))
        history.append({
            "phase": "grid", "iteration": i,
            "r": r, "gap": gap,
            "capital_demand": k_demand, "wage": wage,
        })

    bracket = None
    for j in range(len(grid_results) - 1):
        if grid_results[j][1] * grid_results[j + 1][1] <= 0.0:
            bracket = (grid_results[j], grid_results[j + 1])
            break

    if bracket is None:
        best = min(grid_results, key=lambda row: abs(row[1]))
        r_star, gap, hh, k_demand, labor, wage = best
        return EquilibriumResult(
            params=params,
            interest_rate=r_star,
            wage=wage,
            bond_price=params.bond_price,
            capital_demand=k_demand,
            capital_supply=hh.aggregate_assets,
            labor=labor,
            excess_assets=gap,
            iterations=0,
            bracketed=False,
            search_history=history,
            household=hh,
        )

    (r_lo, gap_lo, _, _, _, _), (r_hi, gap_hi, _, _, _, _) = bracket
    iterations = 0
    last = None
    for it in range(params.ge_max_iter):
        iterations = it + 1
        r_mid = 0.5 * (r_lo + r_hi)
        gap, hh, k_demand, labor, wage = excess_assets_at_r(r_mid, params)
        history.append({
            "phase": "bisect", "iteration": it,
            "r": r_mid, "gap": gap,
            "capital_demand": k_demand, "wage": wage,
        })
        last = (r_mid, gap, hh, k_demand, labor, wage)
        if abs(gap) < params.ge_tolerance:
            break
        if gap * gap_lo <= 0.0:
            r_hi, gap_hi = r_mid, gap
        else:
            r_lo, gap_lo = r_mid, gap

    r_mid, gap, hh, k_demand, labor, wage = last  # type: ignore[misc]
    return EquilibriumResult(
        params=params,
        interest_rate=r_mid,
        wage=wage,
        bond_price=params.bond_price,
        capital_demand=k_demand,
        capital_supply=hh.aggregate_assets,
        labor=labor,
        excess_assets=gap,
        iterations=iterations,
        bracketed=True,
        search_history=history,
        household=hh,
    )


def result_summary(result: EquilibriumResult) -> dict:
    s = {
        "interest_rate": result.interest_rate,
        "wage": result.wage,
        "bond_price": result.bond_price,
        "capital_demand": result.capital_demand,
        "capital_supply": result.capital_supply,
        "labor": result.labor,
        "excess_assets": result.excess_assets,
        "iterations": result.iterations,
        "bracketed": result.bracketed,
        "borrowing_constraint_mass": result.household.borrowing_constraint_mass,
        "aggregate_bonds": result.household.aggregate_bonds,
        "euler_max": result.household.euler_max,
        "euler_p95": result.household.euler_p95,
        "euler_p95_log10": math.log10(max(result.household.euler_p95, 1.0e-30)),
        "params": asdict(result.params),
    }
    return s


def write_result_files(result: EquilibriumResult, build_dir: Path) -> None:
    build_dir.mkdir(exist_ok=True)
    (build_dir / "summary_v2.json").write_text(
        json.dumps(result_summary(result), indent=2), encoding="utf-8"
    )
    with (build_dir / "life_cycle_profiles_v2.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["age_index", "mean_assets", "mean_bonds", "mean_consumption"])
        for age in range(result.params.n_cohorts):
            w.writerow([
                age,
                f"{result.household.mean_assets_by_age[age]:.6f}",
                f"{result.household.mean_bonds_by_age[age]:.6f}",
                f"{result.household.mean_consumption_by_age[age]:.6f}",
            ])
