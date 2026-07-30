from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence


NEG_INF = -1.0e30


@dataclass(frozen=True)
class OLGParams:
    """Small single-asset OLG model calibrated for a fast classroom run."""

    n_cohorts: int = 7
    model_years: int = 72
    beta_yearly: float = 0.96
    gamma: float = 2.0
    alpha: float = 1.0 / 3.0
    delta_yearly: float = 0.10
    asset_grid_size: int = 90
    asset_max: float = 8.0
    retire_age_index: int = 5
    pension_replacement: float = 0.25
    income_states: tuple[float, float, float] = (0.70, 1.00, 1.35)
    income_transition: tuple[tuple[float, float, float], ...] = (
        (0.86, 0.13, 0.01),
        (0.08, 0.84, 0.08),
        (0.01, 0.13, 0.86),
    )
    min_consumption: float = 1.0e-10
    r_min: float = 0.005
    r_max: float = 0.45
    r_grid_search_points: int = 25
    ge_tolerance: float = 5.0e-4
    ge_max_iter: int = 40

    @property
    def period_length(self) -> float:
        return self.model_years / self.n_cohorts

    @property
    def beta(self) -> float:
        return self.beta_yearly**self.period_length

    @property
    def delta(self) -> float:
        return 1.0 - (1.0 - self.delta_yearly) ** self.period_length


@dataclass
class HouseholdSolution:
    asset_grid: List[float]
    value: List[List[List[float]]]
    policy_index: List[List[List[int]]]
    consumption: List[List[List[float]]]
    distribution: List[List[List[float]]]
    mean_assets_by_age: List[float]
    mean_consumption_by_age: List[float]
    aggregate_assets: float
    aggregate_labor: float
    euler_max: float
    euler_p95: float
    borrowing_constraint_mass: float


@dataclass
class EquilibriumResult:
    params: OLGParams
    interest_rate: float
    wage: float
    capital_demand: float
    capital_supply: float
    labor: float
    excess_assets: float
    iterations: int
    bracketed: bool
    search_history: List[dict]
    household: HouseholdSolution


def labor_profile(n_cohorts: int) -> List[float]:
    """Life-cycle labor endowment used in Lab12, translated to standard Python."""

    a_is = int(n_cohorts * 0.6)
    temp_l = 0.6 + (0.6 * 1.27 / (n_cohorts / 2.0) ** 2) * (
        (n_cohorts / 2.0) ** 2 - (a_is - (n_cohorts / 2.0)) ** 2
    )
    values = []
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


def asset_grid(params: OLGParams) -> List[float]:
    if params.asset_grid_size < 3:
        raise ValueError("asset_grid_size must be at least 3")
    return [
        params.asset_max * (i / (params.asset_grid_size - 1)) ** 2.2
        for i in range(params.asset_grid_size)
    ]


def stationary_income_distribution(params: OLGParams) -> List[float]:
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


def aggregate_labor_supply(params: OLGParams) -> float:
    labor = labor_profile(params.n_cohorts)
    z_dist = stationary_income_distribution(params)
    mean_z = sum(p * z for p, z in zip(z_dist, params.income_states))
    return sum(labor[j] for j in range(params.retire_age_index)) * mean_z / params.n_cohorts


def firm_prices_from_r(r: float, params: OLGParams) -> tuple[float, float, float]:
    labor = aggregate_labor_supply(params)
    capital = labor * (params.alpha / (r + params.delta)) ** (1.0 / (1.0 - params.alpha))
    wage = (1.0 - params.alpha) * (capital / labor) ** params.alpha
    return capital, labor, wage


def utility(consumption: float, gamma: float) -> float:
    if consumption <= 0.0:
        return NEG_INF
    if abs(gamma - 1.0) < 1.0e-12:
        return math.log(consumption)
    return (consumption ** (1.0 - gamma) - 1.0) / (1.0 - gamma)


def solve_household(r: float, params: OLGParams) -> HouseholdSolution:
    grid = asset_grid(params)
    n_a = len(grid)
    n_z = len(params.income_states)
    _, _, wage = firm_prices_from_r(r, params)
    labor = labor_profile(params.n_cohorts)
    mean_work_labor = sum(labor[: params.retire_age_index]) / params.retire_age_index
    pension = params.pension_replacement * wage * mean_work_labor

    value = [
        [[0.0 for _ in range(n_z)] for _ in range(n_a)]
        for _ in range(params.n_cohorts)
    ]
    policy_index = [
        [[0 for _ in range(n_z)] for _ in range(n_a)]
        for _ in range(params.n_cohorts)
    ]
    consumption = [
        [[0.0 for _ in range(n_z)] for _ in range(n_a)]
        for _ in range(params.n_cohorts)
    ]

    for age in reversed(range(params.n_cohorts)):
        for ia, assets in enumerate(grid):
            for iz, z in enumerate(params.income_states):
                labor_income = wage * labor[age] * z if age < params.retire_age_index else 0.0
                resources = (1.0 + r) * assets + labor_income
                if age >= params.retire_age_index:
                    resources += pension

                best_value = NEG_INF
                best_idx = 0
                best_c = params.min_consumption
                for next_idx, next_assets in enumerate(grid):
                    c = resources - next_assets
                    current = utility(c, params.gamma)
                    if current <= NEG_INF / 2:
                        continue
                    continuation = 0.0
                    if age < params.n_cohorts - 1:
                        continuation = sum(
                            params.income_transition[iz][jz]
                            * value[age + 1][next_idx][jz]
                            for jz in range(n_z)
                        )
                    candidate = current + params.beta * continuation
                    if candidate > best_value:
                        best_value = candidate
                        best_idx = next_idx
                        best_c = c
                value[age][ia][iz] = best_value
                policy_index[age][ia][iz] = best_idx
                consumption[age][ia][iz] = best_c

    distribution = [
        [[0.0 for _ in range(n_z)] for _ in range(n_a)]
        for _ in range(params.n_cohorts)
    ]
    z0 = stationary_income_distribution(params)
    for iz, prob in enumerate(z0):
        distribution[0][0][iz] = prob / params.n_cohorts

    for age in range(params.n_cohorts - 1):
        for ia in range(n_a):
            for iz in range(n_z):
                mass = distribution[age][ia][iz]
                if mass == 0.0:
                    continue
                next_idx = policy_index[age][ia][iz]
                for jz in range(n_z):
                    distribution[age + 1][next_idx][jz] += (
                        mass * params.income_transition[iz][jz]
                    )

    aggregate_assets = 0.0
    aggregate_labor = 0.0
    mean_assets_by_age = []
    mean_consumption_by_age = []
    borrowing_mass = 0.0
    total_mass = 0.0
    for age in range(params.n_cohorts):
        age_mass = 0.0
        age_assets = 0.0
        age_consumption = 0.0
        for ia, assets in enumerate(grid):
            for iz, z in enumerate(params.income_states):
                mass = distribution[age][ia][iz]
                total_mass += mass
                age_mass += mass
                age_assets += mass * assets
                age_consumption += mass * consumption[age][ia][iz]
                aggregate_assets += mass * assets
                if ia == 0:
                    borrowing_mass += mass
                if age < params.retire_age_index:
                    aggregate_labor += mass * labor[age] * z
        mean_assets_by_age.append(age_assets / age_mass if age_mass else 0.0)
        mean_consumption_by_age.append(age_consumption / age_mass if age_mass else 0.0)

    euler_errors = euler_residuals(
        r=r,
        params=params,
        grid=grid,
        consumption=consumption,
        policy_index=policy_index,
        distribution=distribution,
    )
    euler_errors_sorted = sorted(euler_errors)
    if euler_errors_sorted:
        euler_max = max(euler_errors_sorted)
        p95_idx = min(len(euler_errors_sorted) - 1, int(0.95 * len(euler_errors_sorted)))
        euler_p95 = euler_errors_sorted[p95_idx]
    else:
        euler_max = 0.0
        euler_p95 = 0.0

    return HouseholdSolution(
        asset_grid=grid,
        value=value,
        policy_index=policy_index,
        consumption=consumption,
        distribution=distribution,
        mean_assets_by_age=mean_assets_by_age,
        mean_consumption_by_age=mean_consumption_by_age,
        aggregate_assets=aggregate_assets,
        aggregate_labor=aggregate_labor,
        euler_max=euler_max,
        euler_p95=euler_p95,
        borrowing_constraint_mass=borrowing_mass / total_mass if total_mass else 0.0,
    )


def euler_residuals(
    *,
    r: float,
    params: OLGParams,
    grid: Sequence[float],
    consumption: List[List[List[float]]],
    policy_index: List[List[List[int]]],
    distribution: List[List[List[float]]],
) -> List[float]:
    errors: List[float] = []
    n_z = len(params.income_states)
    for age in range(params.n_cohorts - 1):
        for ia in range(len(grid)):
            for iz in range(n_z):
                mass = distribution[age][ia][iz]
                next_idx = policy_index[age][ia][iz]
                if mass <= 1.0e-12 or next_idx == 0 or next_idx == len(grid) - 1:
                    continue
                c = max(consumption[age][ia][iz], params.min_consumption)
                lhs = c ** (-params.gamma)
                expected_rhs = 0.0
                for jz in range(n_z):
                    c_next = max(
                        consumption[age + 1][next_idx][jz],
                        params.min_consumption,
                    )
                    expected_rhs += params.income_transition[iz][jz] * c_next ** (-params.gamma)
                rhs = params.beta * (1.0 + r) * expected_rhs
                if lhs > 0.0 and rhs > 0.0:
                    errors.append(abs(math.log(rhs / lhs)))
    return errors


def excess_assets_at_r(r: float, params: OLGParams) -> tuple[float, HouseholdSolution, float, float, float]:
    capital_demand, labor, wage = firm_prices_from_r(r, params)
    household = solve_household(r, params)
    return (
        household.aggregate_assets - capital_demand,
        household,
        capital_demand,
        labor,
        wage,
    )


def solve_equilibrium(params: OLGParams | None = None) -> EquilibriumResult:
    params = params or OLGParams()
    history: List[dict] = []
    grid_rs = [
        params.r_min
        + i * (params.r_max - params.r_min) / (params.r_grid_search_points - 1)
        for i in range(params.r_grid_search_points)
    ]

    evaluations = []
    for r in grid_rs:
        gap, _hh, kd, _labor, wage = excess_assets_at_r(r, params)
        row = {"phase": "bracket", "r": r, "gap": gap, "capital_demand": kd, "wage": wage}
        history.append(row)
        evaluations.append((r, gap))

    bracket = None
    for left, right in zip(evaluations, evaluations[1:]):
        if left[1] == 0.0:
            bracket = (left[0], left[0])
            break
        if left[1] * right[1] < 0.0:
            bracket = (left[0], right[0])
            break

    bracketed = bracket is not None
    if not bracketed:
        best_r, _ = min(evaluations, key=lambda item: abs(item[1]))
        gap, household, kd, labor, wage = excess_assets_at_r(best_r, params)
        return EquilibriumResult(
            params=params,
            interest_rate=best_r,
            wage=wage,
            capital_demand=kd,
            capital_supply=household.aggregate_assets,
            labor=labor,
            excess_assets=gap,
            iterations=len(history),
            bracketed=False,
            search_history=history,
            household=household,
        )

    lo, hi = bracket
    if lo == hi:
        r_star = lo
        gap, household, kd, labor, wage = excess_assets_at_r(r_star, params)
    else:
        gap_lo, _hh, _kd, _labor, _wage = excess_assets_at_r(lo, params)
        household = _hh
        kd = _kd
        labor = _labor
        wage = _wage
        gap = gap_lo
        r_star = lo
        for iteration in range(params.ge_max_iter):
            mid = 0.5 * (lo + hi)
            gap_mid, hh_mid, kd_mid, labor_mid, wage_mid = excess_assets_at_r(mid, params)
            history.append(
                {
                    "phase": "bisect",
                    "iteration": iteration + 1,
                    "r": mid,
                    "gap": gap_mid,
                    "capital_demand": kd_mid,
                    "wage": wage_mid,
                }
            )
            r_star, gap, household, kd, labor, wage = (
                mid,
                gap_mid,
                hh_mid,
                kd_mid,
                labor_mid,
                wage_mid,
            )
            if abs(gap_mid) < params.ge_tolerance:
                break
            if gap_lo * gap_mid <= 0.0:
                hi = mid
            else:
                lo = mid
                gap_lo = gap_mid

    return EquilibriumResult(
        params=params,
        interest_rate=r_star,
        wage=wage,
        capital_demand=kd,
        capital_supply=household.aggregate_assets,
        labor=labor,
        excess_assets=gap,
        iterations=len(history),
        bracketed=True,
        search_history=history,
        household=household,
    )


def result_summary(result: EquilibriumResult) -> dict:
    return {
        "interest_rate": result.interest_rate,
        "wage": result.wage,
        "capital_demand": result.capital_demand,
        "capital_supply": result.capital_supply,
        "labor": result.labor,
        "excess_assets": result.excess_assets,
        "iterations": result.iterations,
        "bracketed": result.bracketed,
        "euler_max_log10": math.log10(max(result.household.euler_max, 1.0e-16)),
        "euler_p95_log10": math.log10(max(result.household.euler_p95, 1.0e-16)),
        "borrowing_constraint_mass": result.household.borrowing_constraint_mass,
        "mean_assets_by_age": result.household.mean_assets_by_age,
        "mean_consumption_by_age": result.household.mean_consumption_by_age,
        "params": asdict(result.params),
    }


def write_result_files(result: EquilibriumResult, build_dir: Path) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "summary.json").write_text(
        json.dumps(result_summary(result), indent=2),
        encoding="utf-8",
    )

    with (build_dir / "equilibrium_history.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in result.search_history for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.search_history:
            writer.writerow(row)

    with (build_dir / "life_cycle_profiles.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["age_index", "model_age", "mean_assets", "mean_consumption"],
        )
        writer.writeheader()
        for age in range(result.params.n_cohorts):
            model_age = 20.0 + age * result.params.period_length
            writer.writerow(
                {
                    "age_index": age,
                    "model_age": f"{model_age:.1f}",
                    "mean_assets": f"{result.household.mean_assets_by_age[age]:.8f}",
                    "mean_consumption": f"{result.household.mean_consumption_by_age[age]:.8f}",
                }
            )

    with (build_dir / "distribution.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["age_index", "asset", "income_state", "mass"],
        )
        writer.writeheader()
        for age, age_dist in enumerate(result.household.distribution):
            for ia, assets in enumerate(result.household.asset_grid):
                for iz, _z in enumerate(result.params.income_states):
                    mass = age_dist[ia][iz]
                    if mass > 1.0e-10:
                        writer.writerow(
                            {
                                "age_index": age,
                                "asset": f"{assets:.8f}",
                                "income_state": iz,
                                "mass": f"{mass:.12f}",
                            }
                        )
