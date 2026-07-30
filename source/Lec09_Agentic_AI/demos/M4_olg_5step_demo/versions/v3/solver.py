"""V3 — two-asset OLG with aggregate TFP (numpy-based grid solver).

State: ``(age, ia, ib, iz_idio, iz_tfp)``.

V3 uses an averaged-equilibrium approximation: a single equilibrium ``r*``
(not state-contingent in TFP) such that integrated market clearing holds —
``E[K_supply] = E[K_demand(r*, z_tfp)]`` over the joint stationary
distribution of ``(z_idio, z_tfp)``. Households perceive r* as constant but
TFP affects their realized wage and continuation values.

V3 collapses to V2 when ``n_tfp = 1``: TFP is constant at 1.0, the firm
prices match V2's, and the solver returns the same equilibrium.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np

from tfp import tauchen, tfp_stationary_distribution
from two_asset_spec import (
    TwoAssetParams,
    aggregate_labor_supply,
    asset_grid,
    bond_grid,
    firm_prices_from_r_tfp,
    labor_profile,
    stationary_income_distribution,
)

NEG_INF = -1.0e30
OLGParams = TwoAssetParams


@dataclass
class HouseholdSolution:
    asset_grid: List[float]
    bond_grid: List[float]
    tfp_grid: List[float]
    value: np.ndarray                # [age, ia, ib, iz_idio, iz_tfp]
    policy_a_idx: np.ndarray
    policy_b_idx: np.ndarray
    consumption: np.ndarray
    distribution: np.ndarray
    mean_assets_by_age: List[float]
    mean_bonds_by_age: List[float]
    mean_consumption_by_age: List[float]
    aggregate_assets: float
    aggregate_bonds: float
    aggregate_labor: float
    euler_max: float
    euler_p95: float
    borrowing_constraint_mass: float


@dataclass
class EquilibriumResult:
    params: TwoAssetParams
    interest_rate: float
    bond_price: float
    expected_wage: float                       # E[w(z_tfp)] over stationary TFP dist
    expected_capital_demand: float             # E[K_demand(r*, z_tfp)]
    capital_supply: float
    labor: float
    excess_assets: float
    iterations: int
    bracketed: bool
    search_history: List[dict]
    household: HouseholdSolution
    tfp_grid: List[float]
    tfp_transition: List[List[float]]
    tfp_stationary: List[float]


def utility_arr(c: np.ndarray, gamma: float) -> np.ndarray:
    out = np.full_like(c, NEG_INF)
    valid = c > 0.0
    if abs(gamma - 1.0) < 1.0e-12:
        out[valid] = np.log(c[valid])
    else:
        out[valid] = (c[valid] ** (1.0 - gamma) - 1.0) / (1.0 - gamma)
    return out


def solve_household(r: float, params: TwoAssetParams, tfp_grid: List[float],
                    tfp_transition: List[List[float]]) -> HouseholdSolution:
    a_grid = np.array(asset_grid(params))
    b_grid = np.array(bond_grid(params))
    n_a = len(a_grid)
    n_b = len(b_grid)
    n_z = len(params.income_states)
    n_tfp = len(tfp_grid)
    n_age = params.n_cohorts

    Pi_z = np.array(params.income_transition)
    Pi_tfp = np.array(tfp_transition)
    z_grid = np.array(params.income_states)
    z_tfp_arr = np.array(tfp_grid)

    labor = labor_profile(n_age)
    mean_work_labor = sum(labor[: params.retire_age_index]) / params.retire_age_index

    # Pre-compute wages and pensions per TFP state.
    wages = np.zeros(n_tfp)
    pensions = np.zeros(n_tfp)
    for itfp, z_t in enumerate(tfp_grid):
        _, _, w = firm_prices_from_r_tfp(r, z_t, params)
        wages[itfp] = w
        pensions[itfp] = params.pension_replacement * w * mean_work_labor

    # Storage: [age, ia, ib, iz, itfp]
    shape = (n_age, n_a, n_b, n_z, n_tfp)
    value = np.zeros(shape)
    policy_a_idx = np.zeros(shape, dtype=np.int32)
    policy_b_idx = np.zeros(shape, dtype=np.int32)
    consumption = np.zeros(shape)

    # Vectorized backward induction. For each (age, ia, ib, iz, itfp), evaluate
    # all (ia_next, ib_next) candidates simultaneously.
    psi = params.psi_k
    if psi > 0.0:
        adj_cost = psi * (a_grid[:, None] - a_grid[None, :]) ** 2  # (n_a, n_a) indexed [ia, ia_next]
    else:
        adj_cost = None

    for age in reversed(range(n_age)):
        labor_age = labor[age]

        if age < n_age - 1:
            v_next = value[age + 1]                                       # (n_a, n_b, n_z, n_tfp)
            ev_jz = np.einsum("zj,abjt->abzt", Pi_z, v_next)              # avg over idio
            ev = np.einsum("ts,abzs->abzt", Pi_tfp, ev_jz)                # avg over tfp
        else:
            ev = np.zeros((n_a, n_b, n_z, n_tfp))

        for iz in range(n_z):
            z_val = z_grid[iz]
            for itfp in range(n_tfp):
                w = wages[itfp]
                pension = pensions[itfp]
                income = (w * labor_age * z_val) if age < params.retire_age_index else pension

                # resources[ia, ib] = (1+r) * a_grid[ia] + b_grid[ib] + income
                resources = (1.0 + r) * a_grid[:, None] + b_grid[None, :] + income  # (n_a, n_b)

                # c[ia, ib, ia_next, ib_next]
                c = (resources[:, :, None, None]
                     - a_grid[None, None, :, None]
                     - params.bond_price * b_grid[None, None, None, :])
                if adj_cost is not None:
                    c = c - adj_cost[:, None, :, None]

                # candidate utility + beta * E[V_next at (ia_next, ib_next, iz, itfp)]
                util = utility_arr(c, params.gamma)
                cont = ev[:, :, iz, itfp]              # shape (n_a, n_b)
                # cand[ia, ib, ia_next, ib_next]
                cand = util + params.beta * cont[None, None, :, :]

                # Set invalid (negative consumption) candidates to NEG_INF
                cand = np.where(c > 0.0, cand, NEG_INF)

                # argmax over (ia_next, ib_next): flatten the last 2 dims
                flat = cand.reshape(n_a, n_b, n_a * n_b)
                best_flat_idx = np.argmax(flat, axis=2)
                best_ia_next = best_flat_idx // n_b
                best_ib_next = best_flat_idx % n_b
                best_val = np.max(flat, axis=2)
                # best consumption
                best_c = np.take_along_axis(
                    c.reshape(n_a, n_b, n_a * n_b),
                    best_flat_idx[:, :, None], axis=2,
                ).squeeze(axis=2)

                value[age, :, :, iz, itfp] = best_val
                policy_a_idx[age, :, :, iz, itfp] = best_ia_next
                policy_b_idx[age, :, :, iz, itfp] = best_ib_next
                consumption[age, :, :, iz, itfp] = np.maximum(best_c, params.min_consumption)

    # Distribution propagation forward
    distribution = np.zeros(shape)
    z0 = stationary_income_distribution(params)
    tfp_stationary = tfp_stationary_distribution(tfp_transition)
    ib0 = int(np.argmin(np.abs(b_grid)))
    for iz, p_iz in enumerate(z0):
        for itfp, p_itfp in enumerate(tfp_stationary):
            distribution[0, 0, ib0, iz, itfp] = p_iz * p_itfp / n_age

    for age in range(n_age - 1):
        for iz in range(n_z):
            for itfp in range(n_tfp):
                mass_block = distribution[age, :, :, iz, itfp]   # (n_a, n_b)
                if mass_block.sum() == 0.0:
                    continue
                pa = policy_a_idx[age, :, :, iz, itfp]
                pb = policy_b_idx[age, :, :, iz, itfp]
                # For each (ia, ib), distribute mass to (pa, pb, jz, jtfp)
                for jz in range(n_z):
                    for jtfp in range(n_tfp):
                        prob = Pi_z[iz, jz] * Pi_tfp[itfp, jtfp]
                        # Use np.add.at for scattered accumulation
                        np.add.at(
                            distribution[age + 1, :, :, jz, jtfp],
                            (pa, pb),
                            mass_block * prob,
                        )

    # Aggregates
    a_field = a_grid[None, :, None, None, None]                    # broadcastable
    b_field = b_grid[None, None, :, None, None]
    z_field = z_grid[None, None, None, :, None]
    aggregate_assets = float((distribution * a_field).sum())
    aggregate_bonds = float((distribution * b_field).sum())
    work_mask = np.zeros((n_age, 1, 1, 1, 1))
    for age in range(params.retire_age_index):
        work_mask[age, 0, 0, 0, 0] = 1.0
    labor_arr = np.array(labor)[:, None, None, None, None]
    aggregate_labor = float((distribution * work_mask * labor_arr * z_field).sum())

    mean_assets_by_age: List[float] = []
    mean_bonds_by_age: List[float] = []
    mean_consumption_by_age: List[float] = []
    borrowing_mass = 0.0
    total_mass = float(distribution.sum())
    for age in range(n_age):
        age_dist = distribution[age]
        m = float(age_dist.sum())
        if m > 0.0:
            mean_assets_by_age.append(float((age_dist * a_grid[:, None, None, None]).sum()) / m)
            mean_bonds_by_age.append(float((age_dist * b_grid[None, :, None, None]).sum()) / m)
            mean_consumption_by_age.append(float((age_dist * consumption[age]).sum()) / m)
        else:
            mean_assets_by_age.append(0.0)
            mean_bonds_by_age.append(0.0)
            mean_consumption_by_age.append(0.0)
        borrowing_mass += float(distribution[age, 0, :, :, :].sum())

    # Euler residuals (capital only)
    errors: List[float] = []
    for age in range(n_age - 1):
        for ia in range(n_a):
            for ib in range(n_b):
                for iz in range(n_z):
                    for itfp in range(n_tfp):
                        mass = distribution[age, ia, ib, iz, itfp]
                        if mass <= 1.0e-12:
                            continue
                        ia_next = int(policy_a_idx[age, ia, ib, iz, itfp])
                        ib_next = int(policy_b_idx[age, ia, ib, iz, itfp])
                        if ia_next in (0, n_a - 1):
                            continue
                        c = max(consumption[age, ia, ib, iz, itfp], params.min_consumption)
                        lhs = c ** (-params.gamma)
                        rhs = 0.0
                        for jz in range(n_z):
                            for jtfp in range(n_tfp):
                                cn = max(consumption[age + 1, ia_next, ib_next, jz, jtfp], params.min_consumption)
                                rhs += Pi_z[iz, jz] * Pi_tfp[itfp, jtfp] * cn ** (-params.gamma)
                        rhs *= params.beta * (1.0 + r)
                        if lhs > 0.0 and rhs > 0.0:
                            errors.append(abs(math.log(rhs / lhs)))

    if errors:
        s = sorted(errors)
        euler_max = s[-1]
        euler_p95 = s[min(len(s) - 1, int(0.95 * len(s)))]
    else:
        euler_max = 0.0
        euler_p95 = 0.0

    return HouseholdSolution(
        asset_grid=list(map(float, a_grid)),
        bond_grid=list(map(float, b_grid)),
        tfp_grid=list(tfp_grid),
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
        euler_p95=euler_p95,
        borrowing_constraint_mass=borrowing_mass / total_mass if total_mass else 0.0,
    )


def expected_capital_demand(r: float, params: TwoAssetParams,
                             tfp_grid: List[float], tfp_stationary: List[float]) -> float:
    total = 0.0
    for itfp, z_t in enumerate(tfp_grid):
        K_d, _, _ = firm_prices_from_r_tfp(r, z_t, params)
        total += tfp_stationary[itfp] * K_d
    return total


def excess_assets_at_r(r: float, params: TwoAssetParams,
                        tfp_grid: List[float], tfp_transition: List[List[float]],
                        tfp_stationary: List[float]):
    hh = solve_household(r, params, tfp_grid, tfp_transition)
    K_demand_avg = expected_capital_demand(r, params, tfp_grid, tfp_stationary)
    return hh.aggregate_assets - K_demand_avg, hh, K_demand_avg


def solve_equilibrium(params: TwoAssetParams | None = None) -> EquilibriumResult:
    params = params or TwoAssetParams()
    tfp_grid, tfp_transition = tauchen(params.n_tfp, params.rho_tfp, params.sigma_tfp, params.tauchen_m)
    tfp_stationary = tfp_stationary_distribution(tfp_transition)
    expected_wage_at = lambda r: sum(
        tfp_stationary[i] * firm_prices_from_r_tfp(r, tfp_grid[i], params)[2]
        for i in range(len(tfp_grid))
    )
    history: List[dict] = []
    grid_rs = [
        params.r_min + (params.r_max - params.r_min) * i / (params.r_grid_search_points - 1)
        for i in range(params.r_grid_search_points)
    ]

    grid_results = []
    for i, r in enumerate(grid_rs):
        gap, hh, K_d_avg = excess_assets_at_r(r, params, tfp_grid, tfp_transition, tfp_stationary)
        grid_results.append((r, gap, hh, K_d_avg))
        history.append({"phase": "grid", "iteration": i, "r": r, "gap": gap, "capital_demand": K_d_avg})

    bracket = None
    for j in range(len(grid_results) - 1):
        if grid_results[j][1] * grid_results[j + 1][1] <= 0.0:
            bracket = (grid_results[j], grid_results[j + 1])
            break

    if bracket is None:
        best = min(grid_results, key=lambda row: abs(row[1]))
        r_star, gap, hh, K_d_avg = best
        labor = aggregate_labor_supply(params)
        return EquilibriumResult(
            params=params, interest_rate=r_star, bond_price=params.bond_price,
            expected_wage=expected_wage_at(r_star), expected_capital_demand=K_d_avg,
            capital_supply=hh.aggregate_assets, labor=labor, excess_assets=gap,
            iterations=0, bracketed=False, search_history=history,
            household=hh, tfp_grid=list(tfp_grid),
            tfp_transition=tfp_transition, tfp_stationary=tfp_stationary,
        )

    (r_lo, gap_lo, _, _), (r_hi, gap_hi, _, _) = bracket
    iterations = 0
    last = None
    for it in range(params.ge_max_iter):
        iterations = it + 1
        r_mid = 0.5 * (r_lo + r_hi)
        gap, hh, K_d_avg = excess_assets_at_r(r_mid, params, tfp_grid, tfp_transition, tfp_stationary)
        history.append({"phase": "bisect", "iteration": it, "r": r_mid, "gap": gap, "capital_demand": K_d_avg})
        last = (r_mid, gap, hh, K_d_avg)
        if abs(gap) < params.ge_tolerance:
            break
        if gap * gap_lo <= 0.0:
            r_hi, gap_hi = r_mid, gap
        else:
            r_lo, gap_lo = r_mid, gap

    r_mid, gap, hh, K_d_avg = last  # type: ignore[misc]
    labor = aggregate_labor_supply(params)
    return EquilibriumResult(
        params=params, interest_rate=r_mid, bond_price=params.bond_price,
        expected_wage=expected_wage_at(r_mid), expected_capital_demand=K_d_avg,
        capital_supply=hh.aggregate_assets, labor=labor, excess_assets=gap,
        iterations=iterations, bracketed=True, search_history=history,
        household=hh, tfp_grid=list(tfp_grid),
        tfp_transition=tfp_transition, tfp_stationary=tfp_stationary,
    )


def result_summary(result: EquilibriumResult) -> dict:
    return {
        "interest_rate": result.interest_rate,
        "bond_price": result.bond_price,
        "expected_wage": result.expected_wage,
        "expected_capital_demand": result.expected_capital_demand,
        "capital_supply": result.capital_supply,
        "labor": result.labor,
        "excess_assets": result.excess_assets,
        "iterations": result.iterations,
        "bracketed": result.bracketed,
        "borrowing_constraint_mass": result.household.borrowing_constraint_mass,
        "aggregate_bonds": result.household.aggregate_bonds,
        "tfp_grid": result.tfp_grid,
        "tfp_stationary": result.tfp_stationary,
        "euler_max": result.household.euler_max,
        "euler_p95": result.household.euler_p95,
        "euler_p95_log10": math.log10(max(result.household.euler_p95, 1.0e-30)),
    }


def write_result_files(result: EquilibriumResult, build_dir: Path) -> None:
    build_dir.mkdir(exist_ok=True)
    summary = result_summary(result)
    summary["params"] = asdict(result.params)
    (build_dir / "summary_v3.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
