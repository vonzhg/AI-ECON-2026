"""V3 — residuals: V2's interface plus TFP-conditional capital demand info."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class ResidualReport:
    euler_max: float
    euler_p95: float
    euler_p95_log10: float
    capital_market: float
    distribution_mass: float
    distribution_mass_residual: float
    feasibility_min_consumption: float
    feasibility_residual: float
    aggregate_bonds: float
    bond_price: float
    bond_market_residual: float
    expected_capital_demand: float                # NEW IN V3
    capital_supply: float                          # NEW IN V3
    n_tfp: int                                     # NEW IN V3 (informational)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


DEFAULT_TOLERANCES: Dict[str, float] = {
    "euler_p95_log10": 1.0,
    "capital_market": 5.0e-3,
    "distribution_mass_residual": 1.0e-8,
    "feasibility_residual": 0.0,
    "bond_market_residual": 1.0,
}


def compute_residuals(result) -> ResidualReport:
    household = result.household

    distribution = household.distribution  # numpy array
    distribution_mass = float(distribution.sum())
    distribution_mass_residual = abs(distribution_mass - 1.0)

    min_consumption = float(np.min(household.consumption))
    feasibility_residual = max(0.0, -min_consumption)

    euler_p95_log10 = math.log10(max(household.euler_p95, 1.0e-30))

    return ResidualReport(
        euler_max=household.euler_max,
        euler_p95=household.euler_p95,
        euler_p95_log10=euler_p95_log10,
        capital_market=result.excess_assets,
        distribution_mass=distribution_mass,
        distribution_mass_residual=distribution_mass_residual,
        feasibility_min_consumption=min_consumption,
        feasibility_residual=feasibility_residual,
        aggregate_bonds=household.aggregate_bonds,
        bond_price=result.bond_price,
        bond_market_residual=household.aggregate_bonds,
        expected_capital_demand=result.expected_capital_demand,
        capital_supply=result.capital_supply,
        n_tfp=len(result.tfp_grid),
    )


def check_residuals(
    report: ResidualReport,
    tolerances: Dict[str, float] | None = None,
) -> List[Tuple[str, bool]]:
    tol = dict(DEFAULT_TOLERANCES)
    if tolerances:
        tol.update(tolerances)
    return [
        ("euler_p95_log10 < tol",
         report.euler_p95_log10 < tol["euler_p95_log10"]),
        ("|capital_market| < tol",
         abs(report.capital_market) < tol["capital_market"]),
        ("distribution_mass_residual < tol",
         report.distribution_mass_residual < tol["distribution_mass_residual"]),
        ("feasibility_residual <= tol",
         report.feasibility_residual <= tol["feasibility_residual"]),
        ("|bond_market_residual| < tol (informational)",
         abs(report.bond_market_residual) < tol["bond_market_residual"]),
    ]


def format_report(report: ResidualReport, title: str = "V3 residual report") -> str:
    rows = [
        f"{title}",
        "-" * max(len(title), 60),
        f"  euler_max                       {report.euler_max: .6e}",
        f"  euler_p95                       {report.euler_p95: .6e}",
        f"  euler_p95_log10                 {report.euler_p95_log10: .6f}",
        f"  capital_market (excess assets)  {report.capital_market:+.6e}",
        f"  capital_supply                  {report.capital_supply: .6f}",
        f"  expected_capital_demand         {report.expected_capital_demand: .6f}",
        f"  distribution_mass               {report.distribution_mass: .12f}",
        f"  distribution_mass_residual      {report.distribution_mass_residual: .3e}",
        f"  feasibility_min_consumption     {report.feasibility_min_consumption: .6e}",
        f"  feasibility_residual            {report.feasibility_residual: .3e}",
        f"  bond_price (exogenous in V3)    {report.bond_price: .6f}",
        f"  aggregate_bonds                 {report.aggregate_bonds:+.6e}",
        f"  bond_market_residual            {report.bond_market_residual:+.6e}",
        f"  n_tfp states                    {report.n_tfp}",
    ]
    return "\n".join(rows)
