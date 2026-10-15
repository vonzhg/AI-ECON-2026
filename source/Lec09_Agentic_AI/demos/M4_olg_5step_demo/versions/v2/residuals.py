"""V2 — residual report extending V1 with bond diagnostics.

Same interface as V1: ``ResidualReport`` dataclass, ``compute_residuals``,
``check_residuals``, ``format_report``. V2 adds two fields tracking the
bond market — ``aggregate_bonds`` (sum of household bond holdings,
weighted by mass) and ``bond_market_residual`` (which equals
``aggregate_bonds`` when bonds are in zero net supply).

In V2 bonds are in *elastic* supply at fixed ``bond_price``, so
``bond_market_residual`` is informational only. V4's NN solver enforces
``bond_market_residual = 0`` as part of its loss function.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple


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
    aggregate_bonds: float                # NEW IN V2
    bond_price: float                     # NEW IN V2
    bond_market_residual: float           # NEW IN V2 (= aggregate_bonds when net-supply = 0)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


DEFAULT_TOLERANCES: Dict[str, float] = {
    "euler_p95_log10": 1.0,
    "capital_market": 5.0e-3,
    "distribution_mass_residual": 1.0e-8,
    "feasibility_residual": 0.0,
    "bond_market_residual": 1.0,         # V2 doesn't enforce; V4 will tighten
}


def compute_residuals(result) -> ResidualReport:
    household = result.household

    distribution_mass = 0.0
    for age_dist in household.distribution:
        for asset_dist in age_dist:
            for bond_dist in asset_dist:
                for mass in bond_dist:
                    distribution_mass += mass
    distribution_mass_residual = abs(distribution_mass - 1.0)

    min_consumption = float("inf")
    for age_dist in household.consumption:
        for asset_row in age_dist:
            for bond_row in asset_row:
                for c in bond_row:
                    if c < min_consumption:
                        min_consumption = c
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


def format_report(report: ResidualReport, title: str = "V2 residual report") -> str:
    rows = [
        f"{title}",
        "-" * max(len(title), 60),
        f"  euler_max                       {report.euler_max: .6e}",
        f"  euler_p95                       {report.euler_p95: .6e}",
        f"  euler_p95_log10                 {report.euler_p95_log10: .6f}",
        f"  capital_market (excess assets)  {report.capital_market:+.6e}",
        f"  distribution_mass               {report.distribution_mass: .12f}",
        f"  distribution_mass_residual      {report.distribution_mass_residual: .3e}",
        f"  feasibility_min_consumption     {report.feasibility_min_consumption: .6e}",
        f"  feasibility_residual            {report.feasibility_residual: .3e}",
        f"  bond_price (exogenous in V2)    {report.bond_price: .6f}",
        f"  aggregate_bonds                 {report.aggregate_bonds:+.6e}",
        f"  bond_market_residual            {report.bond_market_residual:+.6e}",
    ]
    return "\n".join(rows)
