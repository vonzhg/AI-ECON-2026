"""V1 — explicit residual diagnostics for the V0 OLG equilibrium.

V0 computes Euler errors and the capital-market gap implicitly inside
``solver.py``. V1 factors them into a uniform ``ResidualReport`` interface
so every later version (V2..V5) can plug in instead of re-inventing.

Public surface:

- ``ResidualReport``        — dataclass holding the four named residuals
- ``compute_residuals``     — build a ``ResidualReport`` from an EquilibriumResult
- ``format_report``         — pretty-print the report as a table
- ``check_residuals``       — apply tolerances; return [(label, ok), ...]
- ``DEFAULT_TOLERANCES``    — the V0 validation-gate tolerances

This module imports nothing beyond the Python stdlib.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ResidualReport:
    """Four named residuals at a candidate equilibrium.

    Each field measures the slack of one equilibrium condition in the V0
    spec (see ``versions/v0/model_spec.md``):

    * ``euler_*``                  — household optimality
    * ``capital_market``           — firm asset demand vs household supply
    * ``distribution_mass_*``      — total mass of the stationary distribution
    * ``feasibility_*``            — consumption non-negativity constraint
    """

    euler_max: float
    euler_p95: float
    euler_p95_log10: float
    capital_market: float
    distribution_mass: float
    distribution_mass_residual: float
    feasibility_min_consumption: float
    feasibility_residual: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


DEFAULT_TOLERANCES: Dict[str, float] = {
    "euler_p95_log10": 1.0,
    "capital_market": 5.0e-3,
    "distribution_mass_residual": 1.0e-8,
    "feasibility_residual": 0.0,
}


def compute_residuals(result) -> ResidualReport:
    """Build a ResidualReport from an EquilibriumResult.

    ``result`` is whatever ``solver.solve_equilibrium`` returns; this function
    relies only on documented public fields, so it works unchanged for V0
    and any future version that preserves the EquilibriumResult API.
    """
    household = result.household

    distribution_mass = 0.0
    for age_dist in household.distribution:
        for asset_dist in age_dist:
            for mass in asset_dist:
                distribution_mass += mass
    distribution_mass_residual = abs(distribution_mass - 1.0)

    min_consumption = float("inf")
    for age_dist in household.consumption:
        for asset_row in age_dist:
            for c in asset_row:
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
    ]


def format_report(report: ResidualReport, title: str = "Residual report") -> str:
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
    ]
    return "\n".join(rows)
