# V1 — Model Specification (Delta from V0)

V1 changes **no economics**. It adds an explicit residual-diagnostic interface that V0 computed implicitly inside `solver.py`. Every later version (V2..V5) plugs into this interface instead of re-inventing one.

The full V0 spec at `versions/v0/model_spec.md` remains in force; the cells below describe only what's new in V1.

## What V1 adds

A `ResidualReport` dataclass (in `versions/v1/residuals.py`) with four named residuals — one per equilibrium condition in V0:

| Residual | Measures the slack of | V0 source |
|---|---|---|
| `euler_max`, `euler_p95`, `euler_p95_log10` | Household Euler equation across active states | `HouseholdSolution.euler_max`, `.euler_p95` |
| `capital_market` | `K_supply − K_demand` (firm/household clearing) | `EquilibriumResult.excess_assets` |
| `distribution_mass_residual` | `|Σ μ − 1|` (consistency of the propagated distribution) | computed in V0's validation cell |
| `feasibility_residual` | `max(0, −min c)` (consumption non-negativity at every state) | not previously surfaced; V0 enforced it inside backward induction |

Plus three helpers:

- `compute_residuals(result)` — build a report from any `EquilibriumResult`.
- `check_residuals(report, tolerances=None)` — apply tolerances; return `[(label, ok), …]`.
- `format_report(report, title)` — pretty-print as a fixed-width table.

## What V1 does NOT add

- New economic objects (no bonds, no aggregate state, no extra cohorts).
- New solver code. V1 reuses V0's `solver.py` byte-for-byte.
- New parameters in `OLGParams`.

## Validation gate

Every residual in V1 must reconcile with V0's reported numbers to **1e-10**. If they don't, the bug is in `residuals.py` (V0 was already validated). The notebook's Section 2 prints both V0's implicit numbers and V1's explicit numbers side-by-side; `tests/test_v1_residuals.py` enforces the reconciliation.
