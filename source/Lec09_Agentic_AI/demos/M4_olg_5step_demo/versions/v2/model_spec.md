# V2 — Model Specification (Delta from V1)

V2 introduces a **second asset** (bonds) alongside capital. Aggregate uncertainty stays out (deferred to V3). Bonds are in *elastic supply* at an exogenous price; bond market clearing arrives in V4.

The full V0 spec at `versions/v0/model_spec.md` and the V1 residual interface at `versions/v1/model_spec.md` remain in force; this delta describes only the new structure.

## What V2 adds

- **Second asset**: `b ∈ [b_min, b_max]` with grid of `bond_grid_size` points.
- **Bond price**: exogenous `p_b` (default 0.98 — bonds give net return ≈ 2% per model period).
- **Borrowing constraint on bonds**: `b' ≥ b_min` (default −0.05 — small borrowing window).
- **Capital adjustment cost**: `psi_k * (a' − a)^2` (default 0; will be raised in V3 to give capital its own friction).

## State, controls, budget

State: `(age, a, b, z)`.

Controls: `(a', b')`.

Budget (working ages):
```
c + a' + p_b * b' + psi_k * (a' - a)^2  =  (1 + r) * a + b + w * labor_age * z
```

Budget (retirement, age ≥ retire_age_index):
```
c + a' + p_b * b' + psi_k * (a' - a)^2  =  (1 + r) * a + b + pension
```

Constraints: `a' ≥ 0`, `b' ≥ b_min`, `c > 0`.

## Equilibrium (delta)

Same as V1: a single-price recursive competitive equilibrium where `r*` clears the **capital market**:
```
∫ a dμ  =  K_demand(r*, w(r*))
```

Bond market clearing is **not** imposed in V2 — bonds are in elastic supply. The aggregate bond position `∫ b dμ` is reported as a diagnostic but not enforced to be zero. V4 introduces endogenous bond pricing where this becomes a residual the NN must drive to zero.

## Functional forms

| Object | Form | New parameter |
|---|---|---|
| Bond return | gross return `1 / p_b` per period | `bond_price` |
| Bond grid | linear | `bond_grid_size`, `bond_min`, `bond_max` |
| Capital adjustment cost | quadratic `psi_k * (a'−a)^2` | `psi_k` |

## Parameter values (new)

| Symbol | Value | Rationale |
|---|---|---|
| `bond_price` | 0.98 | net bond return ≈ 2% per period |
| `bond_min` | −0.05 | small borrowing window so the constraint occasionally binds |
| `bond_max` | 0.5 | upper end of lending |
| `bond_grid_size` | 5 | classroom-fast (5×15 = 75 capital-bond pairs per state) |
| `psi_k` | 0.0 | off by default; V3 raises it |

## Validation gate

| Check | Tolerance |
|---|---|
| **V2 with bonds disabled reduces to V1** (bond_grid_size=1, bond_min=bond_max=0, psi_k=0): identical r and K | 1e-10 |
| Distribution mass = 1 | 1e-8 |
| Feasibility residual = 0 | exact |
| Aggregate bonds in `[b_min, b_max]` | strict |

## Implementation pointer

`versions/v2/{solver.py, two_asset_spec.py, residuals.py}`. The solver uses 4D Python lists for the state space; runtime grows as `O(n_a^2 * n_b^2 * n_z * n_age)`. With default classroom grids (n_a=15, n_b=5, n_z=3, n_age=7), one `solve_equilibrium` call takes < 1 second.

## Why this design

V2's purpose is to demonstrate **state-space expansion** in a controlled way. The grid blowup is real but still tractable on stdlib Python. V3 keeps this structure and adds aggregate TFP; V4 then replaces the grid policy with a neural network when the grid becomes infeasible.
