# V3 — Model Specification (Delta from V2)

V3 introduces **aggregate TFP** as an AR(1) process discretized via Tauchen's method. The household state expands to `(age, a, b, z_idio, z_tfp)`. V2's deterministic two-asset structure is otherwise preserved.

The full V0/V1/V2 specs remain in force. This delta describes only the new TFP layer.

## What V3 adds

- **Aggregate state**: log-TFP follows AR(1) with persistence `rho_tfp` and innovation std `sigma_tfp`.
- **Tauchen discretization** to `n_tfp` states (default 3 — enough for class, fast in numpy).
- **TFP enters production**: `Y = z_tfp * K^α * L^(1-α)`. Wage and capital demand are now functions of `(r, z_tfp)`.
- **Household state expansion**: `(age, a, b, z_idio, z_tfp)`. Households perceive transitions in both idiosyncratic and aggregate states.
- **Numpy-based solver**: V3's solver migrates to numpy arrays for vectorized backward induction. Pure-stdlib Python lists become unworkable at this state-space size.

## Equilibrium concept (delta)

V3 uses an **averaged equilibrium**: a single equilibrium interest rate `r*` satisfies

```math
∫ a dμ  =  E_{stationary z_tfp}[ K_demand(r*, z_tfp) ]
```

Households perceive `r*` as constant across TFP states; the firm's marginal-product condition holds in expectation rather than period-by-period. This is a deliberate simplification — true Krusell-Smith aggregate-uncertainty equilibrium with state-contingent prices arrives in V4 with the NN policy.

## Functional forms

| Object | V3 form | New parameter |
|---|---|---|
| Production | `Y = z_tfp * K^α * L^(1-α)` | (none) |
| Wage | `w(z_tfp) = (1-α) * z_tfp * (K_demand/L)^α` | (none) |
| Capital demand | `K_demand(r, z_tfp) = L * ((r+δ) / (α z_tfp))^(1/(α-1))` | (none) |
| TFP process | AR(1) on log z, discretized via Tauchen | `rho_tfp`, `sigma_tfp`, `n_tfp` |

## Parameter values (new)

| Symbol | Value | Rationale |
|---|---|---|
| `rho_tfp` | 0.85 | persistence in yearly time; per-period persistence after period_length compression |
| `sigma_tfp` | 0.03 | small TFP innovation std |
| `n_tfp` | 3 | minimal grid that shows TFP variation while staying class-fast |
| `tauchen_m` | 3.0 | grid width = ±3 std of unconditional log-TFP |

## Validation gate

| Check | Tolerance |
|---|---|
| **V3 with `n_tfp=1` reduces to V2** (single TFP state at z=1.0) | identical r and K |
| Distribution mass = 1 | 1e-8 |
| Feasibility residual = 0 | exact |
| Bracketed equilibrium found at default settings | strict |

## Implementation pointer

`versions/v3/{solver.py, two_asset_spec.py, tfp.py, residuals.py}`. The 5D state space (`age × a × b × z_idio × z_tfp`) makes pure-Python lists too slow; the solver uses numpy vectorization on the inner candidate-policy loop. Default classroom grids: `(7, 10, 4, 3, 3) = 2520 cells`. One `solve_equilibrium` call: < 0.1 s.

## Why this matters for V4

V3 demonstrates that the grid solver still works at 5D state but is starting to feel the curse of dimensionality. Adding finer TFP discretization, finer asset grids, or more idiosyncratic states would scale runtime non-linearly. V4 replaces the grid with a neural network whose runtime grows linearly in state-space size (just bigger inputs), unlocking the path to a true Krusell-Smith aggregate-uncertainty equilibrium.
