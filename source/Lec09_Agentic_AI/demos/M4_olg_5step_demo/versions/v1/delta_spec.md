# V1 Delta Spec — Seven-Cohort Lifecycle

V1 keeps V0's deep equilibrium net machinery and replaces V0's three-cohort demographics with a seven-cohort lifecycle. The spec for everything not listed below is unchanged from `versions/v0/model_spec.md`.

## Stage 1 — what changes

### Demographics

| Element | V0 | V1 |
|---|---|---|
| Cohorts $N$ | 3 | 7 |
| Period length $\tau$ (years) | $\approx 20$ | $72/7 \approx 10.286$ |
| Working ages | 0, 1 | 0, 1, 2, 3, 4, 5 |
| Retired age | 2 | 6 |

Each cohort indexed $j \in \{0, \ldots, N-1\}$ has unit measure. Cohort age 0 is born with zero wealth; cohort age $N-1$ is retired and consumes everything. Six savers in V1 (vs two in V0).

### Labour profile

A **hump-shaped** vector $\boldsymbol{\varepsilon} = (0.7, 0.9, 1.0, 1.05, 1.0, 0.9, 0.0)$ replacing V0's two-element $(\varepsilon_y, \varepsilon_m) = (0.6, 1.0)$. Aggregate efficiency labour $L = \sum_j \varepsilon_j = 5.35$.

### Discount factor and depreciation

Re-anchored on yearly primitives:

| Yearly | Per-period (V1) |
|---|---|
| $\beta_y = 0.97$ | $\beta = \beta_y^\tau \approx 0.731$ |
| $\delta_y = 0.06$ | $\delta = 1 - (1-\delta_y)^\tau \approx 0.471$ |

V0's $\beta = 0.85$ and $\delta = 0.30$ were calibrated to a 20-year period; the V1 numbers are the natural rescaling for a 10.3-year period at sensible yearly primitives.

### Aggregate state

$s_t = (Z_t, a^1_t, \ldots, a^{N-1}_t) \in \{Z_\text{lo}, Z_\text{hi}\} \times \mathbb{R}_+^{N-1}$. The state dimension grows from 3 (V0) to 7 (V1).

## Stage 2 — what changes

The recursive equilibrium has $N-1 = 6$ Euler equations instead of 2. The cohort age $j$ saves to wealth $a^{j+1}_{t+1}$ for $j \in \{0, \ldots, N-2\}$:

$$
u'(c^j_t) = \beta\,\mathbb{E}_t\!\left[(1+r_{t+1})\,u'(c^{j+1}_{t+1})\right], \quad j = 0, \ldots, N-2.
$$

The aggregate-capital identity becomes $K_t = \sum_{j=1}^{N-1} a^j_t$.

## Stage 3 — what changes

The algorithm is identical in spirit to V0:

- single MLP, sigmoid head, $N-1$ outputs (was 2);
- pretraining target now a single scalar $\bar s = 0.40$ across all cohorts (V0 had different targets per cohort);
- network width raised from 64 to 128 because input dimension grew from 3 to 7;
- training budget 6,000 steps (was 5,000) so the larger problem has more iterations to converge.

Closed-form expectations are preserved (still 2-state TFP at V1).

## Stage 4 — what changes

The Euler-residual loop is vectorised over the cohort dimension. Each `cohort_decisions(...)` call returns tensors of shape `(batch, N-1)` for next-period wealth and `(batch, N)` for consumption. See `pseudocode.md` for the full vectorised form.

## Validation gate

`simulate.validation_gate(sim, losses)` returns:

- `training_progressed`: best MSE drops by ≥ 5× from the first step.
- `procyclical_capital_E[K|hi] > E[K|lo]`.
- `rms_euler_residual_<_10pct`: RMS of best 50 trailing MSE under 0.10.
- `consumption_grows_with_age`: $\bar c_{N-1} > \bar c_0$ on the ergodic distribution (the standard lifecycle pattern under positive interest rates).
- `savings_peak_in_pre_retirement`: peak mean savings rate occurs at cohort age $\geq 2$ (not at the youngest).

The deterministic-SS check (set $Z_\text{lo} = Z_\text{hi} = 1$, retrain briefly) is run by `tests/test_v1_smoke.py`. Path standard deviation of $K$ must be under $10^{-3}$.

## Out of scope at V1

Everything not listed above stays as in V0. In particular: still 2-state TFP (V2 changes that), still one asset (V3 changes that), still no adjustment cost (V4), still no homotopy (V5).
