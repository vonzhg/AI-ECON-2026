# V4 Delta Spec — Capital Adjustment Cost

V4 keeps everything in V3 and adds a convex capital adjustment cost
$\frac{\psi_K}{2}\,(k^{j+1}_{t+1} - k^j_t)^2$ paid out of period-$t$
consumption.  Setting $\psi_K = 0$ recovers V3 exactly.

## Stage 1 — what changes

### Calibration

| Parameter | Value | Notes |
|---|---|---|
| $\psi_K$ | $0.50$ | mild cost; encourages smooth capital paths but doesn't dominate prices |

### Budget

For cohorts age $j \in \{0, \ldots, N-2\}$:

$$
c^j_t + k^{j+1}_{t+1} + p_{b,t}\,b^{j+1}_{t+1} + \frac{\psi_K}{2}\bigl(k^{j+1}_{t+1} - k^j_t\bigr)^2 = I^j_t.
$$

Cohort age 0 enters with $k^0_t = 0$, so its adjustment cost is $\frac{\psi_K}{2}(k^1_{t+1})^2$, i.e. quadratic in start-of-life capital.

## Stage 2 — what changes

The capital Euler equation gains the marginal-adjustment factor:

$$
u'(c^j_t)\bigl(1 + \psi_K\,(k^{j+1}_{t+1} - k^j_t)\bigr) = \beta\,\mathbb{E}_t\!\left[(1+r_{t+1})\,u'(c^{j+1}_{t+1})\right].
$$

The bond Euler is unchanged.

We adopt the **partial-equilibrium-of-investment** simplification: the FOC ignores the indirect effect of today's $k^{j+1}_{t+1}$ on next period's adjustment cost (which would be $\partial[\psi_K/2 \cdot (k^{j+2} - k^{j+1})^2]/\partial k^{j+1}$). For mild $\psi_K$ and short cohort life this is a small distortion; documenting it here keeps the algorithmic claim honest.

## Stage 3 — what changes

| | V3 | V4 |
|---|---|---|
| Capital Euler residual | $1 - \beta\cdot\text{rhs}/u'(c)$ | $(1 + \psi_K\Delta k) - \beta\cdot\text{rhs}/u'(c)$ |
| `cohort_decisions` | budget without adjustment | subtracts $\frac{\psi_K}{2}(\Delta k)^2$ from $c$ |
| HP knob | n/a | `psi_K`: float; default uses `P["psi_K"] = 0.5` |
| Reduce-to-V3 | n/a | set HP `psi_K=0.0` |

Network architecture, training budget, and loss-weight defaults are unchanged.

## Stage 4 — what changes

`cohort_decisions` returns an additional `adj_cost` tensor for diagnostics. The `delta_k` tensor (next-period capital minus current entering wealth, aligned per cohort) flows into `euler_residuals` to construct the marginal-adjustment factor. See `pseudocode.md`.

## Validation gate

`simulate.validation_gate(sim, losses)` returns the V3 booleans plus a `K_volatility` numeric: capital path standard deviation, which should be **lower** than V3's at the same calibration (smoother investment path due to adjustment costs).

The reduce-to-V3 sanity check sets `psi_K=0` everywhere and confirms ergodic moments match V3's run within a few percent. This lives in `tests/test_v4_smoke.py::test_psi_zero_matches_v3`.

## Out of scope at V4

- Stabilising homotopy schedule (V5).
- Endogenous depreciation linked to TFP.
- Quadratic adjustment costs on bonds (skipped — cleaner pedagogy with one mechanism at a time).
