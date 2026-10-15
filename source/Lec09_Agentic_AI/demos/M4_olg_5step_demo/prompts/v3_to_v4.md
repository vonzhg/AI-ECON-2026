# V3 → V4 — Capital Adjustment Cost (one Claude Code session)

Paste this into Claude Code from the demo root.

---

## Context

V3 is a validated seven-cohort, two-asset OLG with an endogenous bond price and zero-net-supply market clearing. Capital paths in V3 can be jumpy because there is no friction discouraging large cohort-to-cohort capital adjustments. V4 adds a convex capital adjustment cost $\frac{\psi_K}{2}(k_\text{next} - k)^2$ — a single new parameter and one new term in the capital Euler equation.

V4's *only* job is the adjustment cost. We will *not* introduce a homotopy schedule; that is V5.

## Stage 1 — Model

Read `versions/v3/delta_spec.md`. Draft `versions/v4/delta_spec.md`:

- Add convex capital adjustment cost $\frac{\psi_K}{2}(k^{j+1}_{t+1} - k^j_t)^2$ paid out of period-$t$ consumption.
- Calibration: $\psi_K = 0.50$ (mild — encourages smooth paths but doesn't dominate prices).
- For cohort age 0: $k^0_t = 0$, so the cost is $\frac{\psi_K}{2}(k^1_{t+1})^2$ — quadratic in start-of-life capital.

Reduce-to-V3 lever: $\psi_K = 0$ recovers V3 exactly.

## Stage 2 — Equilibrium

The capital Euler equation gains the marginal-adjustment factor:

$$
u'(c^j_t)\bigl(1 + \psi_K\,(k^{j+1}_{t+1} - k^j_t)\bigr) = \beta\,\mathbb{E}_t\!\left[(1+r_{t+1})\,u'(c^{j+1}_{t+1})\right].
$$

The bond Euler is unchanged.

We adopt the **partial-equilibrium-of-investment simplification**: the FOC ignores the indirect effect of today's $k^{j+1}_{t+1}$ on the *next* period's adjustment cost. For mild $\psi_K$ and short cohort life this is a small distortion; document it explicitly so the workflow is honest about the simplification.

## Stage 3 — Algorithm

No architectural change. Two narrow code edits:

- `cohort_decisions` subtracts $\frac{\psi_K}{2}(\Delta k)^2$ from consumption and returns `delta_k` for downstream use, where $\Delta k = k_\text{next} - k_\text{aligned}$ and `k_aligned` zero-pads cohort age 0.
- `euler_residuals` constructs `marg_cost = 1 + ψ_K · delta_k` and returns $R_K = \text{marg\_cost} - \beta\,\text{rhs}_K / u'(c)$ instead of V3's $R_K = 1 - \beta\,\text{rhs}_K / u'(c)$.

When $\psi_K = 0$ both edits become no-ops.

## Stage 4 — Pseudo-code

Write `versions/v4/pseudocode.md`. Send to **`domain-reviewer`** with these checks:

- `delta_k[..., 0]` equals `k_next[..., 0]` (cohort age 0 inherits zero capital).
- The capital Euler reduces to V3's $1 - \beta\,\text{rhs}/u'(c)$ when $\psi_K = 0$.
- The adjustment cost subtracts from consumption only at the saver cohorts (`c[..., :N-1]`), not at the retired cohort.

## Stage 5 — Implementation

Most of `versions/v4/` is byte-identical to V3. Plan first. Add `tests/test_v4_smoke.py` with at least:

- `test_psi_K_zero_drops_adjustment_cost` — `cohort_decisions(psi_K=0)` returns zero `adj_cost`.
- `test_marginal_adjustment_term_in_residual` — `euler_residuals` differs between $\psi_K = 0.5$ and $\psi_K = 0$.

## Validation gate

`simulate.validation_gate(sim, losses)` returns the V3 booleans plus a numeric `K_volatility`. The path standard deviation of $K$ should be **lower** than V3's at the same calibration (smoother investment from the cost). All bond-related checks must still pass.

The reduce-to-V3 sanity check (`hp_overrides={"psi_K": 0.0}`) lives in `tests/test_v4_smoke.py::test_marginal_adjustment_term_in_residual` (indirectly).
