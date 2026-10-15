# V0 → V1 — Seven-Cohort Lifecycle (one Claude Code session)

Paste this into Claude Code from the demo root. The session walks Stages 1–5; you (the human) approve each stage before the next begins.

---

## Context

V0 is a validated three-cohort, one-asset OLG with a single-MLP policy trained on Euler residuals (`versions/v0/`). The bottleneck for going further is structural: the lifecycle has only three points (young / middle / old), so we cannot show cohort-specific patterns or middle-age peaks of saving / income. V1's *only* job is to extend the cohort dimension to seven.

We are **not** changing the algorithm, the policy parametrisation, or the TFP discretisation. Just demographics and the labour profile.

## Stage 1 — Model

Read `versions/v0/model_spec.md` and `reference/research_target_notes.md`. Draft `versions/v1/delta_spec.md` describing only what changes:

- Extend cohorts from $N=3$ to $N=7$. Period length $\tau = 72/7 \approx 10.286$ years.
- Re-anchor $\beta$, $\delta$ on yearly primitives ($\beta_y = 0.97$, $\delta_y = 0.06$).
- Hump-shaped efficiency-labour profile $\boldsymbol{\varepsilon} = (0.7, 0.9, 1.0, 1.05, 1.0, 0.9, 0.0)$.
- State expands to $(Z, a^1, \ldots, a^{N-1})$.

Then restate the new model in your own words; I'll check for misunderstandings before we move on.

## Stage 2 — Equilibrium

Update the equilibrium definition: $N - 1$ Euler equations now (one per saver cohort). The cohort-age-$j$ saver's Euler equation reads

$$
u'(c^j_t) = \beta\,\mathbb{E}_t\!\left[(1+r_{t+1})\,u'(c^{j+1}_{t+1})\right] \quad j=0,\ldots,N-2.
$$

Aggregate capital is $K = \sum_{j=1}^{N-1} a^j$. Sketch the fixed-point structure.

## Stage 3 — Algorithm

Same DEQN-style approach:
- single MLP with sigmoid head, but with $N-1$ outputs and input dim $1 + (N-1)$;
- pretraining target a single scalar $\bar s = 0.40$ across all cohorts;
- 2-state TFP unchanged;
- network width raised to 128 (input dim grew from 3 to 7);
- training budget 6,000 steps (was 5,000).

If you propose a different algorithmic change (e.g. cohort-specific pretraining targets), justify the cost in terms of the reduce-to-V0 check.

## Stage 4 — Pseudo-code

Write `versions/v1/pseudocode.md`. The cohort-dimension vectorisation is the only non-trivial bookkeeping. Send the pseudo-code to the **`domain-reviewer`** sub-agent; require it to trace at least one cohort's residual end-to-end against the equilibrium definition before you authorise me to write code.

## Stage 5 — Implementation

Implement `versions/v1/{model.py, network.py, train.py, simulate.py, plotting.py}`. Plan first; show me the plan; only then write code. Add `tests/test_v1_smoke.py`. Run `python3 -m unittest tests.test_v1_smoke` and the validation gate; iterate until all pass.

## Validation gate (do not declare done until all pass)

`simulate.validation_gate(sim, losses)` returns five booleans:

- `training_progressed`.
- `procyclical_capital_E[K|hi] > E[K|lo]`.
- `rms_euler_residual_<_10pct`.
- `consumption_grows_with_age`.
- `savings_peak_in_pre_retirement`.

Plus the deterministic-SS check (set $Z_\text{lo} = Z_\text{hi} = 1$, retrain briefly, confirm $\text{std}(K) < 0.02$). The full check lives in `tests/test_v1_smoke.py::test_consumption_grows_with_age`.

If any gate fails, the bug is in the new code (V0 is validated). Use V0's reproducibility report as your line-by-line diff target.
