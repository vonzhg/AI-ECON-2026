# V1 → V2 — Four-State Markov TFP via Rouwenhorst (one Claude Code session)

Paste this into Claude Code from the demo root.

---

## Context

V1 is a validated seven-cohort lifecycle on a 2-state symmetric Markov TFP. The 2-state shock is a pedagogical placeholder — researchers calibrate to AR(1) processes on log-TFP, and even four states give a markedly richer shock distribution at no algorithmic cost (closed-form expectations remain a finite sum).

V2's *only* job is to upgrade the TFP discretisation. The cohort dimension, the network architecture, and the loss are unchanged.

## Stage 1 — Model

Read `versions/v1/delta_spec.md` and `reference/research_target_notes.md`. Draft `versions/v2/delta_spec.md`:

- Replace 2-state $\{0.95, 1.05\}$ symmetric Markov with a 4-state Rouwenhorst discretisation of an annual AR(1) on log-TFP, primitives $\rho_y = 0.85$, $\sigma_{\varepsilon,y} = 0.03$.
- Per-period AR(1) primitives: $\rho_\tau = \rho_y^\tau$, $\sigma_{\varepsilon,\tau} = \sigma_{\varepsilon,y}\sqrt{(1 - \rho_y^{2\tau})/(1 - \rho_y^2)}$.

The ergodic distribution is the Rouwenhorst stationary; centre the grid in log-space around zero.

## Stage 2 — Equilibrium

The equilibrium definition is unchanged. Only the support of $Z$ and the transition matrix change. The expectation in each Euler equation is now a 4-term sum.

## Stage 3 — Algorithm

Same DEQN-style approach, with two narrow algorithmic edits:

- The expectation loop runs `for jz in range(n_tfp)` with `n_tfp = 4` (was 2).
- `step_cloud` switches from the binary form `(u > probs[:, 0]).long()` to the general categorical inverse-CDF
  ```
  cdf = cumsum(probs, dim=-1)
  z_idx_next = (u > cdf).sum(dim=-1).clamp(max=n_tfp-1)
  ```

Network architecture, training budget, and pretraining target are unchanged.

**Why Rouwenhorst (not Tauchen).** Rouwenhorst matches unconditional first and second moments exactly and degenerates correctly at $\rho \to 1$. With only four nodes this matters more than for fine grids.

## Stage 4 — Pseudo-code

Write `versions/v2/pseudocode.md`. Pseudo-code for `rouwenhorst(n, ρ, σ_ε)` and `aggregate_ar1(ρ_y, σ_y, τ)`. Send to **`domain-reviewer`**; require explicit checks that:

- `P_MAT` rows sum to one to floating-point tolerance;
- `Z_VALS` is symmetric in log-space;
- the unconditional mean matches the analytical $\frac{1}{n}\sum z_j$ for symmetric Rouwenhorst.

## Stage 5 — Implementation

Implement `versions/v2/{model.py, network.py, train.py, simulate.py, plotting.py}`. Most of `network.py` and `train.py` are unchanged from V1. Plan first.

Add `tests/test_v2_smoke.py`. Include a "reduce-to-V1" sanity check: instantiate `make_tfp(n_tfp=2)` and confirm the resulting transition matrix has the structure of V1's symmetric Markov.

## Validation gate

`simulate.validation_gate(sim, losses)` returns:

- `training_progressed`.
- `procyclical_top_state>bottom_state`.
- `rms_euler_residual_<_10pct`.
- `consumption_grows_with_age`.
- `savings_peak_in_pre_retirement`.
- `K_spread_across_TFP_states` (numeric — V2 should produce a wider spread than V1).

If any gate fails, the bug is in the new code.
