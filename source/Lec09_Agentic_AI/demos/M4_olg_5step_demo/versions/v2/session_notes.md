# V2 Session Notes

V2 replaces V1's 2-state symmetric Markov TFP with a 4-state Rouwenhorst discretisation of an annual AR(1) on log-TFP. One Claude Code session, prompt at `prompts/v1_to_v2.md`.

## Stage discipline

- **Stage 1 (Model)** — `delta_spec.md`. Annual primitives $\rho_y = 0.85$, $\sigma_{\varepsilon,y} = 0.03$; per-period via the closed-form aggregation.
- **Stage 2 (Equilibrium)** — equilibrium definition unchanged in form.
- **Stage 3 (Algorithm)** — closed-form expectations preserved (4-term sum). Same hyperparameters.
- **Stage 4 (Pseudo-code)** — `pseudocode.md`. Rouwenhorst recursive construction; general categorical sampling for `step_cloud`.
- **Stage 5 (Implementation)** — `model.py` (Rouwenhorst, AR(1) aggregation, 4-state TFP), unchanged `network.py` and `train.py` cohort logic; `simulate.py` updated to handle ≥3 regimes.

## Decisions worth flagging

- **Rouwenhorst over Tauchen.** Rouwenhorst is exact in unconditional moments and degenerates correctly at $\rho \to 1$. With only four nodes this matters.
- **`n_tfp = 4` is per spec.** A wider grid (e.g. 7 nodes) could be added at marginal cost — closed-form expectations remain a 7-term sum — but the user fixed 4 for this demo.
- **No code change in `train.py`.** The Euler-residual loop already used `for jz in range(n_z)`; making the loop bound dynamic was the only edit. Same for `network.py`.
- **`step_cloud` generalised.** The binary `(u > probs[:, 0])` form had to be replaced by the general inverse-CDF construction. Reviewed by `code-reviewer`.

## Validation result

Typical run:

- `training_progressed` — final MSE ≈ $6 \times 10^{-4}$ (RMS ≈ 2.5%).
- `procyclical_top_state>bottom_state` — TRUE; $\bar K$ at $Z_4 \approx 1.10$ exceeds $\bar K$ at $Z_1 \approx 0.91$ by ~5%.
- `rms_euler_residual_<_10pct` — TRUE.
- `consumption_grows_with_age` — TRUE; lifecycle profile preserved from V1.
- `savings_peak_in_pre_retirement` — TRUE.
- `K_spread_across_TFP_states` ≈ 0.044 (numeric; documents shock-amplification capacity).

The reduce-to-V1 sanity check (run V2 with the TFP grid collapsed to 2 states matching V1's) lives in `tests/test_v2_smoke.py::test_two_state_collapse_matches_v1`.

## Open extensions for V3+

- Second asset (V3 — bonds with endogenous price + market clearing).
- Capital adjustment cost (V4).
- Stabilising homotopy (V5).
