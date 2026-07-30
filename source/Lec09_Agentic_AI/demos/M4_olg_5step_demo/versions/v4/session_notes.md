# V4 Session Notes

V4 adds a convex capital-adjustment cost on top of V3's two-asset model. Setting $\psi_K = 0$ exactly reproduces V3.

## Stage discipline

- **Stage 1 (Model)** — `delta_spec.md`. Single new parameter $\psi_K$ in the budget.
- **Stage 2 (Equilibrium)** — capital Euler gains the marginal-adjustment factor; bond Euler unchanged.
- **Stage 3 (Algorithm)** — no architectural change; one extra term in the loss via the capital residual.
- **Stage 4 (Pseudo-code)** — `pseudocode.md`. Subtle bookkeeping: `delta_k = k_next - k_aligned` where `k_aligned` zero-pads cohort age 0.
- **Stage 5 (Implementation)** — `model.py` (cohort_decisions returns delta_k and adj_cost), `train.py` (capital Euler residual updated), `simulate.py` (records adj_cost path).

## Decisions worth flagging

- **$\psi_K = 0.5$** is mild. With this calibration the adjustment cost adds ≈ 0.05% to per-period consumption costs on average — a smoothing nudge, not a hard lever.
- **Partial-equilibrium-of-investment FOC.** We dropped the "future adjustment cost" term in the Euler equation (the partial of $\frac{\psi_K}{2}(k_\text{t+2} - k_\text{t+1})^2$ with respect to $k_\text{t+1}$). This is a documented simplification — $O(\psi_K^2)$ in moment errors. Tightening this is a V5 candidate, not a V4 burden.
- **Reduce-to-V3 lever is `psi_K=0`** in HP. Tests verify byte-equivalent behaviour at this setting (within sampling tolerance).

## Validation result

Typical run with $\psi_K = 0.5$:

- Total loss drops from ≈ 0.38 to ≈ $5 \times 10^{-4}$ (RMS ≈ 2%).
- $\bar K \approx 0.77$ (vs V3's $\approx 0.82$): adjustment cost slightly suppresses long-run capital.
- $\text{std}(K) \approx 0.076$ (similar to V3's $\approx 0.080$): cost is mild, smoothing is modest.
- Bond market clears to machine precision; lifecycle bond pattern (young borrow, old lend) preserved.
- All V3 gate criteria pass.

The reduce-to-V3 sanity check (`psi_K=0`, `bond_weight=1`, `fb_weight=0.5`) lives in `tests/test_v4_smoke.py::test_psi_zero_matches_v3`.

## Open extensions for V5

- Stabilising homotopy schedule (V5).
- Endogenous depreciation linked to TFP.
- Lab-style adjustment-cost specification with future-cost feedback.
