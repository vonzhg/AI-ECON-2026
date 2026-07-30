# V3 Session Notes

V3 introduces a second asset (one-period zero-coupon bonds, zero net supply, endogenous price), a market-clearing layer (sum-to-zero by construction), and a Fischer–Burmeister soft penalty for the borrowing limit. Single Claude Code session per `prompts/v2_to_v3.md`.

## Stage discipline

- **Stage 1 (Model)** — `delta_spec.md` §1. Bond face-value convention, zero net supply, $b_\text{min} = -0.05$ borrowing limit.
- **Stage 2 (Equilibrium)** — `delta_spec.md` §2. Bond Euler $u'(c) p_b = \beta\,\mathbb{E}[u'(c')]$ pins the bond price as the expected SDF.
- **Stage 3 (Algorithm)** — `delta_spec.md` §3. Mean-subtract market-clearing layer; tanh-bounded raw bond demand; sigmoid-mapped bond price.
- **Stage 4 (Pseudo-code)** — `pseudocode.md`. Two Euler residuals plus FB; loss-weight knobs surface the reduce-to-V2 lever.
- **Stage 5 (Implementation)** — `model.py` (state, FB), `network.py` (market-clearing layer), `train.py` (three-component loss), `simulate.py` (track $b$ paths).

## Decisions worth flagging

- **Mean-subtraction for sum-to-zero.** Considered residualising the last cohort's bond holding (forced from the constraint) but rejected — pedagogically, every cohort participating symmetrically is cleaner.
- **FB proxy multiplier.** Used $|R_B| + 10^{-4}$ as the multiplier in the FB residual so the network doesn't have to output explicit Lagrange multipliers. The penalty stays small in equilibrium (constraint not binding) and grows when bonds drift toward $b_\text{min}$.
- **Bond price range $(0.55, 0.95)$.** Chosen so the implied gross bond return $1/p_b \in (1.05, 1.82)$ brackets the equilibrium capital return.
- **Pretraining adds $b_\text{next} \to 0$ term.** Without it, the network's bond outputs at initialisation are noise that fights the early Euler-loss optimisation.
- **Training budget bumped to 8,000 steps.** Two Euler equations + FB require more iterations than V2's single residual.

## Validation result

Typical run:

- `training_progressed` — total loss drops from ≈ 0.35 to ≈ $2 \times 10^{-3}$ over 3,000 steps; final 50-step mean ≈ $3 \times 10^{-4}$ (RMS ≈ 1.6%).
- `procyclical_top_state>bottom_state` — TRUE.
- `rms_total_loss_<_15pct` — TRUE.
- `bond_market_clears` — TRUE; $\max_t |\sum_j b^j_t| \approx 10^{-8}$ (machine-zero).
- `bond_lifecycle_dispersion` — TRUE; young cohorts (1–3) borrow $\approx -0.05$, old (4–6) lend $\approx +0.05$.
- `consumption_grows_with_age` — TRUE.
- `savings_peak_in_pre_retirement` — TRUE; peak capital savings rate at cohort age 5.
- `bond_price_in_range` — TRUE; $\bar p_b \approx 0.58$, implied bond rate ≈ 71% per period.

The reduce-to-V2 check (`bond_weight=0, fb_weight=0, bonds_off=True`) lives in `tests/test_v3_smoke.py::test_bonds_off_matches_v2`.

## Open extensions for V4+

- Capital adjustment cost $\frac{\psi_K}{2}(k_\text{next} - k)^2$ (V4).
- Stabilising homotopy schedule (V5).
