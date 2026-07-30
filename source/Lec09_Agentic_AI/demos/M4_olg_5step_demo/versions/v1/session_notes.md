# V1 Session Notes

V1 extends V0 from 3 cohorts to 7 cohorts on a hump-shaped lifecycle labour profile. One Claude Code session, prompt at `prompts/v0_to_v1.md`.

## Stage discipline

- **Stage 1 (Model)** — `delta_spec.md` §1. Chose $N=7$, $\tau \approx 10.286$ years, $\beta_y = 0.97$, $\delta_y = 0.06$, hump-shaped $\varepsilon$.
- **Stage 2 (Equilibrium)** — `delta_spec.md` §2. Six Euler equations replacing V0's two.
- **Stage 3 (Algorithm)** — `delta_spec.md` §3. Same DEQN-style approach; widened MLP to 128 hidden; pretraining target a single scalar; 2-state TFP unchanged.
- **Stage 4 (Pseudo-code)** — `pseudocode.md`. Vectorised over cohort dimension; `domain-reviewer` traced one cohort's residual end-to-end before code was written.
- **Stage 5 (Implementation)** — `model.py`, `network.py`, `train.py`, `simulate.py`, `plotting.py`; tests in `tests/test_v1_*.py`; notebook section in `demo.ipynb` Section 2.

## Decisions worth flagging

- **Period length 72/7.** Mirrors a 72-year adult horizon split into 7 cohorts; matches the framing in research-grade two-asset OLG implementations.
- **$\beta_y = 0.97$ instead of $0.96$.** Slightly more patient than common annual values so the model settles with non-degenerate savings; documented in `delta_spec.md`.
- **Hump-shaped $\boldsymbol{\varepsilon}$.** Peak at cohort age 3 (mid-career), zero in retirement. The non-zero $\varepsilon_0 = 0.7$ keeps young able to participate in the asset market (otherwise their corner solution dominates the lifecycle).
- **Single scalar pretraining target.** All 6 saver cohorts pretrained to $\bar s = 0.40$. Cohort-specific targets were considered and rejected as premature optimisation — V2 will revisit if anything looks fragile.

## Validation result

`simulate.validation_gate(sim, losses)` returns five booleans:

- `training_progressed` — final MSE drops to $\sim 10^{-3}$ (RMS $\sim 3\%$) from initial $\sim 0.16$ (RMS $\sim 40\%$).
- `procyclical_capital_E[K|hi] > E[K|lo]` — TRUE on a typical run.
- `rms_euler_residual_<_10pct` — TRUE.
- `consumption_grows_with_age` — TRUE; lifecycle profile rises smoothly from $\bar c_0 \approx 0.25$ to $\bar c_{N-1} \approx 0.50$.
- `savings_peak_in_pre_retirement` — TRUE; peak savings rate at cohort age 5 (just before retirement).

Cohort age 0 frequently pins to the savings floor ($s \approx 10^{-4}$). This is *not* a bug: with $\varepsilon_0 = 0.7 < \varepsilon_1 = 0.9$, the Euler ratio $c_1/c_0 = (\beta(1+r))^{1/\gamma} \approx 1.12$ is satisfied by the labour-income gradient alone, leaving young's savings rate at the boundary. Documented in `delta_spec.md` §3 → "Out of scope".

## Open extensions handled by V2+

- Continuous TFP (V2 → 4-state Markov with Rouwenhorst).
- Second asset (V3).
- Capital adjustment costs (V4).
- Stabilising homotopy (V5).
