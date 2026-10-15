# V0 Session Notes

V0 is the **seed**, not a transition. Its contents were ported from a clean working notebook into the `versions/v0/` module layout used by every subsequent version.

## Stage discipline

- **Stage 1 (Model)** — `model_spec.md` §1.
- **Stage 2 (Equilibrium)** — `model_spec.md` §2.
- **Stage 3 (Algorithm)** — `model_spec.md` §3.
- **Stage 4 (Pseudo-code)** — `pseudocode.md`.
- **Stage 5 (Implementation)** — `model.py`, `network.py`, `train.py`, `simulate.py`, `plotting.py`; tests in `tests/test_v0_*.py`; notebook section in `demo.ipynb` Section 1.

## Validation result (per `model_spec.md` §5)

`simulate.validation_gate(sim, losses)` returns four booleans on a typical run:

- `training_progressed` — final 50-step MSE < starting MSE / 10.
- `procyclical_capital_E[K|hi] > E[K|lo]`.
- `rms_euler_residual_<_8pct` — RMS of final 50 steps' MSE under 8%.
- `lifecycle_hump_co>cm>cy` — typical hump shape.

The deterministic-SS check (criterion 5 in `model_spec.md` §5) is exercised by `tests/test_v0_smoke.py::test_deterministic_ss_collapses` rather than the in-line gate.

## Open extension menu (handled by V1+)

The list is not "V0 missing features" — it is the menu of one-step-richer specs from which V1 picks. See `prompts/v0_to_v1.md` for V1's specific choice (7-cohort lifecycle).

- Idiosyncratic income (per-cohort Markov).
- Second asset (bonds) with endogenous price.
- Continuous TFP via discretisation.
- More cohorts (V1's pick).
- Borrowing constraint replacing the sigmoid trick.
