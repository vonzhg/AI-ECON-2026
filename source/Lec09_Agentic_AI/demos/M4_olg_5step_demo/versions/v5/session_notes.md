# V5 Session Notes

V5 introduces a four-phase stabilising-homotopy training schedule on top of V4's economic model. Single Claude Code session per `prompts/v4_to_v5.md`.

## Stage discipline

- **Stage 1 (Model)** — `delta_spec.md`. No change to the economic model.
- **Stage 2 (Equilibrium)** — `delta_spec.md`. The equilibrium is unchanged; the *path to it* is what we're engineering.
- **Stage 3 (Algorithm)** — `delta_spec.md`. Four-phase schedule with linear bond-weight ramp in Phase 3 and LR cut in Phase 4.
- **Stage 4 (Pseudo-code)** — `pseudocode.md`. `_train_block` helper to keep the four phases readable; per-step residual decomposition is recorded for plotting.
- **Stage 5 (Implementation)** — only `train.py` (homotopy schedule) and `simulate.py` (the gate now consumes `history`) changed substantively. `model.py` and `network.py` are byte-identical to V4 except for headers.

## Decisions worth flagging

- **Why the schedule and not adaptive thresholds.** Considered "advance phase when residual < ε" — rejected for predictability and pedagogy. Fixed step counts are easier to demonstrate in a classroom and produce the same plot every time.
- **Phase 1 uses `bonds_off=True`.** Critical for clean separation. Without this, Phase 1 still computes `bond_cost` from random network outputs and the gradient leaks bond information into the capital problem.
- **Phase 3 ramp is linear in step count, not in residual progress.** Linear is simpler to reason about; the residual will ride along non-monotonically with the weight increase, which is fine.
- **Phase 4 LR cut by 10×.** Standard fine-tuning trick. Considered also lowering the LR linearly during Phase 3 — extra knob, dropped.
- **Reduce-to-V4 lever.** `phase1_steps = phase2_steps = phase3_steps = 0`, `phase4_steps = 8000`, no LR cut. Documented in `delta_spec.md`.

## Validation result

Typical run (full budget = 6,500 steps + 800 pretraining):

- Phase 1: $R_K$ MSE drops from $\sim 0.20$ to $\sim 3 \times 10^{-3}$ (RMS ≈ 5.5%).
- Phase 2: $R_K$ holds at $\sim 2 \times 10^{-3}$; $R_B$ MSE drops from $\sim 0.13$ to $\sim 5 \times 10^{-3}$.
- Phase 3: total loss bumps as bond weight ramps up, then settles in the $10^{-3}$ range.
- Phase 4: fine-tuning lowers total loss to ~$1 \times 10^{-3}$ (RMS ≈ 3.3%).
- Lifecycle bond pattern preserved (young borrow, old lend); $\bar K$ within a few percent of V4.
- All gate criteria pass.

## Wrap-up

V5 reaches the *algorithmic structure* of the silent research-grade target described in `reference/research_target_notes.md`: cloud method, Euler-residual loss, two-asset OLG with market-clearing layer, capital adjustment costs, Fischer–Burmeister borrowing penalty, and stabilising homotopy. Going further (more cohorts, continuous TFP via Gauss–Hermite, GPU-accelerated array libraries) is a separate engineering project — the workflow taught here is the same regardless of where you take the model next.
