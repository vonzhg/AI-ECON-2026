# V4 → V5 — Stabilising Homotopy Schedule (one Claude Code session)

Paste this into Claude Code from the demo root.

---

## Context

V4 is a validated seven-cohort, two-asset OLG with capital adjustment costs, market-clearing layer, and Fischer–Burmeister borrowing penalty — the full economic model. Single-shot training (V4) reaches a workable equilibrium, but the bond price, the borrowing limit, and the adjustment cost are mutually coupled, so naively training the full loss from a blank initialisation often lands the optimiser in a local minimum where bonds pin to the borrowing limit.

V5's *only* job is to introduce a four-phase stabilising homotopy schedule on top of V4's economic model. **The economic model itself does not change.**

## Stage 1 — Model

Read `versions/v4/delta_spec.md` and `reference/research_target_notes.md`'s "Stabilising homotopy" section. Draft `versions/v5/delta_spec.md`:

> Same economic model as V4. Replace single-shot training with a four-phase homotopy schedule.

No new parameters in the calibration table. Spend Stage 1 on motivation rather than spec deltas.

## Stage 2 — Equilibrium

The equilibrium is unchanged. The point is that the *path* to it is what we're engineering. Articulate the local-minimum diagnosis clearly: pull-in toward the borrowing limit because (a) the bond price needs to converge to the SDF, (b) the SDF depends on consumption growth, (c) consumption growth depends on capital savings, (d) capital savings get less attention when bond loss dominates early.

## Stage 3 — Algorithm

Four phases:

| Phase | Steps | Active losses | LR | Effect |
|---|---|---|---|---|
| **1 capital-only** | 2000 | $R_K$ only ($\bonds_off=True$) | $\Gamma$ | Network learns capital savings as if bonds didn't exist. |
| **2 bond pretraining** | 1500 | $R_K + 0.1 \cdot R_B$ ($\bonds_off=False$) | $\Gamma$ | Bond price and bond holdings start to wake up under a small loss weight; capital path stabilises. |
| **3 bond homotopy** | 1500 | $R_K + w_B(t) R_B + w_{FB}(t) FB$, weights ramp linearly | $\Gamma$ | Linearly raises bond importance and turns on the borrowing-limit FB residual. |
| **4 fine-tuning** | 1500 | $R_K + R_B + 0.5 FB$ | $\Gamma / 10$ | Final polish. |

Total budget ~6,500 steps + 800 pretraining steps. Roughly comparable to V4's 8,000 steps but with the work organised so each step targets a tractable sub-problem.

**Reduce-to-V4 lever.** Setting `phase1_steps = phase2_steps = phase3_steps = 0`, `phase4_steps = 8000`, and removing the LR cut collapses the homotopy to a single-block run with V4's loss weights.

## Stage 4 — Pseudo-code

Write `versions/v5/pseudocode.md`. Express the phase machinery as a `_train_block(net, opt, sched, cloud, weights, bonds_off, n, weight_schedule=None)` helper that returns a per-step record of `(R_K_mse, R_B_mse, FB_mse, total)`. The main `homotopy_run` function calls `_train_block` four times with appropriate arguments.

Send to **`domain-reviewer`** with these checks:

- The `bonds_off=True` branch in Phase 1 forces `b_next` to zero in `cohort_decisions` so gradients don't leak through unused outputs.
- `weight_schedule` returns weights *continuous* across phase boundaries (Phase 2's exit $(1, 0.1, 0)$ matches Phase 3's entry).
- Per-phase residuals are recorded as `(R_K_mse, R_B_mse, FB_mse, total)` for the notebook's homotopy plot.

## Stage 5 — Implementation

Most of `versions/v5/` is byte-identical to V4 except for `train.py` and `simulate.py` (the gate now consumes `history`). Plan first. Add `tests/test_v5_smoke.py` with at least:

- `test_phases_run_in_order_and_finite` — four phases run in expected order with finite snapshots.
- `test_validation_gate_returns_expected_keys` — gate returns the expected booleans.

## Validation gate

`simulate.validation_gate(sim, history)` returns:

- `training_progressed`.
- `all_residual_snapshots_finite` — no NaN/Inf at any step.
- `rms_total_loss_<_8pct`.
- `rk_phase4_end_<_half_phase1_start` — capital Euler residual at the end of Phase 4 is no worse than half of where Phase 1 started.
- `bond_market_clears` (machine zero).
- `bond_lifecycle_dispersion`.
- `consumption_grows_with_age`.
- `savings_peak_in_pre_retirement`.
- `bond_price_in_range`.
- `procyclical_top_state>bottom_state`.

Once the gate passes you have reached the *algorithmic structure* of the silent research-grade target described in `reference/research_target_notes.md` — the demo ladder is complete.
