# V5 Delta Spec — Stabilising Homotopy Schedule

V5 keeps V4's full two-asset model (cohorts, TFP, bonds, adjustment cost, FB) and replaces single-shot training with a **four-phase stabilising-homotopy schedule**. The model is identical; only the training procedure changes.

## Stage 1 — what changes

Nothing in the economic model. Same calibration as V4. The change is purely numerical: a research-grade NN solver does not train on the full residual loss from a blank initialisation — it solves a sequence of nested easier problems first.

## Stage 2 — what changes

Equilibrium definition is unchanged. The point is that the *path* to the V4 equilibrium is harder than the equilibrium itself: the bond price and the borrowing limit are coupled, so naively penalising both residuals at once can drive the network to a local minimum where bonds are pinned at the borrowing limit and the price is whatever closes the bond Euler at that corner. The homotopy walks around that local minimum.

## Stage 3 — what changes

| Phase | Steps | Active losses | LR | Effect |
|---|---|---|---|---|
| **1 capital-only** | 2000 | $R_K$ only ($\bonds_off=True$) | base | Network learns capital savings as if bonds didn't exist. |
| **2 bond pretraining** | 1500 | $R_K + 0.1 \cdot R_B$ ($\bonds_off=False$) | base | Bond price and bond holdings start to wake up under a small loss weight; capital path stabilises. |
| **3 bond homotopy** | 1500 | $R_K + w_B(t) R_B + w_{FB}(t) \cdot FB$ with $w_B: 0.1 \to 1.0$, $w_{FB}: 0 \to 0.5$ linear | base | Linearly raises bond importance and turns on the borrowing-limit FB residual. |
| **4 fine-tuning** | 1500 | $R_K + R_B + 0.5 \cdot FB$ at $\Gamma/10$ | $\Gamma / 10$ | Final polish: same loss as V4, lower learning rate. |

Total budget ~6,500 steps + 800 pretraining steps. Roughly comparable to V4's 8,000 steps but with the work organised so each step targets a tractable sub-problem.

### Why the schedule

- **Phase 1** isolates the capital problem so the network has a sane prior on $s_K$ before bonds enter.
- **Phase 2** introduces bonds at $w_B = 0.1$ — a "warm gradient" so the bond price drifts to a sensible value without dominating the loss.
- **Phase 3** is the homotopy proper: continuously raising $w_B$ avoids the discontinuous jump that would push the network into a corner.
- **Phase 4** polishes at lower LR — a standard fine-tuning trick when the optimiser has been operating in a high-curvature regime.

Each phase's start- and end-step total losses are recorded in `history["phases"]` and surfaced by `notebook_report_v5.json`.

## Stage 4 — what changes

`train.homotopy_run` orchestrates the schedule. Internal structure:

```
pretrain(net, 800 steps)
opt = Adam(net.parameters(), lr=Γ); sched = ExponentialLR(γ)

# Phase 1
_train_block(weights=(1, 0, 0), bonds_off=True, n=2000)
# Phase 2
_train_block(weights=(1, 0.1, 0), bonds_off=False, n=1500)
# Phase 3
_train_block(weight_schedule=t→(1, 0.1+0.9·t/n3, 0.5·t/n3), bonds_off=False, n=1500)
# Phase 4
opt.lr ← opt.lr × 0.10
_train_block(weights=(1, 1.0, 0.5), bonds_off=False, n=1500)
```

The `_train_block` helper records per-step `(R_K_mse, R_B_mse, FB_mse, total)` so the notebook can plot the per-phase residual decomposition with phase boundaries shaded.

## Validation gate

`simulate.validation_gate(sim, history)` returns:

- `training_progressed`.
- `all_residual_snapshots_finite` — no NaN/Inf at any step.
- `rms_total_loss_<_8pct`.
- `rk_phase4_end_<_half_phase1_start` — capital Euler residual at the end of Phase 4 is no worse than half of where Phase 1 started (with two assets, the capital residual should not have *gotten worse* by adding bonds).
- `bond_market_clears` (machine zero by construction).
- `bond_lifecycle_dispersion`.
- `consumption_grows_with_age`.
- `savings_peak_in_pre_retirement`.
- `bond_price_in_range`.
- `procyclical_top_state>bottom_state`.

The reduce-to-V4 sanity check is implicit: V5 with `phase1_steps=0`, `phase2_steps=0`, `phase3_steps=0` and `phase4_steps=8000` (no LR cut) replicates V4's training trajectory.

## Out of scope at V5

- Continuous-TFP via Gauss–Hermite (deferred to a hypothetical V6).
- Adaptive phase boundaries (e.g. "advance phase when residual drops below threshold"). The current schedule uses fixed step counts for predictability and pedagogy.
- Forward-looking adjustment-cost FOC (still uses V4's partial-equilibrium simplification).
