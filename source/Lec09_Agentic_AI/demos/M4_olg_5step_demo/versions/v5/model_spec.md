# V5 — Model Specification (Delta from V4)

V5 keeps V4's NN architecture and the underlying model unchanged. The new piece is a **homotopy training schedule** that progressively shifts the loss from V4's imitation MSE toward residual-based objectives (Euler equations + KKT non-negativity). This mirrors Lab12's approach: walk from an easy-to-fit objective to the target objective, keeping training stable.

## What V5 adds

- **`HomotopySchedule`**: a list of `PhaseConfig` entries, each specifying weights for imitation, capital Euler, bond Euler, and KKT losses, plus a learning rate.
- **`run_homotopy(net, schedule, ...)`**: takes V4's pretrained network and runs the schedule, recording residual snapshots at each phase boundary.
- **5-phase default schedule**: imitation refresh → light Euler-k → KKT pressure → bonds on → residual only.

## What V5 does NOT add

- **No new model**: state, budget, equilibrium concept all inherit from V3/V4.
- **No new architecture**: the same `PolicyNet` from V4 is fine-tuned.
- **No new dependencies**: same PyTorch + numpy as V4.

## Default homotopy schedule

| # | Name | Steps | w_imit | w_eu_k | w_eu_b | w_kkt | lr |
|---|---|---|---|---|---|---|---|
| 1 | `1_imitation_refresh` | 200 | 1.0  | 0.0  | 0.0  | 0.0  | 5e-4 |
| 2 | `2_light_euler_k`     | 200 | 1.0  | 0.05 | 0.0  | 10.0 | 5e-4 |
| 3 | `3_kkt_pressure`      | 200 | 0.5  | 0.1  | 0.0  | 50.0 | 5e-4 |
| 4 | `4_bonds_on`          | 200 | 0.2  | 0.1  | 0.05 | 50.0 | 3e-4 |
| 5 | `5_residual_only`     | 200 | 0.0  | 0.1  | 0.05 | 50.0 | 1e-4 |

The schedule is exposed as a dataclass so students can edit phase weights or add their own phases.

## Validation gate (and honest caveat)

| Check | Threshold |
|---|---|
| Residual snapshots are finite (no NaN/Inf) at every phase boundary | strict |
| Capital Euler residual decreases by ≥ 2× from phase 1 start to phase 3 end | qualitative |
| Phase ordering preserved in the history | strict |

**Important caveat:** V5's *default* schedule does NOT necessarily produce a policy that more closely matches V3 than V4's imitation does. In our reference run, V4's imitation gives mean K rel diff to V3 of ~16%, while V5's default homotopy ends at ~88% rel diff (over-saving) — the residual-loss phases pull the policy away from V3's specific equilibrium toward different local minima.

This is a **realistic difficulty** of NN-based macro solvers: residual training is sensitive to schedule design, learning rates, and loss weighting. Lab12-grade results require careful tuning beyond the demo's default schedule.

V5's role in the demo is to **demonstrate the technique** — that a homotopy schedule can stabilize residual-loss training and walk through different objective configurations. Students adapting V5 for their own research will likely need to tune the schedule.

## Implementation pointer

`versions/v5/{homotopy.py, nn_solver.py, two_asset_spec.py, tfp.py}`. The `homotopy` module exports `HomotopySchedule`, `PhaseConfig`, `PhaseSnapshot`, `HomotopyHistory`, and `run_homotopy`. Tests in `tests/test_v5_homotopy.py` verify the schedule runs cleanly on a tiny problem (4 tests, skip if torch missing).
