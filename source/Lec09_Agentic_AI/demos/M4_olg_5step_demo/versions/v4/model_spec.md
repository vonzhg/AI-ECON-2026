# V4 — Model Specification (Delta from V3)

V4 replaces V3's grid backward induction with a **PyTorch MLP policy**, trained by imitation learning on V3's grid solution. The model itself (state, budget, equilibrium concept) is unchanged from V3 — V4 is an **algorithm change**, not a model change.

## What V4 adds

- **Policy network** `PolicyNet`: state → (a', b'), MLP with two SELU hidden layers (default 128 units each), sigmoid-rescaled outputs to keep policies in the asset/bond bounds.
- **Imitation learning**: V3's grid policy serves as supervised targets. The NN learns to map every (age, a, b, z_idio, z_tfp) state to V3's grid policy at the closest grid point.
- **PyTorch dependency** (gated): V4 cells short-circuit if `torch` isn't installed; install with `pip install -e ".[nn]"`.

## What V4 does NOT add

- **No new model**: state, budget, equilibrium concept all inherit from V3.
- **No residual loss yet**: V4 trains by MSE on V3's policy, not by Euler residuals. Residual training arrives in V5 with the homotopy schedule that makes it stable.
- **No new economic objects**: aggregate TFP and bonds are inherited as-is.

## Architecture (Lab12-aligned in spirit)

```text
PolicyNet:
    Linear(state_dim → 128) → SELU
    Linear(128 → 128)      → SELU
    Linear(128 → 2)
    output [0]: sigmoid * a_max          → a'
    output [1]: b_min + sigmoid * (b_max - b_min)  → b'

Final-layer bias initialized to -3.0 so initial sigmoid outputs are ~0.05
(matches the typical "small savings" prior for OLG households).
```

State encoding: continuous (age/n_age, a/a_max, (b-b_min)/(b_max-b_min)) plus one-hot z_idio and z_tfp. State dim = 3 + n_z + n_tfp.

## Training

- **Supervised MSE** on `(a_pred, b_pred)` vs V3 grid targets.
- Sample weighting: V3 stationary mass + 1e-4 floor (so undervisited states still see gradient signal).
- Adam optimizer, default `lr=1e-3`, `n_steps=2000`, `batch_size=256`.
- Runs on MPS (Apple Silicon GPU) when available, else CPU. ~3 s for full training at default settings.

## Validation gate

V4's validation is qualitative — exact match to V3 isn't expected because:

1. The NN smooths V3's spiky grid policy (unavoidable approximation cost).
2. V3's policy was solved at fixed equilibrium r; small NN errors in policy compound over the lifecycle simulation.

| Check | Threshold |
|---|---|
| Imitation loss decreases monotonically | ≥1 order of magnitude reduction over training |
| Final per-state MSE on V3 grid | < 0.1 (in units of `a_max²`) |
| Lifecycle profile shape matches V3 | peak in middle age (not at age 0 or last) |
| Mean K rel diff vs V3 | < 30% (5b: K ≈ 16%, B ≈ 30% with default config) |
| Bond profile qualitative match | NN follows V3's "young borrow, old lend" shape |

V4's value is *demonstration* — the NN can encode the household policy. V5 introduces residual training + homotopy as the more accurate approach.

## Why imitation, not residual loss?

Pure Euler/KKT residual loss on a randomly-initialized NN is notoriously unstable: outputs saturate at extremes, gradient signal is sparse, training diverges. V4's role in the demo is to show **the architecture works** (NN can encode the policy, runtime scales linearly in state dim, etc.). V5 is where residual training + homotopy schedule give the policy direct equilibrium-condition supervision.

## Implementation pointer

`versions/v4/{nn_solver.py, two_asset_spec.py, tfp.py}` (the latter two are V3 copies for self-containment). The `nn_solver` module exports `PolicyNet`, `TrainingConfig`, `train_imitation`, `simulate_lifecycle`, and `collect_v3_targets`. Tests in `tests/test_v4_nn_smoke.py` (4 tests, all skip if torch is missing).
