# V2 Delta Spec — Four-State Markov TFP

V2 keeps everything in V1 and replaces V1's two-state symmetric Markov TFP with a four-state Rouwenhorst discretisation of an annual AR(1) on log-TFP.

## Stage 1 — what changes

### TFP process

| | V1 | V2 |
|---|---|---|
| TFP states | $\{Z_\text{lo}, Z_\text{hi}\} = \{0.95, 1.05\}$ | 4 states from Rouwenhorst |
| Transition matrix | symmetric $p = 0.80$ | $4\times4$ from recursive Rouwenhorst |
| Annual primitives | n/a | $\rho_y = 0.85$, $\sigma_{\varepsilon,y} = 0.03$ |
| Per-period AR(1) | n/a | $\rho_\tau = \rho_y^\tau \approx 0.205$, $\sigma_{\varepsilon,\tau}$ as derived |

The Rouwenhorst grid is symmetric and equally spaced in log-TFP; exponentiated to give $Z$ values.

### Why Rouwenhorst (not Tauchen)

Rouwenhorst's discretisation is exact for the unconditional first and second moments of the AR(1), and degenerates correctly at $\rho \to 1$. With only four states this matters more than for fine grids.

## Stage 2 — what changes

The recursive equilibrium definition is unchanged in form. Only the TFP support and transition matrix are replaced. Closed-form expectations are still possible — the `for jz in range(n_z)` loop in `euler_residuals` now runs four iterations instead of two.

## Stage 3 — what changes

| | V1 | V2 |
|---|---|---|
| `n_z` (loop length in residuals) | 2 | 4 |
| `init_cloud` TFP sampling | uniform on $\{0,1\}$ | uniform on $\{0,1,2,3\}$ |
| `step_cloud` regime sampling | binary inverse-CDF | general categorical via cumulative sum |

Hyperparameters, network architecture, and training budget are unchanged.

## Stage 4 — what changes

`step_cloud` switches from a binary `(u > probs[:, 0]).long()` to the general categorical sampling

```
cdf = cumsum(probs, dim=-1)
z_idx_next = (u > cdf).sum(dim=-1).clamp(max=n_tfp-1)
```

so it works for any number of TFP states. See `pseudocode.md`.

## Validation gate

`simulate.validation_gate(sim, losses)` returns:

- `training_progressed` (≥5× MSE drop).
- `procyclical_top_state>bottom_state` ($\bar K$ in highest-TFP regime exceeds $\bar K$ in lowest).
- `rms_euler_residual_<_10pct`.
- `consumption_grows_with_age`.
- `savings_peak_in_pre_retirement`.
- `K_spread_across_TFP_states` (numeric — should be wider than V1's spread).

The reduce-to-V1 check is: re-instantiate the TFP grid as a 2-state version (using the same `make_tfp` code with `n_tfp=2`) and confirm the trained network's behaviour matches V1's within 1e-3 on key moments. The check lives in `tests/test_v2_smoke.py::test_two_state_collapse_matches_v1`.
