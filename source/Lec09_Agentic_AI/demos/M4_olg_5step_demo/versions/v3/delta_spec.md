# V3 Delta Spec — Bonds + Market-Clearing Layer + Fischer–Burmeister

V3 adds a second asset (one-period zero-coupon bonds in zero net supply) with an endogenous price, a market-clearing layer enforcing $\sum_j b^j = 0$ by construction, and a Fischer–Burmeister soft penalty for the borrowing limit.

## Stage 1 — what changes

### State

| | V2 | V3 |
|---|---|---|
| State | $(Z, k^1, \ldots, k^{N-1})$ | $(Z, k^1, \ldots, k^{N-1}, b^1, \ldots, b^{N-1})$ |
| Dim | 7 | 13 |

$b^j_t$ is the **face value** of bonds held by cohort age $j$ entering period $t$. Bonds pay one consumption unit per face value at maturity (one period later); they trade at the endogenous price $p_b$ at issuance.

### Bonds in zero net supply

$$
\sum_{j=1}^{N-1} b^j_t = 0 \quad \text{for all } t.
$$

### Borrowing limit

$b^j \geq b_\text{min}$ with $b_\text{min} = -0.05$ (face-value units). Enforced softly via a Fischer–Burmeister residual; multiplier proxy is the bond-Euler residual magnitude.

### Calibration additions

| Parameter | Value |
|---|---|
| $b_\text{min}$ | $-0.05$ |
| $b_\text{scale}$ | $0.10$ (tanh range on raw bond demand) |
| $p_{b,\min}, p_{b,\max}$ | $0.55, 0.95$ |

## Stage 2 — what changes

Two Euler equations per saver (capital and bond):

$$
u'(c^j_t) = \beta\,\mathbb{E}_t\!\left[(1+r_{t+1})\,u'(c^{j+1}_{t+1})\right] \quad\text{(capital)}
$$

$$
u'(c^j_t)\,p_{b,t} = \beta\,\mathbb{E}_t\!\left[u'(c^{j+1}_{t+1})\right] \quad\text{(bond)}
$$

The bond Euler pins down the bond price as the expected stochastic discount factor.

Period-$t$ income for cohort age $j \in \{1, \ldots, N-1\}$:

$$
I^j_t = w_t\,\varepsilon_j + (1+r_t)\,k^j_t + b^j_t.
$$

Budget identity (cohorts $0, \ldots, N-2$):

$$
c^j_t + k^{j+1}_{t+1} + p_{b,t}\,b^{j+1}_{t+1} = I^j_t.
$$

The retired cohort consumes $c^{N-1}_t = I^{N-1}_t$ with no savings.

### Fixed-point structure

Adding the bond Euler creates a second residual, but the equilibrium remains a fixed point of the policy on its own training cloud. The market-clearing layer in the policy network removes a degree of freedom from the bond demand vector (mean-subtraction), so no separate fixed-point loop on $p_b$ is needed.

## Stage 3 — what changes

| | V2 | V3 |
|---|---|---|
| Network output dimension | $N - 1$ | $(N - 1) + (N - 1) + 1 = 13$ |
| Network width | 128 | 192 |
| Loss components | one capital Euler MSE | weighted sum: capital Euler + bond Euler + Fischer–Burmeister |
| Pretraining | $\bar s = 0.40$ | $\bar s_K = 0.40$, $\bar p_b = 0.80$, $b_\text{next} \to 0$ |
| Training budget | 6,000 steps | 8,000 steps |

### Loss-weight convention

```
L_total = w_K · mean(R_K²) + w_B · mean(R_B²) + w_FB · mean(FB²)
```

Default: $w_K = w_B = 1.0$, $w_{FB} = 0.5$. The "reduce-to-V2" check sets $w_B = w_{FB} = 0$ and `bonds_off=True` in `cohort_decisions` to suppress the bond term in the budget; the trained-network capital path then matches V2's within the tolerance set in `tests/test_v3_smoke.py`.

### Algorithmic decisions

- **Mean-subtract for market clearing.** Cleaner than residualising one cohort's bond holding; every cohort participates symmetrically.
- **`tanh × b_scale` raw bond demand.** Keeps output in a controlled range so the network doesn't propose face values orders of magnitude larger than wages.
- **`p_b` mapped through sigmoid into $(p_{b,\min}, p_{b,\max})$.** Avoids non-positive prices and absurd discount rates without a hard projection.
- **Fischer–Burmeister with proxy multiplier.** $|R_B| + \varepsilon$ stands in for the multiplier so we don't have to expand the network to output Lagrange multipliers explicitly. The residual is small whenever the borrowing limit isn't binding; pedagogically it documents *where* the multiplier would enter in a fully-rigorous treatment.

## Stage 4 — what changes

`cohort_decisions` now takes both $k$ and $b$ and the bond outputs $(b_\text{next}, p_b)$; budget bookkeeping subtracts $p_b \cdot b_\text{next}$ from the savings residual. `euler_residuals` returns three tensors $(R_K, R_B, FB)$ instead of one. See `pseudocode.md` for the full pseudo-code.

## Validation gate

`simulate.validation_gate(sim, losses)` returns:

- `training_progressed`.
- `procyclical_top_state>bottom_state`.
- `rms_total_loss_<_15pct`.
- `bond_market_clears` — $\max_t |\sum_j b^j_t| < 10^{-4}$ (machine-zero by construction).
- `bond_lifecycle_dispersion` — $\text{std}(\bar b_j) > 10^{-3}$ across cohorts.
- `consumption_grows_with_age`.
- `savings_peak_in_pre_retirement`.
- `bond_price_in_range`.

The reduce-to-V2 sanity check is in `tests/test_v3_smoke.py::test_bonds_off_matches_v2` — running V3 with `bond_weight=0`, `fb_weight=0`, `bonds_off=True` reproduces V2's $\bar K$ within 5%.

## Out of scope at V3

- Capital adjustment cost (V4).
- Stabilising homotopy schedule (V5).
- Explicit Lagrange-multiplier KKT (the FB proxy in V3 is enough for this calibration).
