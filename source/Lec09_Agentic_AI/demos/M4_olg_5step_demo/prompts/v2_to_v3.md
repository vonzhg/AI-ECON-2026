# V2 → V3 — Bonds + Market-Clearing Layer + Fischer–Burmeister (one Claude Code session)

Paste this into Claude Code from the demo root.

---

## Context

V2 is a validated seven-cohort, one-asset, 4-state-TFP OLG. The next ingredient is a second asset — a one-period zero-coupon bond in zero net supply, with an endogenous price $p_b$. This is the largest single jump in the ladder: the state dimension nearly doubles, two Euler equations replace one, and the policy network needs a market-clearing layer for bond positions.

V3's *only* job is to introduce bonds cleanly. We will *not* add adjustment costs (V4) or a homotopy schedule (V5) in this session.

## Stage 1 — Model

Read `versions/v2/delta_spec.md` and `reference/research_target_notes.md`. Draft `versions/v3/delta_spec.md`:

- Add second asset: one-period zero-coupon bonds, face value 1, traded at endogenous price $p_b$.
- Zero net supply: $\sum_{j=1}^{N-1} b^j_t = 0$ for all $t$.
- Soft borrowing limit $b^j \geq b_\text{min} = -0.05$ enforced via Fischer–Burmeister residual.
- State expands from $(Z, k^1, \ldots, k^{N-1})$ to $(Z, k^1, \ldots, k^{N-1}, b^1, \ldots, b^{N-1})$ — dim $1 + 2(N-1) = 13$.
- Calibration additions: $b_\text{min} = -0.05$, $b_\text{scale} = 0.10$, $p_{b,\min} = 0.55$, $p_{b,\max} = 0.95$.

Restate the model in your own words.

## Stage 2 — Equilibrium

Two Euler equations per saver:

$$
u'(c^j_t) = \beta\,\mathbb{E}_t\!\left[(1+r_{t+1})\,u'(c^{j+1}_{t+1})\right] \quad\text{(capital)}
$$

$$
u'(c^j_t)\,p_{b,t} = \beta\,\mathbb{E}_t\!\left[u'(c^{j+1}_{t+1})\right] \quad\text{(bond)}
$$

The bond Euler pins down $p_b$ as the expected SDF.

Period-$t$ income for cohort age $j \in \{1, \ldots, N-1\}$:

$$
I^j_t = w_t\,\varepsilon_j + (1+r_t)\,k^j_t + b^j_t.
$$

Budget identity (savers $j = 0, \ldots, N-2$):

$$
c^j_t + k^{j+1}_{t+1} + p_{b,t}\,b^{j+1}_{t+1} = I^j_t.
$$

## Stage 3 — Algorithm

Three new pieces:

1. **Network outputs** widen to $(N-1) + (N-1) + 1 = 13$: $N-1$ capital savings rates (sigmoid), $N-1$ raw bond demands (tanh × $b_\text{scale}$), one bond price (sigmoid mapped to $(p_{b,\min}, p_{b,\max})$).
2. **Market-clearing layer.** Subtract the cohort-mean from raw bond demand: $b^j_\text{next} = \tilde b^j - \overline{\tilde b}$. Sum-to-zero by construction.
3. **Loss** is a weighted sum: $w_K \cdot \text{mean}(R_K^2) + w_B \cdot \text{mean}(R_B^2) + w_{FB} \cdot \text{mean}(FB^2)$.

Defaults: $w_K = w_B = 1.0$, $w_{FB} = 0.5$. `bonds_off=False`.

**Reduce-to-V2 lever.** Setting $w_B = w_{FB} = 0$ and `bonds_off=True` in `cohort_decisions` should produce the same trajectories as V2.

Network width raised to 192. Training budget 8,000 steps.

Compare your algorithm choice against an alternative — for instance, residualising the last cohort's bond holding from the constraint instead of mean-subtracting. Justify the choice.

## Stage 4 — Pseudo-code

Write `versions/v3/pseudocode.md`. Send to **`domain-reviewer`** with explicit verification asks:

- `b_next.sum(axis=-1)` is zero up to floating-point precision after every forward pass.
- `bonds_off=True` in `cohort_decisions` causes `bond_cost` to drop out of the budget and `b_next` to be returned as a zero tensor.
- Setting $w_B = w_{FB} = 0$ and `bonds_off=True` recovers V2's loss expression exactly.

## Stage 5 — Implementation

Implement `versions/v3/{model.py, network.py, train.py, simulate.py, plotting.py}`. Plan first. Add `tests/test_v3_smoke.py` with at least:

- `test_market_clearing_layer` — sum of bond demands across cohorts is zero.
- `test_fischer_burmeister_definition` — FB function vanishes at the complementarity-feasible set.
- `test_bonds_off_lever` — `cohort_decisions(bonds_off=True)` zeros the bond budget term.

## Validation gate

`simulate.validation_gate(sim, losses)` returns:

- `training_progressed`.
- `procyclical_top_state>bottom_state`.
- `rms_total_loss_<_15pct`.
- `bond_market_clears` ($\max_t |\sum_j b^j_t| < 10^{-4}$ — machine-zero by construction).
- `bond_lifecycle_dispersion` (std of mean cohort bond holding > $10^{-3}$).
- `consumption_grows_with_age`.
- `savings_peak_in_pre_retirement`.
- `bond_price_in_range`.

The expected lifecycle bond pattern is: young cohorts borrow (negative $\bar b$), old cohorts lend (positive $\bar b$). If your trained network does not show this pattern, the bug is in the budget bookkeeping or the loss weights — *not* in the network architecture.
