# V0 Model Spec — Three-Period OLG with a Neural-Network Policy

This document is the **contract** for the V0 implementation in `versions/v0/`. It mirrors Stages 1–4 of the 5-stage Claude Code workflow taught in `Slides/Module4_Agentic_AI/M4_T3_Case_DynamicMacro.tex`. Stage 5 (implement / verify / iterate) lives in the source modules and the **Verification** section (§5) of this document.

V0 is intentionally the simplest interesting OLG that benefits from a neural-network policy. Each subsequent version (V1–V5) is one more piece of economics or one more numerical technique on top of this.

## §1. Stage 1 — Model

### Demographics and timing

One generational period $\approx 20$ years. At each calendar period $t$ three cohorts coexist:

| Cohort $j$ | Wealth at $t$ | Income at $t$ | Saves to | Consumes |
|---|---|---|---|---|
| 1 — young | 0 | $w_t \varepsilon_y$ | $a^m_{t+1}$ | $c^y_t = w_t \varepsilon_y - a^m_{t+1}$ |
| 2 — middle | $a^m_t$ | $w_t \varepsilon_m + (1+r_t) a^m_t$ | $a^o_{t+1}$ | $c^m_t = (\text{income}) - a^o_{t+1}$ |
| 3 — old | $a^o_t$ | 0 | — | $c^o_t = (1+r_t) a^o_t$ |

Each cohort has unit measure. Aggregate efficiency-weighted labour supply is $L = \varepsilon_y + \varepsilon_m$.

### Preferences

CRRA period utility $u(c) = c^{1-\gamma}/(1-\gamma)$. The young's lifetime objective is

$$
U_t = u(c^y_t) + \beta\,\mathbb{E}_t[u(c^m_{t+1})] + \beta^2\,\mathbb{E}_t[u(c^o_{t+2})].
$$

### Technology

Cobb-Douglas, $Y_t = Z_t K_t^\alpha L^{1-\alpha}$. Capital depreciates at rate $\delta$ per generational period.

$$
r_t = \alpha\, Z_t (K_t/L)^{\alpha-1} - \delta, \qquad
w_t = (1-\alpha)\, Z_t (K_t/L)^{\alpha}.
$$

### Aggregate shock

$Z_t$ is a 2-state Markov chain on $\{Z_\text{lo}, Z_\text{hi}\}$ with symmetric persistence $p$:

$$
\Pi = \begin{pmatrix} p & 1-p \\ 1-p & p \end{pmatrix}.
$$

Two states are sufficient because the conditional expectation in the Euler equation is a closed-form 2-term sum — no quadrature.

### Calibration

| Parameter | Value | Rationale |
|---|---|---|
| $\alpha$ | $1/3$ | standard capital share |
| $\beta$ | $0.85$ | per-generational-period discount; $\approx 0.992^{20}$ in annual |
| $\gamma$ | $2$ | standard CRRA |
| $\delta$ | $0.30$ | per-generational depreciation; $\approx 1.8\%$ annual |
| $\varepsilon_y$ | $0.6$ | hump-shaped labour profile |
| $\varepsilon_m$ | $1.0$ | |
| $L$ | $1.6$ | $= \varepsilon_y + \varepsilon_m$ |
| $Z_\text{lo}, Z_\text{hi}$ | $0.95, 1.05$ | symmetric ±5% TFP |
| $p$ | $0.80$ | symmetric persistence |

These are **not** calibrated to macro moments — they are chosen so the trained equilibrium is legible on a single-axis chart.

## §2. Stage 2 — Equilibrium

### Aggregate state

$s_t = (Z_t, a^m_t, a^o_t)$. Tracking $K_t = a^m_t + a^o_t$ alone is enough for *prices* but not for *payoffs* (the old's wealth-at-entry is exactly $a^o_t$, not an arbitrary split).

### Policy

The neural network outputs savings *rates*

$$
(s_y, s_m) = \pi_\theta(Z, a^m, a^o), \quad s_y, s_m \in (0,1),
$$

with savings *levels* given by the cohort budgets

$$
a^m_{t+1} = s_y\,w_t \varepsilon_y, \qquad
a^o_{t+1} = s_m\,(w_t \varepsilon_m + (1+r_t) a^m_t).
$$

### Market clearing

$K_{t+1} = a^m_{t+1} + a^o_{t+1}$ is automatic — the parametrisation makes capital-market clearing free.

### Recursive equilibrium

Policy functions $g_y, g_m: \{Z_\text{lo}, Z_\text{hi}\} \times \mathbb{R}_+^2 \to (0,1)$ together with an ergodic distribution $\lambda^\ast$ such that:

- both Euler equations hold pointwise on the support of $\lambda^\ast$;
- $\lambda^\ast$ is invariant under the transition induced by $(g_y, g_m, \Pi)$.

The two Euler equations (the old's problem is trivial):

$$
u'(c^y_t) = \beta\,\mathbb{E}_t[(1+r_{t+1})\,u'(c^m_{t+1})] \tag{Y}
$$

$$
u'(c^m_t) = \beta\,\mathbb{E}_t[(1+r_{t+1})\,u'(c^o_{t+1})] \tag{M}
$$

## §3. Stage 3 — Algorithm

### Choice

A Deep-Equilibrium-Net-style policy approximation:

- single MLP $\pi_\theta$ with sigmoid head;
- loss = MSE of normalised Euler residuals on a cloud of parallel economies;
- Adam with mild exponential LR decay;
- brief constant-savings pretraining as initialisation;
- periodic random restart of 5% of the cloud (anti-collapse).

### Why this algorithm

| Decision | Rationale |
|---|---|
| Single policy net (no value net) | Three-period horizon ⇒ FOCs alone close the model |
| Sigmoid-rescaled savings rates | $c > 0$ for every cohort by construction; no penalty terms |
| Closed-form expectation over $Z'$ | 2-state Markov ⇒ 2-term sum, no quadrature |
| Cloud of parallel economies | Endogenous training distribution consistent with policy |
| **Normalised** residual $1 - \beta\,\mathbb{E}[\dots]/u'(c)$ | Dimensionless ⇒ one learning rate works across cohorts |
| Adam + decay | Standard for stochastic non-convex; decay trades exploration for late stability |
| Random restart 5%/100 steps | Keeps the cloud's support broader than the ergodic distribution |
| Pretraining (~400 steps) | Anchors Adam in a basin where Euler loss is well-behaved |

### Deliberately excluded at V0

| Excluded | Why |
|---|---|
| Value network | FOCs already close the problem |
| Borrowing constraint | Sigmoid output guarantees positive consumption |
| Multi-stage training schedule | Single Adam pass converges; homotopy belongs in V5 |
| Idiosyncratic risk | Aggregate risk only at this stage |

## §4. Stage 4 — Pseudo-code

See `pseudocode.md` for the full pseudo-code. The implementation lives across `model.py` (primitives), `network.py` (`PolicyNet`), `train.py` (`pretrain`, `train`, `run`), and `simulate.py` (`run`, `validation_gate`).

## §5. Verification (Stage 5 outputs)

V0 is **validated** when a single run produces, in order:

1. **Loss curve.** Monotone-decreasing on log scale from MSE ≈ $0.3$ down to ≈ $10^{-3}$ (RMS residual ≈ 5%).
2. **Procyclical capital.** $\mathbb{E}[K \mid Z_\text{hi}] > \mathbb{E}[K \mid Z_\text{lo}]$.
3. **Lifecycle hump.** $c^o > c^m > c^y$ on average.
4. **Pointwise residuals.** RMS Euler residual on a fresh ergodic cloud below 8%.
5. **Deterministic-SS check.** Setting $Z_\text{lo} = Z_\text{hi} = 1$ and retraining briefly gives a path with constant $K^\ast$ (path std ≪ $10^{-4}$).

If V0 fails any of (1)–(5), the bug is in V0 — do not advance to V1.

### Validation gate (machine-checkable)

`simulate.validation_gate(sim, losses)` returns a dict of booleans for criteria 1, 2, 3, 4. The deterministic-SS check is run separately by the V0 test suite.

## §6. Where the researcher's judgment is indispensable

1. **Tracking $(a^m, a^o)$ separately rather than just $K$.** Wrong payoffs without it.
2. **Sigmoid-rescaling savings rates.** Free feasibility; alternative is penalty terms.
3. **Pretraining instead of homotopy.** Anchors Adam; cheaper than a phase schedule at V0.
4. **Closed-form expectation over $Z'$.** Continuous TFP would force quadrature, which adds noise and tunable knobs.
5. **The deterministic-SS check.** Without it, "loss decreased" is suggestive but not a benchmark.
