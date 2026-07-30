# Case Study: AI-Assisted Development of a Deep Ramsey Solver for Heterogeneous-Agent Economies

## Summary

This case study documents how an AI collaborator was used to extend a computational
Ramsey-taxation project from a representative-agent (RA) neural-network solver toward a
heterogeneous-agent (HA) solver with idiosyncratic employment risk, incomplete markets,
and occasionally binding borrowing constraints. The work spans three connected layers:
the economic model, the numerical algorithm, and the software codebase.

The central lesson is not that the AI solved the economics. It did not. The AI was most
useful as a fast formalization, implementation, and debugging partner. It compressed the
loop between theory, code, diagnostics, and documentation. The researcher still supplied
the economic judgment: choosing which formulations were meaningful, identifying when
plausible-looking equations were wrong, deciding when a simplification was strategically
useful, and accepting or rejecting numerical results.

The self-contained package consists of five source groups:

1. Original RA source/config files: `01_source_code/RA_Ramsey_NN_original/`.
2. First HA source/config implementation: `01_source_code/HA_Ramsey_HA_NN/v1/`.
3. Full HA source/config implementation: `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/`.
4. Model/algorithm growth TeX files: `02_tex_model_algorithm_growth/`.
5. Notes, audits, and local support snapshots: `03_notes_and_audits/` and `01_source_code/HA_1_local_snapshot/`.

The bundled source copies under `01_source_code/` intentionally include only `.py` and `.json` files. Supporting markdown and TeX documents are kept outside the source tree. See `PACKAGE_MANIFEST.md` for the folder-level inventory and source-copy policy.

---

## 1. Source Corpus and Provenance

The case study is supported by both code and theory documents. They are related but not
identical: `Ramsey_NN` is the mature RA implementation, `Ramsey_HA_NN/v1` is the first HA
prototype, `Ramsey_HA_NN/v2_fullmodel` is the larger full HA implementation, and
`01_source_code/HA_1_local_snapshot` is a local teaching/support snapshot close to the early HA code.

### Original RA Code Repository

Path:

```text
01_source_code/RA_Ramsey_NN_original/
```

Important files:

| File | Role |
|---|---|
| `config.json` | Main RA experiment configuration, including pretraining, adaptive sampling, admissibility thresholds, alpha-shape settings, and simulation options. |
| `dashboard.py` | RA training orchestration: loads config, creates networks, samples from history, and runs training. |
| `adaptive_sampling.py` | RA admissibility scoring, adaptive sampling, dynamic boundary updating, and alpha-shape/geometric boundary support. |
| `value_module.py` | RA policy/value networks, policy-value trainer, value simulator, and plotting utilities. |
| `simulation.py` | RA post-training simulation and visualization. |
| `pretrain.py` | Python pretraining code retained as source. The original C++ and text pretraining data are intentionally not copied under the py/json-only source policy. |

### HA Development Repository

Base path:

```text
01_source_code/HA_Ramsey_HA_NN/
```

The HA code history is visible in two bundled version folders. These copies include only `.py` and `.json` source/configuration files.

| Folder | Role |
|---|---|
| `v1/` | First HA implementation. It has the core 5D state, actor-critic networks, FB residuals, alpha-boundary learning, simulation, and visualization, but the trainer is still relatively compact. |
| `v2_fullmodel/` | Full HA implementation. It expands the codebase substantially with versioning, checkpoint loading/saving, adaptive Level-1/Level-2 sampling, target critic updates, alpha-shape projection, boundary and shape penalties, richer diagnostics, and period-0 simulation. |

Both folders contain the same basic source/config module names: `config.json`, `ha_model.py`,
`boundary.py`, `dashboard.py`, `simulation.py`, and `visualization.py`. The file sizes and
symbols show the progression clearly: `v1/dashboard.py` is about 250 lines, while
`v2_fullmodel/dashboard.py` is over 1,200 lines and implements the explicit two-level
fixed-point loop.

Key differences in `v2_fullmodel`:

| Area | Progress from `v1` to `v2_fullmodel` |
|---|---|
| Configuration | Adds `random_seed`, `debug`, `versioning`, checkpoint frequency, target-network settings, boundary/shape penalties, and explicit sampling-method options. |
| Economic model | Keeps the 5D state and 3D controls, but adds a frozen target critic, bounded future-consumption transforms, normalized Euler discrepancies, raw-vs-clamped asset diagnostics, and explicit Q bounds. |
| Boundary learning | Moves from a simpler alpha boundary to a full 5D normalized alpha-shape with Delaunay `QJ` joggling, vectorized simplex membership, KD-tree projection, save/load, and sampling methods (`uniform`, `expanded_shape`, `gaussian`). |
| Training loop | Separates Level 1 domain stabilization from Level 2 policy optimization, keeps sampling until enough valid points are accumulated, projects rollout states back to the learned admissible set, and tracks value, FB, boundary, and shape losses separately. |
| Simulation | Adds a period-0 allocation problem that distinguishes initial controls from continuation states, plus richer Monte Carlo, ergodic, policy-function, report, and data outputs. |
| Visualization | Adds detailed diagnostics and versioned figures for losses, boundaries, policy functions, Q distributions, projections, and training progress. |


### Case-Study Support Documents

The root folder contains this case study and the support documents are organized under `02_tex_model_algorithm_growth/` and `03_notes_and_audits/`.

Important support documents:

| File | Role |
|---|---|
| `deep_ramsey_ai_case_study.md` | This completed case study. |
| `02_tex_model_algorithm_growth/ra_progression/RA_model_refine_3.tex` | Refined RA algorithm reference: recursive formulation, admissibility scoring, fixed-point structure, boundary refinement, adaptive sampling, training, and simulation. |
| `02_tex_model_algorithm_growth/ra_progression/deep_ramsey_refined_v3.tex` | Same refined RA material in a master/reference LaTeX file. |
| `02_tex_model_algorithm_growth/ha_progression/heterogeneous_agents_section_v2.tex` | HA theory section: 5D state, controls, explicit reductions, complementarity, FB formulation, admissibility, boundary learning, and training. |
| `02_tex_model_algorithm_growth/ha_progression/ha_boundary_learning_section_v2.tex` | Focused HA boundary-learning section: high-dimensional admissibility and alpha-shape construction. |
| `03_notes_and_audits/code_vs_document_analysis.md` | Audit comparing code and documentation, including scoring mismatches, boundary-learning details, and scaling issues. |
| `03_notes_and_audits/RA_HA_diaglos_1.md` | Conversation-derived working notes, including code/document alignment and vectorization proposals. |
| `01_source_code/RA_local_refined_v2/` | Local refined/copy versions of the RA code aligned with the refined RA document. |

### Local HA Teaching Snapshot

Path:

```text
01_source_code/HA_1_local_snapshot/
```

This directory is a local support copy close to the early HA implementation. It remains useful for teaching and for checking the core equations, but the fuller progress history is in `01_source_code/HA_Ramsey_HA_NN/v1/` and `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/`.

Important files:

| File | Role |
|---|---|
| `config.json` | HA configuration for log utility and Cobb-Douglas production. Includes economic parameters, state/control bounds, training schedule, FB penalty schedule, boundary settings, admissibility weights, and simulation settings. |
| `ha_model.py` | HA actor/critic networks and economic forward pass. Implements the 5D state, transition probabilities, budget-derived assets, Euler discrepancies, Fischer-Burmeister residuals, and admissibility scoring. |
| `boundary.py` | Alpha-shape boundary learner using Delaunay triangulation, circumradius filtering, normalized coordinates, and KD-tree buffer membership. |
| `dashboard.py` | Main HA training loop: boundary update, critic training, actor training with FB penalty, adaptive FB penalty update, plotting, checkpointing, and optional simulation. |
| `simulation.py` | HA simulation, ergodic statistics, policy-function analysis, plots, and report generation. |
| `run_simulation.py` | Standalone simulation runner for a trained HA model. |
| `visualization.py` | Training, FB penalty, boundary, and policy-function plots. |
| `02_tex_model_algorithm_growth/ha_progression/ha_model_section.tex` | LaTeX theory reference aligned with the HA code. |

---

## 2. Starting Point: The Representative-Agent Deep Ramsey Solver

The project began with a working solver for the Ramsey optimal-taxation problem under a
single representative agent. The RA state is

```text
(B, μ, g)
```

where `B` is government debt, `μ` is the co-state variable summarizing past policy
commitments, and `g` is the exogenous government-expenditure shock. Consumption is
represented through `c = 1 / μ`, matching the recursive formulation in
`02_tex_model_algorithm_growth/ra_progression/RA_model_refine_3.tex`.

The RA solver already contained the methodological core later reused in the HA setting:

| Component | RA implementation |
|---|---|
| Actor-critic architecture | Policy and value networks in `value_module.py`; refined local version in `01_source_code/RA_local_refined_v2/Ramsey_RA_value_module_v2.py`. |
| Two-level fixed point | Training alternates between learning the policy/value functions and refining the feasible domain. |
| Admissibility scoring | Candidate states are scored using policy feasibility, co-state bounds, and debt-bound feasibility. |
| Boundary learning | The RA document uses 2D binning in `μ` and shock-specific debt bounds; the original repository also contains alpha-shape/geometric support in `adaptive_sampling.py`. |
| Simulation | Post-training rollouts in `simulation.py` and local `01_source_code/RA_local_refined_v2/Ramsey_RA_simulation_module_v2.py`. |

In the RA case, the feasible region can be represented cheaply because the continuous
state is low-dimensional. The refined RA document describes debt-bound learning as
functions such as `B_min(μ, g)` and `B_max(μ, g)`, estimated with quantiles within bins.
That representation is interpretable and computationally convenient, but it does not
scale directly once the state includes multiple household asset and consumption variables.

---

## 3. Why Heterogeneity Changes the Problem

Adding idiosyncratic employment risk changes the economic and numerical problem in three
ways.

First, the state becomes high-dimensional. The HA implementations use the five-dimensional
state

```text
s = (K, a^e, a^u, c^e, c^u)
```

where `K` is aggregate capital, `a^e` and `a^u` are assets held by employed and unemployed
agents, and `c^e` and `c^u` are consumption/co-state variables encoding past commitments.
This is documented in `02_tex_model_algorithm_growth/ha_progression/ha_model_section.tex` and implemented directly in
`01_source_code/HA_1_local_snapshot/ha_model.py`, `01_source_code/HA_Ramsey_HA_NN/v1/ha_model.py`, and
`01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/ha_model.py`.

Second, household borrowing constraints bind occasionally and cannot be treated as a soft
box constraint. The optimality condition is a complementarity condition: assets must be
nonnegative, Euler discrepancies must be nonnegative, and at least one side must be zero.
This requires a different loss construction from the RA tax-bound penalties.

Third, the admissible region is no longer a simple pair of debt-bound functions. The early
HA snapshot learned geometry over the derived next-state coordinates, while
`v2_fullmodel` moves the alpha-boundary machinery to the full five-dimensional state space
`(K, a^e, a^u, c^e, c^u)`. A grid or binning scheme becomes sparse quickly in either case;
the HA code therefore uses a geometric boundary representation.

---

## 4. Extending the Economic Model

The first AI-assisted layer was formalizing the HA model so it could be implemented. The
critical modeling decision was to choose policy outputs that make the transition explicit.
The HA actor maps the current state to

```text
(n^e, c'^e, c'^u)
```

where `n^e` is employed labor and `c'^e`, `c'^u` are next-period consumptions. Given those
controls, the remaining quantities are derived in one forward pass.

The HA code uses separable CRRA/log utility and Cobb-Douglas production:

```text
u(c, n) = c^(1-σ)/(1-σ) - n^(1+γ)/(1+γ)        for σ != 1
u(c, n) = log(c) - n^(1+γ)/(1+γ)               for σ = 1
F(K, N) = K^α N^(1-α)
```

The teaching snapshot `01_source_code/HA_1_local_snapshot/config.json` uses the log case `σ = 1.0`. The development
folders `01_source_code/HA_Ramsey_HA_NN/v1/config.json` and `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/config.json` use the
full CRRA setting `σ = 2.0`, with `β = 0.8`, `α = 0.3333`, `γ = 2.0`, `δ = 0.1`,
`π^e = π^u = 0.5`, and a symmetric transition matrix `[[0.5, 0.5], [0.5, 0.5]]`.

The explicit transition is:

```text
K' = K^α (π^e n^e)^(1-α) + (1-δ)K - π^e c^e - π^u c^u
ŵ  = (n^e)^γ (c^e)^σ
Q  = β (c^e)^σ [(c'^e)^(-σ) π^{ee} + (c'^u)^(-σ) π^{ue}]
```

The budget-derived asset transitions are:

```text
a'^e = (1/Q) [ (a^e π^e π^{ee} + a^u π^u π^{eu}) / π^e + ŵ n^e - c^e ]
a'^u = (1/Q) [ (a^e π^e π^{ue} + a^u π^u π^{uu}) / π^u - c^u ]
```

Two corrections are important enough to record explicitly:

1. The asset equations use current consumption, `c^e` and `c^u`, not next-period
   consumption. This matters because the household budget constraint is a current-period
   constraint.
2. The transition probabilities must be used according to the direction of movement
   between employment states. The code in `01_source_code/HA_1_local_snapshot/ha_model.py` stores `π^{ee}`, `π^{eu}`,
   `π^{ue}`, and `π^{uu}` separately and comments the corrected usage.

This is a useful example of the human-AI division of labor. AI accelerated the drafting
and refactoring of the model, but the researcher had to catch economically wrong
formulations that were syntactically plausible.

---

## 5. Complementarity and the Fischer-Burmeister Penalty

The borrowing constraints are:

```text
a'^e >= 0,   a'^u >= 0
```

The Euler discrepancies in the log-utility HA reference are:

```text
φ^e = Q / c^e - β [π^{ee}/c'^e + π^{eu}/c'^u]
φ^u = Q / c^u - β [π^{ue}/c'^e + π^{uu}/c'^u]
```

The complementarity condition for each type is:

```text
a'^i >= 0,   φ^i >= 0,   φ^i a'^i = 0,   i in {e, u}
```

The implementation embeds this in the differentiable objective using the smoothed
Fischer-Burmeister function:

```text
Φ_ε(a, b) = a + b - sqrt(a^2 + b^2 + ε^2)
```

`01_source_code/HA_1_local_snapshot/ha_model.py` implements this as `fischer_burmeister(phi_i, a_prime_i)` and returns
both employed and unemployed FB residuals from `forward_physics`. `01_source_code/HA_1_local_snapshot/dashboard.py`
accumulates `fb_e^2 + fb_u^2` during actor training and adds it to the actor objective:

```text
loss = -value.mean() + λ_FB * mean(FB residuals)
```

The penalty weight follows an augmented schedule. In `01_source_code/HA_1_local_snapshot/config.json`, the current
values are:

| Key | Value |
|---|---:|
| `fischer_burmeister.epsilon` | `0.01` |
| `fischer_burmeister.lambda_initial` | `1.0` |
| `fischer_burmeister.lambda_max` | `100.0` |
| `fischer_burmeister.rho` | `1.5` |
| `fischer_burmeister.penalty_threshold` | `0.05` |

During training, if the average FB penalty remains above the threshold, `dashboard.py`
updates `λ_FB` by multiplying it by `rho` up to `lambda_max`. This starts with a moderate
penalty so the policy can explore and then progressively enforces complementarity.

---

## 6. High-Dimensional Admissibility and Boundary Learning

The HA admissibility score separates local economic feasibility from global domain
membership.

`01_source_code/HA_1_local_snapshot/ha_model.py` computes a continuous score

```text
A(s) = w_K A_K + w_a A_a + w_Q A_Q
```

with three components:

| Component | Meaning |
|---|---|
| `A_K` | Next-period capital `K'` lies within configured capital bounds. |
| `A_a` | Both `a'^e` and `a'^u` lie in `[0, K']`; this checks the borrowing constraint and rules out assets above aggregate capital. |
| `A_Q` | Bond price `Q` lies in `[0, β]`. |

The default weights in `01_source_code/HA_1_local_snapshot/config.json` are approximately one third each:
`w_K = 0.3333`, `w_a = 0.3333`, and `w_Q = 0.3334`. The score uses a smooth power-barrier
function, so near-boundary points receive graded scores instead of a hard zero-one label.

The geometric boundary is handled in `01_source_code/HA_1_local_snapshot/boundary.py` by the `AlphaBoundary` class. The
procedure is:

1. Filter sampled states to points with admissibility score above the threshold.
2. Keep the next-state coordinates `(K', a'^e, a'^u)`.
3. Normalize coordinates using the configured state bounds.
4. Build a Delaunay triangulation of the admissible points.
5. Compute each simplex circumradius and retain only simplices with radius `<= 1 / α`.
6. Use `Delaunay.find_simplex` to test exact membership in the alpha complex.
7. Use a KD-tree buffer in normalized coordinates to admit points near the learned boundary.

The current boundary configuration is:

| Key | Value |
|---|---:|
| `boundary.alpha_param` | `0.8` |
| `boundary.admissibility_threshold` | `0.95` |
| `boundary.inner_loops` | `5` |
| `boundary.n_boundary_samples` | `5000` |
| `boundary.buffer_percent` | `0.10` |

This is the main algorithmic change from the RA version. In the RA model, the domain can
be approximated by shock-specific debt bounds over `μ`. In the HA model, the admissible
region is a learned subset of a higher-dimensional space, so the boundary representation
has to be geometric.

---

## 7. Software Architecture

The completed local codebase has a clean RA-to-HA progression.

### Representative-Agent Code

The original RA repository is organized around:

| Module | Main symbols | Purpose |
|---|---|---|
| `adaptive_sampling.py` | `AdmissibilityScorer`, `AdaptiveSampler`, `AdmissibilityVisualizer` | Feasibility scoring, adaptive sampling, dynamic boundary estimation, and plotting. |
| `value_module.py` | `PolicyValueTrainer`, `ValueSimulator`, `create_policy_network`, `create_value_network` | Policy/value training and value simulation. |
| `simulation.py` | `RamseyT0Optimizer`, `RamseySimulator`, `run_simulations` | Post-training simulation and figures. |
| `dashboard.py` | `run_training` | Main RA orchestration. |
| `pretrain.py` | `run_pretraining` | Optional pretraining from legacy text data. |

The refined local copies in the case-study folder add document-aligned power-barrier
scoring and updated naming. The file `03_notes_and_audits/code_vs_document_analysis.md` is especially useful
because it records where the code and document differed, which issues were correctness
risks, and which were documentation/consistency issues.

### Heterogeneous-Agent Code

The HA implementation appears in three local forms: the support snapshot `01_source_code/HA_1_local_snapshot/`, the first
repository version `01_source_code/HA_Ramsey_HA_NN/v1/`, and the full version `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/`.
They share the same module layout, but `v2_fullmodel` is the most complete code path.

The HA implementation is organized around:

| Module | Main symbols | Purpose |
|---|---|---|
| `ha_model.py` | `HANetworkFactory`, `HAModel` | Actor/critic networks, economic transition, FB residuals, and admissibility scoring. |
| `boundary.py` | `AlphaBoundary` | Delaunay/alpha-shape boundary construction and membership testing. |
| `dashboard.py` | `train` | Boundary discovery, critic training, actor training, FB schedule, plots, checkpointing, and optional simulation. |
| `simulation.py` | `simulate_trajectory`, `run_monte_carlo`, `compute_ergodic_distribution`, `run_simulation` | Simulation, diagnostics, plots, and report generation. |
| `run_simulation.py` | `SimulationRunner` | Standalone model loading and simulation workflow. |
| `visualization.py` | `HAVisualizer` | Training losses, FB penalties, boundary plots, and policy plots. |

The implementation is configuration driven. `01_source_code/HA_1_local_snapshot/config.json`, `01_source_code/HA_Ramsey_HA_NN/v1/config.json`,
and `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/config.json` hold the economic parameters, bounds, learning
rates, batch sizes, rollout horizon, FB schedule, boundary parameters, admissibility
weights, and simulation settings. In `v2_fullmodel`, configuration also controls versioned
model loading/saving, checkpointing, target-network updates, boundary penalties, and
sampling methods. This matters because the research process involved repeated changes in
penalties, bounds, sampling, and visualization; moving those choices into config made
experiments easier to compare.

---

## 8. Development Progression: `v1` to `v2_fullmodel`

The new `Ramsey_HA_NN` folders show how the HA solver matured.

`v1` establishes the core proof of concept. It implements the 5D state, a 3-output actor,
a scalar critic, explicit transition equations, FB complementarity, alpha-boundary
learning, Monte Carlo simulation, and basic visualization. The training loop is still
compact: each iteration updates the boundary from random candidates, trains the critic,
trains the actor with the FB penalty, updates `lambda_fb`, and optionally simulates.

`v2_fullmodel` turns that prototype into a full research code path. The main change is
not a single equation but a more faithful computational workflow. It separates the solver
into two explicit levels: Level 1 stabilizes the learned domain by repeatedly sampling
until enough admissible points are collected for the alpha boundary; Level 2 prepares an
admissible training dataset and optimizes the policy/value networks conditional on that
boundary. This is much closer to the two-level fixed-point structure described in the RA
algorithm document.

Several debugging lessons became permanent code features in `v2_fullmodel`:

| Issue discovered during progress | `v2_fullmodel` response |
|---|---|
| Low pass rates from naive high-dimensional sampling | Adaptive sampling loops keep drawing candidates until enough valid points are accumulated. |
| Boundary estimates can become fragile in 5D | Boundary points are normalized, Delaunay uses `qhull_options="QJ"`, and the alpha boundary can be saved and reloaded. |
| Rollouts can leave the learned admissible set | The boundary module adds `project_to_admissible`, and training penalizes projection distance. |
| Critic bootstrapping can be unstable | `HAModel` adds a frozen target critic with soft updates. |
| Diagnostics need to identify which mechanism is failing | Loss tracking separates value, FB, hypercube boundary, alpha-shape projection, and total actor losses. |
| Initial-period economics differs from continuation dynamics | `simulation.py` adds a period-0 allocation routine where initial consumptions are controls rather than inherited states. |

This progression is important for the case study because it shows the AI-assisted work as
an iterative research process: first make the equations executable, then expose numerical
failures, then encode the fixes as more explicit algorithmic structure.

---

## 9. Training Flow

The HA training loop in `01_source_code/HA_1_local_snapshot/dashboard.py` follows this sequence:

1. Load `config.json` and initialize `HAModel`, `AlphaBoundary`, and `HAVisualizer`.
2. Generate random candidate states from the configured state bounds.
3. Run the economic forward pass and compute admissibility scores.
4. Update the alpha-shape boundary using high-scoring next-state points.
5. Train the critic on boundary-sampled states using finite-horizon rollout targets plus
   the continuation value.
6. Train the actor to maximize discounted welfare plus continuation value while penalizing
   FB residuals.
7. Increase the FB penalty weight when residuals exceed the configured threshold.
8. Save plots and the trained model.
9. Optionally run post-training simulation.

This realizes the two-level fixed-point idea in code. The inner level improves the
policy/value approximation conditional on the current domain. The outer level updates the
learned admissible region based on the states produced by the current policy.

The default HA outputs are:

| Output | Location |
|---|---|
| Trained model | `01_source_code/HA_1_local_snapshot/ha_model_final.pth` |
| Training progress plot | `01_source_code/HA_1_local_snapshot/figures/training_progress.png` |
| FB penalty plot | `01_source_code/HA_1_local_snapshot/figures/fb_penalty.png` |
| Boundary plots | `01_source_code/HA_1_local_snapshot/figures/boundary_3d_iter_*.png`, `01_source_code/HA_1_local_snapshot/figures/boundary_2d_iter_*.png` |
| Simulation report | `01_source_code/HA_1_local_snapshot/results/simulation_report.txt` |
| Simulation figures | `01_source_code/HA_1_local_snapshot/results/figures/` |
| Simulation data | `01_source_code/HA_1_local_snapshot/results/data/` |

---

## 10. Debugging and Validation

The most important debugging episodes were conceptual rather than syntactic.

The first was the asset-transition bug. A plausible formulation used the wrong
consumption timing in the household budget equation. The correct equations use current
consumption, because the budget constraint is a current-period restriction. This is now
visible in both `01_source_code/HA_1_local_snapshot/ha_model.py` and `02_tex_model_algorithm_growth/ha_progression/ha_model_section.tex`.

The second was transition-probability direction. The employed Euler equation must use
transitions from employment, while the unemployed Euler equation must use transitions from
unemployment. The code explicitly stores `π^{ee}`, `π^{eu}`, `π^{ue}`, and `π^{uu}` and
uses comments marking the corrected terms. This is the kind of bug that a type checker or
unit test may not catch unless the test encodes the economic relationship.

The third was complementarity convergence. FB residuals reveal whether the solver is
actually satisfying the borrowing-constraint complementarity conditions. The training
loop records average FB penalties and increases `λ_FB` when residuals remain above
tolerance. This made the residuals a first-class diagnostic rather than an afterthought.

The fourth was boundary bias. Uniform sampling is inefficient in high dimensions, but it
is less likely to reinforce a mistaken early boundary. Focused sampling is efficient, but
it can create feedback loops if the current policy is wrong. The current HA code uses
random candidate states for boundary discovery and a learned boundary for subsequent
training samples; the documentation records the broader design tension and the reason to
prefer conservative exploration when the feasible region is not yet reliable.

Good validation checkpoints for this project are:

| Checkpoint | What to inspect |
|---|---|
| Asset timing | `01_source_code/HA_1_local_snapshot/ha_model.py`: asset transitions subtract `c_e` and `c_u`, not `c_prime_e` or `c_prime_u`. |
| Transition probabilities | `01_source_code/HA_1_local_snapshot/ha_model.py`: bond-price and Euler terms use the appropriate transition probabilities for each current employment state. |
| Complementarity | `01_source_code/HA_1_local_snapshot/ha_model.py` and `01_source_code/HA_1_local_snapshot/dashboard.py`: FB residuals are computed and squared in actor training. |
| Boundary learning | `01_source_code/HA_1_local_snapshot/boundary.py`: Delaunay triangulation is filtered by circumradius and queried with `find_simplex`. |
| Admissibility score | `01_source_code/HA_1_local_snapshot/ha_model.py`: `A_a` checks both employed and unemployed next-period assets. |
| Simulation outputs | `01_source_code/HA_1_local_snapshot/results/simulation_report.txt` and plots under `01_source_code/HA_1_local_snapshot/results/figures/`. |

---

## 11. The AI-Human Division of Labor

The collaboration worked because the roles were distinct.

| Contribution | Primarily AI | Primarily researcher |
|---|---:|---:|
| Drafting and reorganizing LaTeX formulations | Yes |  |
| Checking economic correctness of formulations |  | Yes |
| Translating equations into vectorized PyTorch code | Yes |  |
| Deciding which simplifications are economically acceptable |  | Yes |
| Proposing implementation patterns for FB penalties and boundary learning | Yes |  |
| Choosing which mechanisms belong in the model |  | Yes |
| Summarizing diagnostics and code/document mismatches | Yes |  |
| Catching conceptual bugs in timing and transition probabilities |  | Yes |
| Setting validation benchmarks and accepting results |  | Yes |

The AI's comparative advantage was throughput: it could draft, refactor, align notation,
write vectorized routines, and summarize long code/document inconsistencies quickly. The
researcher's comparative advantage was judgment: recognizing which mathematical objects
were constraints, which were derived reductions, which equations encoded the wrong timing,
and when a numerical result was not economically credible.

That distinction is the main reason the final artifact is useful. The failures are not
incidental; they show where human verification is indispensable.

---

## 12. Outcomes and Current Status

The project produced four concrete outputs.

First, it produced a refined RA reference algorithm in `02_tex_model_algorithm_growth/ra_progression/RA_model_refine_3.tex` and
`02_tex_model_algorithm_growth/ra_progression/deep_ramsey_refined_v3.tex`, aligned with a local refined RA code copy.

Second, it produced a working HA theory section in `02_tex_model_algorithm_growth/ha_progression/ha_model_section.tex` and
`02_tex_model_algorithm_growth/ha_progression/heterogeneous_agents_section_v2.tex`, with a 5D state, explicit transition reductions,
borrowing-constraint complementarity, FB smoothing, actor-critic training, and geometric
boundary learning.

Third, it produced a concrete HA Python implementation sequence: `Ramsey_HA_NN/v1` as the
first prototype, `Ramsey_HA_NN/v2_fullmodel` as the full model, and `01_source_code/HA_1_local_snapshot/` as a local
support/teaching snapshot. Together these contain the economic model, alpha-shape boundary
learner, trainer, simulator, visualization modules, checkpoints/versioning logic, and
post-training diagnostics.

Fourth, it produced a reusable research workflow: write the equations, implement the
forward pass, audit code against equations, expose hyperparameters through configuration,
run diagnostics, revise the model, and preserve the provenance of each correction.

A clarification is important: earlier notes mention a possible linear-technology variant
and a 4D simplification, but no `ha_model_linear.py`, `config_linear.json`, or
`simulation_linear.py` file is present in the supplied local folders. Those linear-variant
claims should not be cited as current local code unless the missing files are added.

---

## 13. Transferable Lessons

Use AI to compress the theory-code-debug loop, not to replace domain judgment. The gains
came from iteration speed, not from delegating economic correctness.

Keep the theory and code close enough that each economic relationship can be checked in
both places. The key bugs were not syntax errors; they were plausible equations using the
wrong timing or wrong transition direction.

Make diagnostics part of the training objective. The FB residual is not merely a plot; it
is a direct measure of whether the complementarity conditions are being enforced.

Separate local optimality from global feasibility. The FB penalty enforces local
borrowing-constraint complementarity. The admissibility score and alpha-shape boundary
learning enforce whether the state belongs to the learned feasible region.

Prefer conservative exploration when the boundary is unknown. Efficient sampling can be
valuable later, but early overfitting to a wrong feasible region can corrupt the entire
fixed-point iteration.

Preserve provenance. The supporting documents are not clutter; they are the audit trail
that explains why the implementation looks the way it does.

---

## Appendix A: Replication Guide

### RA Solver

Source path:

```text
01_source_code/RA_Ramsey_NN_original/
```

Basic training:

```bash
cd "01_source_code/RA_Ramsey_NN_original"
python dashboard.py
```

Optional pretraining:

```bash
python pretrain.py
```

Main configuration file:

```text
config.json
```

Key RA config blocks to inspect:

| Block | Purpose |
|---|---|
| `model_io` | Model input/output numbering and pretraining data. |
| `economic_parameters` | Discount factor, labor curvature, aggregate shock process. |
| `state_bounds` | Co-state/value bounds. |
| `penalty_params` | Tax, debt, and labor penalty bounds/epsilons. |
| `training_iterations` | Number of outer iterations and warmup. |
| `policy_training`, `value_training` | Policy/value optimization settings. |
| `adaptive_sampling` | Cache and candidate-sampling settings. |
| `admissibility_thresholds`, `admissibility_weights` | Global admissibility scoring. |
| `scoring_parameters`, `buffer_parameters`, `hard_bounds`, `safe_bands` | Boundary and scoring details. |
| `simulation` | Post-training simulation length and starting points. |

### HA Solver

Primary full-model source path:

```text
01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/
```

First prototype source path:

```text
01_source_code/HA_Ramsey_HA_NN/v1/
```

Local teaching/support snapshot:

```text
01_source_code/HA_1_local_snapshot/
```

Basic full-model training:

```bash
cd "01_source_code/HA_Ramsey_HA_NN/v2_fullmodel"
python dashboard.py
```

The local `01_source_code/HA_1_local_snapshot/` snapshot can also be run with `python dashboard.py` from its directory.
Its standalone simulation CLI uses:

```bash
python run_simulation.py simulate --model-path ha_model_final.pth --config config.json --output results
```

Main configuration file in each HA folder:

```text
config.json
```

Key HA config blocks to inspect:

| Block | Purpose |
|---|---|
| `economic_parameters` | `β`, `α`, `σ`, `γ`, `δ`, stationary shares, and transition matrix. |
| `state_bounds` | Bounds for `K`, assets, and consumption. |
| `control_bounds` | Bounds/transforms for actor outputs. |
| `training` | Iterations, batch size, actor/critic learning rates, rollout length, plot frequency. |
| `fischer_burmeister` | Smoothing parameter and adaptive penalty schedule. |
| `boundary` | Alpha-shape parameter, threshold, sample count, and buffer size. |
| `admissibility` | Power-barrier parameters and score weights. |
| `network_architecture` | Hidden dimension and layer setting. |
| `simulation` | Number of trajectories, periods, burn-in, seed, and output directory. |

---

## Appendix B: Code-to-Claim Map

| Case-study claim | Primary supporting files |
|---|---|
| RA state is `(B, μ, g)` and uses a recursive Ramsey formulation | `02_tex_model_algorithm_growth/ra_progression/RA_model_refine_3.tex`, `02_tex_model_algorithm_growth/ra_progression/deep_ramsey_refined_v3.tex` |
| RA solver uses actor-critic policy/value networks | `01_source_code/RA_Ramsey_NN_original/value_module.py`, `01_source_code/RA_Ramsey_NN_original/dashboard.py`, `01_source_code/RA_local_refined_v2/Ramsey_RA_value_module_v2.py` |
| RA solver learns/refines an admissible region | `01_source_code/RA_Ramsey_NN_original/adaptive_sampling.py`, `01_source_code/RA_local_refined_v2/Ramsey_RA_adaptive_sampling_v2.py`, `02_tex_model_algorithm_growth/ra_progression/RA_model_refine_3.tex` |
| HA state is `(K, a^e, a^u, c^e, c^u)` | `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/ha_model.py`, `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/ha_model.py`, `01_source_code/HA_1_local_snapshot/ha_model.py`, `02_tex_model_algorithm_growth/ha_progression/ha_model_section.tex`, `02_tex_model_algorithm_growth/ha_progression/heterogeneous_agents_section_v2.tex` |
| HA actor outputs `(n^e, c'^e, c'^u)` | `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/ha_model.py`, `01_source_code/HA_1_local_snapshot/ha_model.py`, `02_tex_model_algorithm_growth/ha_progression/ha_model_section.tex` |
| HA assets are derived from current-period budget constraints | `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/ha_model.py`, `01_source_code/HA_1_local_snapshot/ha_model.py`, `02_tex_model_algorithm_growth/ha_progression/ha_model_section.tex` |
| HA transition probabilities are direction-specific | `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/ha_model.py`, `01_source_code/HA_1_local_snapshot/ha_model.py`, `02_tex_model_algorithm_growth/ha_progression/ha_model_section.tex` |
| Borrowing constraints are enforced through FB complementarity | `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/ha_model.py`, `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/dashboard.py`, `01_source_code/HA_1_local_snapshot/ha_model.py`, `02_tex_model_algorithm_growth/ha_progression/heterogeneous_agents_section_v2.tex` |
| FB penalty has an adaptive schedule | `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/config.json`, `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/dashboard.py`, `01_source_code/HA_1_local_snapshot/config.json`, `01_source_code/HA_1_local_snapshot/dashboard.py` |
| HA admissibility score checks capital, both assets, and bond price | `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/ha_model.py`, `01_source_code/HA_1_local_snapshot/ha_model.py`, `02_tex_model_algorithm_growth/ha_progression/ha_model_section.tex` |
| HA boundary learning uses Delaunay plus alpha-complex filtering | `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/boundary.py`, `01_source_code/HA_1_local_snapshot/boundary.py`, `02_tex_model_algorithm_growth/ha_progression/ha_model_section.tex`, `02_tex_model_algorithm_growth/ha_progression/ha_boundary_learning_section_v2.tex` |
| HA simulation produces reports and figures | `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/simulation.py`, `01_source_code/HA_1_local_snapshot/simulation.py`, `01_source_code/HA_1_local_snapshot/run_simulation.py` |
| Code/document mismatches were audited explicitly | `03_notes_and_audits/code_vs_document_analysis.md`, `03_notes_and_audits/RA_HA_diaglos_1.md` |

---

## Appendix C: Verification Checklist

Before presenting or extending this project, verify these points directly in the local
files.

1. `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/ha_model.py` and `01_source_code/HA_1_local_snapshot/ha_model.py` subtract current `c_e` and `c_u` in the budget-derived asset
   transitions.
2. `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/ha_model.py` uses `π^{eu}` and `π^{ue}` in the correct places for the bond-price
   and Euler equations.
3. `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/ha_model.py` computes both employed and unemployed FB residuals.
4. `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/dashboard.py` includes the squared FB residuals in the actor loss.
5. `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/dashboard.py` increases `lambda_fb` only when the average FB penalty exceeds the
   configured threshold.
6. `01_source_code/HA_1_local_snapshot/ha_model.py` computes the early asset-feasibility score `A_a` from both employed and unemployed assets; `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/ha_model.py` instead combines alpha-geometry membership with Q feasibility.
7. `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/boundary.py` and `01_source_code/HA_1_local_snapshot/boundary.py` filter Delaunay simplices by circumradius before membership tests.
8. `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/boundary.py` and `01_source_code/HA_1_local_snapshot/boundary.py` normalize coordinates before radius and KD-tree buffer checks.
9. `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/config.json` and `01_source_code/HA_1_local_snapshot/config.json` values match the assumptions described in the teaching material.
10. Generated outputs are intentionally not bundled under the source-copy policy; rerun the relevant `dashboard.py` or simulation script if figures, models, or reports are needed.

---

## Appendix D: Conversation Provenance

The local files are the authoritative source for this case study. Historical Claude
conversation links in earlier drafts are useful as provenance, but they are not standalone
file sources and should not replace the local copies.

Recorded historical links from the earlier draft:

| Artifact or discussion | Historical link |
|---|---|
| Original RA algorithm discussion | https://claude.ai/chat/3ead775f-7f3c-4df1-9164-2f08f521ede9 |
| Full HA model code review | https://claude.ai/chat/56bdc991-fe56-428c-8b94-3bae6b3af066 |
| Early RA-to-HA formulation | https://claude.ai/chat/5d6bb12f-5250-4402-a758-2e58c515d595 |
| Linear-technology discussion | https://claude.ai/chat/a5888e86-17b4-44e9-9530-acf688663d28 |
| Linear codebase walkthrough | https://claude.ai/chat/1b4474b3-da75-4062-8fb3-67abffd698ce |
| FB residual/debugging discussion | https://claude.ai/chat/6e660766-e477-48b6-b7bc-4d06eafbb7c1 |
| Edited HA LaTeX section | https://claude.ai/chat/ecf29c9b-e569-402c-a0be-66cfcc583460 |

Use these links only to recover the reasoning history. Use the local files listed in
Appendices A and B for citations, replication, and code verification.
