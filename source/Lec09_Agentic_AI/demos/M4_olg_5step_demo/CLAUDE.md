# CLAUDE.md — Project Instructions

This is the M4 Topic 4.3 classroom demo for the 5-stage Claude Code workflow taught in `Slides/Module4_Agentic_AI/M4_T3_Case_DynamicMacro.tex`. The demo evolves a small OLG model across six versions (V0 → V5), each transition representing one disciplined Claude Code session.

## Canonical entry

```bash
pip install -e .
jupyter notebook demo.ipynb
```

Run All. Every section prints its validation-gate result and writes a reproducibility JSON to `build/notebook_report_v{n}.json`.

## Stage discipline (mandatory)

For every version transition `V_n → V_{n+1}`, the agent must walk Stages 1–5 in order:

1. **Stage 1 — Model.** Read `versions/v{n}/model_spec.md`; draft `versions/v{n+1}/delta_spec.md` describing what's being added/changed/renamed. Restate the new model in plain language; user approves.
2. **Stage 2 — Equilibrium.** Update the equilibrium definition: list new objects and the conditions they must satisfy. Identify the fixed-point structure if it changed.
3. **Stage 3 — Algorithm.** Enumerate candidate methods with trade-offs; user picks deliberately. Record knob values (grid sizes, learning rates, loss weights) and at least one anticipated failure mode.
4. **Stage 4 — Pseudo-code.** Write line-by-line pseudo-code in `versions/v{n+1}/pseudocode.md`. The `domain-reviewer` sub-agent verifies the maths.
5. **Stage 5 — Implementation.** Plan first, implement, run tests, fix what's wrong, score. Do **not** declare done until the validation gate in `delta_spec.md` passes.

Skipping a stage is a process failure even if the resulting code happens to work.

## Version progression (PyTorch + NumPy throughout)

| Ver | Adds | New code (delta-style) |
|---|---|---|
| **V0** | 3-cohort, 1-asset, 2-state Markov TFP, MLP+sigmoid, Euler MSE on cloud, pretraining | seed (full implementation) |
| **V1** | 7 cohorts, hump-shaped labour, period discount $\beta_y^{72/7}$ | extend cohort loop and policy output dims |
| **V2** | 4-state Markov TFP (Rouwenhorst) | regenerate transition matrix; expectation loop runs over 4 outcomes instead of 2 |
| **V3** | bonds (2nd asset) + market-clearing layer + Fischer–Burmeister | dual savings rates per cohort + endogenous bond price + bond Euler equation |
| **V4** | capital adjustment cost $\psi_K(a_K' - a_K)^2 / 2$ | extra term in capital Euler; smoothness penalty on policy |
| **V5** | 4-phase stabilising homotopy (capital-only → bond pretrain → homotopy → fine-tune) | scheduled loss weights; phase boundaries logged to JSON |

PyTorch is a **base** dependency (V0 already needs it). No `[nn]` extra. JAX/Haiku/Optax are intentionally *not* used so the demo runs anywhere.

## Cross-version isolation

Each `versions/v{n}/` is **self-contained**. No runtime imports between versions. The `demo.ipynb` and tests do `sys.path.insert(0, "versions/v{n}")` before importing each version's modules. Validation that compares versions reads JSON reports under `build/`, not Python objects.

## Validation gate per version

A version is "done" when the gate listed in its `model_spec.md` (and at the top of its notebook section) passes. Each subsequent version must also pass a **reduce-to-previous** check — disabling its new feature reproduces the previous version's numbers within tolerance. This is the principal evidence the workflow produced trustworthy code.

## Files Claude is asked to write

For each version `V_{n+1}`, populate `versions/v{n+1}/`:

- `model_spec.md` (or `delta_spec.md` for transitions) — Stage 1 contract.
- `pseudocode.md` — Stage 4 contract.
- `model.py` — primitives (parameters, prices, cohort budgets).
- `network.py` — `PolicyNet`.
- `train.py` — cloud sim + Euler loss + Adam loop.
- `simulate.py` — forward simulation + ergodic statistics.
- `plotting.py` — matplotlib helpers.
- `session_notes.md` — what happened: prompts, fixes, sub-agent calls, validation result.

…plus a `tests/test_v{n+1}_*.py` and a notebook section.

## Sub-agents available

See `AGENTS.md`. Four roles: `designer` (read-only, drafts pseudo-code), `domain-reviewer` (read-only, checks economics), `code-reviewer` (read-only, checks broadcasting / typing), `verifier` (execute, runs tests). Reviewers write *new* review files; the main agent applies fixes.

## The silent benchmark

The research-grade target is described abstractly in `reference/research_target_notes.md`. **Do not name specific external notebooks, paper URLs, or third-party array libraries** in production-facing artifacts (`README.md`, `demo.ipynb` markdown cells, `versions/v*/model_spec.md`, `prompts/v*_to_*.md`). The reference notes mention Azinovic et al. (2022) only because the underlying paper is part of the public literature on Deep Equilibrium Nets.

## Conventions

- Random seeds: `torch.manual_seed(0)` and `np.random.seed(0)` at the top of every training entrypoint, with the seed exposed as a parameter to `train()` for reproducibility.
- Float dtype: `torch.float32` everywhere. Do not mix dtypes silently.
- Devices: detect once via `torch.device("cuda" if torch.cuda.is_available() else "cpu")`; the notebook prints the choice in Section 0.
- Tensors that flow into `cohort_decisions` and `prices` must always be `float`-typed; integer state indices stay `long`.
