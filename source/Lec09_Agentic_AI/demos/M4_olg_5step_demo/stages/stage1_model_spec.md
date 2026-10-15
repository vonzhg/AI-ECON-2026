# Stage 1: Model Specification (Template)

The first stage of every Claude Code workflow session. The artifact is a written specification of the economic model — a contract between you and the agent.

## What this stage produces

A `model_spec.md` (or `delta_spec.md` for transitions) that pins down:

- **Research question.** One paragraph. Why are you solving this model?
- **Optimization problem.** Objective, controls, constraints, state. Maths in LaTeX.
- **Functional forms.** Utility, production, transition, etc., with parameter symbols.
- **Parameter values.** With economic motivation or citation, not just numbers.
- **Expected outputs.** Policy functions, distributions, moments — the things the solver must report.

For a transition `V_n → V_{n+1}`, write only the *delta*: what's added, what's removed, what's renamed. The previous version's spec stays valid for everything not in the delta.

## Why this matters

The slide deck (frame 127–143) puts it bluntly: **code quality is directly proportional to spec quality.** A vague prompt produces a model that *looks right* but solves the wrong problem.

## The Claude Code prompt pattern

```
Read versions/v{n}/model_spec.md (and reference/research_target_notes.md for the long-run target).
Draft model_spec.md (or delta_spec.md) for V_{n+1} that adds:
  - {single concrete model change, e.g. "endogenous bond price p_b alongside capital"}
Pin down: research question, optimization problem with constraints, parameter values
with economic motivation, expected outputs. Keep notation consistent with V_n.
Then restate the new model in your own words; I'll check for misunderstandings.
```

## Validation gate before moving to Stage 2

- The new spec parses without ambiguity (a second person could implement it).
- The agent's restatement matches your intent.
- The delta is small enough that one Claude Code session can implement it (rule of thumb: ≤300 LOC change).
- Every new symbol has a defined parameter value.

## Slide reference

Module 4 Topic 4.3, frames 127–143 (`Stage 1: write down the model before opening Claude Code`).
