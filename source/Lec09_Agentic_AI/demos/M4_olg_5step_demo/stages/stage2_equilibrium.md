# Stage 2: Equilibrium Definition (Template)

Once the model is specified, define the equilibrium concept and the fixed-point structure that any solver must respect.

## What this stage produces

An "equilibrium" section appended to the version's `model_spec.md` (or its own `equilibrium.md`) that lists:

- **The objects.** `⟨V, c, a', μ, r, w, …⟩` — value functions, policies, distributions, prices.
- **The conditions each object satisfies.** Bellman, market clearing, consistency of distribution under policies, transversality, no-Ponzi.
- **The fixed-point structure.** Which loops are inner, which are outer, what closes them.

For OLG with two assets and aggregate uncertainty (the V_final target), this includes the bond market clearing condition, the law of motion for aggregate state, and the consistency of perceived vs realized aggregate dynamics.

## The Claude Code prompt pattern

```
Given versions/v{n+1}/model_spec.md, state the recursive competitive equilibrium.
List ⟨V, c, a', μ, r, w⟩ (and any new objects added in this version) and the conditions
each must satisfy. Identify the inner/outer fixed-point structure. Then push back on
anything I wrote in the spec that's inconsistent with the equilibrium definition.
```

## Validation gate before Stage 3

- Every object in `⟨…⟩` has at least one defining equation.
- Every market or constraint introduced in Stage 1 has a matching clearing/binding condition.
- The fixed-point structure is acyclic (no two objects mutually defined without a wrapping loop).
- The agent identified at least one ambiguity in your Stage 1 spec — if it didn't, push back ("are you sure there's nothing missing?").

## Why this stage matters

Bugs in the equilibrium definition surface as wrong stationary distributions or non-convergent outer loops two stages later. Catching them here costs minutes; later they cost hours.

## Slide reference

Module 4 Topic 4.3, frames 167–187 (`Stage 2: define equilibrium — in chat, before any code`).
