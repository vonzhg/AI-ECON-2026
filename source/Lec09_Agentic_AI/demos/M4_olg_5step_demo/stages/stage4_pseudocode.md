# Stage 4: Pseudo-Code (Template)

The second contract. Algorithm bugs are 10× cheaper to fix on paper. Pseudo-code makes every implicit choice explicit, line by line.

## What this stage produces

A `pseudocode.md` in the version directory containing:

- **Outer loop(s).** Initialization, termination criterion, what's updated each iteration.
- **Inner loop(s).** Backward induction, distribution propagation, NN training step, etc.
- **Data structures.** Grid sizes, tensor shapes, what lives on the heap vs. in closures.
- **Diagnostics.** Every quantity that must be logged for the validation gate.

For NN-based versions (V4, V5), pseudo-code must spell out:
- Input dimension and feature normalization.
- Network architecture (layer sizes, activations).
- Loss function with weighted residual terms.
- Training-loop control (batch size, learning rate, number of steps, when to evaluate).
- Homotopy schedule (V5): phase boundaries, what changes in each phase.

## The Claude Code prompt pattern

```
For versions/v{n+1}, write pseudo-code in pseudocode.md following the algorithm choice
in algorithm.md. Be explicit about: outer/inner loop nesting, data structure shapes,
convergence criterion, and every diagnostic that must be logged. Use indented Python-
style pseudo-code, not prose. Then have the domain-reviewer sub-agent verify the
Bellman/Euler/KKT conditions match versions/v{n+1}/equilibrium.md.
```

## Validation gate before Stage 5

- A second-year PhD student could implement the pseudo-code without asking questions.
- The diagnostics list covers everything the validation gate requires.
- Sub-agent (domain-reviewer) signed off on the math.
- Estimated runtime is acceptable for the classroom (or has a coarse-grid fallback).

## Stage 4 exit test

Read the pseudo-code aloud and defend every choice. If you can't, send it back to the agent.

## Slide reference

Module 4 Topic 4.3, frames 242–260 (`Stage 4: pseudo-code as a second contract`) and frames 262–284 (sub-agents).
