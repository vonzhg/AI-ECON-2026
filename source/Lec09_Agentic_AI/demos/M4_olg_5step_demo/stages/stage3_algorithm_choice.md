# Stage 3: Algorithm Choice (Template)

The agent will offer multiple methods. **You** pick. Stage 3 makes that choice deliberate and documents the rationale.

## What this stage produces

An "algorithm" section in the version's spec (or its own `algorithm.md`) that records:

- **Candidate methods** with their strengths and where they bite.
- **The chosen method** for this version, with explicit rationale.
- **What's deferred** to a later version (e.g., "V2 stays grid; V4 switches to NN policy").
- **Key numerical choices** made up front: convergence tolerance, grid construction style, interpolation order.

For the version ladder in this demo, Stage 3 is where each transition's heavy lifting is announced:

| Transition | Algorithm change |
|---|---|
| V0 → V1 | Same solver, factor residuals into a separate module. |
| V1 → V2 | Add a bond grid dimension to backward induction; everything else unchanged. |
| V2 → V3 | Discretize aggregate TFP via Tauchen; expand state by one dim. |
| V3 → V4 | Replace exhaustive grid policy with PyTorch MLP trained on Euler/bond/KKT residuals. |
| V4 → V5 | Wrap V4 training in a 5-phase homotopy schedule. |

## The Claude Code prompt pattern

```
For versions/v{n+1}/model_spec.md, list the algorithm options with their trade-offs.
I'm picking {method, e.g. "PyTorch MLP with SELU activations and Adam"} because {reason}.
Explain back: where this method may struggle on this specific model, what diagnostics
to log, and what convergence criterion to use. Cite at least one canonical reference.
```

## Validation gate before Stage 4

- The chosen method is named explicitly (not "VFI or whatever").
- Every numerical knob (grid size, tolerance, training-step count, learning rate) has a value or a default.
- The agent flagged at least one failure mode you should be ready for.
- You can defend the choice in a seminar — read the rationale aloud.

## Anti-pattern

Letting the agent silently default. Grid VFI is the comfort food; NN+homotopy is the right tool for V_final. Never get there *by accident*.

## Slide reference

Module 4 Topic 4.3, frames 219–240 (`Stage 3: choosing the algorithm`).
