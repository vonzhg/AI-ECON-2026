# Stage 5: Implementation, Verification, Iteration (Template)

The only stage that lives in Claude Code (not in chat). Write code, run it, check it, fix what's wrong, log what changed. Repeat until the validation gate passes.

## What this stage produces

For each version `V_{n+1}`, a populated `versions/v{n+1}/` directory containing:

- **Source files** (`solver.py`, plus version-specific modules).
- **Updated `model_spec.md`** if Stage 1 changed during implementation.
- **`session_notes.md`** documenting what happened: which prompts, which fixes, which sub-agent calls.
- **Test entry** in `tests/test_v{n+1}_*.py`.
- **Notebook section** in `demo.ipynb` showing the new version's results next to the previous version's.

## The Stage 5 prompt pattern

```
Implement V_{n+1} per the approved pseudo-code in versions/v{n+1}/pseudocode.md.
Include:
  (i)   the diagnostics listed in pseudocode.md;
  (ii)  a comparison cell against versions/v{n}/build/notebook_report_v{n}.json;
  (iii) the validation gate listed in versions/v{n+1}/model_spec.md.
Do NOT add features beyond V_{n+1}. Plan first; show me the plan before writing code.
```

## The Claude Code agent loop (mirrors slide frame 316)

1. **Plan** — agent writes implementation plan to `versions/v{n+1}/plan.md`.
2. **Approve** — you read the plan; approve or course-correct.
3. **Implement** — agent writes `versions/v{n+1}/solver.py` and friends.
4. **Verify** — agent (or `verifier` sub-agent) runs the new code; reports convergence.
5. **Review** — `code-reviewer` and `domain-reviewer` sub-agents read in parallel.
6. **Fix** — apply suggestions Critical → High → Medium.
7. **Score** — Correctness ≥ 8/10? Done. Add notebook section. Run validation gate.

## Validation gate (the only objective stop condition)

The validation gate is whatever the version's `model_spec.md` says it is. For each version in this demo, it's listed in the table at the top of `demo.ipynb`. Examples:

- V0: deterministic-SS collapses to a point + procyclical capital + RMS Euler residual < 8%.
- V2: reduces to V1 numerically when the TFP grid is collapsed to V1's two states.
- V3: bond-loss weight zero reproduces V2's capital path within 1e-4.
- V5: per-phase residual snapshots show monotone improvement; final RMS Euler residual < 5% with positive bond holdings late in life.

If any check fails: the bug is in the new code (versions before V0 were validated). Use the previous-version report as your line-by-line diff target.

## Slide reference

Module 4 Topic 4.3, frames 316–376 (`Stage 5: from pseudo-code to running code` + verification).
