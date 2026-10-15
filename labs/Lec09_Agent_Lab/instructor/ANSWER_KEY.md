# Answer key — planted defects in `aiyagari_solver.py`

*Instructor copy. The point of the lab is not that students find these by
reading, but that they watch three specialist agents find different subsets of
them — which is what makes the division of labour visible.*

The file runs to completion and prints plausible numbers. That is deliberate:
a solver that crashes teaches nothing about silent failure.

## What `code-reviewer` should find

1. **`TOL` is defined and never used** (line ~34, ~46). The loop runs
   `MAX_ITER` times unconditionally. There is no convergence test, so the code
   cannot report non-convergence — it can only report *finishing*. This is the
   canonical silent-failure pattern.
2. **Negative consumption is never masked** (`c = cash[ia] - a_grid`). For
   `sigma = 2`, `u(c) = -1/c`, which is **positive for `c < 0`**. Infeasible
   choices therefore score *higher* than feasible ones:

   | choice | c | u(c) |
   |---|---|---|
   | infeasible | −0.13 | **+9.49** |
   | feasible | +0.50 | −2.00 |

   At `a = 0, z = low`, **144 of 150** grid points imply `c <= 0`. The printed
   `iter 0  V[0,0] = 9.490446` *is* this bug, visible in the program's own
   output.
3. **`c = 0` divides by zero** — a `RuntimeWarning`, not an exception, so it is
   easy to miss.
4. **`aggregate_assets` also loops a fixed 500 times** with no convergence
   check on the distribution.

## What `domain-reviewer` should find

5. **No market clearing.** `R` is a fixed constant. Aiyagari's contribution is
   the general-equilibrium fixed point in `r`; `aggregate_assets` is computed
   and then *discarded* rather than equated to capital demand. The docstring
   says "partial equilibrium", so this is a scope limitation to state, not a
   lie to catch — a good reviewer distinguishes the two.
6. **`utility()` is wrong at `sigma = 1`** (log case): the formula divides by
   `1 - sigma = 0`. The constant is currently 2, so it does not bite, but the
   function is presented as general.

> **Run on 2026-09-12 (rung 3, Claude Code 2.1.270, `scripts/detection_matrix.py`).** Asked whether the printed
> number is the quantity the docstring claims, `econ-agent` wrote: *"This is a stated scope limit, not a defect."*
> That is why the walkthrough's matrix now scores partial equilibrium as a **scope row that must stay silent**,
> not as a defect to catch.

## What `domain-reviewer` should explicitly clear

7. **`EV = P @ V.T` is correct.** `EV[iz, ia']` is
   `sum_jz P[iz,jz] V[ia',jz]` — the right conditional expectation. A reviewer
   that flags this is producing a false positive, which is worth discussing:
   agents that "find" something in every category are not being careful, they
   are being agreeable.
8. Calibration is internally consistent: annual `beta = 0.96` with `r = 0.04`
   gives `beta(1+r) = 0.9984 < 1`, and `E[z] = 1`.

## What `verifier` should report

9. The script **exits 0** and prints `aggregate assets = 10.7631`. It does not
   crash. Runs are reproducible (no RNG). Verifier's job is to say exactly
   that — *and nothing about causes* — which is what makes its report
   combinable with the other two rather than redundant.

## Two leakage channels, both closed

Live testing of the Part B dispatch caught the reviewing agents cheating twice:

1. **`ANSWER_KEY.md` used to live in `review_target/`.** The orchestrator read
   it and had to be told explicitly to ignore it. Now it is in `instructor/`.
2. **The solver's docstring used to say "contains several planted defects."**
   A reviewer keyed on it: *"the module docstring at line 14-15 says this is a
   teaching artifact with planted..."* — turning the review into a scavenger
   hunt with a known count. The docstring is now neutral.

If you add material to this lab, keep both channels closed: **nothing the agent
can read should tell it what it is supposed to find.**

**Residual risk, stated honestly.** `review_target/` is now clean, but three
files outside it still state conclusions — `README.md`, `TERMINAL_WALKTHROUGH.md`
(Step 3 tells students what a good review finds), and this key. An agent with
`Glob`/`Read` over the repository *could* reach them. That is unavoidable
without hiding the teaching material from the students it is written for.

Before treating a classroom run as a clean demonstration, check the transcript
for any `Read` outside `review_target/`. If you want an airtight run, copy
`review_target/aiyagari_solver.py` alone into an empty directory and dispatch
there. Either way, "check what the agent was allowed to see" is the habit the
lab is trying to build.

## Verified after the fix

Re-run with the key relocated and the docstring neutral, `code-reviewer` still
derives finding #2 from first principles — no hint, no key:

> **Line 34-35 — `utility()` has no guard against non-positive consumption, and
> for the calibrated `SIGMA=2.0` this silently *rewards* infeasible choices
> instead of penalizing them.** With `SIGMA = 2.0`, `1 - SIGMA = -1`, so
> `utility(c) = c**(-1)/(-1) = -1/c`. For feasible `c > 0` this is correctly
> negative [...] for **infeasible `c < 0`**, `-1/c` is **positive**.

It also organised the report under this key's own headings ("## 1. Outright
bugs") — those come from the `description` and body of `code-reviewer.md`, which
is T3b's point about prose being the mechanism, visible in the output.

## Teaching use

Findings 1–2 are `code-reviewer`'s and finding 5 is `domain-reviewer`'s, and
neither agent can produce the other's. That is the argument for specialisation.
Finding 7 is the trap that shows why agreement between agents is weak evidence
(T5c's point, made concrete).
