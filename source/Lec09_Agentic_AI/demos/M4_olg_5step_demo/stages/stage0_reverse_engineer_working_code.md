# Stage 0: Reverse-Engineer The Working Code (Bonus Track)

The forward workflow (Stages 1 → 5) starts from a written model and ends at running code. Stage 0 starts from a working solver and infers the economics — useful when you inherit code, or when validating that an AI-written V_n actually solves what its `model_spec.md` claims.

```text
working code -> inferred model -> clean notes -> controlled extensions
```

The goal isn't to trust the code blindly. The goal is to learn how to turn a known-good implementation into an explicit economic contract.

## Start from a known-good baseline

V0 is the baseline in this demo. From the demo folder:

```bash
jupyter notebook demo.ipynb     # Run All — Section 1 walks through V0
# or:
python3 run.py all              # CLI equivalent of V0 only
python3 -m unittest discover -s tests
```

Success means:

- `build/notebook_report_v0.json` exists with `summary.excess_assets` near zero.
- Capital supply and demand match at the chosen `r`.
- Distribution mass = 1.
- Euler residuals are finite and reported.
- The four SVGs in `figures/` regenerate.

## Code files to read

- `versions/v0/solver.py` — economics, dynamic programming, distribution propagation, equilibrium search.
- `versions/v0/plotting.py` — generated classroom figures (hand-rolled SVG).
- `versions/v0/cli.py` — commands exposed through `run.py`.
- `tests/test_v0.py` — regression checks that define the V0 contract.

## Reverse-engineering questions

1. What are the state variables?
2. What are the controls?
3. What is the household budget constraint during working ages?
4. What changes in retirement?
5. What is the firm problem?
6. Which market clears in equilibrium?
7. What numerical method solves the household problem?
8. What numerical method solves the general equilibrium price?
9. What checks prove the current version is still usable?
10. Which simplifying assumptions would become binding in the next version on the ladder (V1, V2, V3, V4, V5)?

## Model inferred from V0 code

Finite-life OLG, state `(age, assets, productivity)`. The household chooses next-period assets; consumption is implied by the budget constraint. Assets stay on a nonnegative grid.

During working ages:

```text
c + a_next = (1 + r) a + w * labor_age * z
```

During retirement:

```text
c + a_next = (1 + r) a + pension
```

Cobb-Douglas firm. For a guessed `r`, the firm block returns `K_demand`, `L`, `w`. Equilibrium = an `r` such that household asset supply equals firm capital demand.

## Algorithm inferred from V0 code

Backward induction on a finite asset grid. The stationary distribution is propagated forward using the policy function and the productivity transition matrix. The general equilibrium loop first searches for a sign-changing bracket in the capital-market gap, then bisects on `r`.

## Output from Stage 0

By the end of Stage 0 a student should be able to write `versions/v0/model_spec.md` from the code alone:

```text
state variables -> controls -> budget constraints -> firm side -> equilibrium condition -> validation checks
```

Stage 0 reads code and infers the model. **Stages 1–5** turn each subsequent inferred-or-specified model into the next validated version.
