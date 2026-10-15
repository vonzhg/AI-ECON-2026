---
name: olg-5step
description: Manually run the local five-step OLG dynamic-macro demo. Use when explicitly invoked to demonstrate Model -> Equilibrium -> Algorithm -> Pseudo-code -> Implement.
argument-hint: "stage number or command"
disable-model-invocation: true
---

# OLG Five-Step Skill

Use this skill to demonstrate the five-stage agentic dynamic-macro workflow.

## Workflow

1. Work from the demo root, the folder containing `run.py`, `stages/`, and `src/`.
2. If the user gives a stage number, show both the stage document and its Claude prompt:

```bash
/usr/bin/python3 run.py stage --stage "$ARGUMENTS"
/usr/bin/python3 run.py prompt --stage "$ARGUMENTS"
```

3. For the full classroom run:

```bash
/usr/bin/python3 run.py stage-map
/usr/bin/python3 run.py all
/usr/bin/python3 run.py check
```

4. Point students to:

```text
stages/
prompts/
figures/stage_map.svg
figures/life_cycle_profiles.svg
figures/equilibrium_gap.svg
figures/asset_distribution.svg
build/summary.json
```

## Constraints

- Do not skip stages 1-4 when the teaching goal is the workflow.
- Use `/usr/bin/python3`.
- Do not require external Python packages.
