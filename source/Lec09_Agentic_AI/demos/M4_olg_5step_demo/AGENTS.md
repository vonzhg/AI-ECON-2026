# Sub-Agent Definitions

The slide deck (frame 262–284) recommends separating writers from checkers via sub-agents. Each agent has a narrow job, read-only or execute permissions, and shares files (not memory) with other agents. This file lists the four sub-agents the demo's prompts assume; configure them under `.claude/agents/` if you want them as named agents in Claude Code, or invoke them inline within prompts.

## designer

| Field | Value |
|---|---|
| **Permissions** | READ-ONLY |
| **Job** | Drafts pseudo-code (Stage 4) and algorithm rationale (Stage 3). Cites textbook references where applicable. |
| **When to use** | Any time a new version's pseudo-code is being drafted from a model spec. |
| **Inputs** | `versions/v{n+1}/model_spec.md`, `stages/stage{3,4}_*.md`, `reference/research_target_notes.md`. |
| **Output** | `versions/v{n+1}/algorithm.md` and `versions/v{n+1}/pseudocode.md`. |

## domain-reviewer

| Field | Value |
|---|---|
| **Permissions** | READ-ONLY |
| **Job** | Verifies that the code/pseudo-code respects the Bellman equation, Euler conditions, KKT for non-negative consumption, market clearing, and any other equilibrium condition in the version's spec. |
| **When to use** | After every Stage 4 pseudo-code draft, and again after every Stage 5 implementation. |
| **Inputs** | `versions/v{n+1}/model_spec.md`, `versions/v{n+1}/pseudocode.md`, `versions/v{n+1}/solver.py`. |
| **Output** | A review markdown listing issues by severity (Critical / High / Medium). |

## code-reviewer

| Field | Value |
|---|---|
| **Permissions** | READ-ONLY |
| **Job** | Checks code quality: vectorization opportunities, type hints, indexing bugs, NumPy/PyTorch broadcasting consistency, error handling at boundaries. Does not opine on economics. |
| **When to use** | Run in parallel with `domain-reviewer` after every Stage 5 implementation. |
| **Inputs** | The new/changed Python files. |
| **Output** | Review markdown listing issues by severity. |

## verifier

| Field | Value |
|---|---|
| **Permissions** | EXECUTE |
| **Job** | Runs the new version's code and the test suite. Reports convergence, NaNs, distribution-mass drift, validation-gate pass/fail. |
| **When to use** | After every Stage 5 implementation, and after every fix from the reviewer agents. |
| **Inputs** | None — operates on the workspace. |
| **Output** | Stdout/stderr from `python -m unittest discover -s tests` and `python run.py all`, plus the notebook's `notebook_report_v{n+1}.json`. |

## Workflow that uses all four (Stage 5)

```text
1. designer drafts pseudocode.md
2. domain-reviewer reads pseudocode.md → review_pseudocode.md
3. you fix Critical+High items
4. main agent implements solver.py
5. verifier runs tests + notebook section
6. domain-reviewer + code-reviewer in parallel → review_implementation.md
7. main agent fixes Critical → High → Medium
8. verifier re-runs; validation gate must pass before this version is "done"
```

The four agents are not magic — they are role-separated prompts. The friction between writer and checker surfaces issues a single thread would miss.

## Notes for Claude Code

- These four roles must NOT mutate each other's files. Reviewers write *new* review files; the main agent applies fixes.
- The `verifier` is the only role with execute permission. Do not give the others execute access; they must work from artifacts.
- Sub-agent calls should be logged in `versions/v{n+1}/session_notes.md` so the audit trail is visible in the demo.
