# Lecture 9 lab — Anatomy of an Agent

Companion to **Topic 9.2b: Anatomy of an Agent — Sessions, Sub-Agents, and
Skills**. Two parts: the mechanism offline, then real dispatch in your terminal.

## Part A — `Lec09_Lab_Agents.ipynb`

```bash
cd labs/Lec09_Agent_Lab
jupyter notebook Lec09_Lab_Agents.ipynb
```

Runs on the course's standard environment (numpy + matplotlib) with **no API
key, no network, and no GPU**. The language model is replaced by a
`ScriptedModel` that replays fixed responses, so the loop around it — which is
the real thing — runs deterministically and for free.

Nine steps: the session as a list, tool schemas, the agentic loop, the cost
curve, sequential relay, orchestrator–worker isolation, blackboard, a capacity
calculation, and an optional live call.

## Part B — `TERMINAL_WALKTHROUGH.md`

Where you dispatch specialist agents at a deliberately flawed Aiyagari solver
and run the four checks from the slides — including the decisive one: narrow an
agent's `description` and watch dispatch stop. Steps 6–7 then re-sort the team
by *background* rather than by review function, have you design one of your own
from the canvas, and make you fill a detection matrix to prove each specialist
catches something the others do not. Step 8 then runs one task at all three
invocation rungs — implicit, named, and scripted — the subject of Topic 9.2c.

Needs `claude` on your PATH and a completed `/login`. No API key.

## Files

| Path | What it is |
|---|---|
| `Lec09_Lab_Agents.ipynb` | Part A, outputs stored so it reads on GitHub |
| `agent_lab.py` | the loop, the stub model, token accounting, CLI probes |
| `TERMINAL_WALKTHROUGH.md` | Part B |
| `AGENT_DESIGN_CANVAS.md` | the eight-cell worksheet for Step 6 — print or copy it |
| `.claude/agents/{code-reviewer,domain-reviewer,verifier}.md` | the trio sorted by **review function** (T4's RA team) |
| `.claude/agents/{tooling-agent,math-agent,econ-agent}.md` | the trio sorted by **background** (Topic 9.2b §2) |
| `review_target/aiyagari_solver.py` | the flawed solver under review |
| `instructor/ANSWER_KEY.md` | **instructor copy** — the planted defects |

`.claude/agents/` is picked up only when you start `claude` from *this* folder,
so it does not affect sessions elsewhere in the repository. That resolution rule
is the point of Topic 9.2c's "Where Those Four Agents Lived": the course's own
reusable specialists (`deck-cartographer`, `boundary-auditor`, `claim-checker`)
therefore live at the **repository root**, in `AI-ECON-2026/.claude/agents/`,
where sessions actually start.

The answer key lives in `instructor/`, **not** in `review_target/`. That is
deliberate: a live test showed the reviewing agents will happily read an answer
key sitting next to the file they were asked to review, and then report its
findings as their own. Keep the two directories separate.

## A note on the solver

`review_target/aiyagari_solver.py` runs to completion and prints plausible
numbers. It is still wrong — that is the point. A solver that crashes teaches
nothing about silent failure. Students should not use it as a template; the
`crra_utility` function in Part A Step 2 shows the guarded version.

## Dependencies

Nothing beyond the repository's `requirements.txt`. `anthropic`, `openai`,
`crewai` and `langgraph` are deliberately **not** required — the real API
surface appears in the notebook as a printed reference sample only.
