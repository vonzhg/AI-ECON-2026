# Lecture 9 lab — Anatomy of an Agent, and the Design Lab

Companion to **Lecture 9, Parts 9.2–9.4**: how an agent runs (T2a), building and invoking agents (T3a, T3b), and the Design Lab (T4). Three parts: the mechanism offline, real dispatch in your terminal, then an agent of your own for an economics task.

**Start here (Step 0):** `python3 scripts/setup_check.py --live` tells you which of Claude Code, Gemini CLI, or Codex CLI works on your machine and routes you through Part B. Part A needs none of them.

## Part A — `Lec09_Lab_Agents.ipynb`

```bash
cd labs/Lec09_Agent_Lab
jupyter notebook Lec09_Lab_Agents.ipynb
```

Runs on the course's standard environment (numpy + matplotlib) with **no API key, no network, and no GPU**. The language model is replaced by a `ScriptedModel` that replays fixed responses, so the loop around it — which is the real thing — runs deterministically and for free.

Nine steps and two optional ones: the session as a list, tool schemas, the agentic loop, the cost curve, a recorded real run replayed through the same loop (Step 4b, optional), sequential relay, orchestrator–worker isolation, blackboard, a capacity calculation, and a live call (Step 9, optional).

## Part B — `TERMINAL_WALKTHROUGH.md`

Dispatch specialist agents at a deliberately flawed Aiyagari solver and run the four checks from T3b — including the decisive one: narrow an agent's `description` and watch dispatch stop. Steps 6–7 re-sort the team by *background* rather than by review function, have you design one of your own from the canvas, and make you fill a detection matrix. Step 8 runs one task at all three invocation rungs — implicit, named, and scripted.

The walkthrough is written for Claude Code. **On Gemini CLI or Codex CLI, take the rung-3 route:** every check that matters runs as a headless call through `scripts/detection_matrix.py`, which works on all three harnesses.

## Part C — the Design Lab (T4)

Pick a scenario, fill `AGENT_DESIGN_CANVAS.md`, write the agent's brief, and measure it at rung 3, three fresh runs per cell:

```bash
python3 scripts/detection_matrix.py scenarios/S1_aiyagari_audit --harness claude --k 3
```

| Scenario | The task | What it teaches |
|---|---|---|
| `scenarios/S1_aiyagari_audit/` | audit the flawed Aiyagari solver | silent numerical failure; scope versus bug |
| `scenarios/S2_leading_question/` | "explain how the 2022–23 rate hikes brought inflation down" | a leading verb on real FRED data; correlation printed as an effect |
| `scenarios/S3_fellows_provenance/` | audit ten rows of the Econometric Society Fellows dataset | provenance, missingness, and a mechanical gate (a hook or a script) |

Each run happens in a fresh folder that holds only the files under review, so nothing can leak the answers. Replies and a `summary.json` land in `results/` (not committed).

## Files

| Path | What it is |
|---|---|
| `Lec09_Lab_Agents.ipynb` | Part A, outputs stored so it reads on GitHub |
| `agent_lab.py` | the loop, the stub model, token accounting, and the harness table for Claude Code, Gemini CLI, and Codex CLI |
| `scripts/setup_check.py` | Step 0: which harness is installed and logged in |
| `scripts/detection_matrix.py` | the rung-3 acceptance test for any scenario, on any harness |
| `TERMINAL_WALKTHROUGH.md` | Part B |
| `AGENT_DESIGN_CANVAS.md` | the eight-cell worksheet — print or copy it |
| `.claude/agents/{code-reviewer,domain-reviewer,verifier}.md` | the trio sorted by **review function** (T5b's RA team) |
| `.claude/agents/{tooling-agent,math-agent,econ-agent}.md` | the trio sorted by **background** (T3a) |
| `review_target/aiyagari_solver.py` | the flawed solver under review |
| `scenarios/` | the Design Lab scenarios, each with a `brief.md` and a `scenario.json` |
| `transcripts/` | recorded runs used on the slides, e.g. the GDP-by-party trace behind T2a (Part A Step 4b replays it offline), and the scenario validation records |
| `instructor/` | **instructor copies** — answer keys and reference briefs |

`.claude/agents/` is picked up only when you start `claude` from *this* folder, so it does not affect sessions elsewhere in the repository. That resolution rule is the point of T2b's "Where an Agent Lives": the course's own reusable specialists (`deck-cartographer`, `boundary-auditor`, `claim-checker`) therefore live at the **repository root**, in `AI-ECON-2026/.claude/agents/`, where sessions actually start. Gemini CLI reads `.gemini/agents/` the same way, but only in a folder you have trusted.

The answer keys live in `instructor/`, **not** next to the files under review. That is deliberate: a live test showed reviewing agents will happily read an answer key sitting next to the file they were asked to review, and then report its findings as their own.

## A note on the solver

`review_target/aiyagari_solver.py` runs to completion and prints plausible numbers. It is still wrong — that is the point. A solver that crashes teaches nothing about silent failure. Students should not use it as a template; the `crra_utility` function in Part A Step 2 shows the guarded version.

## Dependencies

Nothing beyond the repository's `requirements.txt` (S2 also uses pandas, which it lists). `anthropic`, `openai`, `crewai` and `langgraph` are deliberately **not** required — the real API surface appears in the notebook as a printed reference sample only. Part B and the Design Lab need one terminal agent and your own login; no API key goes into any file here.
