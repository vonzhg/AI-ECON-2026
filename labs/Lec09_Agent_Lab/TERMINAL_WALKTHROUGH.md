# Part B — Design and dispatch your own agents

*Lecture 9 lab, part 2. Part A stubbed the model so you could see the loop.
This part uses a real one.*

**Time:** about 30 minutes. **Cost:** a few cents of your Claude Code
subscription. **You need:** `claude` on your PATH and a completed `/login`.

> Everything here works the same way in other terminal agents (Codex CLI,
> Gemini CLI, Aider). The file layout and slash-commands differ; the mechanism
> does not. A portability note is at the end.

---

## What you are about to test

T3b made four claims. You are going to try to break each one:

| # | Claim | The check |
|---|---|---|
| 1 | The "agent registry" is just your folder | `/agents` |
| 2 | Dispatch is an ordinary tool call | watch the transcript |
| 3 | Isolation means findings come back summarised | compare context |
| 4 | **Selection is driven by prose you wrote** | narrow a `description` |

Check 4 is the decisive one. If it behaves as predicted, the "router with a
registry" story cannot be right.

---

## Step 1 — Look at the agents you are about to dispatch

```bash
cd labs/Lec09_Agent_Lab
cat .claude/agents/code-reviewer.md
```

Three specialists ship with this lab, in `.claude/agents/`:

| File | Job | Tools |
|---|---|---|
| `code-reviewer.md` | implementation defects, silent failure | Read, Grep, Glob |
| `domain-reviewer.md` | does the code implement the economics? | Read, Grep, Glob |
| `verifier.md` | run it; report only what happened | Read, Bash, Glob |

These are the same three names T5b uses for the RA team. Now they are
real files.

Each is **YAML frontmatter plus prose**. There is no code in them. Note in
particular the `description:` field — it is the only part the main agent always
sees, and it is what selection runs on.

Note also what each one is told *not* to do. `code-reviewer` is forbidden from
judging economics; `verifier` is forbidden from speculating about causes. That
is the point of specialisation: three overlapping generalists would give you
three versions of the same answer.

## Step 2 — Confirm the registry is your folder (check 1)

```bash
claude
```

Then, inside the session:

```
/agents
```

You should see `code-reviewer`, `domain-reviewer` and `verifier`. Nothing was
registered with any service — the list came from the folder you just looked at.
Delete a file and it disappears from that list.

> Verified on Claude Code 2.1.261. A non-interactive equivalent:
> `claude -p "List the specialist subagents available in this project."`

## Step 3 — Dispatch one specialist (check 2)

The file under review, `review_target/aiyagari_solver.py`, is a value-function
iteration for the Aiyagari household problem. **It runs, it prints plausible
numbers, and it is wrong.** Run it first so you have seen it "work":

```bash
python3 review_target/aiyagari_solver.py
```

Now ask for a review, in the session:

```
Review review_target/aiyagari_solver.py for implementation defects.
```

Watch the transcript. The dispatch appears as a **tool call** in the same
stream as `Read` and `Bash`. On Claude Code 2.1.261 the call is named
**`Agent`** and carries a `subagent_type` naming which specialist to run. It is
not a special channel; it is the loop from Part A Step 3, where one of the
tools happens to start another loop.

A trace of a real run of exactly this prompt (tool calls only):

```
TEXT:     I'll launch the code-reviewer subagent on that file.
TOOL_USE: Bash                       <- ordinary tool
TOOL_USE: Agent   code-reviewer      <- the dispatch, same list
TOOL_USE: Read    review_target/aiyagari_solver.py
```

`Agent` sits in the same list as `Bash` and `Read`. That is the whole claim.

**What a good review finds.** The solver never uses its own `TOL`, and — the
serious one — it never masks infeasible consumption. With `sigma = 2`,
`u(c) = -1/c`, which is *positive* when `c < 0`, so bankruptcy scores higher
than eating:

| choice | c | u(c) |
|---|---|---|
| infeasible | −0.13 | **+9.49** |
| feasible | +0.50 | −2.00 |

That `9.49` is printed by the program itself on iteration 0. Compare with
`crra_utility` in Part A Step 2, which guards `c <= 0` and returns `-inf`.

## Step 4 — Dispatch several, and watch the context (check 3)

```
Review review_target/aiyagari_solver.py: use code-reviewer for implementation
defects, domain-reviewer for whether the economics is faithfully implemented,
and verifier to run it and report what actually happens.
```

Three dispatches, run in parallel — a real trace of this prompt:

```
TEXT:     I'll launch both reviewers in parallel on that file.
TOOL_USE: Read
TOOL_USE: Agent   code-reviewer      <- dispatch 1
TOOL_USE: Agent   domain-reviewer    <- dispatch 2, same turn
TEXT:     Both reviewers agree ... line 51-54, utility(c) is POSITIVE for c<0
```

Then:

```
/context
```

Compare against a run where you paste the file in and ask for all three reviews
yourself, with no subagents. The subagent version leaves your main context far
smaller: each worker read the file into *its own* window, and only a short
finding came back. That asymmetry — Part A Step 6, measured on real tokens — is
the economic case for sub-agents.

Notice the three reports do not overlap much. `verifier` reports an exit status
and output; it does not diagnose. `code-reviewer` never mentions market
clearing; `domain-reviewer` never mentions the unused tolerance. Neither could
produce the other's finding.

> **⚠️ The trap.** All three workers received *your* framing through the task
> string you wrote. If they agree, that is weak evidence — they inherited a
> common prior. T5c makes this argument in full. A clean context window
> buys focus, not independence, and no number of extra workers fixes it.

**A worked example of that trap, from building this lab.** Two earlier drafts
leaked the answers to the reviewers:

1. The instructor answer key sat in `review_target/`, next to the solver. The
   orchestrator found it and had to be told to ignore it.
2. The solver's own docstring said *"this file contains several planted
   defects."* A reviewer quoted that line back and went hunting for a known
   number of bugs.

Both produced reviews that looked excellent and proved nothing. Neither was
visible in the *output* — you only catch it by reading the transcript and
asking what the agent was allowed to see.

> **✏️ Your turn:** this generalises past this lab. If you ask an agent to check
> your regression results and your `NOTES.md` records which result you were
> hoping for, what have you actually measured? Name one thing in your own
> project directory that would leak a conclusion to a reviewing agent.

## Step 5 — The decisive experiment (check 4)

Make one agent unselectable **without touching any code**:

```bash
cp .claude/agents/code-reviewer.md /tmp/code-reviewer.backup
```

Edit `.claude/agents/code-reviewer.md` and replace the `description:` line with
something narrow and irrelevant:

```yaml
description: Formats BibTeX bibliography entries. Use only for .bib files.
```

Start a fresh session and ask exactly what you asked in Step 3:

```
Review review_target/aiyagari_solver.py for implementation defects.
```

**The dispatch stops.** The agent does the review itself, or picks a different
specialist. Nothing else changed: same model, same tools, same file, same
prompt. You edited one English sentence.

Restore it:

```bash
cp /tmp/code-reviewer.backup .claude/agents/code-reviewer.md
```

**What this proves.** Selection is the model reading short descriptions and
choosing — the same mechanism by which it chooses any tool. If a separate
router model with a registry service were making the decision, editing a
sentence inside one Markdown file could not switch the behaviour off.

The same logic explains `disable-model-invocation: true` in this repository's
own skill (`source/Lec09_Agentic_AI/demos/M4_olg_5step_demo/.claude/skills/olg-5step/SKILL.md`):
that flag is only meaningful because, by default, *the model itself* was
deciding.

## Step 6 — Re-sort the team by background, and find the hole

The three agents you have been dispatching are cut by **review function**:
who checks the code, who checks the code against the stated model, who runs it.
Sort the same folder by **which human background supplies the expertise** and
something is missing. `domain-reviewer` checks the code against the *stated*
model. Nobody checks the stated model against the *correct* one — look at
`instructor/ANSWER_KEY.md` after class and you will find a limiting-case defect
filed under `domain-reviewer`, because no file owns the mathematics.

Three more agents ship in this lab, cut that second way (T3a):

| File | Background | Tools |
|---|---|---|
| `tooling-agent.md` | programming + AI — convert, run, reproduce | `Read, Bash, Glob, Write` |
| `math-agent.md` | mathematics — domains, derivations, convergence | `Read, Grep, Glob` |
| `econ-agent.md` | economics — the object, the calibration, the counterfactual | `Read, Grep, Glob` |

Read all three before you dispatch them. Then, in a fresh session:

```
Have math-agent check the mathematics of review_target/aiyagari_solver.py
against the model stated in its docstring. The calibration is annual.
```

Compare what comes back with what `domain-reviewer` said in Step 3. They
overlap less than you expect — and notice that you just used rung 2 (you named
the agent), where Step 3 used rung 1 (the model chose). T3b's three rungs.

Now write a fourth of your own, from the eight cells of
`AGENT_DESIGN_CANVAS.md`. The canvas is the assignment; this is the shape its
answers compile to:

```yaml
---
name: replication-checker
description: <you write this -- it decides when you get dispatched>
tools: Read, Bash, Glob
model: sonnet
---

<your instructions, in prose>
```

Then test it honestly, in a fresh session each time:

1. Write a request it **should** handle. Does it get dispatched?
2. Write an adjacent request it should **not** handle. Does it get dispatched
   anyway? A description that is too broad is the most common failure —
   the agent gets pulled into work it is bad at.
3. Tighten the description until both cases behave. **You are debugging
   English**, which is the whole skill.

---

## Step 7 — Fill the detection matrix (the acceptance test)

An untested reviewer is one you are trusting for no reason. Run these three, one
**fresh session each** (`/clear` between them, or a new terminal), and record
what each one reported — not whether you agree with it.

```
Have tooling-agent run review_target/aiyagari_solver.py and report exactly
what happened, including any warnings.
```

```
Have math-agent check the mathematics of review_target/aiyagari_solver.py
against the model stated in its docstring. The calibration is annual.
```

```
Have econ-agent judge whether the number review_target/aiyagari_solver.py
prints is the quantity its docstring claims.
```

Fresh sessions matter: three reviews in one session share a context, so the
second and third read the first one's findings and you have measured agreement
rather than detection.

| Defect (or control) in the solver | `tooling-agent` | `math-agent` | `econ-agent` |
|---|---|---|---|
| the loop never tests a convergence criterion |  |  |  |
| a utility function evaluated outside its domain |  |  |  |
| *control:* `EV = P @ V.T`, which is **correct** | silent? | silent? | silent? |
| *scope:* partial equilibrium, stated in the docstring --- a limitation to state, **not a bug** | silent? | silent? | silent? |

What the matrix tells you, and what no single report can:

- **An empty column** — that agent's `description` is too narrow. It was never
  dispatched, or it was and found nothing it owns.
- **A full column** — too broad. You do not have a specialist, you have a third
  copy of one generalist.
- **Anything in the control or scope row** — a false positive, and the most important
  result in the lab. An agent that finds something in every category is not
  being careful; it is being agreeable. T5c is about why.
- **A cell where two agents agree** — weak evidence, not strong. They read the
  same file, with a framing you wrote.

Do not consult `instructor/ANSWER_KEY.md` until the matrix is filled, and check
your transcripts for any `Read` outside `review_target/` — an agent that found
the teaching material found the answers, not the defects.

---

## Step 8 — The same task at all three rungs

T3b claims there are exactly three ways to invoke an agent, and that they
differ only in who chooses. Run one task three ways and see it.

**Rung 1 — you describe, the model chooses.** No agent named:

```
Check the mathematics of review_target/aiyagari_solver.py.
```

**Rung 2 — you name it.** The Step 7 string above. Then check which specialist
each rung picked, and whether rung 1 picked the one you would have.

**Rung 3 — you script it.** Outside the session, one process, answer parsed
rather than read:

```bash
claude -p "Check the mathematics of review_target/aiyagari_solver.py." \
       --output-format json \
       --allowedTools Read Grep Glob \
  > /tmp/math_review.json
```

Then fan out over more than one file at a time:

```bash
for f in review_target/*.py; do
    claude -p "Check the mathematics of $f." \
           --output-format json --allowedTools Read Grep Glob \
      > "/tmp/$(basename "$f" .py).json" &
done
wait
```

**What to notice.** Rung 3 never asked your permission and never showed you a
transcript. That is the trade: you stopped spending a turn per dispatch, and in
exchange you now need something other than your own reading to decide whether
each answer is any good. T3b's last section is about choosing which.

*Verified on Claude Code 2.1.269. There is no `--max-turns` at the CLI; the
per-run ceiling is the `model:` line in the agent file.*

---

## Portability: the same thing elsewhere

The pattern outlives any one product (Appendix TA's point).

| | Claude Code | Codex CLI / others |
|---|---|---|
| Specialist definitions | `.claude/agents/*.md` | equivalent config dir |
| Inspect them | `/agents` | the tool's list command |
| Dispatch appears as | an `Agent` tool call | a tool/function call |
| Selection driven by | the `description` prose | the same |

`codex` was **not installed on the machine this lab was written on**, so the
Claude Code path is the tested one and the right-hand column is stated from the
shared mechanism rather than from a run. If you have Codex CLI, repeat Step 3
under it and compare: the vocabulary differs, the loop does not.

---

## What to hand in

1. The transcript excerpt from Step 3 showing the dispatch as a tool call.
2. Your before/after `/context` numbers from Step 4.
3. **The two prompts from Step 5** — the one that dispatched and the one that
   did not — plus the two `description` lines that caused the difference.
4. Your own agent from Step 6, its filled `AGENT_DESIGN_CANVAS.md`, and one
   sentence on what you had to change to stop it being dispatched for the
   wrong task.
5. **Your filled detection matrix from Step 7**, plus one sentence on the cell
   that surprised you.
6. From Step 8: which specialist rung 1 chose when you named none, and your
   `/tmp/*.json` from the scripted run.
7. One sentence: after Step 5, what would you now say to a colleague who says
   "the system automatically picks the right agent for my prompt"?

## Reproducibility

Part B calls a live model; it will **not** reproduce verbatim. Record the date,
`claude --version`, and the model name next to anything you keep — the
chain-of-custody discipline from T5d.
