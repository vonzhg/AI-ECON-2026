# Lec09: Agentic AI and Claude Code — Rewrite Plan

## Context

**Goal.** Rewrite Lecture 9 ("Agentic AI and Claude Code", 1 hour) for the new
2026 slide deck under `2026_New_Slides/Lec09_Agentic_AI/`. The folder is
currently empty. The source material is `3-python/Lec_2026_3A_AI_Coding.tex`
(2070 lines, ~75 frames) which contains far more than fits in one hour and
includes a Krusell-Smith semester project that belongs in Lec06, not here.

**Why this rewrite matters.** Three drivers:

1. **Apply RA + advisor feedback** in `Refinement_2026/Feng_slides_feedback.txt`
   (no agenda slide; redundant tool comparisons; key-bindings clutter; harness
   architecture introduced before the file structure was shown; unclear
   boundary between "what you say in English" vs "what's in SKILL.md"; missing
   limitations/failure-mode discussion).
2. **Integrate two Substack articles** by Paul Goldsmith-Pinkham
   (`getting-started-with-claude-code` and `from-an-empty-folder-to-a-figure`)
   to add a setup workflow and a real end-to-end empirical case study.
3. **Stay consistent with the lecture series.** Lectures 6, 8, 10 all use the
   Aiyagari household problem as the running example; this lecture should too
   (replacing the source's RBC walkthrough and the KS semester project).

**Intended outcome.** A ~60-slide Beamer deck that (a) frames agentic AI vs.
chat LLM vs. RAG with explicit naive-baseline motivation, (b) gives a
complete `.claude/` project tour, (c) traces the Aiyagari workflow file by
file, (d) shows a real case study from the Substack article, (e) lists
current failure modes in econometric terms, and (f) keeps the 5-pillar
workflow visible as a throughline from slide 3 onward.

**Slide-count philosophy.** Each substantive concept gets its own frame, and
every new technique gets a naive-baseline motivation frame before the
technical detail. Per instructor preference, slide count is a soft floor —
never compress two unrelated ideas into one frame just to hit a target. The
original "~40 slides" in `Slides_Plan_2026.md` was a planning estimate; the
actual count lands near 60 frames so every pillar, every harness component,
every motivation, and every step of the case study can be explained without
rushing.

---

## Critical files

| File | Role |
|---|---|
| `2026_New_Slides/Lec09_Agentic_AI/Lec09_Agentic_AI.tex` | **Create** — the new deck |
| `2026_New_Slides/Lec09_Agentic_AI/figures/` | **Create** — figures subfolder |
| `3-python/Lec_2026_3A_AI_Coding.tex` | Source — reuse preamble + many frames |
| `Refinement_2026/Feng_slides_feedback.txt` | Feedback to apply |
| `Refinement_2026/readings/Claude-Code-Presentation-main/Presentations/figures/` | Optional terminal/interface screenshots (local, no download) |

---

## Preamble (reuse from source verbatim)

Reuse the preamble from `Lec_2026_3A_AI_Coding.tex` lines 1–82:
- `documentclass[10pt,english,aspectratio=169]{beamer}` + Metropolis theme
- `listings` + `xcolor` colored Python style (`pythonstyle` definition: blue
  keywords, green comments, red strings, gray frame) — already satisfies the
  "use colored code listings" feedback
- Footer with frame number, navigation symbols suppressed
- Title: `Lec.~9: Agentic AI and Claude Code`

---

## Slide outline (~55 frames, 1 hour — soft target)

### Frames 1–3 — Framing
1. Title
2. **Agenda** (lists all 11 sections + appendices) — applies "add agenda"
   feedback
3. **The 5-Pillar Workflow** as a single visual (Theory → Algorithm →
   Numerics → Advanced → Code Literacy). Annotated: "this is the throughline;
   we will reference it on every section divider." Applies "5-pillar on
   slide 2" feedback.

### Frames 4–10 — What is Agentic AI? (with naive-baseline motivation)
4. **The naive baseline: ChatGPT in a browser tab** — concrete economic
   example. You ask: "Pull FOMC meeting dates since 2008 and plot the
   federal-funds-rate path around each meeting." A plain chatbot can write
   code, but it cannot run it, cannot fetch data, cannot debug its own
   output, and forgets the context after the next prompt. *List the four
   things missing*: (1) execution, (2) tool/file access, (3) iteration on
   real output, (4) persistent state across sessions
5. **The chatbot → RAG → agent spectrum** (3-box TikZ diagram). Use one
   economic question across all three boxes:
   - Plain LLM: "answers from training data only" — may hallucinate the
     2024 FOMC dates
   - RAG: "answers from training + retrieved Fed minutes" — accurate text
     but no computation
   - Agent: "takes actions: downloads CSV from FRED, runs Python, makes
     the plot, validates the dates against the calendar" — produces the
     deliverable
6. **What makes something "agentic"?** — four defining features, each with
   an economic-research example:
   - **Tool use**: calls `pandas`, `curl`, `R`, the shell
   - **Autonomy**: plans a 12-step empirical workflow without re-prompting
   - **Iteration**: sees that a regression failed, fixes the formula, reruns
   - **Persistence**: remembers your project conventions across sessions
7. **Standalone LLM vs RAG vs Agent** — comparison table with columns:
   knowledge source, action capability, state persistence, latency,
   typical economic use case (LLM=brainstorming; RAG=literature search;
   Agent=full empirical pipeline)
8. **The Ladder of AI Coding Maturity** Levels 0–5 (Paul GP's framing) —
   TikZ-stacked rungs. Level 0/1: copy/paste between ChatGPT and editor.
   Level 2: IDE-based agents (Cursor). Level 3: terminal agents (Claude
   Code). Level 4: MCPs (tool integrations). Level 5: unattended
   long-running agents. Note where most economists are today (Level 1–2)
   and where this lecture will move you to (Level 3)
9. **Why "Level 3" is the sweet spot for empirical research** — at
   Level 3 you get tool use + autonomy + iteration but you remain in the
   loop to validate. Below: too manual. Above: too risky to leave
   unattended on research code. Concrete: one Aiyagari V0→V1→V2 cycle is
   exactly the kind of bounded, validated task Level 3 was built for
10. **Claude Code in context**: alternatives table — Gemini CLI, OpenAI
    Codex CLI, Aider, Cursor. One-line on each. Then: "we use Claude Code
    as the concrete example for the rest of the lecture, but the workflow
    transfers to any Level-3 terminal agent"

### Frames 11–15 — The 5-Pillar Workflow Detail (each pillar is your econ-research armor)
11. **Pillar 1: Economic Theory** — formulate problems precisely (objective,
    constraints, equilibrium concept). From source lines 144–169. *Anchor:*
    "without a precisely written household problem, the AI's Aiyagari code
    will solve the wrong model — silently"
12. **Pillar 2: Algorithm Design** — translate theory to computation
    (Bellman, VFI, time iteration, EGM). Source lines 171–196. *Anchor:*
    "the AI may default to grid VFI when EGM would be 50× faster — only
    you know the trade-off"
13. **Pillar 3: Numerical Techniques** — optimization, interpolation,
    discretization, simulation, common pitfalls. Source lines 198–225.
    *Anchor:* "an AI-written Aiyagari with too few asset grid points near
    the borrowing constraint produces wrong precautionary savings"
14. **Pillar 4: Advanced Numerical Methods** — perturbation vs projection,
    speed/accuracy/locality trade-offs. Source lines 227–253. *Anchor:*
    "Smets-Wouters at the ZLB needs global methods; AI defaults to
    perturbation"
15. **Pillar 5: Code Literacy + the You-vs-AI division of labor** — Code
    literacy from source lines 255–282 + the "You Provide / AI Provides"
    table from source lines 286–302. RA2 add: "you also know the data
    and how it's set up"

### Frames 16–18 — Tool Comparison (consistent **Strengths → Workflow** headers)
16. **Tool 1: Chatbox (Claude.ai)** — Strengths / Workflow. Brief per
    feedback (do not dwell on Claude.ai Projects). Econ example: "design
    the Aiyagari Bellman equation in chat before any coding"
17. **Tool 2: IDE Integration (Cursor)** — Strengths / Workflow. Cursor
    key bindings table moved to **appendix A** per feedback. Econ example:
    "fill out a regression specification with Tab completion"
18. **Tool 3: Autonomous Agent (Claude Code)** — Strengths / Workflow.
    Cross-link to alternatives mentioned in frame 10. Econ example:
    "convert a MATLAB Aiyagari codebase to Python overnight"

### Frames 19–21 — Setting Up Claude Code (new content from Paul GP `getting-started`)
19. **Installing Claude Code** — npm
    (`npm install -g @anthropic-ai/claude-code`) vs standalone installer.
    Pro ($20) / Max ($100) / Max20x ($200) pricing tiers and which to pick
    for an empirical research workflow
20. **The recommended terminal stack** — Ghostty (GPU-accelerated terminal,
    `brew install ghostty`) + Zellij (multiplexer, `Ctrl+p d` to split
    panes, `Alt+arrow` to navigate) + Oh My Zsh (`git`/`z`/`virtualenv`
    plugins). Why each one: rendering speed, parallel views (regression
    output beside script), autocompletion
21. **First-session checklist + security warnings** — verification steps
    for your first project. Then the IRB/PII/API-key rule: "if you
    wouldn't put it on Dropbox, don't put it in front of Claude". Files
    stay local; code runs locally; conversation transcripts traverse the
    API. Critical for IRB/HIPAA-protected health-econ datasets

### Frames 22–25 — The Iterative Workflow in Practice
22. **The five-step loop: Design → Implement → Validate → Extend → Repeat**
    — source lines 419–474. Re-anchor to the 5 pillars on the side
23. **Why this workflow works** — separation of design from implementation
    catches conceptual errors early; one-feature-at-a-time gives clear
    attribution; analytical benchmarks catch fundamental errors. Source
    lines 476–500
24. **Effective prompting: vague vs specific** — side-by-side example.
    Vague: "analyze this data". Specific: "Load `employment.csv`, compute
    monthly growth rates by sector, plot in BLS style". From Paul GP
    article
25. **Validation: technical + economic checks** — preserve the 2-column
    layout from source lines 640–672 (RA praised this slide; keep it
    prominent). Add "trust but verify" note for econometric edge cases

### Frames 26–31 — Project Structure (introduced **before** the architecture, per RA3 feedback)
26. **The naive baseline: chatting without a harness** — what happens if
    you just open Claude Code in an empty folder and start chatting?
    Concrete failure modes: (1) you re-explain your project conventions
    every session; (2) Claude forgets the correction you gave yesterday;
    (3) every Aiyagari extension reverts to grid-VFI defaults instead of
    your EGM choice; (4) no audit trail of which decisions were made
    why. *The harness is what fixes all four*
27. **What a Claude Code project looks like** — the file-tree from source
    lines 801–831 (the `my-macro-project/` listing). Apply colored code
    listing style. Anchor: "every component you'll see for the next 10
    slides has a place in this tree"
28. **CLAUDE.md** — "README for AI": what it contains (project context,
    available skills/agents, conventions, current state). Snippet from
    source lines 1329–1354. Read at the start of every session
29. **MEMORY.md** — persistent `[LEARN:tag]` corrections, accumulated
    across sessions. Snippet from source lines 1356–1384. How an entry
    gets created (you correct Claude → Claude saves a learn entry).
    Aiyagari example: `[LEARN:aiyagari-grid] use 200+ points near
    borrowing constraint`
30. **The three layers: Skills, Agents, Rules — overview table** — from
    source lines 1238–1265. Location, loading, purpose, Python analogy
31. **English instructions vs encoded `SKILL.md`** (RA confusion point) —
    new slide. Two-column: left = "you type `Implement Aiyagari with
    income risk`"; right = "you type `/solve-model aiyagari`". When to
    use each: English for one-offs / exploration; slash command when
    you've encoded a workflow you want to reuse exactly

### Frames 32–38 — Skills, Agents, Sub-agents in Detail
32. **Why encode workflows at all? The naive baseline** — what happens
    if you just type the workflow steps in English every session?
    Concrete failures: (1) you forget step 4 the third time; (2) Claude
    interprets "validate" differently each time; (3) no version control
    on your methodology; (4) impossible to reproduce a colleague's
    approach. *Skills fix all four by writing the methodology to disk
    once*
33. **Skills = callable workflows** — Python-function analogy from source
    lines 837–881. Skill name + steps + arguments. Aiyagari example:
    `/solve-model aiyagari v1` runs the same 5-step methodology every
    time
34. **Skill file structure** — `.claude/skills/[name]/SKILL.md` + optional
    `references/` folder for detailed guides. From source lines 1140–1169
35. **Agents = role descriptions, not a dispatch system** — agent is a
    `.md` file in `.claude/agents/`; "spawning" means Claude reads it and
    either adopts the role or launches a subagent via the Task tool.
    From source lines 883–911. Three Aiyagari agents: `code-reviewer`
    (vectorization), `domain-reviewer` (Euler conditions), `verifier`
    (runs the script)
36. **Subagents = parallel child processes** — launched in their own
    context window; multiple can run in parallel like Python
    `multiprocessing`. Result returned to the main conversation.
    Aiyagari example: run `code-reviewer` and `domain-reviewer` in
    parallel after each version, halving review time
37. **Rules = always-on guardrails** — source lines 913–943. Loaded at
    session start, you never invoke them. Constrain *all* work.
    Aiyagari example: `iterative-workflow.md` enforces "never extend
    V0 to V1 until V0's three benchmarks pass"
38. **Hooks + Memory + Settings** — hooks fire on events like decorators
    (configured in `settings.json`); `CLAUDE.md` is bootstrap context;
    `MEMORY.md` is accumulated learnings. Three different persistence
    mechanisms with different lifetimes (per-event / per-session /
    cross-session)

### Frames 39–43 — Tracing Aiyagari End-to-End (running example, replaces RBC)
39. **Aiyagari model recap** — adapt source lines 581–634 (state $(a,z)$,
    prices, borrowing constraint $a' \geq \underline{a}$, three
    deterministic benchmarks based on $\beta(1+r)$, why we build it
    incrementally V0→V1→V2). One-line link to Lec06 for the full DEQN
    treatment
40. **Tracing `/solve-model aiyagari` (1/2): the file flow** — source
    lines 1267–1296. File-tree with reading-order numbers (1: CLAUDE.md,
    2: MEMORY.md, 3: rules/, 4: SKILL.md, 5: aiyagari-guide.md, 6:
    agents spawned later)
41. **Tracing `/solve-model aiyagari` (2/2): the orchestrator loop** —
    source lines 1298–1326. Implement → verify → review → fix → score
    cycle, repeated for V0, V1, V2
42. **Inside `CLAUDE.md` for the Aiyagari project** — full annotated
    snippet showing project description, available skills, available
    agents, model versions
43. **Inside `SKILL.md` and `aiyagari-guide.md`** — `SKILL.md` frontmatter
    + body from source lines 1386–1418, plus a reference-file snippet
    styled after source lines 1421–1446 but adapted to Aiyagari (sigma=2,
    beta=0.96, three benchmark cases)

### Frames 44–47 — Real Case Study (new content from `from-an-empty-folder-to-a-figure`)
44. **Case study: from empty folder to figure** — the question: "How has
    the age distribution of US homeowners changed over the last 50 years?"
    Setup: empty directory, no scripts. **Display the actual final
    homeownership chart** as the target output (downloaded from Substack
    into `figures/homeownership_by_age.png`)
45. **The exact prompts** — three verbatim prompts from the article:
    (1) initial data request with `download_data.py` deliverable;
    (2) figure creation referencing Kieran Healy style;
    (3) iteration request to flip axes
46. **Iteration loop + obstacles** — FRED dead-end → pivot to Census
    Tables 12 and 19; Census 403 error → user-agent header fix;
    sub-agents for web research to preserve context window. End-to-end
    timing: 6 min 21 s
47. **What Claude did vs what stayed human** — two columns. Claude:
    discovered Tables 12 + 19, parsed Excel by sampling, wrote
    `download_data.py`, generated 77-line R script, debugged Cairo font
    rendering. Human: posed the question, requested the axis flip,
    judged the story. Closing line: "Claude pulled the data and made
    the figure, but you brought the question"

### Frames 48–52 — How This Changes Research + Limitations (per feedback)
48. **Tedious tasks accelerated** — data cleaning, scraping, refactoring,
    web requests, variable transformation, format conversion. The new
    bottleneck is *thinking*, not *typing*
49. **Context window mechanics** — what a context window is, the ~200K
    token limit, the 20-turn degradation rule, automatic vs manual
    compaction
50. **The `/compact` strategy** — `/compact remember <focus areas>`,
    write progress to a `.md` file before compacting, start fresh
    sessions reading the file. Works as persistent state between sessions
51. **Where Claude Code still fails** (per feedback) — concrete examples
    students will hit in empirical economics:
    - long-context drift on multi-hour sessions
    - silently wrong econometric edge cases (clustered SEs,
      weak-instrument inference, panel SE corrections)
    - hallucinated function signatures from older library versions
      (e.g., `statsmodels` API changes)
    - infinite loops fighting failing tests
    - "happy-path" code that crashes on missing FRED data points
52. **Security: what NOT to expose** — IRB-protected data, PII, HIPAA-
    regulated datasets, API keys, anything you wouldn't put on Dropbox.
    The code runs locally but the conversation transcript traverses the
    API. Health-econ and labor-econ examples

### Frames 53–54 — Verification & The Architect Mindset
53. **Trust but verify** — checklist: re-derive one statistic by hand
    for one row; cross-check against a known package (e.g.,
    `statsmodels`, `linearmodels`); sanity-check magnitudes against
    priors; never accept code you cannot read line by line. Loop back
    to Pillar 5 (Code Literacy)
54. **You are the Research Architect** — closing thesis. AI handles
    bricks; you design the structure. The 5-pillar workflow is what
    makes you the architect rather than a passive consumer of code

### Frames 55–57 — Wrap-up
55. **Key takeaways** — four legs of the stool: 5-pillar foundations +
    harness engineering + iterative workflow + verification. The
    5-pillar visual appears one last time
56. **What changes for economic research** — shorter distance from idea
    to result; bigger payoff to specifying questions clearly; less
    reward for typing speed; more reward for precise problem formulation
57. **Looking ahead** — Lec10 will show four full case studies; the lab
    will have students replicate one end-to-end with their own
    `.claude/` project

### Frames 58–60 — Appendix
58. **Appendix A: Cursor / Claude Code keybindings reference** — moved
    here per feedback. Cursor: Tab (inline accept), `Cmd+L` (chat),
    `Cmd+I` (composer). Claude Code: `/compact`, `/clear`, `/cost`,
    `Esc` to interrupt, `Shift+Tab` for plan mode
59. **Appendix B: Setting up your first project** — minimal `CLAUDE.md`
    template (5 lines), minimal `SKILL.md` template, where to put files
60. **Appendix C: Resources & links** — Paul GP articles (both URLs),
    Anthropic Claude Code docs, Markus Karbacher's
    Claude-Code-Presentation, Claude Code GitHub repo, Substack archive

**Total: ~60 frames** (soft target — may grow if a concept needs splitting).
The slide-density feedback applies: never compress for count.

---

## Figures to produce/download

Create `2026_New_Slides/Lec09_Agentic_AI/figures/` and populate:

1. **`homeownership_by_age.png`** — the iconic Paul GP chart from
   `from-an-empty-folder-to-a-figure`. **Action:** during implementation,
   `WebFetch` the article URL with prompt "give me the URL of the embedded
   homeownership-by-age figure", then `curl -sSL <url> -o figures/homeownership_by_age.png`
   with a `User-Agent` header (Substack uses CDN paths like
   `substackcdn.com/image/fetch/...`). If download fails, fall back to a TikZ
   sketch placeholder + cite the article.
2. **`claude_code_terminal.png`** *(optional)* — pick one terminal screenshot
   from `Refinement_2026/readings/Claude-Code-Presentation-main/Presentations/figures/`
   that shows a real Claude Code session (avoid the meme/joke files). Copy
   into `figures/`. Use it on frame 14 or frame 28.
3. **TikZ diagrams** (no files needed — inline in `.tex`):
   - Frame 5: chatbot/RAG/agent spectrum (3 boxes + arrows)
   - Frame 8: ladder of AI coding maturity (5 stacked rungs)
   - Frame 22: design→implement→validate→extend loop (cyclic graph)
   - Frame 41: orchestrator loop diagram (implement → verify → review → fix → score)

---

## Reusable assets from the source

**Verbatim or near-verbatim reuse** (still well-written; light edits only):
- Preamble (lines 1–82) → preamble of new file
- Pillar slides (lines 144–282) → frames 11–15 (one pillar per frame, not
  compressed; add econ anchor lines)
- "From foundations to AI-assisted coding" division table (lines 286–302)
  → frame 15
- Iterative workflow steps (lines 419–474) → frame 22
- "Why this workflow works" (lines 476–500) → frame 23
- Validation 2-column slide (lines 640–672) → frame 25 (RA-praised; keep)
- Project file-tree listing (lines 801–831) → frame 27
- Skills function-analogy (lines 837–881) → frame 33
- Agent role-description explanation (lines 883–911) → frame 35
- Rules always-on guardrails (lines 913–943) → frame 37
- CLAUDE.md snippet (lines 1329–1354) → frames 28, 42
- MEMORY.md snippet (lines 1356–1384) → frame 29
- Three-layer comparison table (lines 1238–1265) → frame 30
- Skill file structure (lines 1140–1169) → frame 34
- Tracing file flow (lines 1267–1296) → frame 40
- Tracing orchestrator loop (lines 1298–1326) → frame 41
- SKILL.md frontmatter+body (lines 1386–1418) → frame 43
- Reference-file template (lines 1421–1446) → frame 43 (adapt to Aiyagari)
- Aiyagari model frames (lines 581–634) → frame 39

**Cut entirely:**
- "Two platforms for encoding workflows" (Claude.ai vs Claude Code) —
  feedback says "be brief about Claude.ai Projects"
- All RBC frames (lines 506–575) — replaced by Aiyagari for consistency
- Everything in section "Exercise: AI-Assisted Dynamic Equilibrium Modeling"
  (lines 1614–2068) — KS semester project belongs in Lec06's lab, not here

---

## New content to author from scratch

The following frames have no source-file equivalent. Use the gathered article
notes (Paul GP) and the slide-density memory's pedagogical principles:

- **Frame 4** (naive baseline: ChatGPT in a browser) — list four missing
  capabilities with FOMC rate-path example
- **Frames 5–7** (chatbot→RAG→agent spectrum, agentic features, comparison
  table) — anchor every example in economic-research workflow
- **Frames 8–9** (Ladder of AI coding maturity, Level 3 sweet spot) — Paul GP
  framing
- **Frame 10** (alternatives to Claude Code) — Gemini CLI / Codex CLI / Aider
- **Frames 19–21** (setup): npm/standalone, Pro/Max pricing,
  Ghostty + Zellij + Oh My Zsh stack with exact `brew install` commands,
  IRB/PII security warnings
- **Frame 24** (effective prompting): vague vs specific side-by-side
- **Frame 26** (naive baseline for harness): four failure modes of chatting
  without `CLAUDE.md`
- **Frame 31** (English vs SKILL.md): two-column "what you type" comparison
  with Aiyagari examples (resolves RA confusion)
- **Frame 32** (naive baseline for skills): four failure modes of typing
  workflow steps in English every session
- **Frames 36** (subagents): parallel `code-reviewer + domain-reviewer`
  Aiyagari example
- **Frame 38** (hooks/memory/settings): three persistence mechanisms
- **Frames 44–47** (case study): from `from-an-empty-folder-to-a-figure`
- **Frames 48–50** (research changes + context window): from `getting-started`
- **Frame 51** (failure modes): five concrete econometric failure categories
- **Frame 52** (security): IRB/PII/HIPAA warnings
- **Frames 53–54** (verification + architect mindset): closing thesis
- **Frames 55–57** (wrap-up)
- **Frames 58–60** (appendix)

---

## Verification

After implementation, verify by:

1. **Compile the deck** — run `pdflatex -interaction=nonstopmode Lec09_Agentic_AI.tex`
   twice (Beamer needs two passes for frame counts) from inside the lecture
   folder. Confirm zero `! Undefined` and zero `! Missing` errors. Final PDF
   must have ≈60 pages.
2. **Frame count check** — `grep -c '\\begin{frame}' Lec09_Agentic_AI.tex`
   should return roughly 60 (range 55–65 is acceptable; the slide-density
   memory allows growth as long as each frame carries one clean idea).
3. **Figure check** — `ls figures/` must contain at least
   `homeownership_by_age.png` (and any terminal screenshot used). Open the
   PDF and confirm both display at readable resolution.
4. **Feedback audit** — walk through `Feng_slides_feedback.txt` line by line
   and confirm each item is addressed:
   - [ ] Agenda slide present (frame 2)
   - [ ] 5-pillar visible early (frame 3) and re-referenced on section
         dividers
   - [ ] Tool comparison frames 16–18 use identical headers (Strengths /
         Workflow)
   - [ ] Cursor key-bindings moved to appendix A (frame 58)
   - [ ] Source's redundant Tool 2 / Tool 3 slides consolidated into 17–18
   - [ ] Project structure (frame 27) appears **before** Skills/Agents/Rules
         deep dive (frames 33–37)
   - [ ] Frame 31 explicitly addresses "English vs SKILL.md" confusion
   - [ ] Frame 51 lists current failure modes
   - [ ] All `lstlisting` blocks use `style=pythonstyle` (colored code)
5. **Naive-baseline audit** (per slide-density memory):
   - [ ] Frame 4 motivates against the chatbot-only baseline before
         introducing agents
   - [ ] Frame 26 motivates against "no harness" baseline before introducing
         the project structure
   - [ ] Frame 32 motivates against "type instructions every time" before
         introducing skills
6. **Economic anchoring audit** (per slide-density memory):
   - [ ] Each pillar frame (11–15) has an Aiyagari/empirical anchor
   - [ ] Each tool comparison frame (16–18) has an econ-research example
   - [ ] Skills/Agents/Rules detail frames (33–37) all reference Aiyagari
         workflow components
   - [ ] Failure-mode frame (51) lists *econometric* edge cases
         specifically (clustered SEs, weak instruments, panel SEs)
7. **Article integration check** — frames 8, 19–21, 24, 44–47, 49–50 all
   draw from the Paul GP articles. Verbatim prompts on frame 45
8. **Aiyagari consistency check** — frames 39–43 use Aiyagari (not RBC).
   `grep -c -i 'rbc' Lec09_Agentic_AI.tex` should return 0 (or only return
   lines that explicitly say "we use Aiyagari instead of RBC for
   consistency"). No standalone RBC walkthrough remains
