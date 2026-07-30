# Capstone Projects — *AI for Economic Research: Dynamic Models, Language, and Agents*

**Instructor:** Zhigang Feng · Summer 2026 (July intensive) · 24-hour graduate course

This document defines the **capstone** for the course. It is the place where the
four competencies you built across the ten lectures — theoretical mastery,
technical fluency, AI-augmented implementation, and critical validation — come
together on a real research problem.

There are **four tracks**. Each serves a different research taste; each connects
to material we covered and, in three cases, to research the instructor is
actively pursuing. You pick **one**.

> **The one rule that governs all four tracks: _replicate first, explore second._**
> Your baseline deliverable is to faithfully **reproduce and verify** a known
> result — a classical solution, a published number, or the instructor's
> reported finding — on public data or at small scale. Finding "something more
> interesting and insightful" is the **stretch goal**, and it only counts once
> the replication validates. This mirrors how real research works, and it is how
> Lecture 9–10 taught you to use AI: build the trusted benchmark, *then* extend.

---

## 1. How capstones embody the course

Every lecture pushed one idea: **you are the Research Architect.** AI handles the
bricks — syntax, boilerplate, data wrangling, refactoring — while you own the
theory, the validation, and the interpretation. The capstone is graded in that
spirit. We are not impressed by code an agent wrote; we are impressed by a
correct result you can defend, a benchmark you verified by hand, and a clear
account of who did what.

The **5-pillar workflow** from Lecture 9 is your scaffold on every track:

1. **Economic theory** — state the problem precisely (objective, constraints,
   equilibrium or estimand).
2. **Algorithm design** — choose the method (VFI vs. Euler iteration; WLS;
   embedding + retrieval; a typed graph).
3. **Numerical / empirical technique** — discretization, weighting, sampling,
   chunking.
4. **Advanced method & trade-offs** — accuracy vs. speed, bias vs. variance,
   retrieval precision vs. recall.
5. **Code literacy & validation** — read every line; benchmark; sanity-check
   against economics.

### Pick your track

| Track | One-line pitch | Best if you like… | Builds on (lectures) | Connects to instructor research |
|------|----------------|-------------------|----------------------|---------------------------------|
| **A — Optimal Growth via Deep Learning** | Solve the workhorse growth model with a neural net; verify it against the classical solution | Macro modeling, dynamic programming, PyTorch | Lec 1, 3, 4, 5 | (thematic) AI-as-capital extension |
| **B — Building an AI-Exposure Index** | Construct & validate a task-level AI exposure index from public data | Empirical work, data engineering, labor/macro | Lec 2, 7, 10 | **AI-index project** (WIP) |
| **C — Text as Economic Data** | Turn a text corpus into a validated quantitative measure with a small RAG | NLP, measurement, finance/policy text | Lec 7, 8 | (adjacent) text-as-data |
| **D — AI-for-Research Infrastructure** | Build a small typed-retrieval system or a validated Claude Code skill | Tooling, retrieval, methodology | Lec 8, 9, 10 | **GRAM / GraphRAG project** (WIP) |

Tracks **B, C, and D touch ongoing, unpublished research** — read §3 before
choosing one.

---

## 2. Shared guidelines — the "spirit of the class"

These apply to **every** track. The rubric (§4) scores them directly.

**AI is encouraged — as a partner, not a ghostwriter.** Use Claude Code, an IDE
agent, or a chat model across the whole pipeline: scoping, data acquisition,
coding, debugging, drafting. That is the point of the course. What is *not*
allowed is shipping anything you cannot read, explain, and defend.

**Keep an AI-collaboration log (`AI_LOG.md`).** Commit a short running log of the
substantive ways you used AI: the prompts that mattered, the decisions you made,
where the AI was wrong and how you caught it, and an honest split of human vs.
AI contribution. This is a graded artifact, not busywork — it is the evidence
that you were the architect. A trimmed transcript export is fine as an appendix.

**Validation discipline (non-negotiable).** For every headline result:
- Re-derive **at least one** number by hand or in a second tool.
- Benchmark against a **known** quantity — a classical solution, a published
  figure, or a standard package.
- Run **economic sanity checks** (e.g. `β ∈ (0,1)`; Euler-equation errors below
  threshold; an index's occupation ranking passes the smell test; signs match
  theory).
- **Never** submit code you have not read end to end.

**Security & ethics.** Public data only (see each track's data section). No PII,
IRB-, or HIPAA-protected data. No API keys committed to the repo (`.env` +
`.gitignore`). Cite every source. Distinguish your contribution from prior work
honestly.

**Reproducibility.** One command should reproduce your results. Provide:
git history, fixed random seeds, an `environment.yml`/`requirements.txt`, a
`CLAUDE.md` describing the project for the next agent (and human), and a
`README` with run instructions.

---

## 3. Confidentiality & work-in-progress (Tracks B, C, D)

Tracks **B**, **C**, and **D** build on the instructor's **active, unpublished
research** — the AI-index project and the GRAM / GraphRAG project. Choosing one
means agreeing to the following:

- **Treat anything the instructor shares as confidential** — specifications,
  preliminary results, intermediate data, or supervised access to the GRAM
  infrastructure. Do not redistribute, post publicly, or use outside this course.
- **Label your findings preliminary / work-in-progress.** These are open
  research questions; your results are a contribution to an ongoing effort, not
  a settled conclusion.
- **Clear any public release first.** A public GitHub repo, a poster, a blog
  post, or a submitted abstract that draws on this work must be approved by the
  instructor before it goes out.
- **You run on public data by default.** Everything you do independently uses the
  public sources listed in each track. The *only* contact with private
  infrastructure is the supervised, confidential GRAM comparison in Track D, and
  it is mediated by the instructor — you are never handed the private corpus or
  repo.

If you prefer to avoid confidentiality obligations entirely, choose **Track A**,
which is built purely on public/classical material.

---

## 4. Deliverables & grading

Capstones are done in **teams** (suggested 2–4). Each team produces a full
**reproducibility package**:

1. **Research paper** (~8–12 pp): question, method, replication result,
   validation, stretch finding (if any), limitations. Written like a short
   working paper.
2. **Code repository** with `CLAUDE.md`, `README`, pinned environment, fixed
   seeds, and one-command reproduction.
3. **`AI_LOG.md`** — the AI-collaboration log (§2).
4. **Presentation** (~12–15 min) to the class.
5. **Provenance / reproducibility package** — data-acquisition scripts (no
   manual downloads), a data dictionary, and a record of every external source
   with access date.
6. **Referee report on another team's project** — a structured, confidence-gated
   peer review in the style of the Lecture 10 `/review-paper` skill: cite exact
   passages, assert only what you can support, separate "must fix" from
   "suggestions," and stay constructive.

### Rubric (100 pts)

| Dimension | Pts | What we look for |
|-----------|-----|------------------|
| Economic framing & question | 15 | A precise, well-motivated question; correct estimand/equilibrium concept |
| Replication correctness | 20 | The baseline reproduces the benchmark/known result and is verified |
| Validation rigor | 20 | Hand-checks, benchmarks, sanity checks; honest about what failed |
| Reproducibility & provenance | 15 | One-command repro; clean data pipeline; pinned env |
| AI-collaboration quality | 10 | Architect, not consumer; informative `AI_LOG.md`; honest attribution |
| Communication | 10 | Clear paper & talk; figures that tell the story |
| Peer review quality | 10 | Specific, fair, well-evidenced referee report |
| **Stretch finding** | +10 (bonus) | A credible, validated extension beyond replication |

A project that *only* replicates — but does so correctly, with rigorous
validation and clean reproduction — earns a strong grade. The stretch is upside.

---

## 5. The Four Tracks

Each track is written to the same template: **Pitch · Who it's for · Connections
· Goal · Baseline (replicate & verify) · Stretch · Data & infrastructure ·
Suggested tools · AI-usage guidance · Validation checklist.**

---

### Track A — Deep-Learning Solution of the Optimal Growth Model

**Pitch.** Solve the single workhorse of dynamic macro — the optimal growth
model — with a neural network, and prove it is right by lining it up against the
classical value-function / Euler-iteration solution.

**Who it's for.** Students drawn to macro modeling, dynamic programming, and
PyTorch; people who want to *understand* a deep-learning solver rather than
treat it as a black box.

**Connections.** Lecture 1 (Bellman, VFI vs. Euler iteration), Lecture 3 (neural
nets, training), Lecture 4 (the optimal growth model is the running example;
Euler-equation-as-supervised-loss; actor–critic), Lecture 5 (RL framing). The
Lecture 4 dynamic-model labs (`Lab*_Dynamic_Models.ipynb`) are your starting
point.

**Goal.** A neural solution to the deterministic (then optionally stochastic)
optimal growth model whose **Euler-equation errors are at or below the classical
benchmark**, with a written account of the method and its accuracy.

**Baseline — replicate & verify.**

1. **Set up the model.** Standard problem: maximize `Σ βᵗ u(cₜ)` subject to
   `kₜ₊₁ = f(kₜ) − cₜ`, with CRRA utility and Cobb–Douglas production. State the
   Euler equation `u'(cₜ) = β f'(kₜ₊₁) u'(cₜ₊₁)`.
2. **Classical benchmark first.** Solve it the old way — value-function iteration
   and/or Euler-equation (time) iteration on a grid. This is your ground truth.
   (QuantEcon's optimal-growth / dynamic-programming lectures are a clean
   reference for the classical VFI and time-iteration solution.)
3. **Neural solution.** Parameterize the policy `c = πθ(k)` (and/or value `Vθ(k)`)
   with a small network. Train it two ways and compare:
   - **Euler-equation-as-supervised-loss**: minimize the mean squared Euler
     residual on sampled states (Lecture 4, T3).
   - **Actor–critic**: critic learns `V`, actor improves `π` (Lecture 4, T2).
4. **Verify.** Overlay the neural and classical policy functions; report
   Euler-equation errors across the state space; check steady state and the
   `β → ` deterministic limit. Document where (if anywhere) the network is less
   accurate and why.

**Stretch (modest, pick one).**
- Add a **stochastic TFP** shock (`zₜ` Markov) and re-validate — the model now
  has two states; show the neural solver scales gracefully where the grid starts
  to strain.
- Add a **second state** (e.g. a second capital good or a simple labor choice).
- An **AI-as-capital / automation** twist: introduce a productivity shift that
  raises the return to capital (a stylized "AI" technology) and trace its effect
  on the savings policy — a thematic bridge to the labor research in Track B.

**Data & infrastructure (all public).**
- Course Lecture 4 lab notebooks (`Lab*_Dynamic_Models.ipynb`).
- [QuantEcon — Cass–Koopmans optimal growth](https://python.quantecon.org/cass_koopmans_1.html)
  and the [QuantEcon lecture series](https://quantecon.org/lectures/) (see the
  optimal-growth / dynamic-programming lectures for the classical VFI and
  time-iteration benchmarks).
- Method references (read at least one): Maliar, Maliar & Winant (2021), *Deep
  learning for solving dynamic economic models* (JME); Fernández-Villaverde et
  al. on deep learning for macro.

**Suggested tools.** Python, PyTorch (`torch.nn`, `torch.optim`, autograd for
Euler residuals), NumPy/SciPy for the classical benchmark, Matplotlib. Claude
Code to scaffold the solver and the validation harness — then you read and
verify every line.

**AI-usage guidance.** Let the agent write the boilerplate (network, training
loop, plotting). *You* own: the Euler-residual loss derivation, the choice of
sampling distribution over states, and the accuracy comparison. A great
`AI_LOG.md` here shows the agent proposing a subtly wrong loss and you catching
it with the classical benchmark.

**Validation checklist.**
- [ ] Classical VFI/Euler benchmark solved and saved.
- [ ] Neural policy overlays the classical policy within tolerance.
- [ ] Euler-equation error curve reported across the state grid.
- [ ] Steady state and parameter limits behave correctly.
- [ ] Results stable across seeds and network sizes (report sensitivity).

---

### Track B — Building & Validating an AI-Exposure Index *(WIP · confidential — see §3)*

**Pitch.** Reconstruct, on fully public data, a **task-level index of AI exposure
across occupations**, and verify it reproduces the structure the instructor finds
in the ongoing *AI-index* project.

**Who it's for.** Empiricists and data engineers; students interested in AI &
labor markets, the economics of automation, and measurement.

**Connections.** Lecture 2 (AI as cognitive capital; measuring AI exposure via
task-level deployment, Anthropic Economic Index × O\*NET), Lecture 10 (Case 2
MEPS — the agentic empirical pipeline pattern: discover data → harmonize → weight
→ validate against an external benchmark). This track *is* a small, public-data
version of the instructor's AI-index work.

**Goal.** A reproducible pipeline that produces an occupation-level AI-exposure
index and a short validation showing it behaves sensibly and matches the
reported structure.

**Baseline — replicate & verify.**

1. **Acquire the public inputs** (scripted, not by hand):
   - **Anthropic Economic Index** task- and occupation-level usage
     (`job_exposure.csv`, `task_pct_*.csv`, `automation_vs_augmentation_*.csv`).
   - **O\*NET** task ratings and the task-to-occupation (SOC) structure.
2. **Build the index.** For each occupation, combine task-level Claude-usage
   shares with O\*NET task importance to compute:
   - a **deployment-concentration** measure — the Gini-style index `G` (signed)
     and its magnitude `|G|` of how concentrated AI usage is across an
     occupation's tasks, plus an **effective task fraction** (1 / Herfindahl);
   - an **importance-weighted usage intensity** for the occupation.
3. **Verify the index.** Rank occupations; check the extremes pass the smell test
   (which occupations are most vs. least exposed/concentrated?); confirm the
   qualitative structure the instructor reports (e.g. the usage-by-concentration
   pattern). Cross-check occupation employment counts against published BLS/Census
   numbers so you know your crosswalks are sound.

**Stretch (find something more interesting, pick ≥1).**
- **Link to employment.** Merge with **IPUMS-CPS** (or BLS OES) occupation
  employment and estimate whether the index predicts employment-share change
  (WLS, robust SEs). Then probe the open threads from the research: is the effect
  an interaction (usage × concentration)? Is it concentrated in some demographic
  groups? Does it appear in the household survey but not payroll?
- **Improve the index.** Add O\*NET **task-frequency** weighting; build a
  **time-varying** version across AEI releases; or test robustness to weighting
  and time-window choices.

**Data & infrastructure (all public).**
- [Anthropic Economic Index — Hugging Face dataset](https://huggingface.co/datasets/Anthropic/EconomicIndex)
  and the [report page](https://www.anthropic.com/economic-index).
- [O\*NET database](https://www.onetcenter.org/database.html) /
  [O\*NET OnLine](https://www.onetonline.org/).
- [IPUMS-CPS](https://cps.ipums.org/cps/) and
  [BLS OES](https://www.bls.gov/oes/).
- [FRED](https://fred.stlouisfed.org/) for macro context.
- Potential-exposure ratings: Eloundou, Manning, Mishkin & Rock (2023),
  [*GPTs are GPTs*](https://arxiv.org/abs/2303.10130).

**Suggested tools.** Python, pandas, statsmodels/linearmodels (WLS, robust SEs),
the crosswalk discipline from Lecture 10. Claude Code for the data-acquisition
and harmonization pipeline — exactly the MEPS-case workflow.

**AI-usage guidance.** This track lives or dies on **crosswalk and weighting
correctness**, which is precisely where agents make silent errors (Lecture 9–10).
Let the agent build the download + merge pipeline; *you* verify every join,
re-derive one occupation's index by hand, and validate totals against an external
source. Your `AI_LOG.md` should highlight a crosswalk or weighting bug you caught.

**Validation checklist.**
- [ ] Public inputs acquired by script with recorded access dates.
- [ ] Occupation employment totals match published BLS/Census within tolerance.
- [ ] One occupation's index value re-derived by hand.
- [ ] Index rankings pass an economic smell test.
- [ ] (Stretch) Regression uses weights + robust SEs; results survive a
      window/weighting robustness check.
- [ ] Findings labeled **preliminary / WIP**; no private specs or numbers used.

---

### Track C — Text as Economic Data: A Validated Measurement Instrument *(may touch WIP — see §3)*

**Pitch.** Turn a corpus of economic text into a **quantitative measure** — a
policy-stance index, a firm-level exposure score, a narrative index — and prove
it works by validating against an external benchmark. Build a **small RAG**
pipeline along the way.

**Who it's for.** Students interested in NLP, measurement, finance, and
monetary/policy text.

**Connections.** Lecture 7 (tokenization, embeddings, transformers, LLMs as
measurement instruments; FinBERT vs. CentralBankRoBERTa; the validation
checklist), Lecture 8 (RAG: chunking → embeddings → vector store → retrieval;
RAGAS evaluation). The Session-5 FOMC/LLM lab is your starting point.

**Goal.** A measurement instrument that reproduces a *known* text-based result
and is validated against an external benchmark — plus a small RAG pipeline over
the same corpus for question-answering / retrieval.

**Baseline — replicate & verify (pick one corpus & target).**

- **Central-bank policy stance** (hawkish/dovish) from FOMC statements, minutes,
  or speeches → validate against an existing classifier
  (CentralBankRoBERTa) and/or subsequent policy-rate moves.
- **Firm-level AI / risk exposure** from SEC 10-K Item 1A risk factors or
  earnings-call text → validate against a published exposure measure or market
  reaction.
- **A news-based narrative index** (e.g. an "AI narrative" index) → validate
  against an established index or a macro/market outcome.

Steps: (1) acquire and clean the corpus; (2) build the measure two ways —
a **lexical / TF-IDF baseline** and an **embedding-based** version — and a
**small RAG** pipeline (chunk → embed → store in FAISS/Chroma → retrieve →
grounded answer); (3) **validate**: embedding-geometry sanity checks, correlation
with the external benchmark, and an **out-of-sample / held-out-period** test;
(4) audit for bias and, for the RAG component, evaluate **faithfulness** with
RAGAS.

**Stretch.** A novel index, a multi-corpus comparison, or a head-to-head that
surfaces where the embedding measure beats (or loses to) the lexical baseline and
why.

**Data & infrastructure (all public).**
- FOMC materials: [Federal Reserve FOMC calendars & statements](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm).
- [SEC EDGAR full-text search](https://www.sec.gov/edgar/search/) for 10-Ks
  (the `efts.sec.gov` JSON API backs this UI if you want to script it).
- [BIS central bankers' speeches](https://www.bis.org/cbspeeches/).
- [FRED](https://fred.stlouisfed.org/) for outcomes.
- Models: [FinBERT (ProsusAI/finbert)](https://huggingface.co/ProsusAI/finbert),
  [CentralBankRoBERTa](https://huggingface.co/Moritz-Pfeifer/CentralBankRoBERTa-sentiment-classifier),
  [sentence-transformers](https://www.sbert.net/).
- RAG stack: [FAISS](https://github.com/facebookresearch/faiss),
  [Chroma](https://www.trychroma.com/), [RAGAS](https://docs.ragas.io/),
  the original RAG paper (Lewis et al. 2020,
  [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)).

**Suggested tools.** Python, Hugging Face `transformers`, `sentence-transformers`,
FAISS or Chroma, RAGAS, pandas. Claude Code for the ingestion + chunking +
retrieval pipeline.

**AI-usage guidance.** Embeddings and RAG make it easy to get a plausible-looking
number that means nothing. *You* own validation: does the embedding geometry pass
known analogies? Does the measure correlate with the benchmark out of sample? Is
the RAG answer faithful to the retrieved text (RAGAS), or hallucinated? Document a
case where the naive measure looked fine but failed validation.

**Validation checklist.**
- [ ] Lexical baseline and embedding measure both built.
- [ ] Small RAG pipeline returns grounded, source-attributed answers.
- [ ] Embedding-geometry / tokenization sanity checks pass.
- [ ] Correlation with an external benchmark reported, **out of sample**.
- [ ] RAGAS faithfulness reported for the RAG component.
- [ ] Bias audit done; sources cited; access dates recorded.

---

### Track D — AI-for-Research Infrastructure: Typed Retrieval or a Validated Skill *(WIP · confidential — see §3)*

**Pitch.** Build a piece of *research infrastructure*: either a small
**condition-aware GraphRAG** over an economics literature, or a **reusable,
validated Claude Code skill** for a recurring research task.

**Who it's for.** Students who like tooling, retrieval systems, evaluation
methodology, and the meta-question of *how AI should be wired into research*.

**Connections.** Lecture 8 (RAG, GraphRAG, the GRAM case study — "wrong-by-
prominence" vs. condition-aware retrieval), Lecture 9 (project harness; skills,
agents, rules; validation gates), Lecture 10 (Case 1 paper-review skill —
multi-phase pipeline, confidence gates, quote verification). This track engages
the instructor's **GRAM / GraphRAG** research.

**Goal.** A small but genuinely working artifact with a measured comparison
against a sensible baseline.

**Choose one flavor.**

**D1 — Mini-GRAM (typed retrieval).**
1. Assemble a **public corpus of ~15–30 papers** in one macro subfield (e.g.
   Aiyagari/Bewley incomplete-markets, or RBC) from arXiv/NBER/RePEc.
2. Build two retrieval systems over it: a **flat/dense RAG** (chunk → embed →
   vector store) and a **typed, condition-aware GraphRAG** — design a *small*
   schema (a handful of element types and edge types capturing model primitives,
   equilibrium concept, solution method, and their dependencies) and extract
   structured instances per paper.
3. **Ablation.** On a set of held-out modeling / code-generation questions (e.g.
   "which solution method is appropriate given these model conditions, and why?"),
   compare flat RAG vs. typed GraphRAG. Score relevance and condition-awareness.
4. **Supervised, confidential GRAM touchpoint.** With the instructor mediating,
   run your held-out questions **together with the instructor's existing GRAM
   infrastructure** as an additional comparison/ablation arm. You do **not**
   receive the private corpus or repo — access is supervised and confidential
   (§3). Report how your mini-system compares and what the typed structure buys.

**D2 — A validated Claude Code skill.**
1. Pick a recurring research task: e.g. `/replicate-table` (re-run a paper's
   headline table from public data), `/solve-model` (the 5-step model→equilibrium
   →algorithm→pseudocode→code workflow), or `/referee` (a paper-review skill).
2. Author it as a **multi-phase skill** in the Lecture 10 style: explicit phases,
   **confidence gates** ("assert only what you can derive"), **quote
   verification**, and a **validation harness** (regression tests that a known
   input yields a known output).
3. **Evaluate.** Compare skill output to a human-done version of the same task on
   2–3 cases; report where it helps, where it fails, and how the validation gates
   caught failures.

**Data & infrastructure (all public).**
- Corpus: arXiv / [NBER](https://www.nber.org/) / RePEc PDFs.
- Extraction: [MinerU](https://github.com/opendatalab/MinerU) or `pypdf`.
- Retrieval: open embedding models (`sentence-transformers`, `bge`),
  [FAISS](https://github.com/facebookresearch/faiss) /
  [Chroma](https://www.trychroma.com/),
  [Microsoft GraphRAG](https://github.com/microsoft/graphrag),
  [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) for PPR-style retrieval.
- [Claude Code](https://www.claude.com/claude-code) and the project-harness
  patterns from Lecture 9 (`CLAUDE.md`, skills, agents, rules).

**Suggested tools.** Python, an embedding model, a vector DB, a graph library;
Claude Code for both building and (in D2) being the thing you build.

**AI-usage guidance.** This is the most "meta" track — you are engineering how AI
does research. The discipline from Lecture 9–10 *is* the deliverable: confidence
gates, quote verification, validation harnesses, honest evaluation. Your
`AI_LOG.md` should read like an engineering log of the harness itself.

**Validation checklist.**
- [ ] (D1) Both retrieval systems run on the same corpus; ablation is apples-to-
      apples.
- [ ] (D1) Retrieval precision / condition-awareness scored on held-out questions.
- [ ] (D1) Supervised GRAM comparison run with the instructor; results reported
      without exposing private material.
- [ ] (D2) Skill has explicit phases, confidence gates, and quote verification.
- [ ] (D2) Validation harness passes on known inputs; human-vs-skill comparison
      reported.
- [ ] Findings labeled **preliminary / WIP**; confidentiality respected (§3).

---

## 6. Logistics

**Teams.** 2–4 students. Mixed skill sets encouraged (one strong on theory, one
on data/code).

**Suggested milestones.**

| Milestone | Deliverable |
|-----------|-------------|
| **M1 — Proposal** | One page: track, question, the *specific* result you will replicate, data sources, division of labor. |
| **M2 — Data / infra up** | Acquisition pipeline runs end-to-end; benchmark/ground-truth in hand; `CLAUDE.md` + repo scaffolded. |
| **M3 — Replication verified** | Baseline result reproduced **and validated** (the core grade). Stretch scoped. |
| **M4 — Paper, repo, talk, peer review** | Full reproducibility package submitted; referee report on another team delivered. |

**How to scope with AI (starter prompt).** Use your agent to *plan*, then you
decide. For example:

```text
I'm doing Capstone Track <A/B/C/D>. The result I want to replicate first is
<X>. Help me: (1) restate it precisely (estimand or equilibrium concept);
(2) list the exact public data/inputs and how to fetch them by script;
(3) propose a minimal pipeline; (4) propose how I will VALIDATE the
replication against a known benchmark before I attempt any extension.
Do not write the full solution yet — I want the plan, and I will verify it.
```

**Picking a track.** A → macro modeling, no confidentiality. B → empirical AI &
labor (WIP). C → NLP / measurement. D → research tooling / GraphRAG (WIP).
Tracks B/C/D require agreeing to §3. When unsure, talk to the instructor at the
M1 checkpoint.

**Office hours / checkpoints.** Bring your `AI_LOG.md` and your validation
results to each checkpoint — that is what we will discuss.

---

## Appendix — consolidated resources

**Course foundations**
- Course outline: `2026_lecture_outline.md`; syllabus: `syllabus_2026.html`.
- Lecture 4 dynamic-model labs; Session-5 FOMC/LLM lab; Lecture 10 MEPS lab.
- Textbook: [*机器学习与数量宏观经济学*](https://book.douban.com/subject/37885381/).
- Classical methods video series:
  [Bilibili 2026](https://space.bilibili.com/2142649036/lists/7180709).
- Agentic-AI guides: Paul Goldsmith-Pinkham,
  [Getting Started with Claude Code](https://paulgp.substack.com/p/getting-started-with-claude-code),
  [From an Empty Folder to a Figure](https://paulgp.substack.com/p/from-an-empty-folder-to-a-figure).

**Data**
- [Anthropic Economic Index (HF)](https://huggingface.co/datasets/Anthropic/EconomicIndex) ·
  [report](https://www.anthropic.com/economic-index)
- [O\*NET](https://www.onetcenter.org/database.html) ·
  [IPUMS-CPS](https://cps.ipums.org/cps/) ·
  [BLS OES](https://www.bls.gov/oes/) ·
  [FRED](https://fred.stlouisfed.org/)
- [FOMC](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) ·
  [SEC EDGAR](https://www.sec.gov/edgar/search/) ·
  [BIS speeches](https://www.bis.org/cbspeeches/)

**Models, tools & methods**
- [QuantEcon lectures](https://quantecon.org/lectures/) ·
  [Cass–Koopmans optimal growth](https://python.quantecon.org/cass_koopmans_1.html)
- [FinBERT](https://huggingface.co/ProsusAI/finbert) ·
  [CentralBankRoBERTa](https://huggingface.co/Moritz-Pfeifer/CentralBankRoBERTa-sentiment-classifier) ·
  [sentence-transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss) ·
  [Chroma](https://www.trychroma.com/) ·
  [RAGAS](https://docs.ragas.io/) ·
  [Microsoft GraphRAG](https://github.com/microsoft/graphrag) ·
  [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) ·
  [MinerU](https://github.com/opendatalab/MinerU)
- [Claude Code](https://www.claude.com/claude-code)

**Key papers**
- Eloundou, Manning, Mishkin & Rock (2023), *GPTs are GPTs* —
  [arXiv:2303.10130](https://arxiv.org/abs/2303.10130).
- Lewis et al. (2020), *Retrieval-Augmented Generation* —
  [arXiv:2005.11401](https://arxiv.org/abs/2005.11401).
- Maliar, Maliar & Winant (2021), *Deep learning for solving dynamic economic
  models*, JME.
