# Plan: Lecture 10 — Case Studies in Agentic AI for Quantitative Macro

## Context

Lecture 10 is the capstone of a 10-lecture graduate AI/ML for Economists course. It demonstrates two complete research workflows using agentic AI, focused on quantitative macroeconomics. The lecture is 2 hours + 1 hour lab. The Lec10_Case_Studies/ folder exists but is empty.

The user specified:
- Case studies should focus on **quantitative macro**
- **Case Study 1**: Build a Claude Code skill to review research articles — using the existing `review-paper` skill from `coarse-dev/claude-skills/` as the worked example, with `MPE_Hyper_v44.tex` (Feng & Santos, Markov Equilibria with Quasi-Hyperbolic Discounting) as the demo paper
- **Case Study 2**: Large-dataset analysis with MEPS data (following PaulGP's HMDA article pattern) — generate plots on US uninsurance rates 2000–2025
- Slide flow follows PaulGP article arc: Problem → Infrastructure → Architecture → Analysis → Insights

---

## Deliverables

| # | File | Description |
|---|------|-------------|
| 1 | `Lec10_Case_Studies/Lec10_Case_Studies.tex` | Beamer slide deck (~65–75 frames) |
| 2 | `Lec10_Case_Studies/Lab10_MEPS_Uninsurance.ipynb` | Jupyter notebook for MEPS lab |
| 3 | `Lec10_Case_Studies/review-paper-skill/` | Copy of the skill (verbatim) |
| 4 | `Lec10_Case_Studies/figures/` | Generated MEPS plots (6 figures) |
| 5 | `Lec10_Case_Studies/Lec10_Plan.md` | This plan, saved in lecture folder |

---

## Files to Read / Reuse

- **Preamble template**: `Lec09_Agentic_AI/Lec09_Agentic_AI.tex` lines 1–103 (copy verbatim, change only `\title`)
- **Skill source**: `/Users/zfeng/Library/Mobile Documents/com~apple~CloudDocs/iCloud-Git-Projects/Research/coarse-dev/claude-skills/review-paper/` (entire directory)
- **Demo paper**: `/Users/zfeng/Library/Mobile Documents/com~apple~CloudDocs/iCloud-Git-Projects/Research/Feng-Santos/Hyperbolic-RBC/draft/MPE_Hyper_v44.tex`
- **PaulGP article flow** (already read via WebFetch): Problem → Download → Format conversion → Schema harmonization → Database assembly → Aggregation → Visualization → Insights

---

## Slide Outline

### Front Matter (Frames 1–3)
1. Title slide
2. Lecture Agenda (two-column, matching Lec09 format)
3. Where We Are — course arc Lec01→10, the 5-pillar workflow recap

### `\section{Why Case Studies?}` (Frames 4–7)
4. The gap between tools and research — knowing skills ≠ using them
5. The PaulGP framework: Problem → Infrastructure → Architecture → Capability → Insights (TikZ flow)
6. What Lec09 concepts recur today (table: skills, agents, shared directives, CLAUDE.md)
7. Two case studies, one framework — side-by-side comparison table

### `\section{Case Study 1: The Problem}` (Frames 8–12)
8. **Naive baseline**: paste paper into ChatGPT → generic praise, no structure, no quotes
9. What a good referee report looks like — the refine.ink format (show reference example header)
10. The demo paper: Feng & Santos, "Markov Equilibria with Quasi-Hyperbolic Discounting" — what it's about
11. What we want the skill to produce (overview issues + detailed comments + recommendation)
12. The design challenge: decompose into agent-sized phases

### `\section{Case Study 1: Skill Architecture}` (Frames 13–20)
13. File layout: TikZ tree of `review-paper/` directory (SKILL.md + references/)
14. YAML frontmatter: name, description, triggers, arguments
15. 7-phase pipeline overview: TikZ flow diagram showing all phases
16. Why 7 phases, not 1? Context window, specialization, quality control
17. Sequential dependencies: Phase 1→2→3→4→5→6→7 (parallelism within phases)
18. The orchestrator pattern: show Phase 2 excerpt from SKILL.md (3 parallel agents)
19. Reference files = specialized prompts: table mapping file → phase → role
20. Why reference files live outside SKILL.md (on-demand loading, context efficiency)

### `\section{Case Study 1: Prompt Engineering}` (Frames 21–30)
21. Shared directives: tone, banned vocabulary, confidence gate, steelman protocol
22. Why humanization matters: AI-sounding vs. human-sounding side-by-side
23. The confidence gate: only assert errors you can derive; the contribution inversion test
24. Section routing: 6 specialized prompts (proof, methodology, results, literature, discussion, general)
25. Proof verification chain: two-pass adversarial architecture
26. The MPE paper through Phase 1: section extraction and classification
27. The MPE paper through Phases 2–3: calibration, literature search, overview
28. The MPE paper through Phase 4: section reviews with specialized prompts
29. The editorial filter: 7 steps (remove generic, contradiction check, verify quotes, severity, notation cap, humanize, reorder)
30. Quote verification: exact match → keep, fuzzy >80% → accept, no match → drop

### `\section{Case Study 1: Running and Output}` (Frames 31–36)
31. Installation: copy folder to `.claude/commands/`
32. Invocation: `/review-paper MPE_Hyper_v44.pdf` + flags
33. Under the hood: timeline (~8–10 min for 7 phases)
34. The output file: show refine.ink header + example overview issue + example detailed comment
35. Quality assessment: compare to reference example (R3D review)
36. What you do with it: verify math claims, check domain judgment, add perspective → Pillar 5

### `\section{Case Study 2: The Problem}` (Frames 37–41)
37. **Naive baseline**: download 24 MEPS CSVs by hand, figure out changing variable names, merge in Stata
38. The research question: Who is uninsured in the US, 2000–2023? Why it matters for macro (incomplete markets, ACA welfare analysis)
39. What PaulGP did with HMDA (brief parallel — the model to follow)
40. Our workflow: side-by-side HMDA↔MEPS mapping table
41. Three-phase architecture: Data Acquisition → Analysis → Visualization

### `\section{Case Study 2: Data Architecture}` (Frames 42–48)
42. The Claude Code prompt for Phase A (download + harmonize)
43. What Claude does autonomously: discovers AHRQ URLs, handles format changes, builds variable crosswalk
44. Schema harmonization: the hard part (INSCOV01 vs INSCOV22, pre/post-ACA coding)
45. The merged dataset: ~500K–1M rows, standardized columns
46. The prompt for Phase B (weighted statistics by year × group)
47. Weighted statistics: PERWT numerator/denominator, why weighting matters
48. Validation: cross-check vs Census Bureau known rates (2013 ≈ 13.4%, 2019 ≈ 8–9%)

### `\section{Case Study 2: Visualization and Insights}` (Frames 49–56)
49. The prompt for Phase C (6 publication-quality figures)
50. **Figure 1**: Overall uninsurance rate 2000–2023 (annotate ACA 2010/2014, COVID 2020)
51. **Figure 2**: By age group — 18-25 spike, ACA dependent-coverage provision effect
52. **Figure 3**: By poverty category — poor/near-poor largest ACA gains
53. **Figure 4**: By race/ethnicity — persistent disparities, post-2014 convergence
54. **Figure 5**: By employment status — employer-coverage gap
55. **Figure 6**: Regional heatmap — Medicaid expansion vs non-expansion states
56. What Claude did vs what stayed human (two-column, matching Lec09 pattern)

### `\section{Lessons and Patterns}` (Frames 57–62)
57. Common patterns across both case studies (decomposition, validation, domain knowledge encoding)
58. When to build a skill vs use ad hoc prompts (frequency, complexity, quality control)
59. Prompt engineering principles extracted from the skill (5 principles from shared-directives.md)
60. The division of labor table (You provide / AI provides)
61. Failure modes to watch for (both case studies)
62. Looking back at the course — the full Lec01→10 arc

### `\section{Lab: MEPS Data Analysis}` (Frames 63–70)
63. Lab overview (1 hr, replicate Case Study 2)
64. Lab setup: create project, write CLAUDE.md
65. Step 1: data acquisition prompt
66. Step 2: analysis prompt
67. Step 3: visualization prompt
68. Step 4: validation against Census Bureau stats
69. Checkpoints: expected state at 15/30/45/60 min
70. Common issues and fixes table

### Appendix (Frames 71–73)
71. Full MEPS variable crosswalk
72. Complete skill file tree with line counts
73. Resources and references

**Total: ~73 frames** (65 lecture + 8 lab), consistent with the density principle.

---

## Skill Files: Copy Strategy

Copy the entire `review-paper/` directory **verbatim** into `Lec10_Case_Studies/review-paper-skill/`. The user said "make it simpler if necessary" — but the skill is already modular and well-organized. Simplifying it would undermine the teaching point that real skills require this level of detail. Students benefit from seeing the actual artifact.

---

## MEPS Data Analysis: Technical Plan

**Data source**: AHRQ MEPS Full Year Consolidated files (HC series), 2000–2023
**Key variables**: INSCOV/INSCOVY (insurance coverage 1=Private, 2=Public, 3=Uninsured), AGELAST, RACETHX, POVCAT, REGION, EMPST, SEX, PERWT/PERWTF
**Schema challenge**: variable names are year-suffixed (INSCOV01, INSCOV22, PERWT01F, PERWT22F) — need a crosswalk dictionary

**6 figures to generate**:
1. Overall uninsurance rate time series (with ACA/COVID annotations)
2. By age group (18-25, 26-34, 35-44, 45-54, 55-64, 65+)
3. By poverty category (Poor, Near Poor, Low Income, Middle, High)
4. By race/ethnicity
5. By employment status
6. Regional heatmap (year × Census region)

**Python stack**: pandas, pyreadstat (for .ssp files), matplotlib, numpy

**Validation**: compare overall rates to Census Bureau CPS ASEC published statistics

---

## Implementation Sequence

1. Create directory structure: `figures/`, `review-paper-skill/`
2. Copy review-paper skill from iCloud path
3. Copy MPE_Hyper_v44.tex/.pdf to Lec10 folder
4. Write `Lec10_Case_Studies.tex` (preamble from Lec09, content per outline above)
5. Create `Lab10_MEPS_Uninsurance.ipynb` (notebook for MEPS analysis)
6. Run the notebook to generate MEPS figures → save to `figures/`
7. Compile .tex file and verify
8. Save `Lec10_Plan.md` in the lecture folder

---

## Verification

- [ ] .tex compiles cleanly with pdflatex
- [ ] Slide count ≥ 65 frames (excluding appendix)
- [ ] Title → Agenda → content structure matches Lec01–09 pattern
- [ ] `\section{}` commands produce Metropolis section dividers
- [ ] Colored code listings used for all code examples
- [ ] 6 MEPS figures generated and included
- [ ] MEPS uninsurance rates cross-checked against Census Bureau stats
- [ ] Skill files copied to `review-paper-skill/`
- [ ] Lab notebook runs end-to-end
- [ ] Plan saved as `Lec10_Plan.md` in lecture folder
