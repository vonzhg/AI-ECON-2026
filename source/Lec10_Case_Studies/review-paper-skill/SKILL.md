---
name: review-paper
description: >
  Review an academic paper using the coarse multi-stage peer review pipeline.
  Produces a refine.ink-format markdown report with macro-level issues and
  detailed comments with verbatim quotes. Use when the user wants to review
  a paper, get feedback on a manuscript, or generate a referee report.
  Triggers on: "review this paper", "peer review", "referee report",
  "/review-paper path/to/paper.pdf".
---

# Review Paper — Full Pipeline

Review an academic paper using the coarse review pipeline. This replicates the multi-stage peer review process: extraction, calibration, overview, section-level review, editorial filtering, and synthesis into a refine.ink-format markdown report.

## File Layout

This skill's reference files live in the `references/` subdirectory relative to this SKILL.md. When agents need to read a prompt file, construct the path as:
```
<skill_directory>/references/<filename>.md
```
where `<skill_directory>` is the directory containing this SKILL.md. Use the Glob tool to locate it if needed:
```
Glob("**/claude-skills/review-paper/SKILL.md")
```

## Arguments

`$ARGUMENTS` is the path to a paper file (PDF, TXT, MD, TeX, DOCX, HTML, EPUB).

Examples:
- `/review-paper data/papers/my-paper.pdf`
- `/review-paper ~/Downloads/draft.pdf`

Optional flags in `$ARGUMENTS`:
- `--cheap` — produce a shorter, faster review (fewer section agents, skip literature search)
- `--no-literature` — skip the literature search step
- `--output path/to/output.md` — custom output path (default: `{paper_stem}_review.md`)

## Pipeline Overview

```
Phase 1: Extract & Structure     — Read paper, parse sections, classify types
Phase 2: Calibrate (parallel)    — Domain calibration + literature + contribution extraction
Phase 3: Overview                — 4-8 macro issues + completeness + assumption check
Phase 4: Section Reviews (parallel) — 1-5 comments per section, specialized prompts
Phase 5: Cross-Section Synthesis — Discussion claims vs formal results
Phase 6: Editorial Filter        — Dedup, quality gate, quote verify, severity, humanize
Phase 7: Synthesis               — Render final markdown in refine.ink format
```

## Execution

### Phase 1: Extract & Structure

Read the paper at `$ARGUMENTS` using the Read tool. Claude reads PDFs natively via vision — this replaces the Mistral OCR + extraction QA stages from the Python pipeline. If the paper is a text format (MD, TXT, TeX), read it directly.

**Extraction quality**: When reading a PDF, pay attention to:
- Garbled math/LaTeX (equations with wrong symbols or missing terms)
- Missing content (text visible in the PDF but absent from extraction)
- Table layout errors (columns merged incorrectly)
If you notice extraction issues, note them but do NOT treat them as author errors in the review.

Produce a structured analysis containing:

1. **Paper metadata**:
   - Title (exact, from first page)
   - Domain (e.g., "social_sciences/economics", "computer_science/machine_learning")
   - Taxonomy (e.g., "academic/research_paper", "academic/working_paper")
   - Abstract (full text)

2. **Section list** — for each section:
   - Section number (e.g., 1, 2.1, A)
   - Title
   - Type: one of `abstract`, `introduction`, `related_work`, `methodology`, `results`, `discussion`, `conclusion`, `appendix`, `references`, `other`
   - `math_content`: true if section contains proofs, derivations, formal definitions, theorem statements with arguments, algebraic manipulations supporting claims
   - Key claims (theorems, lemmas, propositions — extracted verbatim)
   - Formal definitions introduced

3. **Classification rules for section types** (use heading keywords):
   - "abstract" -> abstract
   - "introduction", "overview" -> introduction
   - "related", "literature", "prior", "background" -> related_work
   - "method", "framework", "model", "approach", "design", "setup", "identification", "estimation" -> methodology
   - "result", "experiment", "empirical", "finding", "simulation", "application", "monte carlo" -> results
   - "discussion", "implication" -> discussion
   - "conclusion", "summary", "closing" -> conclusion
   - "appendix" -> appendix
   - "reference", "bibliography" -> references

Save the full structured analysis. You will need this for all subsequent phases.

**Important**: The section TEXT must be the verbatim content from the paper — not summarized. You need the full text for quote verification later.

---

### Phase 2: Calibrate (3 Parallel Agents)

Launch 3 agents simultaneously using the Agent tool. Each agent should read its reference file from this skill's `references/` directory.

**Agent 1: Domain Calibration**
- Read `references/calibration-prompt.md` for instructions
- Produce a domain-specific review calibration for this paper
- Input: title, domain, abstract, section titles list

**Agent 2: Literature Search**
- Read `references/literature-prompt.md` for instructions
- Use WebSearch to find related papers
- Input: title, first 1500 chars of abstract
- Skip if `--no-literature` or `--cheap` flag is present

**Agent 3: Contribution Extraction**
- Read `references/contribution-prompt.md` for instructions
- Extract the paper's stated contributions (reading comprehension, not evaluation)
- Input: title, abstract, intro section text (max 8000 chars), conclusion text (max 3000 chars)

Collect all 3 results before proceeding.

---

### Phase 3: Overview

Read these reference files:
- `references/shared-directives.md` — tone, confidence gates, quote rules
- `references/overview-prompt.md` — overview generation instructions

Produce the overview feedback using the full paper content, domain calibration, literature context, and contribution context:

1. **Overview Issues** (4-8): Each with a title and body paragraph (4-8 sentences). Reference specific sections, equations, theorems. End each with a concrete remediation suggestion.

2. **Completeness Assessment** (0-4 additional issues): Structural gaps — missing examples, simulations, implications, inference discussion. Merge into the overview issues (max 12 total). Deduplicate by checking for word overlap between titles.

3. **Assumption Consistency Check** (0-3 additional issues): Formal assumptions vs actual data structure. Merge into overview.

4. **Recommendation**: accept / minor revision / major revision / reject, with justification.

5. **Revision Targets**: 2-5 specific items if not "accept".

6. **Assessment**: 2-3 sentence assessment of the paper's contribution and what it does well.

---

### Phase 4: Section Reviews (Parallel Agents)

Identify reviewable sections: exclude "references" sections and appendix sections shorter than 500 characters. Cap at 25 sections.

For each reviewable section, determine its review focus:
- `math_content` is true -> "proof"
- Type is "methodology" -> "methodology"
- Type is "results" -> "results"
- Type is "related_work" -> "literature"
- Type is "discussion" or "conclusion" -> "discussion"
- Otherwise -> "general"

Launch parallel agents — one per section (or batch 2-3 sections per agent to stay within agent limits). Each agent should read:
- `references/shared-directives.md`
- `references/section-prompts.md`

Then review the section using the appropriate prompt variant (proof, methodology, results, literature, discussion, or general), producing 1-5 comments each with: title, verbatim quote, feedback, severity, confidence.

Include in each agent's context:
- Paper title and abstract
- Overview issues (for context only — do NOT restate)
- Domain calibration (if available)
- Literature context (only for intro/related_work sections)
- Cross-section claims & definitions (from Phase 1, cap at 60 items)
- Introduction/conclusion leniency note (for those section types)

**Proof verification chaining**: For sections with focus "proof" that produced comments, run a second agent pass:
- Read `references/proof-verify-prompt.md`
- Re-derive each claimed error independently; keep if valid (confidence "high"), drop if not
- Find 0-3 NEW issues the first pass missed
- Return COMPLETE merged list

Collect ALL section comments.

---

### Phase 5: Cross-Section Synthesis

Check if the paper has BOTH:
- Results/methodology sections with math_content = true
- Discussion/conclusion sections

If yes, read `references/cross-section-prompt.md` and check whether discussion claims are supported by formal results. Produce 0-3 additional comments. Add to the section comments list.

---

### Phase 6: Editorial Filter

Read `references/editorial-prompt.md`.

Apply the full editorial filter to ALL collected detailed comments. You have:
- The full paper text
- The overview issues (for deduplication)
- The paper's stated contributions (for contradiction checking)
- All draft detailed comments

Follow ALL 7 steps from the editorial prompt:
1. Remove low-value comments
2. Contradiction check
3. Verify against full paper text (if a comment claims something is "never defined", search the paper)
4. Quality and severity assignment
5. Notation capping (max 2-3 notation-level comments)
6. Humanize language
7. Order by importance (critical -> major -> minor, high confidence first)

Renumber surviving comments from 1.

**Fallback**: If the editorial filter produces unusable results (drops all comments, incoherent output), fall back to a two-step process:
1. **Crossref pass**: Deduplicate near-identical comments, remove low-value ones, renumber
2. **Critique pass**: Evaluate each surviving comment for specificity, accuracy, actionability

**Quote Verification**: For each surviving comment, verify the quote against the paper text:
- **Exact match**: Search for the quote as a substring (case-insensitive). Keep as-is.
- **Fuzzy match**: If not found exactly, search for the passage with highest word overlap. Accept if similarity > 80% (> 92% for math-heavy quotes with LaTeX, equations, or >30% digits/operators).
- **Correction**: Replace paraphrased quotes with the verbatim passage from the paper.
- **Drop**: If no match above threshold, drop the comment entirely.
- **Safety fallback**: If quote verification would drop ALL comments, skip verification and keep originals.

---

### Phase 7: Synthesis

Read `references/output-format.md` for the exact output format.

Render the final review markdown and write it to a file:
- Default path: `{paper_stem}_review.md` in the same directory as the paper
- Or the path specified by `--output` flag

The output must contain:
- Header block with title, date (today in MM/DD/YYYY format), domain, taxonomy
- Overall Feedback section with all overview issues
- Recommendation + revision targets
- Detailed Comments section with all surviving editorial-filtered comments

Report to the user:
- Output file path
- Number of overview issues
- Number of detailed comments
- Recommendation

---

## Parallelization Strategy

Use the Agent tool to run independent work in parallel:

- Phase 2: All 3 calibration agents run simultaneously
- Phase 4: Section review agents run simultaneously (batch into groups of 5-8)
- Phase 4 proof verification: Runs after first-pass section review for proof sections only

Sequential dependencies:
- Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 -> Phase 5+6 -> Phase 7

---

## Quality Targets

The goal is to produce a review matching the quality and format of the reference example in `references/reference-example/`:
- 4-8 substantive overview issues (not generic)
- 8-25 detailed comments with verbatim quotes
- Each comment identifies a concrete issue with a specific fix
- Natural, human-sounding language (not AI-generated feel)
- Every quote is a verbatim substring of the paper
