---
name: review-paper-readme
description: Documentation for the review-paper skill
---

# review-paper

Claude Code skill that replicates the coarse review pipeline for academic paper peer review.

## Usage

```
/review-paper path/to/paper.pdf
```

Optional flags:
- `--cheap` — shorter, faster review (fewer sections, skip literature search)
- `--no-literature` — skip web search for related papers
- `--output path/to/review.md` — custom output path

## Installation

Copy the `review-paper/` directory into your project's `.claude/commands/` or register it as a skill:

```bash
# Option A: as a command directory
cp -r claude-skills/review-paper /path/to/project/.claude/commands/

# Option B: symlink
ln -s "$(pwd)/claude-skills/review-paper" /path/to/project/.claude/commands/review-paper
```

## File Structure

```
review-paper/
├── SKILL.md                   # Main orchestrator (7-phase pipeline)
├── README.md                  # This file
└── references/
    ├── shared-directives.md   # Tone, confidence, humanization, quote rules
    ├── overview-prompt.md     # Overview + completeness + assumption check
    ├── section-prompts.md     # 6 variants: general, proof, methodology, results, literature, discussion
    ├── proof-verify-prompt.md # Adversarial proof verification (chained after proof sections)
    ├── cross-section-prompt.md# Discussion claims vs formal results
    ├── editorial-prompt.md    # 7-step editorial filter
    ├── calibration-prompt.md  # Domain-specific review criteria
    ├── contribution-prompt.md # Extract paper's stated contributions
    ├── literature-prompt.md   # Literature search via WebSearch
    ├── output-format.md       # refine.ink output format specification
    └── reference-example/     # Gold standard output
        └── feedback-regression-discontinuity-design-...md
```

## Pipeline (matches coarse Python pipeline)

| Phase | Python coarse | This skill |
|-------|--------------|------------|
| Extract | Mistral OCR + Docling + vision QA | Claude reads PDFs natively |
| Structure | `structure.py` (parse + LLM metadata) | Phase 1: Extract & Structure |
| Calibrate | `calibrate_domain()` | Phase 2 Agent 1 |
| Literature | `literature.py` (Perplexity Sonar) | Phase 2 Agent 2 (WebSearch) |
| Contribution | `extract_contribution()` | Phase 2 Agent 3 |
| Overview | `overview.py` + `completeness.py` | Phase 3 |
| Section review | `section.py` (parallel) | Phase 4 (parallel agents) |
| Proof verify | `verify.py` (chained) | Phase 4 chained |
| Cross-section | `cross_section.py` | Phase 5 |
| Editorial | `editorial.py` / crossref+critique | Phase 6 |
| Quote verify | `quote_verify.py` (fuzzy match) | Phase 6 (inline) |
| Synthesis | `synthesis.py` | Phase 7 |
