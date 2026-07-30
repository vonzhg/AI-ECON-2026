# Lecture Sources

The LaTeX sources, figures, and lab material behind the decks published in
[`../slides/`](../slides/). Everything here is *source*: the site ships compiled, watermarked PDFs,
and this tree is what they are built from.

One folder per lecture, each holding the `.tex` files for its topic decks plus their figures:

| Folder | Lecture | Published decks |
|---|---|---|
| `Lec01_Introduction/` | 1 · Introduction — AI and Economic Research | `Lec01_T1`–`T3` |
| `Lec02_What_is_AI/` | 2 · What is AI? | `Lec02_T1`, `T2` |
| `Lec03_ML_for_Macro/` | 3 · Machine Learning for Quantitative Macroeconomists | `Lec03_T1`–`T4` |
| `Lec04_DL_RL_Macro/` | 4 · Solving Macro Models via Deep Learning & RL | `Lec04_T1`–`T4` |
| `Lec05_RL_Nutshell/` | 5 · Reinforcement Learning in a Nutshell | `Lec05_T1`–`T3` |
| `Lec06_HA_Models/` | 6 · Heterogeneous Agent Models via RL | *placeholder on the site* |
| `Lec07_LLM/` | 7 · Large Language Models & Text as Economic Data | `Lec07_T1`–`T3b` |
| `Lec08_RAG/` | 8 · Retrieval-Augmented Generation | `Lec08_T1`–`T3` |
| `Lec09_Agentic_AI/` | 9 · Agentic AI for Research Workflows | `Lec09_T1`–`T5` |
| `Lec10_Case_Studies/` | 10 · Case Studies | *placeholder on the site* |
| `Lec01_Quant_Macro/` | — | Earlier drafts, superseded by `Lec01_Introduction/` |

`shared_preamble.tex` holds the common Beamer preamble; `build_all_topic_decks.sh` and
`qa_page_numbers.sh` are the batch build and page-count QA helpers. `Labs/` and `Notebooks/` carry
the lab material (the published subset is mirrored into `../labs/`), `Case_Study/` the Deep-Ramsey
case-study package, and `Docs/` the capstone brief and lecture-script style guide.

## Not every master is published

`source/` holds **71 more PDFs than the site serves** — earlier drafts, superseded variants, and a
few decks removed from the site on purpose. The important ones:

- **`Lec08_T4_Grounded_Generation_vs_FineTuning.pdf`** and
  **`Lec08_T5_Research_Applications_Demo_Eval.pdf`** were deliberately dropped from the site
  (commit `2b10bbc`, "Lec08 site: keep T1-T3, remove T4"). They build fine; they are just not
  linked. Don't "restore" them without meaning to.
- `Lec06_HA_Models/` and `Lec10_Case_Studies/` have real decks, but the site shows
  *In preparation* placeholders for those two lectures.

`tools/build_slides.py` therefore rebuilds only what is already in `slides/`. Use `--adopt` to start
publishing a master that is not yet there.

## Building a deck

All decks are Beamer with the Metropolis theme, sharing `shared_preamble.tex`.

```bash
cd Lec08_RAG
pdflatex Lec08_T2_Index_Chunking_Embeddings.tex
pdflatex Lec08_T2_Index_Chunking_Embeddings.tex   # second pass for refs and the frame counter
```

Then publish it — this stamps the copyright watermark and updates the manifest:

```bash
cd ../..
python3 tools/build_slides.py Lec08_T2
```

Build artifacts (`.aux .log .nav .snm .toc .out .vrb .synctex.gz`) are git-ignored and never
committed. Decks print their build timestamp on the title slide, so a recompile is visible in the
PDF itself — which is how the stale-deck problem below was found.

## Provenance

Imported from `AI-ML-2026/FullCourse_10Lecs/` on 2026-07-29, so the site and its sources live
together. Excluded on the way in, deliberately:

- `Lec10_Case_Studies/Exercise10_ES_Fellows/` — 240 MB of raw exercise data, not needed to build
  any deck
- `Combined_PDFs/`, `*_pdf_build/` — derived output
- `_archive/`, `_claude_backup/` — superseded copies
- Internal planning and revision notes, an unpublished working paper, and a photo carrying EXIF —
  moved to the instructor's local `_private-readings/`, outside this repo

At import, three published decks turned out to be **stale**: `Lec08_T1`, `T2` and `T3` had been
rewritten on 2026-07-10 but the site was still serving the 07-08 build. Confirmed as a real content
change, not just a recompile — the chunking-strategy slide had been reworded — and republished. All
29 non-placeholder decks now match their masters; `tools/build_slides.py --check` verifies it.
