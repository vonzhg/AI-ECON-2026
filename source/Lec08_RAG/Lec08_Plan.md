# Plan: Lec08 — RAG (Retrieval Augmented Generation)

## Context

The 2026 graduate course "AI/ML for Economists" has 10 lectures. Lec01 (Quant Macro) and Lec02 (What is AI) are already drafted and compiled. Lec08 is a 1-hour lecture on RAG that sits between Lec07 (LLM internals) and Lec09 (Agentic AI / Claude Code). Its job: explain *why static LLMs are insufficient for real research workflows* and *how RAG bridges them to live, domain-specific knowledge.* The economist hook is high-volume institutional text — FOMC minutes, congressional records, central bank speeches, NBER working papers — exactly the kind of corpus that motivates retrieval over fine-tuning.

**Pedagogical priorities (from instructor feedback, reinforced today):**
1. **Slide count is a soft floor, not a ceiling.** Length is a later concern. Don't compress.
2. **Always motivate techniques against the naive baseline.** Specifically: explain why naive prompt engineering (pasting documents into the prompt) hits a wall before introducing RAG. This deserves its own dedicated section.
3. **Anchor every technical concept in economic research.** Every embedding, every vector DB feature, every chunking decision needs an FOMC / Congressional Record / NBER / 10-K example — not just at the end.
4. **Explain all concepts well, even at the cost of more slides.**

**Critical research finding:** The local source materials (HTML readings, Claude-Code-Presentation main.tex) contain almost **no RAG-specific content** — they're about Claude Code as an agentic system, not about embeddings, vector stores, or retrieval pipelines. The X.com URL the user supplied (`https://x.com/meer_aiit/status/2030755726235422919?s=46`) returns HTTP 402 (paywalled). Therefore Lec08 must be **written from scratch** using general RAG knowledge, with the local materials providing only thin scaffolding (context-window framing carried over from Lec07).

---

## Output

**File:** `/Users/zfeng/Library/CloudStorage/OneDrive-Personal/Teachings/AI_ML/Lecture_notes_2026/2026_New_Slides/Lec08_RAG/Lec08_RAG.tex`

**Figures subfolder:** `2026_New_Slides/Lec08_RAG/figures/` — created during implementation. Architecture diagrams will be drawn inline in TikZ (no external assets needed). The folder exists only for any later screenshots the user wants to drop in.

The folder `Lec08_RAG/` already exists and is empty.

---

## Template anchor — match Lec01/Lec02 exactly

The plan I was originally given (`Slides_Plan_2026.md`) tells me to copy the preamble from `16_LLM/Lec11_LLM.tex`, but the user has already drafted Lec01/Lec02 with a **customized** preamble. Lec08 must match Lec01/Lec02's preamble — not the raw Lec11_LLM.tex one — so the new deck looks identical to what's already approved.

**Reference files (preamble source):**
- `2026_New_Slides/Lec01_Quant_Macro/Lec01_Quant_Macro.tex` (lines 1–94)
- `2026_New_Slides/Lec02_What_is_AI/Lec02_What_is_AI.tex` (lines 1–92)

**Preamble must include (verbatim from Lec01):**
- `\documentclass[10pt,english,aspectratio=169]{beamer}`
- `metropolis` theme + `professionalfonts`
- Colors: `LightGray (230,230,230)`, `RedTitle (0,0,200)`, `VeryLightGray (245,245,245)`, `DarkText (0,0,0)`
- `\lstset{...}` block with colored code listings (blue keywords, green comments, orange strings, gray-8 background, single frame)
- Footer: `\insertframenumber/\inserttotalframenumber`
- `\setbeamertemplate{navigation symbols}{}`
- `\setbeamersize{text margin left=5mm, text margin right=5mm}`
- `\graphicspath{{figures/}}` (Lec08 has no upstream source folder, so it points at its own `figures/`)
- TikZ packages: `\usepackage{tikz}` + `\usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit,calc,backgrounds}` for the RAG architecture diagrams

**Title block:**
```
\title[Workshop on AI \& ML for Economists]{Workshop on AI, Machine Learning for Economists\\
8: Retrieval Augmented Generation (RAG)\\[0.5em]}
\author{Zhigang Feng}
\date{\today}
```

---

## Lecture structure — ~62 frames in 11 sections

Section dividers use the same `%====` comment style Lec01 uses. Frame counts target a 1-hour lecture pace at ~1 minute/slide (matching Lec01's 51 frames in 1 hour).

### Frame 1 — Title (`\makebeamertitle`)

### Frame 2 — Lecture Agenda
Numbered list of the 11 sections below, em-dash one-liners.

---

### Section I — Where We Are: From LLMs to RAG (frames 3–6)

- **Frame 3 — Quick recap of Lec07.** What an LLM is: parametric memory, static weights, pretrained on a frozen corpus.
- **Frame 4 — What an LLM knows vs what it can know.** Knowledge cutoff. FOMC December 2025 example.
- **Frame 5 — The economist's pain points.** (i) stale knowledge, (ii) hallucinated citations, (iii) no source attribution. Real example: LLM inventing fake NBER paper title.
- **Frame 6 — Two ways to add knowledge.** Fine-tuning vs Retrieval. One sentence each.

---

### Section II — The Naive Approach: Prompt Engineering Alone (frames 7–13) **[INSTRUCTOR PRIORITY]**

- **Frame 7 — What "naive prompt engineering" means.** Define: pasting full documents directly into the prompt. Show tiny example.
- **Frame 8 — A tempting first attempt.** Walk through FOMC use case naively. Paste 5 years of FOMC minutes into one prompt.
- **Frame 9 — Limit #1: Context window.** Real numbers. 200K tokens; 1 year of FOMC ≈ 144K tokens. Visual: token budget bar chart.
- **Frame 10 — Limit #2: Cost.** Pricing math. 720K input tokens × $3/M = $2.16/query. Prohibitive for thousands of queries.
- **Frame 11 — Limit #3: Latency & lost-in-the-middle.** Liu et al. (2024) "Lost in the Middle" U-shaped accuracy curve. Cramming hurts.
- **Frame 12 — Limit #4: No scalability or persistence.** Can't grow knowledge base. No reuse, no indexing, no incremental updates.
- **Frame 13 — The wall.** Side-by-side: naive vs research needs. Conclusion: need to *separate indexing from querying*.

---

### Section III — RAG: The Principled Solution (frames 14–18)

- **Frame 14 — The key insight.** Pre-process documents once; retrieve only relevant pieces at query time.
- **Frame 15 — Definition.** Lewis et al. (2020) NeurIPS citation.
- **Frame 16 — End-to-end architecture.** Full TikZ diagram. User Query → Embedder → Vector DB → Retrieved Chunks → LLM → Grounded Answer.
- **Frame 17 — Two phases: offline indexing.** Documents → Chunker → Embedder → Vector Store. "Library catalog" analogy.
- **Frame 18 — Two phases: online query.** Query → Embedder → Vector Store → Top-K → Augmented Prompt → LLM → Answer. "Ask the librarian" analogy.

---

### Section IV — Step 1: Document Ingestion & Chunking (frames 19–24)

- **Frame 19 — Why chunk at all?** Three reasons.
- **Frame 20 — Chunking strategies — overview.** Table: fixed/recursive/semantic/sentence-window/document-aware.
- **Frame 21 — Fixed-size vs recursive — example.** Same FOMC paragraph chunked two ways.
- **Frame 22 — Semantic chunking.** Cluster sentences by similarity, split on cluster boundaries.
- **Frame 23 — Practical defaults + Python code.** 500–1000 tokens, 10–20% overlap, metadata. Code listing.
- **Frame 24 — The economist's chunking gotcha.** FOMC speaker turns, congressional hierarchical metadata, 10-K item numbers.

---

### Section V — Step 2: Embeddings (frames 25–31)

- **Frame 25 — From text to vectors — what & why.** Mapping text → fixed-length float vector.
- **Frame 26 — Geometric intuition.** 2D scatter (TikZ): "Fed raised rates" near "FOMC tightened policy"; "GDP growth slowed" elsewhere.
- **Frame 27 — Why this works (lineage).** Word2Vec → GloVe → Sentence-BERT → modern transformer embedders.
- **Frame 28 — Modern embedding models — comparison table.** OpenAI 3-small/large, Cohere v3, MiniLM, bge-large, voyage-3.
- **Frame 29 — Economists' embedding example.** Embed Fed Chair speeches 2010–2025. Cluster hawkish/dovish/pivot. Hansen, McMahon & Prat (2018); Shapiro & Wilson (2021).
- **Frame 30 — Domain-adaptation problem.** Generic embeddings underperform on economics jargon. Fine-tune vs hybrid.
- **Frame 31 — Code: embed FOMC chunks.** Python listing.

---

### Section VI — Step 3: Vector Storage & Retrieval (frames 32–39)

- **Frame 32 — Why a vector database?** Flat numpy array doesn't scale.
- **Frame 33 — Vector DB landscape — table.** FAISS, Chroma, Pinecone, Weaviate, Qdrant, Milvus, pgvector.
- **Frame 34 — Similarity search 101.** Cosine vs dot vs Euclidean. Geometric picture.
- **Frame 35 — Approximate nearest neighbor (ANN).** HNSW, IVF. Why exact search doesn't scale.
- **Frame 36 — Hybrid retrieval: dense + sparse.** BM25 + embeddings via reciprocal rank fusion.
- **Frame 37 — Re-ranking with cross-encoders.** Two-stage: bi-encoder top-50 → cross-encoder top-5.
- **Frame 38 — Metadata filters.** "All FOMC discussions of unemployment from 2020 onwards, excluding press conferences."
- **Frame 39 — Code: build a Chroma index from FOMC chunks + query.** Python listing.

---

### Section VII — Step 4: Augmented Generation (frames 40–44)

- **Frame 40 — From retrieved chunks to a final answer.** Insert into prompt template, send to LLM.
- **Frame 41 — Prompt template — anatomy.** Real RAG prompt template verbatim, annotated.
- **Frame 42 — Source attribution mechanics.** `[doc_id:page]` tokens, post-process to inline links.
- **Frame 43 — Refusal & faithfulness.** "I don't know" is a feature.
- **Frame 44 — Worked example: full pipeline run.** Powell 2023 inflation expectations end-to-end.

---

### Section VIII — RAG vs Fine-Tuning (frames 45–48)

- **Frame 45 — Decision matrix.** 7 dimensions × RAG vs FT.
- **Frame 46 — When fine-tuning still makes sense.** Style, format, narrow vocab, latency.
- **Frame 47 — The hybrid pattern.** Fine-tune embeddings + general LLM + RAG glue.
- **Frame 48 — Cost & operational comparison.** Real numbers.

---

### Section IX — RAG for Economic Research: Applications (frames 49–55) **[INSTRUCTOR PRIORITY]**

- **Frame 49 — Use case 1: FOMC minutes & press conferences.** 1993–present. Hansen, McMahon & Prat (2018).
- **Frame 50 — Use case 2: NBER working papers.** ~30K papers as KB. Literature review automation.
- **Frame 51 — Use case 3: Congressional Record & legislative text.** ~150 years.
- **Frame 52 — Use case 4: SEC 10-K filings (Item 1A risk factors).** Cross-firm comparison. Lopez-Lira & Tang (2023).
- **Frame 53 — Use case 5: Cross-country central bank speeches.** BIS, ECB, BoE, BoJ.
- **Frame 54 — Use case 6: Combining structured + unstructured.** FRED time series + Fed speeches.
- **Frame 55 — Synthesis.** Three workflow patterns.

---

### Section X — Limitations, Evaluation & Failure Modes (frames 56–59)

- **Frame 56 — When RAG fails.** Six failure modes.
- **Frame 57 — Mitigations.** Per-failure-mode fixes.
- **Frame 58 — Evaluation: RAGAS framework.** Faithfulness, answer relevance, context precision/recall. Es et al. (2023).
- **Frame 59 — Open challenges.** Multi-hop, long-doc QA, table extraction, multilingual, eval cost.

---

### Section XI — Bridge & Summary (frames 60–62)

- **Frame 60 — Single-shot RAG vs agentic RAG (preview Lec09).** Linear pipeline vs loop.
- **Frame 61 — Summary: 4-step pipeline + the why.**
- **Frame 62 — Closing.** RAG = institutional text as queryable research instrument. `\end{document}`.

---

## Frame budget summary

| Section | Frames | Count |
|---------|--------|-------|
| Title + Agenda | 1–2 | 2 |
| I. From LLMs to RAG | 3–6 | 4 |
| II. Naive Prompt Engineering | 7–13 | 7 |
| III. RAG: Principled Solution | 14–18 | 5 |
| IV. Chunking | 19–24 | 6 |
| V. Embeddings | 25–31 | 7 |
| VI. Vector Storage & Retrieval | 32–39 | 8 |
| VII. Augmented Generation | 40–44 | 5 |
| VIII. RAG vs Fine-Tuning | 45–48 | 4 |
| IX. Economic Applications | 49–55 | 7 |
| X. Limitations & Evaluation | 56–59 | 4 |
| XI. Bridge & Summary | 60–62 | 3 |
| **Total** | | **62** |

---

## Verification

1. **Compile check.** `pdflatex -interaction=nonstopmode Lec08_RAG.tex` (twice). PDF appears, no error-level messages in `.log`.
2. **Frame count.** `grep -c "begin{frame}" Lec08_RAG.tex` returns 62 ± 3.
3. **Visual diff against Lec01.** Title page color, frametitle background, footer page-number format, code-listing colors must be identical.
4. **Agenda frame check.** Frame 2 must be the "Lecture Agenda" matching Lec01/Lec02 pattern.
5. **TikZ render check.** Architecture diagram (Frame 16) and embedding-space scatter (Frame 26) render without overflow.
6. **Pedagogical sanity check.** Section II should convince a student that prompt engineering hits a wall. Section IX should be concrete and corpus-grounded.

---

## Things this plan is explicitly NOT doing

- Not touching Lec01/Lec02. Already compiled and approved.
- Not creating Lec03–Lec07, Lec09, Lec10. Scope is Lec08 only this round.
- Not downloading external figures (X.com URL fails HTTP 402). All architecture diagrams drawn inline in TikZ.
- Not writing a lab notebook (Lec08 has no lab in the 2026 outline).

---

## Open question

The X.com URL (`https://x.com/meer_aiit/status/2030755726235422919?s=46`) returned HTTP 402 — paywalled. If the user wants content from that post, they'll need to paste it into chat. The lecture is complete without it.
