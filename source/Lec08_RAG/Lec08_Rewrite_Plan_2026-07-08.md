# Lec08 RAG Rewrite — One Map, Five Acts (2026-07-08)

## Context

Lecture 8 (RAG) is the last full lecture still on the retired "Topic N.k Agenda + Arc line" style. The user asked to rewrite it to (1) follow the Lec07/Lec09 format — the house **"one map, N acts"** roadmap convention (`\LecNMap{k}` "you are here" frame, `\sectiondivider`s, "Bridge:" closers), (2) **scale back complexity** for economics students — the deck should answer *what is RAG* and *how does it improve economic research*, not train retrieval engineers, and (3) **improve the flow logic**.

Current state (verified 2026-07-08): five decks, **99 authored frames** (+15 auto section pages ≈ 114 PDF pages) — T1 Where_We_Are_Prompting (16), T2 RAG_Chunking_Embeddings (23, has a `\appendix` that silently kills the house footer), T3 Vector_Retrieval_GraphRAG (26 — ANN/HNSW/IVF internals, an RRF formula, a 17-frame beyond-dense/GraphRAG block), T4a Augmented_Gen_Apps (19), T4b Demo_Failures_Wrap (15). No map, no dividers, no Bridge frames; only `takeawaybox` used. Known defects: T2 f21 points at `wiki_demo/knowledge/` + a TF-IDF retriever that don't exist there (the TF-IDF retriever lives in `demos/M4_rag_olg_demo/`); T1 f7's arithmetic is self-contradictory (650K tokens "does not fit" a 1M window); model/pricing facts stamped "June 2026"; several "(Section N)" pointers left over from the pre-split monolith.

**Deadline:** Lecture 8 is taught **Friday July 10** (Session 5, shared with Lec07). The syllabus (both `Docs/syllabus_2026.html` and public `AI-ECON-2026/syllabus.html`) promises: the RAG pipeline (chunking → embedding → vector storage → augmented generation), the RAG-vs-fine-tuning decision matrix, hybrid retrieval + **GraphRAG with the GRAM case study ("wrong-by-prominence" vs condition-aware)**, and **failure modes + RAGAS**. Binding on the rewrite.

**Scope decisions (user-confirmed 2026-07-08):**
- **Rename to five act-aligned decks T1–T5** (old five archived); RAG-vs-fine-tuning stays with generation (Act 4); the six econ use cases join demo + evaluation (Act 5).
- **GraphRAG block compressed to exactly 7 frames**, GRAM case + syllabus promises intact.
- **Target ≈ 75 authored frames** — design lands at **74** (13/16/15/11/19).
- **Regenerate `Lec08_讲稿.md` in this task** after the decks pass QA.

## Architecture — one map, five acts

| Act | New deck file | Student question | Display title (`Lecture 8.k:` line) |
|-----|---------------|------------------|-------------------------------------|
| 8.1 WALL | `Lec08_T1_The_Wall_Naive_Prompting.tex` | why not just paste it all in? | The Wall — Why Naive Prompting Fails |
| 8.2 INDEX | `Lec08_T2_Index_Chunking_Embeddings.tex` | how does my corpus become searchable? | Index — Chunking and Embeddings |
| 8.3 RETRIEVE | `Lec08_T3_Retrieval_Vector_GraphRAG.tex` | how does the right passage come back? | Retrieve — Vector Search and GraphRAG |
| 8.4 GENERATE | `Lec08_T4_Grounded_Generation_vs_FineTuning.tex` | how do I get citations, not hallucinations? | Generate — Grounded Answers and RAG vs Fine-Tuning |
| 8.5 APPLY | `Lec08_T5_Research_Applications_Demo_Eval.tex` | what does this change for my research? | Apply — Research Applications, Demo, and Evaluation |

## Case threads (named in every map-frame caption)

- **Thread A — FOMC / central-bank text:** 8.1 the December-2025 FOMC question a frozen model can't answer + corpus-size arithmetic → 8.2 FOMC minutes chunked (speaker-turn gotcha), embedded (hawk/dove clusters), coded → 8.3 the Section 13(3) hybrid story + metadata filters + Chroma FOMC index → 8.4 worked example: Powell 2023 pressers → cited paragraph → 8.5 Use Case 1 (FOMC) + Use Case 6 (FRED + minutes).
- **Thread B — the OLG literature:** 8.1 teaser in map caption (157 papers — queried by graph in 8.3, by hand in 8.5) → 8.2 bridge preview of `demos/M4_rag_olg_demo` (TF-IDF = sparse cousin of embeddings) → 8.3 GRAM case study (research-grade) → 8.5 classroom demo ("What did Diamond (1965) add to the OLG approach?", offline, refusal test).

## `lec08_map.tex` spec (new shared file)

Clone `lec09_map.tex` geometry with `lec07_map.tex`'s spine-caption variant. Filename **lowercase** so the build glob `Lec*_T*.tex` never compiles it standalone. `\LecEightMap{k}`, k∈{0..5}; `\ifnum` precomputes `\tikzset` styles **outside** TikZ option lists (comma-split breaks `\ifnum` inside `[...]`); current act `fill=orange!20, draw=RedTitle, very thick` + "you are here ▼"; k=0 all-green reprise + "all five acts complete".

- Boxes at x = 0/2.85/5.7/8.55/11.4: `\textbf{8.1 WALL}\\ \tiny naive prompting fails` · `\textbf{8.2 INDEX}\\ \tiny chunk & embed once` · `\textbf{8.3 RETRIEVE}\\ \tiny vector, hybrid, graph` · `\textbf{8.4 GENERATE}\\ \tiny cited, refusable answers` · `\textbf{8.5 APPLY}\\ \tiny corpora, demo, evaluation`.
- Student questions (tiny italic, y=−0.82): "why not just paste it all in?" / "how does my corpus become searchable?" / "how does the right passage come back?" / "how do I get citations, not hallucinations?" / "what does this change for my research?"
- Bottom band 3 zones: **THE PROBLEM** (act 1, blue!10) / **THE PIPELINE** (acts 2–4, green!10) / **YOUR RESEARCH** (act 5, orange!15).
- Spine caption (y≈−1.83): `the pipeline: corpus → chunk → embed → index ‖ query → retrieve → augment → cited answer`.
- Dashed satellites: left *"before 8.1: **Lec07** the language engine: frozen weights, finite context — RAG is its grounded memory"* (echoes lec07_map's own "Lec08 RAG (grounded memory)" satellite); right *"after 8.5: **Lec09** agents — retrieval becomes one tool the model chooses to call"*.

## Per-deck frame outlines

Tags: [KEEP] as-is · [EDIT] reworked (old source named) · [MOVE] relocated · [NEW]. `old <deck> fN` = Nth `\begin{frame}`, agenda = f1. All 99 old frames accounted for. Counts exclude title + dividers.

### T1 — 13 frames (from old T1's 16). Sections: "Where We Stand" {the language engine: frozen weights, finite context} / "The Wall" {four walls: window, middle, meter, memory} / "The Human-Scale Fix: A Personal Wiki" {retrieval by hand — the pre-formal RAG}

1. [NEW] **Lecture 8 in One Map** — `\LecEightMap{1}`; route caption + both threads.
2. [EDIT: f2+f3] **The Frozen Brain: What Lecture 7 Left Us** — merge recap + knows-vs-can-know; cutoff as concept (drop the 4-model cutoff list); keep December-2025 FOMC question, hallucinate-or-refuse, private-corpora point; callback → Lec. 7 T4/T5a.
3. [EDIT: f4] **Three Pain Points for the Economist** — keep stale knowledge / hallucinated citations (fake NBER Phillips-curve papers) / no attribution.
4. [EDIT: f5] **Two Ways to Add Knowledge** — fine-tune vs retrieve; fix stale "Section VIII" → "the decision matrix comes in Act 8.4"; promise line rewritten to the five acts.
5. [EDIT: f6] **A Tempting First Attempt: Paste It All In** — keep document-stack TikZ + four-reasons sidebar; fix callbacks (sidebar "Lec. 7 T3b" → T5a; prompting "Lec. 7 T3a" → T4).
6. [EDIT: f7+f9] **Hard Walls: The Window and the Middle** — de-pin window list ("≈1M tokens at the mid-2026 frontier; many models far less"). **Fix the broken arithmetic WITHOUT inventing numbers**: keep the verified ~650K-token figure for 5 yrs of statements+minutes, then scale honestly — 30 years ≈ 3.9M, plus transcripts/speeches/NBER papers pushes the research corpus far past any window; and even what fits is re-paid per query and suffers lost-in-the-middle. Keep Liu et al. (2024, TACL) U-curve, re-pointed → Lec. 7 T5a "Limit 4 — Lost in the Middle"; latency one line.
7. [EDIT: f8+f10] **Economic Walls: The Meter and the Fresh Start** — drop all per-token dollar figures and the "$5/M since Opus 4.5" footnote; keep: every query re-pays the full corpus bill ×1000 exploratory queries; no reuse, no incremental updates; punchline "separate the cost of ingesting documents from the cost of asking questions"; rates pointer → Lec. 7 T4.
8. [EDIT: f11] **The Wall** — keep 5-row can/need table + "*This is what RAG does.*"
9. [EDIT: f12] **Before the Formalism: A Personal Wiki** — keep Karpathy (June 2025) quote (attribution date, not freshness) + 3-step workflow.
10. [KEEP: f13] **The Wiki Pipeline** — build-once/query-many TikZ.
11. [EDIT: f15] **Hands-On: A Mini Wiki for This Course** [fragile] — keep directory listing + `wiki_demo/` pointer (path verified correct); tighten.
12. [EDIT: f14+f16] **Wiki = Manual RAG: Now Automate It** — compress 6-row table to ~4 rows; keep naive→wiki→RAG progression; close "wiki = manual RAG; now automate it."
13. [NEW] **Bridge: From Hand-Curated to Machine-Indexed** — recap + next-act bullets + arc line (**Act 1 WALL** bold).

CUT: f1 agenda (→map). Merged: f3→2, f9→6, f8/f10→7, f16→12/13. All 16 accounted.

### T2 — 16 frames (from old T2's 23). Sections: "RAG: The Principled Solution" {index once, query many} / "Chunking" {what is a retrieval unit?} / "Embeddings" {text becomes geometry}

1. [NEW] **Map: Act 8.2** — `\LecEightMap{2}` + route caption.
2. [EDIT: f3] **The Key Insight: Index Once, Query Many** — keep 3-step + two-phase framing + cost-proportional-to-need punchline.
3. [EDIT: f2+f5+f6+f7] **The Whole Pipeline in One Picture** — THE act-2 anchor (flow fix #1): f2's offline/online TikZ enlarged; f5's five components named with act pointers replacing stale "(Section V/VI/VII)" → "(this act)/(Act 8.3)/(Act 8.4)"; library-catalog + ask-the-librarian analogies one line each; **same-embedder-for-index-and-query** warning as cautionbox.
4. [EDIT: f4] **Definition and Origin** — Lewis et al. (2020, NeurIPS); "the LLM is the reasoner, the vector store the memory, the retriever the librarian."
5. [EDIT: f8] **Why Chunk at All?** — three reasons + "most consequential decision" trade-off.
6. [EDIT: f9+f11] **Chunking Strategies and the Trade-Off** — keep bubble chart; semantic chunking reduced to one idea-line ("split where adjacent-sentence similarity drops; only if recursive is measurably noisy").
7. [EDIT: f10] **Fixed-Size vs. Recursive: An FOMC Example** — keep contrast verbatim.
8. [EDIT: f12] **Practical Defaults + Code** [fragile] — keep LangChain listing + preserve-metadata bullet.
9. [EDIT: f13] **The Economist's Chunking Gotcha** — keep FOMC speaker turns / Congressional hierarchy / 10-K Item 1A (seeds T5 use cases).
10. [EDIT: f14] **From Text to Vectors: What and Why** — keep E: text→R^d; re-point recap → Lec. 7 T3b "Token Embeddings vs. Document Embeddings (Bridge to Lec08)".
11. [EDIT: f15+f16] **Geometric Intuition: Trained So Similar Texts Land Nearby** — keep hawkish/dovish/growth cluster picture; replace pooling/contrastive-loss mechanics with one intuition sentence + "Sentence-BERT lineage, Reimers & Gurevych (2019)"; callback → Lec. 7 T3b.
12. [EDIT: f17] **Choosing an Embedder (mid-2026)** — keep 4-row table; replace $-prices with tier labels (free · local / metered API); add `\textit{\footnotesize Model facts current as of mid-2026.}`; drop appendix pointer.
13. [EDIT: f18] **Economists' Embedding Example: Fed Chair Speeches** — keep UMAP story + Hansen–McMahon–Prat, Shapiro–Wilson, Gentzkow–Kelly–Taddy.
14. [EDIT: f19] **The Domain-Adaptation Problem** — keep "patient"/CUSIP/LSAP/"Section 13(3)" + two fixes; "(Section VI)" → "(Act 8.3)".
15. [EDIT: f20] **Code: Embed FOMC Chunks** [fragile] — keep.
16. [NEW: absorbs f22, salvages f21] **Bridge: The Corpus Is Now Geometry** — recap three load-bearing decisions; **corrected hands-on pointer** (flow fix #3): browse `wiki_demo/` (no retriever there) + preview `demos/M4_rag_olg_demo` (TF-IDF retriever, sparse cousin of embeddings — full run in Act 8.5); arc line.

CUT: f1 (→map); f5/f6/f7 (three near-duplicate architecture diagrams → one anchor, new 3); f11 (→6); f16 (→11); f21 (its `wiki_demo/knowledge/`+TF-IDF claims are wrong on disk — salvaged correctly into new 16); f22 (→16); f23 appendix table (house rule: no `\appendix`). All 23 accounted.

### T3 — 15 frames (from old T3's 26; GraphRAG block = new 8–14 = exactly 7). Sections: "Vector Search in Practice" {from similarity math to a working index} / "When Similarity Is Not Enough" {relationship questions need edges, not echoes}

1. [NEW] **Map: Act 8.3** — `\LecEightMap{3}` + route caption.
2. [EDIT: f2] **Why a Vector Database?** — flat-numpy baseline + four problems/solutions.
3. [EDIT: f4+f5] **Similarity Search 101 — and Why Approximate** — cosine/dot/L2 + normalized⇒dot + geometric picture; O(N·d) in one line; HNSW/IVF compressed to "names to recognize — your library picks the default; ~1% recall traded for milliseconds."
4. [EDIT: f3] **Vector DB Landscape: Scale vs. Hosting** — keep scatter TikZ + economist picks (Chroma/FAISS, pgvector, Pinecone/Qdrant).
5. [EDIT: f6+f7] **Hybrid Retrieval, Plus a Light Re-Rank** — keep Section 13(3) story + BM25; **drop the RRF formula** ("merge the two ranked lists — rank fusion is a solved problem your library implements"); re-rank in 2–3 bullets (bi-encoder shortlist → cross-encoder reorder; `bge-reranker-large`); "2025" → "mid-2026".
6. [EDIT: f8] **Metadata Filters** — keep in full; "(Section IV gotcha)" → "(the Act 8.2 gotcha)".
7. [EDIT: f9] **Code: Build a Chroma Index and Query It** [fragile] — keep.
8. [EDIT: f10] **Where Vanilla Vector RAG Starts to Break** — three missing structures + both economist examples.
9. [EDIT: f11] **Concrete Break Point: The Phillips-Curve Question** — two-column trace + takeawaybox rule ("about a relationship, not just about a topic") verbatim.
10. [EDIT: f12+lines from f13/f14] **Knowledge Graphs: Adding Structure to Retrieval** — `<head, relation, tail>` + `knowledge_graph_example.png` (verified present); one offline/online line from f13; one econ triple from f14 (`<Paper B, cites, Paper A>`).
11. [EDIT: f15] **Vanilla RAG vs. GraphRAG: A Direct Comparison** — 5-row table + "start with vector + BM25 + re-rank" rule.
12. [EDIT: f16+f17] **Case Study: GRAM — Four Mechanisms on One OLG Query** — 1–2 setup bullets (instructor's own GRAM; 157-paper OLG corpus; research-grade escalation of the Act 8.5 demo) above the four-mechanism TikZ **kept intact** (outcome labels: silent structural failure / bounded by curation / wrong-by-prominence / ranked + condition-aware); solver-admissibility query as header line.
13. [EDIT: f18+2 lessons from f20] **Wrong-by-Prominence vs. Condition-Aware Retrieval** — Diamond-vs-Auerbach–Kotlikoff, Aiyagari-vs-Krusell–Smith; "the vocabulary is present; the applicability filter is absent"; **honest-framing takeawaybox essentially verbatim** (design goal / planned three-condition evaluation / not yet a reported win); two durable corpus lessons from f20 (post-cutoff share; citation connectivity) + one-line "one in-progress, unpublished pipeline" caveat.
14. [EDIT: f21+one-liner from f22] **Choosing the Retrieval Architecture** — 4-row decision table + rule of thumb; footnote: "Names to recognize: Microsoft GraphRAG, LightRAG, HippoRAG, RAPTOR — Edge et al. (2024); Guo et al. (2024); Gutiérrez et al. (2024); Sarthi et al. (2024)."
15. [NEW: absorbs f26's agentic pointer] **Bridge: The Right Passage Is Back — Now Answer From It** — recap; "eventually an *agent* chooses vector vs. graph vs. web itself — Act 8.5 bridges to Lec09"; arc line.

CUT: f1 (→map); f5 (→3); f7 (→5); f13 offline-vs-online (redundant with new 10 + T2 anchor; frees `graphrag_pipeline.png`); f14 triples table (one exemplar survives); f19 "Elements, Not Chunks" (schema internals too deep); f20 "Corpus Construction as Engineering" (2 lessons fold into 13; tier counts/9-of-11 audit/MinerU/Qwen3 die); f22 Representative Systems (→footnote on 14); f23+f24 Visual Snapshots; f25 benchmarks; f26 Future Directions (pointer → 15). All 26 accounted.

### T4 — 11 frames (from old T4a f1–f10+f19). Sections: "Grounded Generation" {from retrieved evidence to a cited, refusable answer} / "RAG vs. Fine-Tuning" {facts need retrieval; style needs weights}

1. [NEW] **Map: Act 8.4** — `\LecEightMap{4}` + route caption.
2. [EDIT: f2] **From Retrieved Chunks to a Cited Answer** — lead reframed as the generation contract (grounded, cited, refusable); keep 4-part anatomy preview + "the only place a generative model is used".
3. [EDIT: f3] **Prompt Template — Anatomy** [fragile] — keep SYSTEM/CONTEXT/QUESTION/INSTRUCTIONS + `[doc_id:page]`.
4. [EDIT: f4] **Source Attribution Mechanics** — keep 4-step enforcement + before/after.
5. [EDIT: f5] **Refusal and Faithfulness** — keep "'I don't know' is a feature"; "(T4b)" → "(Act 8.5)".
6. [EDIT: f6] **Worked Example: Full Pipeline Run** — keep 6-step Powell-2023 trace + cited answer; "Send to GPT-4o" → "send to the generator LLM (temperature 0.1)".
7. [EDIT: f7] **Decision Matrix: RAG vs. Fine-Tuning** — keep colortbl matrix + reading paragraph (syllabus-promised).
8. [EDIT: f8] **When Fine-Tuning Still Makes Sense** — keep style/format/vocabulary/latency + "Fine-tuning ≠ teaching new facts" caveat (consistent with Lec07 T5b's "fine-tuning stores style, not facts").
9. [EDIT: f9] **The Hybrid Pattern** — keep; "in 2025" → "mid-2026". (Fallback if deck runs long: merge 8+9.)
10. [EDIT: f10] **Cost: Index Once, Query Many (Visual)** — keep two-curve TikZ + crossover; callback → Lec. 7 T4 "Tokens, Cost, and Reproducibility".
11. [NEW: absorbs f19] **Bridge: The Pipeline Is Complete — Now Make It Earn Its Keep** — recap contract + matrix; next-act bullets (six corpora, demo, failure modes, RAGAS); arc line.

Disposition of old T4a: f1→map; f2–f10 edited here; f11–f16 MOVE → T5 2–7; f17 → T5 8; f18 → T5 9; f19 → new 11. All 19 accounted.

### T5 — 19 frames (old T4b's 15 + 8 moved from T4a). Sections: "Six Research Corpora" {the same pipeline, six research designs} / "Classroom Demo: OLG RAG" {a pipeline you can run offline today} / "Failure Modes and Evaluation" {when it breaks — and how to prove it works} / "From RAG to Agents" {retrieval becomes one tool}

1. [NEW] **Map: Act 8.5** — `\LecEightMap{5}` + route caption.
2.–7. [MOVE: T4a f11–f16] **Use Cases 1–6** — FOMC minutes & pressers (Hansen–McMahon–Prat) / NBER working papers / Congressional Record (gotcha pointer → "the Act 8.2 gotcha"; Gentzkow–Shapiro–Taddy) / SEC 10-K Item 1A (Loughran–McDonald; Lopez-Lira) / cross-country central-bank speeches (BIS) / structured+unstructured (FRED + minutes). Kept intact — the econ payoff; contiguous so the instructor can pace them as one sweep.
8. [MOVE+EDIT: T4a f17] **Three Roles for RAG Evidence: Signal, Outcome, Instrument** — keep 5×3 table; **retitle** (old title hardcodes a wrong callback) and re-point → Lec. 7 T1 "Three Roles of Text in Economic Research" (+ T5b Application Zoo as companion).
9. [MOVE+EDIT: T4a f18] **Synthesis: Three Workflow Patterns** — keep + "queryable research instrument" punchline.
10. [EDIT: T4b f2+f3] **Demo: One Question Through the Whole Pipeline** [fragile] — Diamond-1965 question (verbatim — matches demo code), 5-box pipeline TikZ, three launch commands verbatim (`run_demo.py`, `run.py ask ...`, `PYTHONPATH=src ... unittest`), compact artifacts table; keep "offline: no API key, no network" + `./demos/M4_rag_olg_demo` path.
11. [EDIT: T4b f4] **Demo: Audit the Corpus Before You Index** — keep audit + two-minute rule + `build/chunks.json`.
12. [EDIT: T4b f5] **Demo: Read the Retrieval Report First** — keep anchor checks + PASS/FAIL + "the diagnostic is the publication-grade artifact."
13. [EDIT: T4b f7, retires f6] **Demo: Answer vs. Refusal** — keep two-column + takeawaybox + report path; add one line "the grounded prompt behind this is exactly Act 8.4's template, with line-range citations `[file:La-b]`". Do NOT reproduce f6's phantom "cite-then-claim (Lec. 7 T3b)" quote.
14. [EDIT: T4b f8] **When RAG Fails: Six Failure Modes** — keep all six (syllabus-promised).
15. [EDIT: T4b f9] **Mitigations — One per Failure Mode** — keep (HyDE, hybrid, stricter prompts, overlap/parent-doc, incremental ingest, query rewriting); optionally absorb f12's "evaluation at scale stays expensive — hence the small gold set next".
16. [EDIT: T4b f10] **Evaluation: The RAGAS Framework** — keep Es et al. (2023, EACL) + four metrics + 50-question eval-set workflow (syllabus-promised).
17. [EDIT: T4b f11 + checklist from f15] **Validate Like an Economist: Cohen's κ and Disclosure** — re-point replication cite → Lec. 7 T5b "Yin, Vu, Persico (NBER 35110, 2026)"; keep gold-standard + second annotator + κ ≥ 0.75 + inner/outer loop; pull the AEA disclosure checklist up from f15.
18. [EDIT: T4b f13] **From RAG to Agentic RAG: Retrieval Becomes One Tool** — keep contrast + tool-fan TikZ + patterns; "JSON tool calls Lec. 7 T3a" → Lec. 7 T4 "Structured Output"; "(Lec. 7 T5)" → "(Lec. 7 T4/T5a)". Fulfills old T3 f26's pointer too.
19. [EDIT: T4b f14+f15] **Close: One Map, Complete — Pick the Right Tool** — `\LecEightMap{0}` reprise; four-step summary in one line under the map; pick-the-right-tool table (incl. agentic RAG (Lec. 9) + long-context row) + "queryable research instrument" headline + brain/memory/librarian close (unify metaphor wording with T2 4) + hand-off to Lec09 (Session 6).

Disposition of old T4b: f1→map; f3→10; f6 cut (duplicates T4 3; pointer survives in 13); f12 cut (grab-bag; each item already covered; eval-cost line optionally → 15); f15→17+19. All 15 accounted.

## Integrity constraints

- **Syllabus promises** (`Docs/syllabus_2026.html:405–411`) — every phrase teachable from the decks; "wrong-by-prominence" and "condition-aware" must appear on-slide (T3 12–13).
- **Verbatim survivals:** GRAM honest-framing takeawaybox ("…gap a planned three-condition evaluation (naive / RAG / GraphRAG) is designed to measure — a design goal and hypothesis, not yet a reported win") + "one research pipeline (the instructor's)… unpublished, in-progress" caveat; demo question "What did Diamond (1965) add to the OLG approach?" + all three launch commands + artifact paths (verified on disk; retriever class `TfidfIndex` confirmed); T3 takeawaybox rule; Karpathy quote; "I don't know based on the provided sources." refusal string.
- **Callback re-pointing table (grep-verified against rebuilt Lec07):** T1 f6 prompting → **T4** (:209); T1 f8 / T4a f10 tokens-cost → **T4** (:136); T1 f6/f9 lost-in-middle → **T5a** (:140); T2 f16 embeddings-bridge → **T3b** (:417); T4a f17 three-roles → **T1** (:125, companion T5b:566); T4b f13 JSON tools → **T4** (:270); T4b f11 Yin–Vu–Persico → **T5b** (:468); T4b f13 sycophancy/drift → **T4 + T5a**. Phantom "cite-then-claim (Lec. 7 T3b)": does not exist in rebuilt Lec07 — dies with f6, do not resurrect.
- **Lec07's inbound hooks to echo:** T3a:193 "the 'keys' are your corpus" (→ new T3 3); T2b:361 "find me speeches like this one" (→ new T2 11/13); T2a:76 context forces chunking (→ new T1 6); T5a:267 "Lecture 8's opening question" (→ new T1 2); T5b:720 "fine-tuning stores style, not facts" (→ new T4 8).
- **Stale "(Section N)" pointers to fix:** T1:115; T2:155–158, :541, :579 (dies with f21); T3:231; T4a:487.
- **Dated strings:** "as of June 2026" ×4 → "mid-2026" only where the claim survives; de-pin T1 cutoff/window lists; remove T1 dollar arithmetic + "$5/M since Opus 4.5"; T2 f17 prices → tier labels; "2025" production-default claims → "mid-2026"; "GPT-4o" → generic. **Invent no new numbers** (verify-claims rule); Karpathy "(June 2025)" stays.
- House rules: no `\framesubtitle`, no `\appendix`, `[fragile]` on listings, `-{-}` in `\texttt`, lead/source lines start with a command not a brace, `RedTitle` stays blue, pdflatex-only English.

## Deck management, manifest, cross-file edits

1. Archive old five decks (tex+pdf) + old `Lec08_讲稿.md` → `Lec08_RAG/archive_pre_split/pre_map_rebuild_2026-07-08/` (never a new top-level archive dir — the build glob crosses `/`).
2. New deck headers: fresh provenance line ("Rebuilt 2026-07-08 per Lec08_Rewrite_Plan_2026-07-08.md; sources: <old deck(s)>") — do not carry forward the old Pass-B lineage comments (archive is the provenance chain).
3. `Lec08_manifest.txt` → the five new PDFs in teaching order.
4. `Lec09_Agentic_AI/Lec09_T1_Framing_Setup.tex:266`: "(Lec08 T4b)" → "(Lec08 T5)". (Note in final report: Lec09_Rewrite_Plan's own reference to "Lec08_T4b backref" becomes historical — nobody should "restore" T4b.)
5. `Docs/syllabus_2026.html:413–417`: Lec08 deck links → new five PDFs (content bullets 405–411 unchanged). While in the file, optionally also fix the already-stale Lec07 deck links (:394–401, pre-rebuild names) — flag in final report either way.
6. `plans-comments/2026_lecture_outline.md:161–165`: new deck names.
7. Leave in place: `Graph-rag-share.pptx`, orphaned TikZ fragments (`infra_stack/interface/retrieval_flow.tex`), all `figures/graph_rag_share/*.png` (only `knowledge_graph_example.png` remains referenced; archived decks reference the rest), `wiki_demo/`, `demos/`.

## Build & verification

1. Compile each deck as written; final scoped build: `bash /u/zfeng2/Github/AI-ML-2026/FullCourse_10Lecs/build_all_topic_decks.sh /u/zfeng2/Github/AI-ML-2026/FullCourse_10Lecs/Lec08_RAG` (absolute path — bare relative silently builds zero decks). pdflatex ×2 for the k/Total footer.
2. Footer QA: `qa_page_numbers.sh` same arg (needs pdftotext/pdfinfo; if absent fall back to `gs -sDEVICE=txtwrite`) — no `/100`, footer present on every numbered page.
3. Visual QA via Ghostscript (`gs -sDEVICE=png16m`, the only rasterizer here): each deck's map frame (correct act highlighted), one divider per deck, each Bridge frame, T5's `\LecEightMap{0}` reprise, the GRAM four-mechanisms TikZ, T2's pipeline anchor; sweep pdflatex logs for overfull boxes (tolerate pre-existing-scale slack only).
4. Content greps: zero `\framesubtitle`/`\appendix`/"Topic 8." agenda titles/"T4a|T4b" self-references/"Section V|VI|VII|VIII" pointers/"June 2026"; every Bridge frame carries the arc line; callbacks match the re-pointing table; "wrong-by-prominence" + "condition-aware" present.
5. Frame counts: `grep -c 'begin{frame}'` = 13/16/15/11/19 (74 total, ±2 tolerance per deck).

## 讲稿 regeneration (same task, after decks pass QA)

- One agent per new deck (5, parallel), following `Docs/讲稿_style_guide.md` + master glossary; frames-only `### 幻灯片 N` numbering (the style-guide default; Lec07's divider-inclusive variant was for its interleaved-PDF pipeline only); format `## Topic k：<file>（共 N 张）`.
- Front matter (本讲概览 / 学习目标 / 术语表) updated to the five-act structure.
- Sync check: per-Topic `grep -c '^### 幻灯片'` == that deck's frame count.
- Combined PDFs (`幻灯片_合集`, `讲稿_合集`) NOT rebuilt here (already awaiting Lec09 讲稿 regen).

## Implementation steps (runbook)

0. Copy this plan → `FullCourse_10Lecs/Lec08_RAG/Lec08_Rewrite_Plan_2026-07-08.md`.
1. Archive old decks (tex+pdf) + 讲稿 per §Deck management.
2. Create `lec08_map.tex`; smoke-test compile inside a minimal T1 stub.
3. Write decks in order T1→T2→T3→T4→T5 (sources readable from the archive), compiling each on completion.
4. Cross-file edits (§Deck management 3–6).
5. Full scoped build + all QA gates (§Build & verification).
6. Regenerate `Lec08_讲稿.md` (5 parallel agents) + sync check.
7. Update memory (map-convention roster + 讲稿 status + desync ledger) and write the final report: frame counts, QA results, what was cut, follow-ups.

## Risks / notes

- **Clock:** 74 frames + 5 titles + 14 dividers ≈ 93 pages for a shared 1-hour session is still generous; the six use-case frames are contiguous by design so they can be paced as one sweep. Flow and coverage over clock-fit (Lec09 precedent).
- T1's window arithmetic must be fixed honestly with existing verified numbers only (see T1 6).
- Old T4a f17's *title* hardcodes the wrong callback — retitling is mandatory (讲稿 keys off titles).
- `\metroset{sectionpage=none}` + dividers are additions, not carryovers — omitting either gives double/unstyled section pages.
- Wider syllabus staleness (Lec07 links) is pre-existing; fix opportunistically or flag.

## Follow-ups (out of scope)

- Publish Lec08 to the website (`AI-ECON-2026/` still shows "Soon" + placeholder PDF) — publish recipe in memory.
- Rebuild `Combined_PDFs/` 合集s (blocked on Lec09 讲稿 regen); optional `Lec08_pdf_build/` interleaved handout (Lec07-style).
- Lab 7/8 remains a "Coming soon" placeholder — intentionally untouched.
