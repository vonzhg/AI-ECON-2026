# Lec07_LLM Rebuild — Follow chapter14's Logic + Lec09's Roadmap Format

> **Changelog 2026-07-07 (review pass):** amended after full-source review — archive path fixed (build-glob sweep), build/QA commands scoped to Lec07_LLM, two source-deck attributions corrected (positional encodings are in current T2a; TF-IDF trio is in current T1b), all orphan frames assigned, priority tiers added, three chapter items added (translate-vs-native thinkbox, §14.7 starter paths, FFN key-value memory), numbers-hygiene notes on YVP/pricing, map-caption fallback.

## Context

**Goal.** Rebuild the `Lec07_LLM` slide set so it (a) follows the *logic* of the book chapter `chapter14.tex` — "文本表示与语言模型基础 / Text Representation and Language Model Foundations" — and (b) adopts the *format* of `Lec09_Agentic_AI`, which the user singled out for its "good roadmap for each deck." Audience = **economists** who want to use textual analysis and LLMs in their own research.

**Why now.** The current 7 decks (125 frames) are already econ-rich and share deep DNA with chapter14 (the chapter was a source for the old plan), but three things are missing versus the chapter's *logic* and Lec09's *format*:
1. Chapter14 is an **"econometrics sandwich"**: it opens with a measurement-framing section, walks the ML ladder in the middle (each rung motivated by a *measurement defect* of the previous one), then closes with limits + applications + validation, landing on the thesis **"识别先于测量 / identification precedes measurement."** In the current deck this framing is diffused (buried mid-`T1a`) rather than a clean opening act.
2. Chapter14's central device — the **text data-generating process** `x_i = g(s_i,a_i,c_i)+ε_i` feeding a regression `y_i = β·m(x_i)+γz_i+u_i`, which generates the two recurring econometric hazards (endogeneity via strategic framing `a_i`; non-classical measurement error) — already exists in the deck (`T1a` "An Identification-First View of Text") but is **not a recurring spine**.
3. There is **no roadmap/map system** — no `lec07_map.tex`, no section dividers, no "Bridge:" closers — that Lec09 uses to make a deck-set read as one flow.

**Decisions locked with the user.** (1) Full **5-act / 8-deck** reorganization; (2) **keep** the richer 2024–26 content that exceeds the book; (3) **add China parallels** (CNRDS/CSMAR/Wind, PBOC reports, China EPU, local Qwen/DeepSeek + PIPL/DSL) alongside the US cases; (4) **English + pdflatex** (match Lec09 and the current deck; Chinese stays in the 讲稿; render Chinese data sources romanized, e.g. "People's Daily", "PBOC Monetary Policy Report").

**Outcome.** Eight decks organized as five acts — **FRAME → REPRESENT → ATTEND → PRETRAIN → MEASURE** — with a recurring `\LecSevenMap{k}` "you are here" strip, the text-DGP as an explicit recurring spine, and a close on "identification precedes measurement." Heavy reuse of existing frames; net new authored content is the Act-1 FRAME deck, the map file, the scaffolding (dividers/bridges), and the chapter14/China additions folded into each act.

---

## Target structure (5 acts / 8 decks)

| Act (map box) | Deck | New filename | chapter14 § | Built from |
|---|---|---|---|---|
| **1 FRAME** — text as measurement (7.1) | T1 | `Lec07_T1_Text_as_Measurement.tex` | §14.1 | framing frames carved out of current `T1a` **(NEW deck)** |
| **2 REPRESENT** — tokens→vectors (7.2) | T2a | `Lec07_T2a_Preprocessing_BoW.tex` | §14.2, §14.3.1 | tokenization/preprocessing frames from current `T1a` + BoW; TF-IDF trio relocated from current `T1b` |
| | T2b | `Lec07_T2b_Word_Embeddings.tex` | §14.3.2–3 | current `T1b` (Embeddings) |
| **3 ATTEND** — attention & Transformer (7.3) | T3a | `Lec07_T3a_Attention.tex` | §14.4.1 | current `T2a` (Attention) |
| | T3b | `Lec07_T3b_Transformer.tex` | §14.4.2–3 | current `T2b` (Transformer) + positional-encoding frames from current `T2a` |
| **4 PRETRAIN** — LMs & alignment (7.4) | T4 | `Lec07_T4_Pretraining_Alignment.tex` | §14.5 | current `T3a` (How LLMs Work) |
| **5 MEASURE** — limits, apps, validation (7.5) | T5a | `Lec07_T5a_Limits.tex` | §14.6.1 | current `T3b` (Limits) |
| | T5b | `Lec07_T5b_Applications_Validation.tex` | §14.6.2–4, §14.7 | current `T4` (Applications) |

T-number doubles as the 7.x section number (Lec09 convention). Acts 2, 3, 5 span two decks each; **both decks in an act call the same `\LecSevenMap{k}`** and distinguish themselves with a one-line italic "Route: … (part 1/2 …)" under the map (per Lec09's T-deck pattern).

**Two persistent case threads** (Lec09-style, named in the T1 map-frame footnote and revisited across acts):
- **Thread A — the EPU index** (Baker–Bloom–Davis 2016): the transparent dictionary baseline; the opening hook; returns in T5b as the validation gold standard and in the China parallel (People's Daily EPU, Hartley 2025 19th-c. back-extension).
- **Thread B — central-bank communication / FOMC hawk–dove**: runs through the worked attention example (T3a), the hawk–dove embedding direction (T2b), BERT-vs-GPT (T4), and the monetary-policy-stance application (T5b); China parallel = PBOC Monetary Policy Report.

---

## The recurring spine (chapter14's organizing device)

Make the **text-DGP the visible thread**, shown once in full in T1 and recalled as a one-liner on each act's map frame and each divider where relevant:

```
x_i = g(s_i, a_i, c_i) + ε_i        (observed language = latent state + strategic framing + institutional context)
        │  m(·) = representation      (this is what Acts 2–4 build: counts → embeddings → contextual)
        ▼
y_i = β · m(x_i) + γ z_i + u_i       (the downstream regression — where the two hazards bite)
```

Two hazards restated at each act: **(i) endogeneity** — if framing `a_i` correlates with `u_i`, `m(x_i)` is endogenous; **(ii) non-classical measurement error** — error correlates with the true state, biasing `β̂`. The lecture's thesis, landed on T5b's penultimate frame: **"Identification precedes measurement" (识别先于测量, rendered in English)** — a better text model reduces measurement error but never substitutes for a research design (DID/IV/RD).

Map bottom band = the sandwich zones (analogue of Lec09's HOW-IT-WORKS / HOW-TO-USE-IT-WELL / YOUR-ROLE):
`[Act 1: WHY TEXT = MEASUREMENT]  ·  [Acts 2–4: HOW THE MACHINE READS TEXT]  ·  [Act 5: MEASURE, VALIDATE, DISCLOSE]`, with the DGP one-liner `x=g(s,a,c)+ε → m(x) → y=β·m(x)+γz+u` printed as the caption under the band.

---

## New file: `lec07_map.tex`

Adapt directly from `Lec09_Agentic_AI/lec09_map.tex` (full source already read). Changes:
- **Rename macro** `\LecNineMap` → `\LecSevenMap`; keep the `\ifnum#1=0..5` structure (`k∈{1..5}` highlights the current act with `lecmapcur` = `draw=RedTitle, very thick, fill=orange!20` + blue "you are here ▼"; `k=0` = green all-complete reprise for T5b's close).
- **Five act boxes:** `7.1 FRAME` (what & why — text as measurement), `7.2 REPRESENT` (tokens → vectors), `7.3 ATTEND` (attention & Transformer), `7.4 PRETRAIN` (LMs & alignment), `7.5 MEASURE` (limits, apps, validation), each with its student-question line (`what generates this text?` / `how do we turn it into numbers?` / `how does context get encoded?` / `where does a pretrained model come from?` / `is my measure valid — and defensible?`).
- **Bottom band:** three sandwich zones above; DGP caption underneath. **Density fallback:** the Lec09 map is already dense — if the first visual QA shows the caption crowding the band, keep the DGP one-liner on dividers/bridges only and drop it from the map.
- **Dashed satellites → neighbouring lectures** (not Lec07's own decks, since all 8 are acts): upstream `before 7.1: Lec03–04 gave you neural nets, backprop & PyTorch`; downstream `after 7.5: Lec08 RAG (grounded memory) → Lec09 agents (grounded action) → Lec10 cases` (mirrors chapter14's forward pointers to Ch15/Ch16).
- **Filename must NOT match the build glob `Lec*_T*.tex`** — `lec07_map.tex` satisfies this (like `lec09_map.tex`), so `build_all_topic_decks.sh` never compiles it standalone.

---

## Per-deck plan

Every deck follows the Lec09 skeleton (verified in `shared_preamble.tex` + `Lec09_T1`): `\input{../shared_preamble.tex}` → `\metroset{sectionpage=none}` → `\graphicspath{{./pic/}{../../MiniCourse_8hr/Slides/Module3_LLM_Text/pic/}{../}}` → `\usetikzlibrary{positioning,arrows.meta,shapes,calc}` → `\input{lec07_map.tex}` → `\title[AI for Econ Research]{...\\ Lecture 7.x: ...}` → `\begin{document}` → `\makebeamertitle` → **"Lecture 7 in One Map"** frame calling `\LecSevenMap{act}` (replaces the old "Topic 7.x Agenda") → `\section{}`+`\sectiondivider{}{}` per act-section → house-voice frames → **"Bridge: …"** closer with the persistent arc line. **No `\framesubtitle`**; lead/source lines start with a command, never a brace (`\textit{\footnotesize …}`).

### T1 — FRAME · Text as Economic Measurement `\LecSevenMap{1}` (NEW)
- **Reuse from current `T1a`:** "Text as Economic Data — Big Picture" (TikZ), "Why Economists Care About LLMs", "Three Roles of Text" (`tab:text_roles`), **"An Identification-First View of Text"** (the DGP — promote to the deck's centerpiece), "A Brief History", "The Autoregressive Principle", "Intuition: Summarizing a 10-K", "Intuition: FOMC Stance".
- **Add from §14.1:** the *feasibility-window* figure (`fig:text_feasibility`: storage-cost↓ vs model-scale↑, 2008–2023 — redraw as inline TikZ/pgfplots); the **three classic cases as a unit** — EPU full recipe (`epu_index`, hand-coding audit r≈0.86/0.93), FOMC transparency reform (Hansen–McMahon–Prat 2018), firm-level political risk (Hassan–Hollander, >90% firm-level variance); the **endogeneity thinkbox** (two firms, same `s_i`, different `a_i` → `cautionbox`).
- **China parallel:** China EPU (People's Daily / Guangming Daily, 1949–); data infrastructure CNRDS/CSMAR/Wind vs EDGAR/Compustat/CRSP; SOE-vs-private disclosure-incentive asymmetry as a validity issue.
- **Bridge → Act 2:** BoW's orthogonality can't see "央行 ≈ 联储" → we need representations.

### T2a — REPRESENT (1/2) · Preprocessing & Bag-of-Words `\LecSevenMap{2}`
- **Reuse from current `T1a`:** "Tokenization: First Step", "Vocabulary and the Tokenization Map", "GDP rose by 2%" example, "Subword Tokenization: BPE/WordPiece/SentencePiece" (`tab`), "Tokenization Is a Preprocessing Choice" (Denny–Spirling specification curve), "The Seven Preprocessing Choices".
- **Relocate from current `T1b` (correction — these frames are in T1b, not T1a):** the TF / IDF / TF-IDF worked trio, condensed to ≤2 frames. TF-IDF's home is this deck; T2b keeps only a one-line recall. Fold the look-ahead-bias caveat (`tfidf`) into the relocated frames.
- **Add from §14.2–14.3.1:** Chinese word segmentation (Jieba/THULAC/LTP, user dictionaries) as its own frame; the DTM + **sparsity & orthogonality** two-consequences frame; "when BoW is still the right tool" (transparency/auditability); the tokenization-boundary thinkbox (negation "not sufficient" flips a sign).
- **China parallel:** segmentation is the first modeling decision for Chinese corpora; user dictionaries for policy terms ("稳健"→"prudent" ambiguity); the §14.2.2 **translate-vs-native thinkbox** (machine-translate to English vs. multilingual model on the original — what "稳健"/"适度宽松" lose in translation).
- **Bridge → T2b:** counts are orthogonal; dense embeddings put "hawkish"/"dovish" near each other.

### T2b — REPRESENT (2/2) · Word Embeddings `\LecSevenMap{2}`
- **Reuse from current `T1b`** largely intact **minus the TF/IDF/TF-IDF trio (relocated to T2a; keep only a one-line recall here)**: Word2Vec/Skip-gram/SGNS math + walkthrough, GloVe, 3D toy (`word_examples.png` in `pic/`), **Arrow–Debreu analogy** (already present — matches chapter14 thinkbox), analogies/bias, "Limits of Static Embeddings", "From Embeddings to Macro Variables", "Macro Application Gallery".
- **Add from §14.3.2–3:** `tab:representation_comparison` (count / TF-IDF / Word2Vec / BERT across dim, sparsity, polysemy, cost, econ use); the **two time-series pitfalls** (IDF look-ahead; linguistic regime breaks vs Bai–Perron); the **hawk–dove direction** `u_hike − u_cut` as a construct; the separability-assumption thinkbox.
- **China parallel:** seed-word + nearest-neighbor dictionary expansion on a Chinese policy corpus.
- **Bridge → Act 3:** static vectors can't disambiguate polysemy/negation → context.

### T3a — ATTEND (1/2) · Attention `\LecSevenMap{3}`
- **Reuse from current `T2a`:** **"How the Transformer Works in Plain Language" + "The Transformer as a Function" as the deck openers**, then self-attention key idea, Q/K/V projection + roles, scaled dot-product, softmax, weighted sum, project-back, **"Worked Example Part 1/2"** (the 9-token FOMC numeric walkthrough — matches chapter14's `attention_walkthrough`), how Q/K/V are learned, multi-head. While porting the "Q/K/V at a Glance" TikZ, **fix its arrow semantics** (currently draws weight arrows Q→K and K→V; the Q·K match sets the weight *applied to* V).
- **Add from §14.4.1:** the IR-analogy table (query/key/value ≈ search); the **attention-entropy thinkbox** (Shannon entropy of weights; PBOC party-speak diffuse vs FOMC focused) as an identification aside.
- **Bridge → T3b:** one attention head is a building block; a Transformer layer wraps it with position, residual, and an MLP.

### T3b — ATTEND (2/2) · The Transformer `\LecSevenMap{3}`
- **Reuse from current `T2b`:** residual+LayerNorm, FFN/MLP (**enrich with chapter14's Geva 2021 key-value-memory reading**: FFN ≈ pattern-keyed store holding most factual knowledge, ~2/3 of a layer's parameters — one added bullet, not a new frame), one full layer, stacking L layers, Transformer-vs-Word2Vec, three variants + "picking a variant", "Attention Is All You Need" architecture, the **10-K application arc** (Kim et al. 2024 paragraph-level attention — matches chapter14 §14.4.3 case), token-vs-document embeddings (Lec08 bridge).
- **Reuse from current `T2a` (correction — these frames live in T2a, not T2b):** the positional-encoding trio ("Why They Are Needed", "Sinusoidal/Learned/RoPE", trade-off table); plus **"From Output Embeddings to the Next Token"** and **"Real LLM Scale"** near the close (or as T4 openers — executor's choice). Drop "Inside a Layer (Preview)" — superseded by this deck's full treatment.
- **Add from §14.4.2–3:** the **positional-encoding techbox** (absolute→relative: RoPE, ALiBi — recommend for long docs; keep the RoPE trade-offs the user chose to retain); `tab:transformer_variants` (encoder/decoder/enc-dec by econ use); the **"why Transformer > RNN" techbox** (parallelism + long-range).
- **Bridge → Act 4:** the architecture is fixed; capability comes from *pretraining* it at scale.

### T4 — PRETRAIN · Pretrained LMs, Fine-tuning & Alignment `\LecSevenMap{4}`
- **Reuse from current `T3a`** (already ≈ this act): pretraining (self-supervised), next-token generation, tokens/cost/reproducibility, sampling strategies, prompting (zero/few-shot, ICL), ICL-as-implicit-Bayes, fine-tuning for economics, LoRA intuition+mechanics+workflow, three stages (base→SFT→RLHF), SFT/instruction tuning, RLHF (HHH), Kaplan + Chinchilla scaling, emergence + "mirage" critique, Transformer-vs-LDA, **structured JSON output** (kept per user), three ways to adapt, and **"Revisiting: Why One Equation Enables So Much" as the deck closer before the bridge**. **Refresh the pricing-snapshot frame** with a dated, checked snapshot (the GPT-4o row and the self-contradicting Gemini footnote both need it).
- **Add from §14.5:** the **family-tree figure** (`fig:llm_family_tree`: architecture→families→aligned products — redraw TikZ); `tab:bert_vs_gpt` (MLM/understanding vs CLM/generation, econ angle); `tab:pretrained_model_comparison` incl. **Qwen-7B / DeepSeek-V3**; the alignment econ-analogies — **Bradley–Terry ≙ Luce–McFadden logit**, **KL penalty ≙ ridge λ**; the MLM corpus-choice thinkbox (DAPT on CSMAR MD&A vs generic — "corpus choice is a causal question").
- **China parallel:** local deployment of LLaMA/Qwen/DeepSeek/ChatGLM under PIPL/DSL for micro-data; open-weights ecosystem.
- **Bridge → Act 5:** a capable model is still a fallible instrument — five structural limits first.

### T5a — MEASURE (1/2) · Limits of LLMs `\LecSevenMap{5}`
- **Reuse from current `T3b`:** knowledge cutoff, hallucination risk, context-window limits, **lost-in-the-middle**, prompt sensitivity, plus **"The Brain Metaphor"** as the act's capstone image.
- **Reframe to §14.6.1's "five structural limits,"** each = *architectural root → research implication*: cutoff (static weights), hallucination (max-likelihood over form), O(n²)/context (attention cost), lost-in-the-middle (positional saliency), prompt/primacy (Brucks–Toubia). Capstone with the **YVP (2026) reproducibility thinkbox** (three LLMs score the same task: 3.6× mean gap, κ≈0.36, DiD sign flip) → "model version = measurement instrument; disclose it." **Verify the YVP figures and annotator model names against NBER WP 35110 before authoring** — the current deck chart and chapter14 trace to a single transcription.
- **Bridge → T5b:** knowing the limits, here is what embeddings *can* already measure — and how to defend it.

### T5b — MEASURE (2/2) · Applications & Validation `\LecSevenMap{5}` → close `\LecSevenMap{0}`
- **Reuse from current `T4`** (applications) and the measurement-instrument material from current `T3b`: three featured cases (**ECB Word2Prices** with the Python skeleton, **Kogan patents**, **FinBERT/CentralBankRoBERTa/PaECTER**), why generic LLMs underperform on financial text (Loughran–McDonald), domain-vs-frontier "no single winner", the **2024–25 frontier survey** (central-bank comms + asset pricing — kept), the **AI-exposure instrument comparison** (Felten/Webb/Eloundou three generations + YVP replication), the **application zoo by econometric role** (`tab:embedding_applications`), validation caveats, the **six-step workflow**, disclosure norms, **"When NOT to Use an LLM Rater — A Decision Rule" (current `T3b` — keep; one of the set's strongest frames)**, and the "Grounded Generation" cite-then-claim frame at the close. ("Next Step: Extending the Brain" merges into the closing bridge — its retrain/fine-tune/retrieve triple duplicates "three ways to adapt", kept in T4.)
- **Add from §14.6.2–4 + §14.7:** semantic similarity as economic distance + **cosine's three confounds** (industry/template/length) with the permutation-test thinkbox; the **generated-regressor problem** (bootstrap the whole text→embedding→regression pipeline, not just the last stage); supervised vs **zero-shot classification** (`zeroshot_classification`, SBERT) for label-less concepts (climate risk, AI adoption); the **auditable-LLM-call triple** (fixed seed + SHA-256 of input + archived output, τ=0); the **three starter paths for macro researchers** (§14.7: central-bank stance measurement / EPU discipline / nowcasting) as the practical penultimate frame; the nine key conclusions compressed to a takeaway.
- **China parallel:** corporate-culture / nowcasting / monetary-stance cases on Chinese data; PBOC-report stance measure.
- **Close:** thesis frame **"Identification precedes measurement"** (proxy ≠ causal ID; still need DID/IV/RD — IV example: exogenous CEO turnover; Romer–Romer narrative shocks) → `\LecSevenMap{0}` reprise → **Bridge to Lec08 (RAG/grounded memory) → Lec09 (agents/grounded action) → Lec10 (cases)**, reusing the existing tagline "Language engine (Lec07) + grounded memory (Lec08) + grounded action (Lec09) = auditable economic measurement."

---

## Priority tiers for the new content (cut from Tier 3 upward if time compresses)

The "Add from §14.x" items above total ~25–30 new content frames beyond the new T1 deck — projected ≈155 numbered frames across the 8 decks — so tier them:
- **Tier 1 — must-have:** `lec07_map.tex` + map frames + dividers + bridges; the T1 FRAME deck; the thesis close (T5b); the TF-IDF relocation (T1b→T2a); `tab:representation_comparison`; the family-tree figure; `tab:bert_vs_gpt`; the five-limits reframe (T5a).
- **Tier 2 — high value:** zero-shot classification; the generated-regressor frame; cosine's three confounds + permutation test; the feasibility-window figure; one China parallel per act.
- **Tier 3 — nice-to-have (cut first):** attention-entropy thinkbox; positional-encoding additions beyond the existing RoPE frames; the alignment econ-analogies beyond a one-line mention; the MLM corpus-choice thinkbox.

---

## Format conventions to apply (from `shared_preamble.tex` / Lec09, verified)

- **House voice:** bold lead sentence under each `\frametitle`; `conceptbox` (blue) / `cautionbox` (red) / `examplebox` (green) / `takeawaybox` (gray+blue) for asides; sources named on-slide in a `\textit{\footnotesize Source: …}` body line.
- **`\sectiondivider{Title}{subtitle}`** after each `\section{}` (already in `shared_preamble.tex`); `\metroset{sectionpage=none}` suppresses metropolis's auto section page.
- **Bridge closer** ends every deck: bold recap + forward pointer + persistent arc line `T1 FRAME → T2 REPRESENT → **T3 ATTEND** → T4 PRETRAIN → T5 MEASURE` (current deck bolded).
- **Gotchas:** `RedTitle` is defined as **blue** (RGB 0,0,200) — do not "fix" it. Literal `--` inside `\texttt` must be written `-{-}`. `[fragile]` on any frame with `lstlisting` (the ECB Python skeleton). No watermark exists in the source (memory was stale) — do not add one.
- **Figures:** reuse `pic/transformer-diagram.png` and `pic/word_examples.png` (the only two images; both resolve). Chapter14's six figures are all TikZ with no source images — **redraw inline** (feasibility window, family tree, alignment pipeline are the three worth adding; Q/K/V-roles and transformer-layer already have inline equivalents).

---

## Deck management, manifest, 讲稿

- **Archive** the current 7 `.tex` decks **and their 7 PDFs** into `Lec07_LLM/archive_pre_split/pre_ch14_rebuild_2026-07-07/` (preserve history; do not delete). ⚠ Do **not** invent a new top-level archive folder name: `find -path '*/Lec*/Lec*_T*.tex'` wildcards cross `/` (verified by dry-run), so anything outside the pruned `archive_pre_split/` tree gets re-compiled by `build_all_topic_decks.sh` and its PDFs re-scanned by `qa_page_numbers.sh`. Author the 8 new decks fresh, copying reused frames verbatim so provenance headers can cite the old file.
- **Update `Lec07_manifest.txt`** to the new 8 PDF names in teaching order.
- **讲稿 note:** `Lec07_讲稿.md` will desync heavily. Flag it (like the Lec05/Lec09 desync notes in memory) for regeneration *after* the deck rebuild — **out of scope for this change**, a separate follow-up before Lec07 is taught.

## Build & verification

1. **Build (scoped):** `bash FullCourse_10Lecs/build_all_topic_decks.sh FullCourse_10Lecs/Lec07_LLM` — the script takes ROOT_DIR as `$1`; unscoped it rebuilds all ten lectures and re-stamps every deck's `\date{\today\ \DTMcurrenttime}` footer (~40 PDFs of diff noise). ⚠ Pass a path with at least one leading component (or absolute): a bare `Lec07_LLM` run from inside `FullCourse_10Lecs/` makes `find` emit paths with no `/` before `Lec07_LLM`, the `-path '*/Lec*/...'` pattern matches nothing, and the script silently builds zero decks (observed 2026-07-07). Globs `Lec*_T*.tex`, runs `pdflatex ×2` for the `n/total` footer; `lec07_map.tex` is correctly excluded. Given 8 decks, run **detached via tmux** (long-build convention) and poll.
2. **Compile QA:** all 8 decks must reach `\end{document}` with no fatal errors; scan pass-2 logs for overfull/badness explosions on the new/edited frames.
3. **Visual QA (gs is the only rasterizer here — `pdftoppm`/`pdftocairo` not installed):** `gs -sDEVICE=png16m -r120 -dFirstPage=N -dLastPage=N -o frame_%d.png deck.pdf` on: each deck's map frame (correct act highlighted; T5b shows the `k=0` green reprise), a `\sectiondivider`, a Bridge closer, and the T1 DGP centerpiece. Confirm the "you are here ▼" sits over the right box and the satellites/spine render.
4. **Footer QA (scoped):** `bash FullCourse_10Lecs/qa_page_numbers.sh FullCourse_10Lecs/Lec07_LLM` — no deck's footer may read `/100` (stale-total canary); totals must match actual frame counts.
5. **Content spot-check:** `gs -sDEVICE=txtwrite` extract to confirm the DGP equation, the three-roles table, the thesis line, and the China-parallel asides are present and that no lead line was swallowed as a dropped `\framesubtitle`.

**Not in scope:** regenerating `Lec07_讲稿.md`; publishing to the course website (both are follow-ups once the decks are approved).
