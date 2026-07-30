# Plan: Write `Lec07_LLM.tex` — Large Language Models

## Context

Lecture 7 of a 10-lecture graduate course "AI & Machine Learning for Economists" covers Large Language Models. The directory `2026_New_Slides/Lec07_LLM/` exists but is empty. Per the outline header `### Lec07: Large Language Models (1 hr)` the slot is nominally one hour, but the same outline block and the master `Slides_Plan_2026.md` both state **"Draw from both `16_LLM/Lec11_LLM.tex` AND the book chapters `chapter14.tex` and `chapter15.tex` to fill 2 hours. Target: ~60 slides."** We build for the 2-hour, ~60-slide target.

The instructor's stated pedagogical arc (quoted verbatim from `2026_lecture_outline.md`): *"the foundation of large language model, but this is only a brain, knowledge was a cut off during training. Next step is extend this brain with the help of RAG as we all discuss next topic."* So the lecture must land the Transformer machinery AND set up the RAG motivation for Lec08, without giving away RAG mechanism. Lec02 already makes forward promises ("full treatment in Lecture 7", "Solution: RAG --- Lecture 8") that Lec07 must fulfill.

## File to create

`/Users/zfeng/Library/CloudStorage/OneDrive-Personal/Teachings/AI_ML/Lecture_notes_2026/2026_New_Slides/Lec07_LLM/Lec07_LLM.tex`

## Source material

| File | Role | Key content used |
|---|---|---|
| `2026_New_Slides/Lec02_What_is_AI/Lec02_What_is_AI.tex` | **Preamble template** (copy verbatim) | Has `tcolorbox` + `listings` + `\lstset` already configured; matches the 2026 style guide |
| `2026_New_Slides/Lec01_Quant_Macro/Lec01_Quant_Macro.tex` | Style reference | Comment-barrier convention, image sizing `width=12cm,height=8cm`, agenda/roadmap patterns |
| `16_LLM/Lec11_LLM.tex` (2544 lines) | **Primary content source** | Tokenization (lines 195–318), embeddings (333–1035), Transformer (1434–2175), self-attention (1611–2013), applications (1042–1343, 1964–2515) |
| `面向经济学家的人工智能/chapters/chapter14.tex` | Econ-flavored supplement | Text DGP identification framing, three-roles-of-text table, static-embedding limits, LoRA/DAPT, validation caveats |
| `面向经济学家的人工智能/chapters/chapter15.tex` | Econ-flavored supplement | Lopez-Lira ChatGPT stocks, Hartley 19th-century EPU, Brucks-Toubia primacy bias, six-step workflow |

**Images to reuse** (confirmed to exist):
- `/Users/zfeng/Library/CloudStorage/OneDrive-Personal/Teachings/AI_ML/Lecture_notes_2026/16_LLM/word_examples.png`
- `/Users/zfeng/Library/CloudStorage/OneDrive-Personal/Teachings/AI_ML/Lecture_notes_2026/16_LLM/transformer-diagram.png`

Set `\graphicspath{{../../16_LLM/}}` so these resolve with bare filenames.

## Preamble — copy verbatim from Lec02

Copy lines 1–89 of `Lec02_What_is_AI.tex` exactly, changing only:
- Header comment: `%% Lecture 7: Large Language Models` + `%% Sources: 16_LLM/Lec11_LLM.tex + chapters 14–15`
- `\graphicspath{{../../16_LLM/}}` (instead of `{{../../1-2_intro/}}`)
- Title block: `7: Large Language Models\\[0.5em]`

Do **not** add a `\section{}` macro; use `%===...===` (77 equals) comment barriers as in Lec01/Lec02.

## Slide structure (10 parts, 63 frames, ~126 min)

| Part | Title | Frames | Slots |
|---|---|---|---|
| — | Title + Agenda | 2 | 1–2 |
| I | Why LLMs for Economists? Motivation and History | 5 | 3–7 |
| II | Tokenization: Text to Integers | 5 | 8–12 |
| III | Word Embeddings: TF-IDF → Word2Vec → GloVe | 9 | 13–21 |
| IV | The Transformer: High-Level View | 5 | 22–26 |
| V | Self-Attention: Heart of the Transformer | 10 | 27–36 |
| VI | Full Transformer Layer: Residual, Norm, MLP, Multi-Head | 5 | 37–41 |
| VII | Text Generation and Training (pretrain + fine-tune) | 6 | 42–47 |
| VIII | Economic Applications: Three Case Studies | 9 | 48–56 |
| IX | Limitations: Hallucination, Cutoff, Context, Prompt | 4 | 57–60 |
| X | Transition to Lec08: Extending the Brain with RAG | 2 | 61–62 |
| — | Summary | 1 | 63 |

## Frame-by-frame content map

### Part I — Why LLMs? (frames 3–7)

- **3** `Why Economists Care About LLMs` — text as data (10-K, FOMC, news), applications, course arc preview [Lec11 147–168]
- **4** `Three Roles of Text in Economic Research` — measurement tool / behavioral outcome / treatment variable [chapter14 Table 1.1, unique]
- **5** `A Brief History: RNN → Transformer → GPT` — timeline: Bengio LM 2003, Word2Vec 2013, Vaswani 2017, BERT/GPT 2018, ChatGPT 2022–, refreshed to 2026 landscape [Lec11 125–143 expanded]
- **6** `The Autoregressive Principle` — $P(t^{(n+1)} \mid t^{(1)},\ldots,t^{(n)})$ [Lec11 172–189]
- **7** `An Identification-First View of Text` — DGP $x_i = g(s_i, a_i, c_i) + \varepsilon_i$; why econ grads must think about text as endogenous outcome [chapter14 unique]

### Part II — Tokenization (frames 8–12)

- **8** `Tokenization: The First Step` — raw chars → discrete tokens; why not space-split (`S&P500`, `CO2`, `stagflation`) [Lec11 195–210]
- **9** `Vocabulary and the Tokenization Map` — $\mathcal{K}: A^* \to \{1,\ldots,M\}$, context-length constraints [Lec11 214–277]
- **10** `Illustrative Example: "GDP rose by 2%"` — token IDs; injectivity ("3.5%" vs "3.6%") [Lec11 281–318]
- **11** `Subword Tokenization (BPE/WordPiece)` — OOV handling, `QuantitativeEasing` → subwords [Lec11 238–255]
- **12** `Tokenization as a Preprocessing Choice` — brief note on Denny-Spirling (2018) specification curve (128 preprocessing combos → LDA variation) [chapter14 unique]

### Part III — Embeddings (frames 13–21)

- **13** `Why Embed Tokens into Vectors?` — numeric inputs, semantic space [Lec11 333–357]
- **14** `Three Approaches: TF-IDF, Word2Vec, GloVe` — taxonomy [Lec11 383–426]
- **15** `TF-IDF: The Classical Sparse Approach` — TF, IDF, TF-IDF formulas [Lec11 405–426]
- **16** `Word2Vec: Dense Embeddings from Context` — distributional hypothesis, Skip-gram/CBOW high-level, "king − man + woman ≈ queen" [Lec11 433–499 compressed]
- **17** `Skip-Gram in One Slide: The Math` — one-frame: objective + softmax; SGD details cut [Lec11 555–735 heavily compressed]
- **18** `GloVe: Global Co-occurrence` — $u_i \cdot v_j + b_i + b_j \approx \log X_{ij}$; one-sentence contrast with Word2Vec [Lec11 875–928]
- **19** `3D Toy Example` — image frame: `\includegraphics[width=12cm,height=8cm]{word_examples.png}`; Animal / Royalty / Fruit axes [Lec11 956–977]
- **20** `Analogy: Arrow-Debreu Securities` — semantic states, dot-product as correlation [Lec11 981–999] — *optional; instructor can skip*
- **21** `Limits of Static Embeddings` — polysemy, negation ("GDP growth NOT slowing" ≈ "slowing"), corpus dependence ("mask" pre/post-2020); motivates Transformers [chapter14 unique + Lec11 790–810]

### Part IV — Transformer high-level (frames 22–26)

- **22** `The Transformer as a Function` — $\mathcal{T}: (\mathbb{R}^d)^n \to (\mathbb{R}^d)^n$ [Lec11 1434–1456]
- **23** `Positional Encodings` — $e^{(i)} = \mathcal{E}(t^{(i)}) + p(i)$, sinusoidal vs learned [Lec11 1003–1031]
- **24** `From Output Embeddings to Next Token` — logits, softmax, sampling [Lec11 1460–1485]
- **25** `Inside a Layer (preview)` — two-column frame: bullets left, `transformer-diagram.png` in a `0.5\textwidth` column right [Lec11 1489–1515]
- **26** `Real LLM Scale` — layers, $d$, context length, vocab, params; refresh from GPT-3.5 era to 2026 models (GPT-4/4o, Claude 4, LLaMA-3, DeepSeek) [Lec11 1573–1601 refreshed]

### Part V — Self-attention deep dive (frames 27–36) — the core of the lecture

- **27** `Self-Attention: The Key Idea` — intuition only [Lec11 1635–1653]
- **28** `Step 1 — Project to Q, K, V` — $Q_i = W_Q e^{(i)}$, $K_i = W_K e^{(i)}$, $V_i = W_V e^{(i)}$; question / description / content mnemonic [Lec11 1639–1653]
- **29** `Step 2 — Scaled Dot-Product Scores` — $s_j^{(i)} = \langle Q_i, K_j\rangle/\sqrt{k}$; why $\sqrt{k}$ (gradient stability) [Lec11 1657–1680]
- **30** `Step 3 — Softmax → Attention Weights` — $\alpha_j^{(i)} = \mathrm{softmax}(s_j^{(i)})$; sum to 1; causal mask [Lec11 1684–1701]
- **31** `Step 4 — Weighted Sum of Values` — $\mathrm{AttnVec}_i = \sum_j \alpha_j^{(i)} V_j$ [Lec11 1705–1717]
- **32** `Step 5 — Project Back to d` — $\tilde{e}^{(i)} = W_O\,\mathrm{AttnVec}_i$ [Lec11 1721–1737]
- **33** `Worked Example — Part 1` — "The central bank raised interest rates amid rising inflation"; focus "interest"; show $\langle Q_6,K_7\rangle=2.3$ ("rates"), $\langle Q_6,K_{10}\rangle=1.9$ ("inflation") [Lec11 1741–1805]
- **34** `Worked Example — Part 2` — softmax yields $\alpha_7^{(6)}\approx 0.5$, $\alpha_{10}^{(6)}\approx 0.4$; weighted sum → contextualized "interest" [Lec11 1810–1845]
- **35** `How Q, K, V Are Learned` — backprop through attention; language modeling vs classification loss [Lec11 1877–1912]
- **36** `Multi-Head Attention` — $H$ heads each with own $(W_Q^{(h)},W_K^{(h)},W_V^{(h)})$; finance analogy (one head = policy links, another = temporal phrases, another = hawkish/dovish stance) [Lec11 1988–2013]

### Part VI — Full layer (frames 37–41)

- **37** `Residual + Layer Norm` — $u^{(i)} = \mathrm{Norm}(e^{(i)} + \tilde{e}^{(i)})$; why residuals + LN [Lec11 1935–2067]
- **38** `Feed-Forward Network (MLP)` — $\mathrm{MLP}(x) = W_2\sigma(W_1 x + b_1) + b_2$; $d \to 4d \to d$ expansion [Lec11 1917–1931, 2071–2086]
- **39** `One Layer — Putting It All Together` — flow diagram in text: $e \to \tilde e \to u \to f \to e'$ [Lec11 2089–2116]
- **40** `Stacking L Layers` — output-of-layer = input-of-next; final $e'^{(n)}$ used for prediction [Lec11 2120–2175]
- **41** `Transformer vs Word2Vec` — comparison table (tabular or tcolorbox): static vs contextual, local vs global, $10^7$ vs $10^{11}$ params [Lec11 1387–1422]

### Part VII — Generation, pretrain, fine-tune (frames 42–47)

- **42** `Generating the Next Token` — focus $e'^{(n)}$, project to vocab, softmax, sample, append, repeat [Lec11 2192–2227]
- **43** `Sampling Strategies` — greedy / top-$k$ / top-$p$ (nucleus) / temperature [Lec11 2213–2218 expanded]
- **44** `Transformer vs LDA` — one-slide comparison [Lec11 2232–2278]
- **45** `Pretraining: Self-Supervised LM` — $\mathcal{L} = -\sum_t \log P(x_{t+1} \mid x_{\le t})$ vs Masked LM (BERT); trillions of tokens; Adam [Lec11 2289–2313]
- **46** `Fine-Tuning for Economics` — full FT vs adapters vs LoRA ($W + AB^\top$ with rank-$r$ updates); DAPT (domain-adaptive pretraining); econ task examples [Lec11 2317–2340 + chapter14 unique]
- **47** `Prompt vs Fine-Tune vs RAG (preview)` — three ways to adapt an LLM; one sentence on RAG forward pointer to Lec08. *Bridge slide; fills prompt-engineering coverage gap.*

### Part VIII — Economic applications (frames 48–56)

Featured in depth (3 papers, 6 frames):

- **48** `ECB Word2Prices (Araujo et al. 2025) — Setup` — motivation, data (ECB press conf 2002Q1–2023Q4), BVAR framework
- **49** `ECB Word2Prices — Python Skeleton` — `[fragile]` frame with `\begin{lstlisting}[language=Python, basicstyle=\scriptsize\ttfamily]` showing quarterly Word2Vec retraining + BVAR loop [Lec11 1158–1204]
- **50** `ECB Word2Prices — Results` — BVAR+text beats baseline at 1–4 Q horizons; LLM look-ahead bias caveat; policy implication
- **51** `Patent Text: Kogan et al. (2019)` — TF-IDF + cosine similarity → worker tech exposure → earnings effects [Lec11 1042–1079]
- **52** `CentralBankRoBERTa (Pfeifer & Marohl 2023)` — fine-tuned transformer → agent-level sentiment → predicting inflation/employment
- **53** `PaECTER Patent Embeddings (Ganguli et al. 2024)` — citation-enhanced fine-tuning; patents increasingly diverge semantically

Brief mentions and methodology:

- **54** `The Application Zoo` — one-sentence bullets: Baker-Bloom-Davis EPU (2016), Hoberg-Phillips (2016) product market, Li et al. (2021) company culture, Lopez-Lira & Tang (2023) ChatGPT stock prediction, Hartley (2025) 19th-century EPU [chapter15 unique]
- **55** `Validation Caveats for Econometricians` — industry jargon / template / length confounds in cosine similarity; permutation tests; LM vs Harvard IV-4 dictionary (Loughran-McDonald 2011) [chapter14 unique]
- **56** `A Six-Step Workflow for LLM-Assisted Measurement` — pre-alignment → annotation → gold-standard audit → category reporting → external validity → disclosure (temperature=0, version freeze) [chapter15 unique]

### Part IX — Limitations (frames 57–60) — critical gap-fills

- **57** `The Knowledge Cutoff` — static corpus, fixed date, events after cutoff invisible, refresh-training is expensive. *Key frame for Lec08 setup.*
- **58** `Hallucination Risk` — plausible-but-false text; confabulated citations, wrong statistics; tcolorbox: "looks confident" vs "is correct"
- **59** `Context Window Limits` — even 128k+ context forces chunking on long 10-Ks and FOMC transcripts; attention cost $O(n^2)$
- **60** `Prompt Sensitivity and External Validity` — Brucks-Toubia (2025) primacy bias (GPT-4 picks first option 64–91% when should be 50/50), parrot psychology, WEIRD bias, model version instability [chapter15 unique]

### Part X — Transition to RAG (frames 61–62) — the bridge

- **61** `The Brain Metaphor` — two-column tcolorbox: *What the brain knows* (general language, broad economics, pre-cutoff literature) vs *What it doesn't* (post-cutoff events, your private FOMC corpus, a specific firm's internal docs, tomorrow's data release)
- **62** `Next Step: Extending the Brain with RAG` — trichotomy: retrain (expensive) / fine-tune (not for new facts) / **retrieve** (cheap, dynamic). One-sentence preview of Lec08 (RAG = frozen LLM + queryable store); one-sentence preview of Lec09 (agentic AI lets the brain *act*)

Critical design rule for Part X: **do not mention vector databases, chunking for retrieval, or the RAG architecture itself.** That is Lec08 material. Only name the problem and the conceptual solution.

### Summary (frame 63)

- **63** `Summary` — 6-item enumerate: (i) tokenization + embeddings + positional encoding = input pipeline, (ii) self-attention is the core operation (Q·K → softmax → weighted V), (iii) stacking layers produces contextualized tokens, (iv) generation is autoregressive, (v) three representative econ applications (Kogan, Araujo, CentralBankRoBERTa), (vi) LLMs are brains with a fixed cutoff — next: RAG

## Math depth policy

**Keep in full** (the central mechanics): scaled dot-product attention (frames 29–32), autoregressive loss (frame 45), TF-IDF formulas (frame 15), residual/norm/MLP equations (frames 37–38), the 5-step self-attention derivation.

**Compress**: Skip-gram derivation collapses from ~10 source frames to 2 (frames 16–17); drop the 6-word toy example, the "Why two matrices W and W'?" frame, and the separate CBOW slide. Drop the per-dimension gradient walk-through entirely — econ grads can trust SGD.

**Drop**: Ayyar et al. (2024) gender conformity (4 source frames → 0); second copy of the layer summary; "Why is FFN 4d?" detailed bottleneck comparison; the generic "Why LLMs outperform" frame.

## Key formatting conventions (from Lec01/Lec02)

- `\documentclass[10pt,english,aspectratio=169]{beamer}` + metropolis + professionalfonts
- Colors: `RedTitle` (0,0,200), `VeryLightGray` (245,245,245), `DarkText` (0,0,0)
- Comment-barrier parts (77 `=` signs), no `\section{}`
- Agenda/roadmap: `enumerate` with `\textbf{Topic} --- description`, `\medskip` between items
- Content frames: `itemize` with `\medskip` between items; max 2 levels of nesting
- Images: `\begin{center}\includegraphics[width=12cm,height=8cm]{filename}\par\end{center}`
- Code: `\begin{frame}[fragile]{Title}` with `\begin{lstlisting}[language=Python, basicstyle=\scriptsize\ttfamily]` (only Frame 49)
- Comparison boxes: `columns[T]` with two `0.48\textwidth` `tcolorbox` (green/red for pros/cons, blue for neutral)
- Footer: auto via template already in preamble
- Notation consistency: use $e^{(i)}, \tilde{e}^{(i)}, u^{(i)}, f^{(i)}, e'^{(i)}$ throughout — do not mix in $h_i$ / $x_i$ alternates

## Verification

After writing `Lec07_LLM.tex`:

1. **Compile check** — run `pdflatex` twice (for TOC/refs) in the `Lec07_LLM/` directory:
   ```
   cd "/Users/zfeng/Library/CloudStorage/OneDrive-Personal/Teachings/AI_ML/Lecture_notes_2026/2026_New_Slides/Lec07_LLM"
   pdflatex Lec07_LLM.tex && pdflatex Lec07_LLM.tex
   ```
   Expect a clean PDF; fix any errors.
2. **Image resolution** — open the PDF and verify both `word_examples.png` (Frame 19) and `transformer-diagram.png` (Frame 25) render correctly. If missing, confirm `\graphicspath{{../../16_LLM/}}`.
3. **Frame count** — `\inserttotalframenumber` footer should read ≈63 on the last page. Below 55 = too thin; above 70 = trim Part III or Part VIII.
4. **Agenda on frame 2** — title on frame 1, agenda on frame 2, Part I begins frame 3.
5. **Gap-fill check** — grep the .tex for the four gap terms: "knowledge cutoff" (Frame 57), "hallucination" (Frame 58), "RAG" (Frames 47, 62), "prompt" (Frames 47, 60). All must be present.
6. **Code listing** — Frame 49 must compile (one `[fragile]` frame with `lstlisting`).
7. **Lec02 promise check** — confirm Lec07 delivers on the forward promises Lec02 made (full Transformer treatment in Parts IV–VI; knowledge-cutoff → RAG setup in Parts IX–X).
8. **Visual walkthrough** — scan the compiled PDF page-by-page; verify no frame overflows, notation is consistent, and the 5-step self-attention build-up (frames 27–32) reads cleanly.
