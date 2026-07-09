# Lecture 7 Lab — LLMs & Text as Economic Data

Hands-on companion to **Lecture 7 (Large Language Models)**, in two notebooks:

- **Lab 7A — Text as Data**: `tokenize → count → embed`. What a token is (word-level
  and a toy BPE you train yourself), Bag-of-Words, TF-IDF, a mini EPU-style
  uncertainty index, word embeddings from co-occurrence (PPMI + SVD), a hawk–dove
  stance score — **and an honest validation of where each method works and fails**.
- **Lab 7B — Attention & a Minimal GPT**: `attend → stack → train → generate`. The
  5-step attention formula on the lecture's worked FOMC sentence, causal masking,
  multi-head attention, then a complete **0.6M-parameter GPT trained from scratch**
  on 30 years of FOMC statements — same architecture as GPT-2, small enough to
  read, fast enough for a laptop.

Every formula and worked example comes from the Lecture 7 slides; the notebooks
reproduce the slides' numbers and `assert` them.

Built for students in **China**: fully offline — the data (every FOMC post-meeting
statement, 1994–2026, public domain) is **bundled in `data/fomc/`**. No GPU, no
API key, no network. Packages: `numpy`, `pandas`, `matplotlib` (7A) + `torch` (7B):

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy pandas matplotlib torch
```

## Run it

1. Finish `../Lec01_02_Lab_Getting_Started.ipynb` first (it sets up Python via the
   Tsinghua mirror).
2. Open **`Lab7A_Text_as_Data.ipynb`** in VS Code, select the `aiml2026` kernel,
   and **Run All** (a couple of minutes).
3. Then **`Lab7B_Attention_MiniGPT.ipynb`**. Training the mini-GPT takes
   **~10 minutes on CPU**; set `FAST_MODE = True` in Part 4 for a ~2-minute demo run.

Optional command-line sanity check (no notebook needed):

```bash
python3 fomc_data.py
```

## What you do

**Lab 7A — Text as Data**

| Part | You learn / do |
|---|---|
| 1 | Word-level tokenizer + vocabulary; **train a toy BPE** on the corpus; ✏️ subwords vs OOV |
| 2 | Document–term matrix, TF, IDF, TF-IDF — the slides' worked examples, asserted; ✏️ what TF-IDF does to similarity |
| 3 | A mini **EPU-style uncertainty index** (and its blind spot: the March 2020 statement scores *zero*) |
| 4 | Word embeddings from **PPMI + SVD**; nearest neighbors; a hawk–dove direction — **validated against real rate cycles, where it fails**; ✏️ beat it with a 10-line regex |
| 5 | δ_t = how much each statement changed — "markets read the diff", quantified |

**Lab 7B — Attention & a Minimal GPT**

| Part | You learn / do |
|---|---|
| 1 | The 5-step attention formula on the lecture's 9-token FOMC sentence; ✏️ implement `scaled_dot_attention` |
| 2 | Causal masking (why GPT can't peek), multi-head, attention entropy; ✏️ `causal_mask` |
| 3 | A minimal GPT in PyTorch, every hyperparameter mapped to lecture notation (d, k, H, L, n, M) |
| 4 | Train it: autoregressive loss from ln M ≈ 4.6 down to ~1.3 nats/char, live samples along the way |
| 5 | Generate FOMC-ese at temperature 0.5 / 0.8 / 1.2; inspect what the heads learned; ✏️ top-k sampling |

## Files

```text
Lec07_LLM_Lab/
├── Lab7A_Text_as_Data.ipynb        # start here
├── Lab7B_Attention_MiniGPT.ipynb   # then this
├── fomc_data.py                    # shared corpus loader (pure stdlib)
├── README.md
└── data/
    ├── README.md                   # provenance & regeneration notes
    └── fomc/                       # 245 statements, fomc_YYYYMMDD.txt (~580 KB)
```

## Where the data comes from

`fomc_data.py` finds the statement folder automatically, first hit wins:

1. `$FOMC_DATA_DIR` (if set);
2. the bundled `data/fomc/` next to the notebooks (works out of the box).

FOMC statements are U.S. government works (public domain), fetched from
federalreserve.gov on 2026-07-09 — see `data/README.md` for the full provenance
and how to regenerate or extend the corpus.

## Notes for instructors

- The two labs are sequenced but self-contained; 7A is numpy-only, 7B introduces
  PyTorch (students saw it in Lecture 3's lab).
- Deliberate teaching arcs: the uncertainty dictionary **misses March 2020**
  (dictionaries count words, not situations); the embedding hawk–dove score
  **fails validation** on the 2004–06 cycle ("removing policy *accommodation*" —
  static vectors can't read direction), and a targeted regex then beats it —
  the lecture's "identification precedes measurement" thesis, live on real data.
  7B's generated text is fluent but invents policy content — the hallucination
  bridge into the Lecture 8 RAG lab.
- The committed notebooks include full-run outputs (loss curve ≈ 4.6 → ~1.3
  nats/char), so the decks can quote them without re-running.
