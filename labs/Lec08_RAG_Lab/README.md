# Lecture 8 Lab — Retrieval-Augmented Generation (RAG)

Hands-on companion to **Lecture 8 (RAG)**. You build the full RAG pipeline —
`corpus → chunk → index → retrieve → audit → grounded answer` — over a corpus of
**405 public Ren Zhengfei (任正非) speeches and interviews, 1994–2019**, and see
why grounding an answer in retrieved evidence beats a naive model answer.

Built for students in **China**: the **default path uses only the Python standard
library** — no GPU, no API key, no network, no extra packages. Optional sections
(semantic embeddings, Claude Code) are clearly marked and skip gracefully.

There are **two notebooks** in this folder:

- **`Lec08_Lab_RAG.ipynb`** — the main RAG lab (Ren Zhengfei corpus), below.
- **`Lec08_Lab_MiniGPT_RAG.ipynb`** — *Part B*: wire a **from-scratch mini-GPT**
  (the Lab 7B model) to a RAG over the FOMC corpus and compare the answer **with vs
  without RAG** — then *measure* why a tiny generator can't read its own context.
  See [Part B](#part-b--your-own-mini-gpt--rag) below. Needs `torch` (Lab 7B's dep).

## Run it

1. Finish `../Lec01_02_Lab_Getting_Started.ipynb` first (it sets up Python via the
   Tsinghua mirror).
2. Open **`Lec08_Lab_RAG.ipynb`** in VS Code, select the `aiml2026` kernel, and
   **Run All**. It runs top-to-bottom offline.

Optional command-line sanity check (no notebook needed):

```bash
python3 -c "import rag_ren, questions; \
  idx, chunks, docs = rag_ren.build_index(verbose=True); \
  print('chunks:', len(chunks))"
```

## What you do

| Step | You learn / do |
|---|---|
| 1 | Audit the knowledge base (how many docs, which years) |
| 2 | Chunk the corpus and build a transparent **TF-IDF** index |
| 3–4 | Read the ten discussion questions; **audit retrieval quality** before any generation |
| 5 | Walk through one question's retrieved evidence; **write your own question** |
| 6 | Produce a **grounded answer with `[year · title]` citations** (offline, no LLM) |
| 7 | **✏️ Implement `overlap_search`** — a minimal retriever — and compare it to TF-IDF |
| 8 | *(optional)* semantic **embedding** retrieval — where keywords fail |
| 9 | *(optional)* **naive LLM vs RAG** comparison via local Claude Code |

The notebook's narration is English; the ten questions are shown **bilingually**
(Chinese + English gloss). The Chinese question strings are kept verbatim because
they are the *retrieval queries* against a Chinese corpus — translating them
would break TF-IDF matching.

## Part B — your own mini-GPT + RAG

`Lec08_Lab_MiniGPT_RAG.ipynb` fuses **Lab 7B** (a 0.6M-parameter GPT you train from
scratch) with **RAG**, both over the FOMC statement corpus (bundled next door in
`../Lec07_LLM_Lab/data/`). The point is the honest experiment behind the RAG hype:

| Part | You do |
|---|---|
| 0 | Split the corpus: the model **trains on part**, the RAG indexes **all** of it — so held-out meetings are facts the weights never saw |
| 1 | Train the mini-GPT (context widened 128→256 to hold evidence); watch it **hallucinate** rate numbers |
| 2 | Build a transparent **TF-IDF** RAG over every statement; audit retrieval |
| 3 | Compare the **answer without RAG** (fluent, fabricated, no source) vs **with RAG** (a retrieved sentence + `[date]` citation) — then **measure** whether the model can read the evidence in its context |
| 4 | Honest limits + ✏️ exercises |

**The finding students measure:** a 0.6M char-model has **no in-context copying
(induction) ability**, so it can't lift a fact out of retrieved context — grounding
therefore comes from *retrieval + citation*, and *using* the context is a capability
that only appears with **scale** (ties back to Lab 7B's model-size table). Runs
**offline on CPU**; full training ~12 min (set `FAST_MODE = True` for ~3 min). Needs
`torch numpy matplotlib`.

## Files

```text
Lec08_RAG_Lab/
├── Lec08_Lab_RAG.ipynb            # the main RAG lab (start here)
├── Lec08_Lab_MiniGPT_RAG.ipynb    # Part B: from-scratch mini-GPT + RAG (needs torch)
├── rag_ren.py                     # corpus loading, chunking, TF-IDF + optional backends
├── questions.py                   # ten curated business-strategy discussion questions
├── README.md
└── build/                         # generated diagnostics / caches (created on first run)
```

## Where the corpus comes from

`rag_ren.py` finds the corpus automatically, first hit wins:

1. `$REN_CORPUS_DIR` (if set);
2. `./corpus/` next to the notebook (drop a copy here for a fully standalone lab);
3. the shared in-repo corpus at `MiniCourse_8hr/demos/Ren-master/`.

With the full course repo, option 3 works out of the box — nothing to download.
To ship this lab folder by itself, copy `Ren-master/` in as `corpus/`.

## Optional extras (safe to skip)

- **Semantic embeddings** (Step 8): `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple sentence-transformers`
  — first run downloads a ~120 MB multilingual model; whether that reaches you in
  China depends on your network. The lab skips this section cleanly if it is absent.
- **Claude Code comparison** (Step 9): off by default. Set `ENABLE_LLM = True` only
  after running `claude` in a terminal and doing `/login`.

## Notes for instructors

Derived from `MiniCourse_8hr/demos/M4_rag_ren_demo/` and adapted into an
exercise-driven, China-ready lab. Teaching point: **RAG changes the *evidence*
available to the model, and constrains the answer to an auditable corpus** — it is
not about making the model "smarter." The appendix contains the `overlap_search`
solution; the visible gap between naive token-overlap and TF-IDF motivates IDF.
