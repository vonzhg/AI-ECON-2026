# Labs — AI/ML for Macroeconomics (2026)

Hands-on notebooks, numbered by **lecture**. Built for students in **China**: no
Google/Colab required, and setup uses **Tsinghua (TUNA) mirrors**. Every notebook runs
**top-to-bottom offline**.

## Notebooks

| Notebook | Lectures | What you do |
|---|---|---|
| `Lec01_02_Lab_Getting_Started.ipynb` | 1–2 | Install Python + PyTorch in VS Code; run your first Python; plot Stanford **AI Index** data — AI vs. human experts, inference cost, adoption speed |
| `Lec03_Lab_ML_Basics.ipynb` | 3 | Train neural networks in PyTorch; predict the 10-year Treasury yield; **build** a hawkish/dovish central-bank text classifier |
| `Lec07_LLM_Lab/Lab7A_Text_as_Data.ipynb` | 7 | Tokens & a toy **BPE**; TF-IDF; a mini **EPU** index; word embeddings (PPMI+SVD) and a hawk–dove score **validated against real rate cycles** — on 30 years of bundled FOMC statements |
| `Lec07_LLM_Lab/Lab7B_Attention_MiniGPT.ipynb` | 7 | The 5-step **attention** formula by hand; causal masking; then train a **minimal GPT from scratch** (0.6M params, ~10 min CPU) and generate FOMC-ese with temperature/top-k |
| `Lec08_RAG_Lab/Lec08_Lab_RAG.ipynb` | 8 | Build a **RAG** pipeline over 405 Ren Zhengfei speeches: chunk → TF-IDF index → retrieve → audit → **cited** grounded answer; **implement** a minimal retriever. Runs offline, stdlib-only |

**Start with the Getting Started lab** — it sets up the environment used by every later lab.

## Setup (summary — full steps are inside the Getting Started notebook)

1. In VS Code, install the **Python** and **Jupyter** extensions.
2. Install **Miniconda** from the Tsinghua mirror: <https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/>
3. Point pip at the mirror (one time):
   ```bash
   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
   ```
4. Create the course environment and install packages:
   ```bash
   conda create -n aiml2026 python=3.11 -y
   conda activate aiml2026
   pip install -r requirements.txt
   ```
5. Open a notebook in VS Code → **Select Kernel** → **aiml2026**.

## `data/`

Bundled CSVs so the Getting Started lab runs offline:

- `ai_performance_vs_human.csv` — AI MMLU scores vs. the human-expert baseline (89.8%)
- `inference_cost.csv` — USD per million tokens over time
- `adoption_rates.csv` — years to ~50% adoption (PC / internet / generative AI)

**Sources:** Stanford AI Index 2025 (hai.stanford.edu), Epoch AI (epoch.ai), and Our
World in Data (ourworldindata.org). Values are illustrative figures drawn from these
reports for teaching — see each file's `Source` column. The Getting Started lab also
includes an **optional** cell that fetches the latest data live from Our World in Data
(which is reachable in China).

## Notes for instructors

These labs reuse ideas from the existing `../Notebooks/` (Lab 2A intro, Lab 5A ML
basics, Lab 5B FOMC text) but are self-contained, China-ready, and exercise-driven.
Verify with:

```bash
jupyter nbconvert --to notebook --execute Lec01_02_Lab_Getting_Started.ipynb
jupyter nbconvert --to notebook --execute Lec03_Lab_ML_Basics.ipynb
jupyter nbconvert --to notebook --execute Lec08_RAG_Lab/Lec08_Lab_RAG.ipynb
```

The **Lecture 8 RAG lab** lives in its own subfolder (`Lec08_RAG_Lab/`) because it
ships a small retrieval engine (`rag_ren.py`) alongside the notebook. Its default
path is **standard-library only** — nothing in `requirements.txt` is needed for it
to run; the optional embedding section asks for `sentence-transformers` separately.
