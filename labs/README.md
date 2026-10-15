# Labs — AI for Economic Research (2026)

Hands-on notebooks, one per **session** (`SessionNN_Lab.ipynb`, with its helpers and data in
`sessionNN_assets/`). Each is posted here before its session, so clone the course once and pull
before every session. Built for students in **China**: no Google/Colab required, and setup uses
the **Tsinghua (TUNA) mirror**. Every Session lab runs **top to bottom, offline, on a laptop CPU**,
with fixed seeds.

## Setup (the same steps as Session 1's last deck, *The Workbench*)

1. Install Python 3.11, git, and VS Code with Microsoft's **Python** and **Jupyter** extensions.
2. Clone the course, then create and activate an environment inside the course folder:
   ```bash
   python3 -m venv .venv                # Windows: py -3.11 -m venv .venv
   source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
   ```
3. Install the packages every Session lab uses, through the mirror:
   ```bash
   python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
       numpy pandas matplotlib scipy scikit-learn torch jupyter ipykernel
   ```
4. Open a notebook in VS Code → **Select Kernel** → the `.venv` environment. Run the first cells of
   the Session 1 lab: they print READY, or name what is missing.

## Notebooks from the Summer 2026 version (Zhejiang University)

These are numbered by **lecture** of the ten-lecture version taught in July 2026; their decks are
on [the archive page](../archive/summer-2026.html).

| Notebook | Lectures | What you do |
|---|---|---|
| `Lec01_02_Lab_Getting_Started.ipynb` | 1–2 | Install Python + PyTorch in VS Code; run your first Python; plot Stanford **AI Index** data — AI vs. human experts, inference cost, adoption speed |
| `Lec03_Lab_ML_Basics.ipynb` | 3 | Train neural networks in PyTorch; predict the 10-year Treasury yield; **build** a hawkish/dovish central-bank text classifier |
| `Lec07_LLM_Lab/Lab7A_Text_as_Data.ipynb` | 7 | Tokens & a toy **BPE**; TF-IDF; a mini **EPU** index; word embeddings (PPMI+SVD) and a hawk–dove score **validated against real rate cycles** — on 30 years of bundled FOMC statements |
| `Lec07_LLM_Lab/Lab7B_Attention_MiniGPT.ipynb` | 7 | The 5-step **attention** formula by hand; causal masking; then train a **minimal GPT from scratch** (0.6M params, ~10 min CPU) and generate FOMC-ese with temperature/top-k |
| `Lec08_RAG_Lab/Lec08_Lab_RAG.ipynb` | 8 | Build a **RAG** pipeline over 405 Ren Zhengfei speeches: chunk → TF-IDF index → retrieve → audit → **cited** grounded answer; **implement** a minimal retriever. Runs offline, stdlib-only |
| `Lec09_Agent_Lab/` | 9 | **Part A** (offline notebook): a session as a list, tools, the agent loop, cost and multi-agent patterns, all against a scripted model. **Part B** and the **Design Lab** need one terminal agent (Claude Code, Codex CLI, or Gemini CLI) and network access — see the folder's README |

The Getting Started notebook sets up a conda environment for these labs; the environment above also
has every package in `requirements.txt`.

## `data/`

Bundled CSVs so the Getting Started lab runs offline:

- `ai_performance_vs_human.csv` — AI MMLU scores vs. the human-expert baseline (89.8%)
- `inference_cost.csv` — USD per million tokens over time
- `adoption_rates.csv` — years to ~50% adoption (PC / internet / generative AI)

**Sources:** Stanford AI Index 2025 (hai.stanford.edu), Epoch AI (epoch.ai), and Our
World in Data (ourworldindata.org). Values are illustrative figures drawn from these
reports for teaching — see each file's `Source` column. The Getting Started lab also
includes a cell that fetches the latest data live from Our World in Data (which is reachable
in China).

## Notes for instructors

These labs reuse ideas from the existing `../Notebooks/` (Lab 2A intro, Lab 5A ML
basics, Lab 5B FOMC text) but are self-contained, China-ready, and exercise-driven.
Verify with:

```bash
jupyter nbconvert --to notebook --execute Lec01_02_Lab_Getting_Started.ipynb
jupyter nbconvert --to notebook --execute Lec03_Lab_ML_Basics.ipynb
jupyter nbconvert --to notebook --execute Lec08_RAG_Lab/Lec08_Lab_RAG.ipynb
jupyter nbconvert --to notebook --execute Lec09_Agent_Lab/Lec09_Lab_Agents.ipynb
```

The **Lecture 8 RAG lab** lives in its own subfolder (`Lec08_RAG_Lab/`) because it
ships a small retrieval engine (`rag_ren.py`) alongside the notebook. Its default
path is **standard-library only** — nothing in `requirements.txt` is needed for it
to run; its embedding section asks for `sentence-transformers` separately.
