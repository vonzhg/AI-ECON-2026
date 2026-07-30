# RAG Demo: OLG Models in Macroeconomics

This is a single-notebook classroom demo of retrieval-augmented generation over Spear and Young's OLG paper:

```text
data/source/Spear-Young_OLG_final_preprint.md
```

The demo follows the same teaching style as `M4_T1_rag_ren_demo`: start from the corpus, build chunks, inspect retrieval, construct a grounded prompt, and only then discuss generation.

## Main Artifact

Open and run the notebook with Jupyter:

```bash
./launch_notebook.sh
```

On macOS you can also double-click:

```text
OPEN_NOTEBOOK.command
```

Do not run the notebook with `python demo.ipynb`; `.ipynb` files are JSON notebook documents, not Python scripts. On this machine, also avoid the bare `jupyter` command if it points to pyenv:

```bash
which jupyter
# likely: /Users/zfeng/.pyenv/shims/jupyter
```

If Jupyter is not installed in a local `.venv`, the launcher will print setup instructions. You can always run the notebook-equivalent terminal flow immediately with Apple's system Python:

```bash
/usr/bin/python3 run_demo.py
```

`run_demo.py` executes the same default offline flow as the notebook from a terminal.

The notebook is designed to run top to bottom without API keys or network access. The default path uses:

```text
local markdown -> line-addressable chunks -> TF-IDF retrieval -> extractive grounded answer
```

An optional Claude Code comparison cell is included but skipped by default. Set `ENABLE_LLM = True` in the notebook if you want to compare a naive Claude answer with a retrieval-grounded Claude answer.

## Quick Smoke Tests

From this folder:

```bash
/usr/bin/python3 run_demo.py
/usr/bin/python3 run.py ingest
/usr/bin/python3 run.py ask "What did Diamond 1965 add to the OLG approach?"
PYTHONPATH=src /usr/bin/python3 -m unittest discover -s tests
```

## What Students Should Learn

- RAG is a pipeline, not a magic model setting.
- Retrieval should be audited before generation.
- Citations are an answer contract: claims should trace back to retrieved chunks.
- A sparse TF-IDF retriever is enough to teach the architecture, even though production systems usually use neural embeddings.
- A grounded system should refuse questions not supported by the source paper.

## Notebook Flow

`demo.ipynb` walks through:

1. setup and source audit;
2. chunking the markdown into line-addressable evidence;
3. building a local TF-IDF index;
4. reviewing curated classroom questions;
5. retrieval diagnostics with anchor checks;
6. a single-question walkthrough;
7. the exact grounded prompt;
8. local extractive answering;
9. optional Claude Code naive-vs-RAG comparison;
10. refusal behavior for unsupported questions;
11. a retrieval report written to `build/notebook_retrieval_report.csv`.

## File Structure

```text
rag_olg_demo/
├── README.md
├── AGENTS.md
├── CLAUDE.md
├── demo.ipynb
├── run_demo.py
├── questions.py
├── run.py
├── data/source/Spear-Young_OLG_final_preprint.md
├── src/rag_olg/
│   ├── chunking.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── llm.py
│   └── cli.py
├── tests/test_retrieval.py
└── build/
```

## Optional Claude Code Mode

The optional notebook cell uses local `claude -p`, so it depends on Claude Code being installed and logged in with `/login`. It does not require an Anthropic API key.

The local Claude skill remains available for command-line demos:

```text
/olg-rag How do Spear and Young distinguish OLG from ILA models?
```

For the notebook-first class flow, students do not need to invoke the skill.
