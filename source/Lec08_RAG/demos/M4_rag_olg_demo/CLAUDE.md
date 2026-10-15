# OLG RAG Demo Instructions

This folder is a standalone classroom demo for retrieval-augmented generation over a macroeconomics paper.

The main handout is `demo.ipynb`. It mirrors the style of `M4_T1_rag_ren_demo`: source audit, chunking, TF-IDF retrieval, retrieval diagnostics, grounded prompt, local extractive answer, and optional Claude Code comparison.

## What is portable here

- The source markdown is copied locally at `data/source/Spear-Young_OLG_final_preprint.md`.
- The Claude Code skill is local at `.claude/skills/olg-rag/SKILL.md`.
- The Python retriever uses only the standard library.
- No API key, network access, vector database, or external embedding model is required.

## Important teaching point

The default retriever is not a neural embedding model. It is a sparse TF-IDF vector retriever, which keeps the demo portable and deterministic. It still demonstrates the RAG architecture:

document -> chunks -> vectors -> nearest chunks -> grounded answer with citations

For production semantic RAG, replace the TF-IDF retriever with an embedding model and a vector database.

## Default behavior: no automatic RAG

Do not run retrieval automatically just because the user asks about the paper. The teaching contrast is:

1. Ask the agent normally and observe the ungrounded/default response.
2. Invoke the local skill explicitly with `/olg-rag <question>`.
3. Compare that against retrieved chunks, line citations, and the grounded prompt.

Only run `run.py ask ...` when the user explicitly asks to run RAG/retrieval, asks to use the skill, or invokes `/olg-rag`.

## Common commands

Open the notebook:

```bash
./launch_notebook.sh
```

Run the same default flow from a terminal:

```bash
/usr/bin/python3 run_demo.py
```

Do not run `python demo.ipynb`; use Jupyter for the notebook or `run_demo.py` for terminal execution.

Run ingestion:

```bash
/usr/bin/python3 run.py ingest
```

Ask a question with local retrieval and extractive answering:

```bash
/usr/bin/python3 run.py ask "How do Spear and Young distinguish OLG from ILA models?"
```

Show the prompt that can be passed to Claude Code or Codex:

```bash
/usr/bin/python3 run.py ask "What role did Lucas 1972 play?" --show-prompt
```

Show full retrieved context:

```bash
/usr/bin/python3 run.py ask "Why did ILA become dominant?" --show-context
```

Run tests:

```bash
PYTHONPATH=src /usr/bin/python3 -m unittest discover -s tests
```

On a machine with a healthy `python3`, replace `/usr/bin/python3` with `python3`.

## Agent behavior

For ordinary questions, do not automatically use the RAG pipeline. If the user wants a grounded answer, tell them to invoke `/olg-rag <question>` or explicitly ask you to run the local RAG command. When RAG is invoked, cite retrieved chunks using the `S1`, `S2`, etc. labels or source line numbers printed by the CLI.
