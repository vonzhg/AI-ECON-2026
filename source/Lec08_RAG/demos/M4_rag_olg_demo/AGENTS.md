# AGENTS.md

This folder is a standalone RAG classroom demo over Spear and Young's OLG paper.

The primary classroom artifact is now `demo.ipynb`, a single step-by-step notebook.

## Portable Assets

- Source document: `data/source/Spear-Young_OLG_final_preprint.md`
- Notebook: `demo.ipynb`
- Curated questions: `questions.py`
- Local Claude Code skill: `.claude/skills/olg-rag/SKILL.md`
- Knowledge indexes: `knowledge/data_structure.md`, `knowledge/olg/data_structure.md`
- No external source path is required.

## Retriever

The default retriever is local TF-IDF sparse-vector retrieval, not neural embeddings.

This is intentional for portability:

- no API key
- no network
- no vector database
- no external Python packages

It still demonstrates the RAG architecture: chunk, vectorize, retrieve, construct a grounded prompt, answer with citations.

## Default Agent Behavior

Do not use RAG automatically. This folder is designed to show students why explicit retrieval matters:

1. First, let the agent answer normally if the user asks a plain question.
2. Then, when the user explicitly asks for RAG or invokes the local skill, run the retriever.
3. Compare the default answer with retrieved, cited evidence.

Only run `/usr/bin/python3 run.py ask ...` when the user explicitly asks to run the RAG demo, asks for retrieval, or asks to use the local skill.

## Commands

Use the root wrapper script:

```bash
/usr/bin/python3 run.py ingest
/usr/bin/python3 run.py ask "How do Spear and Young distinguish OLG from ILA models?"
/usr/bin/python3 run.py ask "What role did Lucas 1972 play?" --show-prompt
/usr/bin/python3 run.py ask "Why did ILA become dominant?" --show-context
PYTHONPATH=src /usr/bin/python3 -m unittest discover -s tests
```

For class, prefer:

```bash
./launch_notebook.sh
```

If Jupyter is not being used, run the notebook-equivalent terminal flow:

```bash
/usr/bin/python3 run_demo.py
```

Do not run `python demo.ipynb`; notebooks are not Python scripts, and this machine's `python` may point to a broken pyenv shim.

If the machine has a healthy `python3`, `python3 run.py ...` is also fine.

## Agent Instructions

When RAG is explicitly invoked:

1. Use the local retriever before reading large parts of the markdown.
2. Cite retrieved source labels (`S1`, `S2`, etc.) or line citations.
3. Do not invent claims not supported by retrieved chunks.
4. If retrieval is weak or irrelevant, say the evidence is insufficient and suggest a sharper query.
