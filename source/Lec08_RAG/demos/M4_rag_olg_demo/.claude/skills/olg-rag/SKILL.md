---
name: olg-rag
description: Manually run the portable local RAG demo for Spear and Young's overlapping-generations macro paper. Use only when explicitly invoked to demonstrate retrieval.
argument-hint: "question about the OLG paper"
disable-model-invocation: true
---

# OLG RAG Skill

Use this skill to answer questions from the local Spear-Young OLG markdown in this folder. This skill is intentionally manual: it should run only when the user invokes `/olg-rag ...`, so students can compare default LLM behavior with retrieval-grounded behavior.

## Workflow

1. Work from the demo root, the folder containing `CLAUDE.md`, `src/`, and `data/source/Spear-Young_OLG_final_preprint.md`.
2. Prefer the local retriever instead of reading the full paper:

```bash
/usr/bin/python3 run.py ask "$ARGUMENTS" --top-k 5 --show-prompt
```

3. Use the retrieved `[S1]`, `[S2]`, etc. passages as the evidence.
4. Answer only from retrieved context. Cite every substantive claim.
5. If the retrieved context is weak or irrelevant, say that the local evidence is insufficient and propose a sharper query.

## Useful variants

Show only retrieval plus the local extractive answer:

```bash
/usr/bin/python3 run.py ask "$ARGUMENTS" --top-k 5
```

Show full retrieved chunks for classroom discussion:

```bash
/usr/bin/python3 run.py ask "$ARGUMENTS" --top-k 5 --show-context
```

Rebuild the local index after editing the source markdown:

```bash
/usr/bin/python3 run.py ask "$ARGUMENTS" --rebuild
```

If `/usr/bin/python3` is not available, use a healthy `python3` executable.

## Constraints

- Do not use the original source path outside this folder.
- Do not require an API key.
- Do not claim this is neural semantic embedding retrieval. The portable default is TF-IDF sparse-vector retrieval.
- Do not invent citations. Use the `S` labels and line citations printed by the tool.
