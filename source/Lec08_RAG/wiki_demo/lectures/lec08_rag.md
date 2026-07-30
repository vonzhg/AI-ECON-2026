# Lecture 8: Retrieval Augmented Generation (RAG)

> Part of: AI/ML for Quantitative Macroeconomics (Spring 2026)
> Instructor: Zhigang Feng
> Prerequisite: [Lecture 7 — LLMs](lec07_llm.md)
> Next: Lecture 9 — Agentic AI

## Overview

This lecture addresses a fundamental limitation of LLMs identified in Lecture 7:
their knowledge is frozen at training time. RAG solves this by fetching relevant
documents at query time and injecting them into the prompt — separating the cost
of ingesting documents from the cost of asking questions.

## Lecture Arc

1. **Why naive prompt engineering fails**: Pasting all documents into the prompt
   hits context-window limits (~200K tokens), costs too much per query, degrades
   quality ("lost in the middle"), and provides no persistence or scalability.

2. **Personal wiki as intuition**: Karpathy's LLM knowledge base — the same
   "separate ingestion from querying" idea, but using LLM-maintained markdown
   files instead of embeddings. Works at ~400K words; breaks at larger scale.

3. **The four-step RAG pipeline**: Chunk -> Embed -> Store/Retrieve -> Generate.

4. **Deep dives**: Chunking strategies (fixed-size, recursive, semantic),
   embedding models (comparison table), vector databases (Chroma, Qdrant, FAISS),
   hybrid retrieval (dense + sparse), re-ranking with cross-encoders.

5. **RAG vs. fine-tuning**: Decision matrix for when to use each approach.

6. **Economic applications**: FOMC minutes, NBER papers, Congressional Record,
   SEC 10-K filings, cross-country central bank speeches.

7. **Failure modes and evaluation**: When RAG fails (retrieval miss, wrong chunk
   boundaries, unfaithful generation) and how to measure quality (RAGAS framework).

## Key Takeaways

1. **Separate indexing from querying**: Documents are chunked, embedded, and
   stored once; each query retrieves only the relevant pieces.

2. **The four-step pipeline**: Chunk -> Embed -> Store/Retrieve -> Generate.

3. **For economists**: RAG turns institutional text (FOMC, NBER, SEC) into a
   queryable research instrument with source attribution — something a vanilla
   LLM cannot provide.

4. **RAG vs. fine-tuning**: RAG is cheaper, updatable, and provides attribution.
   Fine-tuning is better for style/format adaptation. The hybrid pattern
   (fine-tune for domain style + RAG for facts) often wins.

## Economic Applications Covered

- FOMC minutes and press conferences (sentiment over time)
- NBER working papers (literature review at scale)
- Congressional Record and legislative text (policy tracking)
- SEC 10-K filings, Item 1A risk factors (firm-level risk measurement)
- Cross-country central bank speeches (comparative monetary policy)

## Connections to Other Lectures

- **[Lecture 7](lec07_llm.md)**: LLM architecture, token embeddings, context
  windows — the limitations that motivate RAG
- **Lecture 9**: Agentic RAG extends this pipeline with planning loops, tool
  use, and multi-step research workflows

## Key Concepts (linked)

- [Embeddings](../concepts/embeddings.md)
- [RAG](../concepts/rag.md)
- [Transformers](../concepts/transformers.md)
