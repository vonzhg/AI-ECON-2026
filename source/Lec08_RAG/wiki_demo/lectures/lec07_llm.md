# Lecture 7: Large Language Models (LLMs)

> Part of: AI/ML for Quantitative Macroeconomics (Spring 2026)
> Instructor: Zhigang Feng
> Next: [Lecture 8 — RAG](lec08_rag.md)

## Overview

This lecture covers the architecture and mechanics of large language models —
what they are, how they work, and what they can (and cannot) do for economic
research. The central message: an LLM is a next-token predictor whose knowledge
is frozen in its weights at training time.

## Key Takeaways

1. **An LLM maps token sequences to probability distributions** over the next
   token. Everything else — conversation, reasoning, code generation — emerges
   from this simple objective.

2. **The transformer architecture** (see [Transformers](../concepts/transformers.md)):
   Self-attention allows each token to attend to every other token. This is what
   makes LLMs powerful — and what limits their context window.

3. **Knowledge is parametric**: Everything the model "knows" is encoded in its
   weights, frozen at the end of training. This creates the knowledge-cutoff
   problem addressed in [Lecture 8](lec08_rag.md).

4. **Tokenization matters**: The model doesn't see words — it sees tokens.
   Subword tokenization (BPE) balances vocabulary size with coverage. This
   affects cost (you pay per token) and performance (rare words get split
   into multiple tokens).

5. **The pipeline**: Tokenization -> Token [embeddings](../concepts/embeddings.md)
   -> Transformer layers (attention + feed-forward) -> Output distribution.

## Limitations Identified (Motivating Lecture 8)

- **Knowledge cutoff**: Cannot access information published after training.
- **Hallucination**: Generates plausible but fabricated facts when uncertain.
- **No source attribution**: Cannot tell you where a claim came from.
- **Private data blindness**: Never saw your working papers, internal memos,
  or proprietary datasets.

## Connections

- **Lecture 8 ([RAG](lec08_rag.md))**: Solves the knowledge-cutoff problem
  by retrieving documents at query time
- **Lecture 9 (Agentic AI)**: Uses LLMs as reasoning engines with tool access
- **Lecture 7 concepts used later**: Token embeddings -> document embeddings
  (Lecture 8); context windows -> chunking constraints (Lecture 8)
