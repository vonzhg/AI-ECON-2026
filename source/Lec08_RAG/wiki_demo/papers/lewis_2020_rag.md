# Lewis et al. (2020) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

> Published: NeurIPS 2020
> Authors: Lewis, Perez, Piktus, Petroni, Karpukhin, Goyal, Kuttler, Lewis,
>          Yih, Rocktaschel, Riedel, Kiela
> Relevance: [RAG concept](../concepts/rag.md), [Lecture 8](../lectures/lec08_rag.md)

## Summary

This paper introduced RAG as a general-purpose architecture for combining
**parametric** memory (neural network weights) and **non-parametric** memory
(retrieved documents). The key innovation: instead of relying solely on
knowledge stored in model weights, retrieve relevant passages from a corpus
and condition generation on them.

## Architecture

- **Retriever**: Dense Passage Retrieval (DPR) — encodes queries and documents
  as dense vectors, retrieves via maximum inner product search (MIPS).
- **Generator**: BART (a seq2seq transformer) conditioned on the concatenation
  of the query and retrieved passages.
- **End-to-end training**: Both the retriever and generator are trained jointly,
  so the retriever learns to find passages that help the generator.

## Two Variants

1. **RAG-Sequence**: The same set of retrieved passages is used for generating
   the entire output sequence. Simpler and more common in practice.
2. **RAG-Token**: Different passages can be retrieved for each output token.
   More flexible but computationally more expensive.

## Key Results

- State-of-the-art on open-domain QA benchmarks (Natural Questions, TriviaQA,
  WebQuestions)
- Generated answers are more factual and specific than pure generative models
- The retriever learns to find relevant passages without passage-level supervision
- Outperforms models with 10x more parameters on knowledge-intensive tasks

## Connection to Modern RAG (2025)

The 2020 paper established the *pattern*. The 2025 ecosystem industrialized it:

| 2020 (paper) | 2025 (production) |
|-------------|-------------------|
| DPR retriever | `text-embedding-3-large`, Cohere Embed v3 |
| BART generator | GPT-4, Claude, Gemini |
| FAISS index | Chroma, Qdrant, Pinecone, Weaviate |
| Academic benchmarks | FOMC analysis, legal discovery, medical Q&A |

The core insight is unchanged: **retrieve first, then generate**.

## Relevance to This Course

This paper is the intellectual origin of everything in Lecture 8. The four-step
pipeline (chunk, embed, store/retrieve, generate) is a direct descendant of the
architecture proposed here.

For economists: this paper showed that grounding generation in retrieved evidence
dramatically reduces hallucination — exactly the property needed for research-grade
work with institutional text.

## Citation

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N.,
Kuttler, H., Lewis, M., Yih, W., Rocktaschel, T., Riedel, S., & Kiela, D.
(2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
*Advances in Neural Information Processing Systems*, 33.
