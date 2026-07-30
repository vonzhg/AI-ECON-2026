# Retrieval Augmented Generation (RAG)

> See also: [Embeddings](embeddings.md), [Transformers](transformers.md)
> Covered in: [Lecture 8](../lectures/lec08_rag.md)
> Origin paper: [Lewis et al. (2020)](../papers/lewis_2020_rag.md)

## Definition

RAG is a system that, before answering a query, **retrieves** relevant documents
from an external knowledge store and supplies them to a language model as
additional context for **generation**.

The LLM is the *reasoner*. The vector store is the *memory*. The retriever is
the *librarian*.

## The Four-Step Pipeline

1. **Chunk**: Split documents into self-contained pieces (~500 tokens each).
   Chunking strategy matters: too large and embeddings become noisy; too small
   and context is lost.

2. **Embed**: Map each chunk to a vector using an embedding model (e.g.,
   OpenAI `text-embedding-3-large`, 3072 dimensions). Use the *same* model
   for indexing and querying.

3. **Store & Retrieve**: Index vectors in a vector database (Chroma, Qdrant,
   FAISS). At query time, embed the question, find the top-K nearest chunks
   via approximate nearest neighbor search.

4. **Augmented Generation**: Insert retrieved chunks into a prompt template
   that instructs the LLM to answer *only* from the provided evidence, cite
   sources, and refuse when evidence is insufficient.

## Why RAG Matters for Economists

| Problem | How RAG Solves It |
|---------|-------------------|
| **Stale knowledge** (training cutoff) | Retrieve post-cutoff documents at query time |
| **Hallucinated citations** | LLM answers from real retrieved text |
| **No source attribution** | Chunk metadata provides document/page citations |
| **Scale** | Index millions of tokens once; query cheaply forever |

## RAG vs. Fine-Tuning

| Dimension | RAG | Fine-Tuning |
|-----------|-----|-------------|
| Cost to update | Low (add new chunks) | High (retrain model) |
| Attribution | Yes (chunk metadata) | No |
| Hallucination risk | Lower (grounded in evidence) | Higher |
| Best for | Factual Q&A over documents | Style/format adaptation |
| Latency | Slightly higher (retrieval step) | Lower |

## RAG vs. Personal Wiki

RAG automates what a personal wiki does manually:

- Wiki: LLM reads index files, picks relevant articles → answers
- RAG: Embedding similarity search finds top-K chunks → answers

The wiki works at ~400K words. RAG scales to millions of documents.
Both separate ingestion from querying — the fundamental insight.

## Connections

- [Embeddings](embeddings.md) are the bridge between text and vector search
- [Transformers](transformers.md) power both the embedding model and the generator
- Agentic RAG (Lecture 9) extends this pipeline with planning loops and tool use

## Sources

- Lecture 8 slides
- Lewis et al. (2020), "Retrieval-Augmented Generation for Knowledge-Intensive
  NLP Tasks," NeurIPS
