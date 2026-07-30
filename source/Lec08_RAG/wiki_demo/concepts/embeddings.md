# Embeddings

> See also: [Transformers](transformers.md), [RAG](rag.md)
> Covered in: [Lecture 7](../lectures/lec07_llm.md), [Lecture 8](../lectures/lec08_rag.md)

## What Are Embeddings?

An **embedding** is a mapping from discrete objects (words, sentences, documents)
to dense vectors in R^d, where geometric proximity encodes semantic similarity.

- "monetary policy" and "interest rate decision" map to nearby vectors, even
  though they share no words.
- Every input, regardless of length, maps to the same d-dimensional vector
  (typical d = 768 or 1536).

## Two Roles in This Course

### 1. Token Embeddings (Lecture 7 — Inside the LLM)

The first layer of a transformer maps each token ID to a learned vector. These
embeddings are the model's "vocabulary" in continuous space. They are trained
end-to-end as part of the language model and are never used directly by the user.

### 2. Document/Chunk Embeddings (Lecture 8 — RAG Retrieval)

A separate embedding model (e.g., `text-embedding-3-large`) maps entire text
chunks to vectors. These are stored in a vector database and used for similarity
search at query time.

**Key difference:** Token embeddings are internal to the LLM. Document embeddings
are external tools used by the RAG pipeline.

## Why Geometry Matters

If two FOMC statements discuss similar themes (say, labor market tightness), their
embedding vectors will be close in R^d — even if they use different words. This is
what makes semantic search possible: you search by meaning, not by keywords.

**Contrast with keyword search (TF-IDF, BM25):**
- Keyword search: "inflation expectations" matches only documents containing those
  exact words.
- Embedding search: "inflation expectations" also matches documents about "price
  stability outlook" or "anticipated CPI trajectory."

## Economic Application

Embedding FOMC statements lets you measure the "distance" between any two meetings'
language — a continuous, high-dimensional sentiment measure that goes beyond simple
keyword counting or dictionary-based approaches.

## Sources

- Lecture 7 slides, "Token Embeddings" section
- Lecture 8 slides, "From Text to Vectors" section
- Mikolov et al. (2013), "Efficient Estimation of Word Representations in Vector Space"
