# Transformers

> See also: [Embeddings](embeddings.md)
> Covered in: [Lecture 7](../lectures/lec07_llm.md)
> Origin paper: [Vaswani et al. (2017)](../papers/vaswani_2017.md)

## What Is a Transformer?

A transformer is a neural network architecture based on **self-attention** — the
ability of each token in a sequence to attend to every other token. Introduced
in 2017, it replaced recurrent (LSTM/GRU) and convolutional architectures for
most NLP tasks within two years.

## Key Components

1. **Token embeddings**: Map discrete token IDs to continuous vectors in R^d.
2. **Positional encoding**: Inject sequence-order information, since attention
   is permutation-invariant by default.
3. **Self-attention layers**: Each token computes a weighted average over all
   other tokens' representations, using learned Query/Key/Value projections.
4. **Multi-head attention**: Multiple attention "heads" run in parallel, each
   learning different relationship patterns.
5. **Feed-forward layers**: Pointwise transformations applied after attention.
6. **Layer normalization**: Stabilizes training across deep stacks of layers.

## The Attention Formula

For a single head:

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

- Q (query), K (key), V (value) are linear projections of the input.
- The softmax produces attention weights — how much each token attends to
  every other token.
- d_k is the key dimension (scaling prevents softmax saturation).

## Why It Matters for Economists

The transformer is the engine inside every modern LLM (GPT-4, Claude, Gemini).
Understanding its architecture helps you reason about:

- **Context windows**: Attention is O(n^2) in sequence length, creating a hard
  cap on how much text the model can process at once.
- **"Lost in the middle"**: Attention tends to focus on the start and end of
  the input, under-weighting information in the middle of long contexts.
- **Scaling laws**: Larger models (more parameters, more data) systematically
  improve performance — the empirical regularity driving the LLM revolution.

## Connections

- [Embeddings](embeddings.md) are the transformer's input representation
- [RAG](rag.md) compensates for the transformer's fixed context window
  and frozen parametric knowledge
- Lecture 9 (Agentic AI) uses the transformer as a reasoning engine with tools

## Sources

- Lecture 7 slides
- Vaswani et al. (2017), "Attention Is All You Need," NeurIPS
