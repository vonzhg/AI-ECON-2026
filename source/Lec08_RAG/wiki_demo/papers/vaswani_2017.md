# Vaswani et al. (2017) — Attention Is All You Need

> Published: NeurIPS 2017
> Authors: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin
> Relevance: [Transformers](../concepts/transformers.md), [Lecture 7](../lectures/lec07_llm.md)

## Summary

This paper introduced the **transformer** architecture — a neural network based
entirely on attention mechanisms, dispensing with recurrence and convolutions.
It is the architectural foundation of every modern large language model.

## Key Innovations

1. **Self-attention**: Each position in a sequence attends to all other positions,
   computing relevance weights via learned Query, Key, and Value projections.
   This replaces the sequential processing of RNNs with parallel computation.

2. **Multi-head attention**: Multiple attention "heads" run in parallel, each
   learning different types of relationships (syntactic, semantic, positional).
   Outputs are concatenated and linearly projected.

3. **Positional encoding**: Since attention is permutation-invariant, sinusoidal
   functions inject sequence-order information so the model knows which token
   comes first.

4. **Encoder-decoder architecture**: The original transformer has an encoder
   (for input) and a decoder (for output). Modern LLMs (GPT, Claude) use
   decoder-only variants.

## Impact

- Enabled scaling to billions of parameters (GPT-2/3/4, BERT, Claude, Gemini)
- Made parallel training feasible (unlike sequential RNNs), unlocking GPU
  utilization
- Created the architectural foundation for the 2023-2026 LLM revolution
- Spawned an entire ecosystem: BERT (encoder-only), GPT (decoder-only),
  T5 (encoder-decoder)

## Relevance to This Course

Understanding the transformer is prerequisite for understanding:
- Why **context windows** exist (attention is O(n^2) in sequence length)
- How **embeddings** work (the first layer of a transformer)
- Why **RAG** is necessary (the model's knowledge is frozen in weights
  learned during training — no mechanism to update them at inference time)
- Why **"lost in the middle"** happens (attention weight distribution)

## Citation

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N.,
Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. *Advances in
Neural Information Processing Systems*, 30.
