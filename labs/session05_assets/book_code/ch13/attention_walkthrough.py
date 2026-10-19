"""
attention_walkthrough.py — A Self-Contained Self-Attention Computation
======================================================================

Companion to Chapter 14, §14.4 (Attention Mechanism and Transformer
Architecture).  Walks through the scaled dot-product attention for one
token in a short economic sentence, mirroring the worked example in the
chapter text.

What this script does
---------------------
For the Chinese sentence "央行 上调 利率 以 遏制 通胀" (6 tokens), or
for the English parallel "The central bank raised interest rates amid
rising inflation" (9 tokens), this script:

  1) Assigns each token a small toy embedding e^(i) in R^d (here d=8).
  2) Builds randomly initialised Q/K/V projection matrices W_Q, W_K, W_V
     in R^{k x d} (here k=4).
  3) Computes Q_i, K_j, V_j for every position.
  4) Picks one query position (default: "利率" / "interest"), reports
     the 9 unnormalised scores s_j^(i) = (Q_i . K_j) / sqrt(k), the
     softmax weights, and the resulting attention output vector.
  5) Prints the top-3 attended positions with their weights — these
     should concentrate on monetary-policy semantically related tokens
     when the projection matrices are seeded so that this is the case.

The toy weights are designed so the qualitative pattern in the chapter
text shows up.  This is a *teaching demo*, not a real Transformer.

Dependencies: numpy  (no PyTorch / no internet)
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------
# 1. Toy embeddings for two parallel sentences
# ----------------------------------------------------------------------
def chinese_sentence() -> tuple[list[str], np.ndarray]:
    """Token list and toy 8-d embedding matrix for the Chinese sentence."""
    tokens = ["央行", "上调", "利率", "以", "遏制", "通胀"]
    rng = np.random.default_rng(0)
    base = rng.standard_normal((len(tokens), 8))
    # Hand-engineer a "monetary policy" axis: tokens 0, 1, 2, 5
    # get a positive component on dim 0; tokens 3, 4 (filler) get near-zero.
    monetary = np.array([+1.5, +1.2, +1.6, 0.0, +0.2, +1.3])
    base[:, 0] += monetary
    return tokens, base


def english_sentence() -> tuple[list[str], np.ndarray]:
    tokens = ["The", "central", "bank", "raised",
              "interest", "rates", "amid", "rising", "inflation"]
    rng = np.random.default_rng(1)
    base = rng.standard_normal((len(tokens), 8))
    monetary = np.array([0.0, +1.0, +1.4, +1.1,
                         +1.6, +1.6, +0.1, +0.3, +1.5])
    base[:, 0] += monetary
    return tokens, base


# ----------------------------------------------------------------------
# 2. The actual scaled dot-product self-attention
# ----------------------------------------------------------------------
def attention_for_query(
    tokens: list[str],
    e: np.ndarray,
    query_idx: int,
    seed: int = 7,
    k: int = 4,
) -> dict:
    """Compute scores, weights, and attention output for one query token."""
    d = e.shape[1]
    rng = np.random.default_rng(seed)
    # Initialise the three projection matrices.
    W_Q = rng.standard_normal((k, d)) * 0.3
    W_K = rng.standard_normal((k, d)) * 0.3
    W_V = rng.standard_normal((k, d)) * 0.3

    # Encourage Q and K to pick up the "monetary policy" axis (dim 0).
    W_Q[:, 0] += 0.6
    W_K[:, 0] += 0.6

    Q = e @ W_Q.T                          # (n, k)
    K = e @ W_K.T                          # (n, k)
    V = e @ W_V.T                          # (n, k)

    q = Q[query_idx]                       # (k,)
    scores = (K @ q) / np.sqrt(k)          # (n,)
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    attn_vec = weights @ V                 # (k,)

    return {
        "tokens": tokens,
        "query_token": tokens[query_idx],
        "scores": scores,
        "weights": weights,
        "attn_vec": attn_vec,
    }


# ----------------------------------------------------------------------
# 3. Pretty-print one self-attention pass
# ----------------------------------------------------------------------
def report(out: dict) -> None:
    print(f"\nQuery position: {out['query_token']!r}\n")
    header = f"{'pos':<4} {'token':<14} {'score':>8} {'weight':>10}"
    print(header)
    print("-" * len(header))
    order = np.argsort(-out["weights"])
    for rank, idx in enumerate(order):
        marker = "  <-- top-3" if rank < 3 else ""
        print(f"{idx:<4} {out['tokens'][idx]:<14} "
              f"{out['scores'][idx]:>+8.3f} {out['weights'][idx]:>10.3f}{marker}")
    print(f"\nattention output (k=4): {np.round(out['attn_vec'], 3)}")


def main() -> None:
    print("=" * 60)
    print("Chinese sentence — query token: '利率' (interest rate)")
    print("=" * 60)
    tok, e = chinese_sentence()
    out = attention_for_query(tok, e, query_idx=tok.index("利率"))
    report(out)

    print("\n" + "=" * 60)
    print("English sentence — query token: 'interest'")
    print("=" * 60)
    tok, e = english_sentence()
    out = attention_for_query(tok, e, query_idx=tok.index("interest"))
    report(out)


if __name__ == "__main__":
    main()
