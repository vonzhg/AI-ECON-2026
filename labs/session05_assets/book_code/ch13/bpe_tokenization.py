"""
bpe_tokenization.py — Subword Tokenizers on Economic Text
=========================================================

Companion to Chapter 14, §14.2 (Text Preprocessing).  Demonstrates how
BPE / WordPiece / SentencePiece tokenizers split English and Chinese
economic sentences, including the "economic precision" example
"GDP rose by 2%" / "GDP rose by 3.5%" / "GDP rose by 3.6%".

What this script does
---------------------
For each tokenizer family, prints the token IDs and human-readable
token strings for:

    * "GDP rose by 2%"          (5 plain words, single-digit percent)
    * "GDP rose by 3.5%"        (decimal — tokenized into 3 pieces)
    * "GDP rose by 3.6%"        (only the last digit differs!)
    * "央行 将 基准利率 上调 25 个 基点"   (Chinese monetary-policy sentence)
    * "央行 将 基准利率 上调 15 个 基点"   (one digit difference)

Dependencies
------------
At least one of:
    * tiktoken         (for GPT-4 BPE)             -- pip install tiktoken
    * transformers     (for BERT WordPiece)        -- pip install transformers
    * sentencepiece    (for SentencePiece)         -- pip install sentencepiece

If a backend is not installed the script skips it and reports the gap.

Tested with: tiktoken 0.7, transformers 4.40, sentencepiece 0.2
"""

from __future__ import annotations

from textwrap import indent
from typing import Callable


ENGLISH_SAMPLES = [
    "GDP rose by 2%",
    "GDP rose by 3.5%",
    "GDP rose by 3.6%",
]
CHINESE_SAMPLES = [
    "央行将基准利率上调25个基点",
    "央行将基准利率上调15个基点",
]


def tokenize_with(tokenizer_name: str, encode_fn: Callable[[str], list[int]],
                  decode_fn: Callable[[int], str],
                  samples: list[str]) -> None:
    print(f"\n--- {tokenizer_name} ---")
    for s in samples:
        ids = encode_fn(s)
        toks = [decode_fn(i) for i in ids]
        print(f"  {s!r}")
        print(indent(f"ids   = {ids}", "    "))
        print(indent(f"toks  = {toks}", "    "))


def run_tiktoken() -> None:
    try:
        import tiktoken
    except ImportError:
        print("\n(skipping tiktoken — `pip install tiktoken` to enable)")
        return
    enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 BPE

    def encode(s):
        return enc.encode(s)

    def decode(i):
        return enc.decode([i])

    tokenize_with("tiktoken / cl100k_base  (GPT-4 BPE)",
                  encode, decode, ENGLISH_SAMPLES + CHINESE_SAMPLES)


def run_bert_wordpiece() -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("\n(skipping BERT WordPiece — `pip install transformers` to enable)")
        return
    try:
        tok_zh = AutoTokenizer.from_pretrained("bert-base-chinese")
        tok_en = AutoTokenizer.from_pretrained("bert-base-uncased")
    except Exception as exc:  # network failure, no cache, etc.
        print(f"\n(BERT WordPiece could not load: {exc})")
        return

    def en_encode(s):
        return tok_en.encode(s, add_special_tokens=False)

    def en_decode(i):
        return tok_en.convert_ids_to_tokens([i])[0]

    def zh_encode(s):
        return tok_zh.encode(s, add_special_tokens=False)

    def zh_decode(i):
        return tok_zh.convert_ids_to_tokens([i])[0]

    tokenize_with("bert-base-uncased  (English WordPiece)",
                  en_encode, en_decode, ENGLISH_SAMPLES)
    tokenize_with("bert-base-chinese  (Chinese WordPiece, char-level)",
                  zh_encode, zh_decode, CHINESE_SAMPLES)


def run_sentencepiece() -> None:
    try:
        import sentencepiece as spm  # noqa: WPS433
    except ImportError:
        print("\n(skipping SentencePiece — `pip install sentencepiece` to enable)")
        return
    # SentencePiece requires a model file.  Most users have one shipped
    # with their LLaMA / T5 / XLM-R checkpoints.  We skip if absent.
    print("\n(SentencePiece demo requires a model file, e.g. tokenizer.model "
          "from a downloaded LLaMA / T5 / XLM-R checkpoint.  Skipping; "
          "see comments inside this script for the call signature.)")
    # Typical usage:
    # sp = spm.SentencePieceProcessor(model_file="path/to/tokenizer.model")
    # sp.encode("GDP rose by 2%") -> [list of token ids]


def main() -> None:
    print("=" * 70)
    print("Chapter 14, §14.2: Subword Tokenizer Comparison on Economic Text")
    print("=" * 70)
    run_tiktoken()
    run_bert_wordpiece()
    run_sentencepiece()


if __name__ == "__main__":
    main()
