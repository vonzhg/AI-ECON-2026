from __future__ import annotations

import re

from .retrieval import SearchResult, tokenize


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def build_prompt(question: str, retrieved: list[SearchResult]) -> str:
    context_blocks = []
    for idx, item in enumerate(retrieved, start=1):
        chunk = item.chunk
        context_blocks.append(
            "\n".join(
                [
                    f"[S{idx}] {chunk.citation}",
                    f"Section: {chunk.section}",
                    f"Retrieval score: {item.score:.3f}",
                    chunk.text,
                ]
            )
        )

    context = "\n\n---\n\n".join(context_blocks)
    return f"""You are answering a macroeconomics question using retrieved evidence from Spear and Young's OLG paper.

Rules:
- Answer only from the retrieved context.
- Cite every substantive claim with source labels such as [S1] or [S2].
- If the retrieved context is insufficient, say so directly.
- Do not use outside knowledge.

Retrieved context:
{context}

Question:
{question}

Grounded answer:"""


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for piece in SENTENCE_RE.split(text):
        piece = re.sub(r"\s+", " ", piece).strip()
        if not piece:
            continue
        if piece.count("|") >= 2:
            continue
        if piece.startswith("![]"):
            continue
        sentences.append(piece)
    return sentences


def extractive_answer(
    question: str,
    retrieved: list[SearchResult],
    *,
    min_top_score: float = 0.035,
    max_sentences: int = 4,
) -> str:
    question_terms = set(tokenize(question))
    retrieved_terms = set(tokenize(" ".join(item.chunk.text for item in retrieved)))
    coverage = (
        len(question_terms & retrieved_terms) / len(question_terms)
        if question_terms
        else 0.0
    )

    if (
        not retrieved
        or retrieved[0].score < min_top_score
        or (retrieved[0].score < 0.075 and coverage < 0.35)
    ):
        return (
            "I do not have enough retrieved evidence from the OLG paper to answer "
            "that question. Try a question using terms from the paper, such as OLG, "
            "ILA, Samuelson, Diamond, Lucas, Barro, or stochastic OLG."
        )

    candidates: list[tuple[float, int, str]] = []

    for source_idx, item in enumerate(retrieved, start=1):
        for sentence in split_sentences(item.chunk.text):
            sentence_terms = set(tokenize(sentence))
            overlap = len(question_terms & sentence_terms)
            if overlap == 0:
                continue
            score = overlap + 2.5 * item.score
            candidates.append((score, source_idx, sentence))

    if not candidates:
        return (
            "The retriever found nearby passages, but the local extractive answerer "
            "could not identify a well-supported answer. Run again with --show-prompt "
            "and ask Claude Code or Codex to synthesize from the retrieved context."
        )

    candidates.sort(key=lambda row: row[0], reverse=True)

    selected: list[str] = []
    seen = set()
    for _, source_idx, sentence in candidates:
        normalized = re.sub(r"\W+", "", sentence.lower())
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(f"{sentence} [S{source_idx}]")
        if len(selected) >= max_sentences:
            break

    return " ".join(selected)
