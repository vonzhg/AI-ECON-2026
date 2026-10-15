from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk


TOKEN_RE = re.compile(r"[a-z][a-z0-9'-]*|\d+(?:\.\d+)?", re.IGNORECASE)

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "more",
    "most",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "out",
    "over",
    "own",
    "paper",
    "papers",
    "same",
    "say",
    "says",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


@dataclass
class TfidfIndex:
    """A tiny sparse-vector index for classroom RAG.

    This is not a neural embedding index. It is an old-school vector-space
    retriever: every chunk becomes a sparse TF-IDF vector, and search is cosine
    similarity against the query vector.
    """

    idf: dict[str, float]
    vectors: list[dict[str, float]]

    def to_dict(self) -> dict[str, object]:
        return {"idf": self.idf, "vectors": self.vectors}

    @classmethod
    def from_dict(cls, row: dict[str, object]) -> "TfidfIndex":
        vectors = [{str(k): float(v) for k, v in vector.items()} for vector in row["vectors"]]
        idf = {str(k): float(v) for k, v in row["idf"].items()}
        return cls(idf=idf, vectors=vectors)


def normalize_token(token: str) -> str:
    token = token.lower().strip("'")
    if token.endswith("'s"):
        token = token[:-2]
    return token


def tokenize(text: str) -> list[str]:
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("_", " ")
    words = [normalize_token(match.group(0)) for match in TOKEN_RE.finditer(text)]
    words = [word for word in words if len(word) > 1 and word not in STOPWORDS]

    expanded = list(words)
    if "olg" in words:
        expanded.extend(["overlapping", "generations", "overlapping_generations"])
    if "ila" in words:
        expanded.extend(["infinite", "lived", "agent", "infinite_lived", "lived_agent"])
    if "solg" in words:
        expanded.extend(["stochastic", "olg", "stochastic_olg"])
    if "rck" in words:
        expanded.extend(["ramsey", "cass", "koopmans", "ramsey_cass", "cass_koopmans"])
    words = expanded

    bigrams: list[str] = []
    for left, right in zip(words, words[1:]):
        if left not in STOPWORDS and right not in STOPWORDS:
            bigrams.append(f"{left}_{right}")

    return words + bigrams


def l2_normalize(weights: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in weights.values()))
    if norm == 0:
        return weights
    return {term: value / norm for term, value in weights.items()}


def build_tfidf_index(chunks: list[Chunk]) -> TfidfIndex:
    tokenized = [tokenize(chunk.text) for chunk in chunks]
    doc_count = len(tokenized)
    document_frequency: Counter[str] = Counter()

    for terms in tokenized:
        document_frequency.update(set(terms))

    idf = {
        term: math.log((1 + doc_count) / (1 + freq)) + 1.0
        for term, freq in document_frequency.items()
    }

    vectors: list[dict[str, float]] = []
    for terms in tokenized:
        counts = Counter(terms)
        if not counts:
            vectors.append({})
            continue
        max_tf = max(counts.values())
        weights = {
            term: (0.5 + 0.5 * count / max_tf) * idf[term]
            for term, count in counts.items()
        }
        vectors.append(l2_normalize(weights))

    return TfidfIndex(idf=idf, vectors=vectors)


def query_vector(question: str, idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokenize(question))
    if not counts:
        return {}
    max_tf = max(counts.values())
    weights = {
        term: (0.5 + 0.5 * count / max_tf) * idf[term]
        for term, count in counts.items()
        if term in idf
    }
    return l2_normalize(weights)


def dot(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(weight * right.get(term, 0.0) for term, weight in left.items())


def search(
    chunks: list[Chunk],
    index: TfidfIndex,
    question: str,
    *,
    top_k: int = 5,
) -> list[SearchResult]:
    qvec = query_vector(question, index.idf)
    scored = [
        SearchResult(chunk=chunk, score=dot(qvec, vector))
        for chunk, vector in zip(chunks, index.vectors)
    ]
    scored.sort(key=lambda row: row.score, reverse=True)
    return scored[:top_k]


def save_index(index: TfidfIndex, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index.to_dict(), ensure_ascii=False), encoding="utf-8")


def load_index(path: Path) -> TfidfIndex:
    return TfidfIndex.from_dict(json.loads(path.read_text(encoding="utf-8")))
