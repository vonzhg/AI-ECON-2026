#!/usr/bin/env python3
"""RAG demo on the Ren Zhengfei (任正非) speech corpus (1994-2019).

Pedagogical goal: show why RAG beats a naive LLM on a Chinese-language,
partially out-of-training corpus. The pipeline mirrors demos/rag_demo/rag_pipeline.py
but adds:

1. Multi-file Markdown corpus loader (walks Ren-master/**/*.md).
2. Chunks carry source filename + year so the answer can cite "[2019-06-24
   FT采访]" rather than an opaque chunk id.
3. Two retrieval backends:
     - tfidf  : zero dependency, no API key, deterministic. Good for smoke tests.
     - embed  : OpenAI-compatible embeddings, semantic. Cached to disk after the
                first build so re-runs are free.
4. naive_llm_answer(): the baseline LLM answer with no retrieval, used to make
   the RAG-vs-naive contrast obvious.
5. compare(): one call returns both sides for the notebook.

Run (CLI, no API key):
    python rag_ren.py --question "什么是灰度？" --llm-backend extractive

Run (LLM generation through Claude Code subscription):
    python rag_ren.py --question "海思备胎战略起源？" --llm-backend claude-code
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Chunk:
    """A text chunk that can be retrieved and cited."""

    chunk_id: int
    text: str
    source_file: str           # e.g. "2019/20190521深圳答记者问.md"
    year: int                  # e.g. 2019
    title: str                 # filename without date prefix and .md suffix
    start_char: int
    end_char: int


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


# --------------------------------------------------------------------------- #
# Step 1: corpus loading
# --------------------------------------------------------------------------- #


SKIP_TOPLEVEL = {"README.md", "SUMMARY.md", "R&D.md"}
DATE_PREFIX_RE = re.compile(r"^(\d{4})(\d{2})?(\d{2})?_?")


def parse_filename(rel_path: str) -> tuple[int, str]:
    """Return (year, human-readable-title) from a path like '2019/20190521深圳答记者问.md'."""
    parts = rel_path.split("/")
    year_dir = parts[0]
    fname = parts[-1]
    name = fname[:-3] if fname.endswith(".md") else fname
    m = DATE_PREFIX_RE.match(name)
    title = name[m.end():] if m else name
    try:
        year = int(year_dir)
    except ValueError:
        year = 0
    return year, title or name


def load_corpus(corpus_dir: Path) -> list[tuple[str, int, str, str]]:
    """Walk corpus_dir, return list of (rel_path, year, title, text).

    Skips top-level metadata files (README/SUMMARY/R&D) so only speeches are
    indexed.
    """
    out: list[tuple[str, int, str, str]] = []
    for md in sorted(corpus_dir.rglob("*.md")):
        rel = md.relative_to(corpus_dir).as_posix()
        if "/" not in rel and md.name in SKIP_TOPLEVEL:
            continue
        text = md.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        year, title = parse_filename(rel)
        out.append((rel, year, title, text))
    return out


# --------------------------------------------------------------------------- #
# Step 2: chunking
# --------------------------------------------------------------------------- #


def split_document_into_chunks(
    text: str,
    source_file: str,
    year: int,
    title: str,
    starting_id: int,
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[Chunk]:
    """Split one document into overlapping character chunks (CJK-safe)."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("invalid chunk parameters")

    chunks: list[Chunk] = []
    start = 0
    cid = starting_id
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            nl = text.rfind("\n", start, end)
            if nl > start + chunk_size // 2:
                end = nl
        body = text[start:end].strip()
        if body:
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    text=body,
                    source_file=source_file,
                    year=year,
                    title=title,
                    start_char=start,
                    end_char=end,
                )
            )
            cid += 1
        if end >= n:
            break
        prev = start
        start = end - overlap
        if start <= prev:
            start = end
    return chunks


def build_chunks(
    docs: list[tuple[str, int, str, str]],
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[Chunk]:
    """Chunk every document; assign globally unique chunk_ids."""
    chunks: list[Chunk] = []
    next_id = 1
    for rel, year, title, text in docs:
        new = split_document_into_chunks(
            text, rel, year, title, next_id, chunk_size=chunk_size, overlap=overlap
        )
        chunks.extend(new)
        next_id += len(new)
    return chunks


# --------------------------------------------------------------------------- #
# Tokenizer (used by both TF-IDF retrieval and extractive answer)
# --------------------------------------------------------------------------- #


def tokenize(text: str) -> list[str]:
    """CJK unigram + bigram, plus English/numeric word tokens."""
    text = text.lower()
    tokens: list[str] = []
    for m in re.finditer(r"[一-鿿]+|[a-z0-9_]+", text):
        piece = m.group(0)
        if re.fullmatch(r"[一-鿿]+", piece):
            tokens.extend(list(piece))
            tokens.extend(piece[i : i + 2] for i in range(len(piece) - 1))
        else:
            tokens.append(piece)
    return tokens


# --------------------------------------------------------------------------- #
# Step 3a: TF-IDF backend (no API key)
# --------------------------------------------------------------------------- #


@dataclass
class TfidfIndex:
    chunks: list[Chunk]
    df: Counter = field(default_factory=Counter)
    idf: dict[str, float] = field(default_factory=dict)
    vectors: list[dict[str, float]] = field(default_factory=list)
    norms: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        N = len(self.chunks)
        tokenized = [tokenize(c.text) for c in self.chunks]

        # document frequency
        for toks in tokenized:
            for t in set(toks):
                self.df[t] += 1
        self.idf = {t: math.log((N + 1) / (df + 1)) + 1.0 for t, df in self.df.items()}

        # vectors
        for toks in tokenized:
            tf = Counter(toks)
            length = max(len(toks), 1)
            vec = {t: (count / length) * self.idf[t] for t, count in tf.items()}
            self.vectors.append(vec)
            self.norms.append(math.sqrt(sum(v * v for v in vec.values())) or 1.0)

    def _query_vec(self, question: str) -> tuple[dict[str, float], float]:
        toks = tokenize(question)
        tf = Counter(toks)
        length = max(len(toks), 1)
        vec = {t: (count / length) * self.idf.get(t, 0.0) for t, count in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return vec, norm

    def search(self, question: str, top_k: int = 6) -> list[RetrievedChunk]:
        qvec, qnorm = self._query_vec(question)
        scored: list[RetrievedChunk] = []
        for chunk, vec, norm in zip(self.chunks, self.vectors, self.norms):
            # sparse cosine: iterate over the smaller vector
            if len(qvec) < len(vec):
                dot = sum(qv * vec.get(t, 0.0) for t, qv in qvec.items())
            else:
                dot = sum(v * qvec.get(t, 0.0) for t, v in vec.items())
            score = dot / (qnorm * norm)
            scored.append(RetrievedChunk(chunk=chunk, score=score))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


# --------------------------------------------------------------------------- #
# Step 3b: embedding backend (OpenAI-compatible, with disk cache)
# --------------------------------------------------------------------------- #


def get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    if not key:
        raise RuntimeError(
            "Embedding/LLM backend needs OPENAI_API_KEY or LLM_API_KEY in env."
        )
    return key


def get_base_url() -> str | None:
    return os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL") or None


def get_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("`pip install openai` first.") from exc
    return OpenAI(api_key=get_api_key(), base_url=get_base_url())


def embed_texts(texts: list[str], model: str, batch_size: int = 64) -> list[list[float]]:
    """Embed in batches to stay under per-request size limits."""
    if not texts:
        return []
    client = get_openai_client()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        ordered = sorted(resp.data, key=lambda d: d.index)
        out.extend(list(d.embedding) for d in ordered)
    return out


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


@dataclass
class EmbeddingIndex:
    chunks: list[Chunk]
    embedding_model: str
    cache_path: Path | None = None
    chunk_vectors: list[list[float]] = field(default_factory=list)

    def build(self, verbose: bool = True) -> None:
        if self.cache_path and self.cache_path.exists():
            try:
                cached = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if (
                    cached.get("model") == self.embedding_model
                    and cached.get("n") == len(self.chunks)
                ):
                    self.chunk_vectors = cached["vectors"]
                    if verbose:
                        print(f"[embed] loaded {len(self.chunk_vectors)} cached vectors from {self.cache_path}")
                    return
                if verbose:
                    print("[embed] cache mismatch; rebuilding")
            except Exception as exc:
                if verbose:
                    print(f"[embed] cache unreadable ({exc}); rebuilding")

        if verbose:
            print(f"[embed] embedding {len(self.chunks)} chunks with {self.embedding_model} ...")
        self.chunk_vectors = embed_texts(
            [c.text for c in self.chunks], model=self.embedding_model
        )
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(
                    {
                        "model": self.embedding_model,
                        "n": len(self.chunks),
                        "vectors": self.chunk_vectors,
                    }
                ),
                encoding="utf-8",
            )
            if verbose:
                print(f"[embed] cached vectors to {self.cache_path}")

    def search(self, question: str, top_k: int = 6) -> list[RetrievedChunk]:
        qvec = embed_texts([question], model=self.embedding_model)[0]
        scored = [
            RetrievedChunk(chunk=c, score=cosine(qvec, v))
            for c, v in zip(self.chunks, self.chunk_vectors)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


# --------------------------------------------------------------------------- #
# Step 3b: local embedding backend (offline, multilingual)
# --------------------------------------------------------------------------- #
#
# Pedagogical companion to EmbeddingIndex above. Same API, but vectors come
# from a sentence-transformers model that runs entirely on the user's CPU --
# no API key, no network at search time. Default model handles Chinese and
# English in the same vector space, which matches the Ren corpus.

DEFAULT_LOCAL_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class LocalEmbeddingIndex:
    """Dense semantic index using a local sentence-transformers model.

    Mirrors the EmbeddingIndex API so the rest of the pipeline
    (build_prompt, generate_extractive_answer, generate_rag_answer) can swap
    backends without changes. Vectors cache to disk in the same JSON shape.
    """

    chunks: list[Chunk]
    embedding_model: str = DEFAULT_LOCAL_EMBED_MODEL
    cache_path: Path | None = None
    chunk_vectors: list[list[float]] = field(default_factory=list)
    _model: object = field(default=None, repr=False, compare=False)

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                'Install with: pip install -e ".[embed]"'
            ) from exc
        self._model = SentenceTransformer(self.embedding_model)
        return self._model

    def build(self, verbose: bool = True) -> None:
        if self.cache_path and self.cache_path.exists():
            try:
                cached = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if (
                    cached.get("model") == self.embedding_model
                    and cached.get("n") == len(self.chunks)
                ):
                    self.chunk_vectors = cached["vectors"]
                    if verbose:
                        print(f"[local-embed] loaded {len(self.chunk_vectors)} cached vectors from {self.cache_path}")
                    return
                if verbose:
                    print("[local-embed] cache mismatch; rebuilding")
            except Exception as exc:
                if verbose:
                    print(f"[local-embed] cache unreadable ({exc}); rebuilding")

        model = self._load_model()
        if verbose:
            print(f"[local-embed] encoding {len(self.chunks)} chunks with {self.embedding_model} ...")
        vectors = model.encode(
            [c.text for c in self.chunks],
            batch_size=32,
            show_progress_bar=verbose,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        self.chunk_vectors = [list(map(float, v)) for v in vectors]
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(
                    {
                        "model": self.embedding_model,
                        "n": len(self.chunks),
                        "vectors": self.chunk_vectors,
                    }
                ),
                encoding="utf-8",
            )
            if verbose:
                print(f"[local-embed] cached vectors to {self.cache_path}")

    def search(self, question: str, top_k: int = 6) -> list[RetrievedChunk]:
        if not self.chunk_vectors:
            raise RuntimeError("LocalEmbeddingIndex.build() must be called before search().")
        model = self._load_model()
        qvec = list(map(float, model.encode([question], convert_to_numpy=True)[0]))
        scored = [
            RetrievedChunk(chunk=c, score=cosine(qvec, v))
            for c, v in zip(self.chunks, self.chunk_vectors)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


# --------------------------------------------------------------------------- #
# Step 4: prompt + generation
# --------------------------------------------------------------------------- #


def format_citation(chunk: Chunk) -> str:
    return f"[{chunk.year} · {chunk.title}]"


def build_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    ctx_blocks = []
    for r in retrieved:
        cite = format_citation(r.chunk)
        ctx_blocks.append(
            f"{cite} (chunk {r.chunk.chunk_id}, score={r.score:.3f})\n{r.chunk.text}"
        )
    context = "\n\n".join(ctx_blocks)
    return f"""你是一个严谨的中文问答助手，专门基于任正非历年讲话回答问题。
请只根据下面【检索上下文】回答【用户问题】。
要求：
1. 引用具体年份和讲话标题，例如 [2019 · 接受金融时报采访]。
2. 如果上下文中没有充分答案，请明确说明 “检索语料中没有找到直接证据”，不要编造。
3. 用简体中文回答。

【检索上下文】
{context}

【用户问题】
{question}

【回答】"""


def split_sentences(text: str) -> list[str]:
    sents: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for m in re.findall(r"[^。！？!?]+[。！？!?]?[”\"']?", line):
            s = m.strip()
            if s:
                sents.append(s)
    return sents


def generate_extractive_answer(
    question: str, retrieved: list[RetrievedChunk]
) -> str:
    """Pick top evidence snippets from retrieved chunks; no LLM call.

    This fallback is intentionally conservative. It is not trying to write a
    polished synthesis; it gives students auditable evidence snippets before any
    generative model is allowed to paraphrase the corpus.
    """
    qtoks = set(tokenize(question))
    cand: list[tuple[float, Chunk, str]] = []
    for r in retrieved:
        for s in split_sentences(r.chunk.text):
            if len(s) < 8:
                continue
            if "？" in s or "?" in s:
                continue
            if re.match(r"^(问|记者|主持人|[0-9]+[、.])", s):
                continue
            stoks = set(tokenize(s))
            overlap = len(qtoks & stoks)
            if overlap == 0:
                continue
            score = overlap + r.score * 8.0
            if "任正非" in s or "我们" in s:
                score += 0.5
            cand.append((score, r.chunk, s))
    if not cand:
        return "检索到的片段中没有与问题直接匹配的句子。(No retrieved sentence directly matches the question.)"
    cand.sort(key=lambda x: x[0], reverse=True)

    seen: set[str] = set()
    out: list[str] = []
    for _, chunk, sent in cand:
        key = re.sub(r"\s+", "", sent)
        if key in seen:
            continue
        seen.add(key)
        out.append(f"{sent} {format_citation(chunk)}")
        if len(out) >= 4:
            break
    bullets = "\n".join(f"- {item}" for item in out)
    return "根据检索，最相关的证据片段是 (top evidence sentences from retrieval):\n" + bullets


def generate_llm_answer(prompt: str, model: str) -> str:
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你只基于提供的上下文回答问题。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def naive_llm_answer(question: str, model: str) -> str:
    """Baseline: ask the LLM the question with NO retrieval context."""
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你是一个中文问答助手。请基于你的知识回答用户问题，"
                "尽量给出具体的年份、场合和原文引用。",
            },
            {"role": "user", "content": question},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------- #
# Claude Code subscription backend (uses local `claude -p`, no API key needed)
# --------------------------------------------------------------------------- #


# Both system prompts are Chinese on purpose: the corpus, the retrieval
# queries, and the expected answers are all Chinese.
NAIVE_SYS_PROMPT = (
    "你是一个中文商业问答助手。回答的问题往往涉及商业战略、组织管理、科技竞争。"
    "请基于你的常识与训练知识给出回答，结构清晰、3-6 句话即可，不需要引用具体来源。"
)
RAG_SYS_PROMPT = (
    "你只能基于下面的【检索上下文】回答用户问题。"
    "回答时必须引用具体的年份和讲话标题，例如 [2019 · 接受金融时报采访]。"
    "如果上下文中没有充分依据，请明确说明 “检索语料中没有找到直接证据”，不要编造。"
    "用简体中文，3-6 句话回答。"
)


def claude_code_available() -> bool:
    """Return True if the `claude` CLI is on PATH."""
    return shutil.which("claude") is not None


def claude_code_auth_check(timeout: int = 30) -> tuple[bool, str]:
    """Probe `claude -p` with a tiny prompt. Returns (ok, message)."""
    if not claude_code_available():
        return False, (
            "未检测到 claude 命令。请先安装 Claude Code：\n"
            "(claude command not found — install Claude Code first:)\n"
            "  https://docs.anthropic.com/claude-code\n"
            "安装后在终端运行 `claude` 并执行 /login 完成订阅认证。\n"
            "(After installing, run `claude` in a terminal and /login.)"
        )
    try:
        out = claude_code_answer(
            "Reply with only the word: ok",
            model="haiku",
            system_prompt=None,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, (
            "claude 命令超时，可能需要重新登录（终端运行 `claude` 然后 /login）。"
            "(claude timed out — you may need to /login again.)"
        )
    except Exception as exc:
        return False, (
            f"claude 命令存在但调用失败：{exc}\n"
            "请在终端运行 `claude`，执行 /login 完成 OAuth 登录，然后重启本 notebook。\n"
            "(claude exists but the call failed — run `claude`, /login, then restart this notebook.)"
        )
    return True, f"Claude Code 已认证 (authenticated; probe response: {out[:60]!r})"


def claude_code_answer(
    prompt: str,
    *,
    model: str = "sonnet",
    system_prompt: str | None = None,
    timeout: int = 180,
) -> str:
    """Call local `claude -p` headless. Uses Claude Code subscription OAuth.

    `--tools ""` disables every built-in tool so this is pure text Q&A.
    `--no-session-persistence` keeps the session out of disk history.
    `--system-prompt` (when provided) replaces the default system prompt so the
    user's CLAUDE.md / memory does not leak into the demo answer.
    """
    cmd = [
        "claude", "-p",
        "--model", model,
        "--tools", "",
        "--no-session-persistence",
        "--output-format", "text",
    ]
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])
    cmd.append(prompt)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or "(no output)"
        raise RuntimeError(f"claude CLI failed (rc={proc.returncode}): {msg}")
    return proc.stdout.strip()


# --------------------------------------------------------------------------- #
# Backend dispatchers
# --------------------------------------------------------------------------- #


def generate_naive_answer(
    question: str, *, llm_backend: str, model: str | None
) -> str:
    if llm_backend == "claude-code":
        return claude_code_answer(
            question, model=model or "sonnet", system_prompt=NAIVE_SYS_PROMPT
        )
    if llm_backend == "openai":
        return naive_llm_answer(question, model=model or "gpt-4o-mini")
    raise ValueError(f"naive answer not available for backend {llm_backend}")


def generate_rag_answer(
    prompt: str, *, llm_backend: str, model: str | None
) -> str:
    if llm_backend == "claude-code":
        return claude_code_answer(
            prompt, model=model or "sonnet", system_prompt=RAG_SYS_PROMPT
        )
    if llm_backend == "openai":
        return generate_llm_answer(prompt, model=model or "gpt-4o-mini")
    raise ValueError(f"RAG answer not available for backend {llm_backend}")


# --------------------------------------------------------------------------- #
# High-level API for notebook
# --------------------------------------------------------------------------- #


def find_corpus_dir() -> Path:
    """Locate the Ren Zhengfei speech corpus (China-friendly, offline).

    Search order, first hit wins:
      1. ``$REN_CORPUS_DIR`` if the environment variable is set.
      2. ``./corpus`` next to this file (bundled copy — fully self-contained lab).
      3. ``MiniCourse_8hr/demos/Ren-master`` anywhere above this file (the shared
         in-repo corpus — no duplication when you have the whole course repo).

    Returns the first existing directory, or candidate (2) as a friendly default
    so error messages point students at the bundle-it option.
    """
    env = os.environ.get("REN_CORPUS_DIR")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    bundled = here.parent / "corpus"
    if bundled.is_dir():
        return bundled

    for parent in here.parents:
        candidate = parent / "MiniCourse_8hr" / "demos" / "Ren-master"
        if candidate.is_dir():
            return candidate

    return bundled


DEFAULT_CORPUS_DIR = find_corpus_dir()
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "build"


def build_index(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    backend: str = "tfidf",
    embedding_model: str = "text-embedding-3-small",
    chunk_size: int = 500,
    overlap: int = 80,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    verbose: bool = True,
):
    """Build (or load cached) retrieval index. Returns (index, chunks, docs)."""
    docs = load_corpus(corpus_dir)
    if verbose:
        print(f"[corpus] loaded {len(docs)} speeches from {corpus_dir}")
    chunks = build_chunks(docs, chunk_size=chunk_size, overlap=overlap)
    if verbose:
        print(f"[corpus] produced {len(chunks)} chunks (chunk_size={chunk_size}, overlap={overlap})")

    if backend == "tfidf":
        idx = TfidfIndex(chunks=chunks)
        if verbose:
            print(f"[tfidf] vocabulary size = {len(idx.idf)}")
    elif backend == "embed":
        cache_path = cache_dir / f"emb_{embedding_model}_{chunk_size}_{overlap}.json"
        idx = EmbeddingIndex(
            chunks=chunks, embedding_model=embedding_model, cache_path=cache_path
        )
        idx.build(verbose=verbose)
    else:
        raise ValueError(f"unknown backend: {backend}")
    return idx, chunks, docs


def corpus_summary(docs: list[tuple[str, int, str, str]], chunks: list[Chunk]) -> dict:
    """Return compact corpus/index metadata for README, CLI, and notebooks."""
    year_counts = Counter(year for _rel, year, _title, _text in docs if year)
    total_chars = sum(len(text) for _rel, _year, _title, text in docs)
    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "characters": total_chars,
        "year_min": min(year_counts) if year_counts else 0,
        "year_max": max(year_counts) if year_counts else 0,
        "years": len(year_counts),
    }


def compare(
    question: str,
    index,
    *,
    top_k: int = 6,
    llm_backend: str = "claude-code",
    llm_model: str | None = None,
    skip_naive: bool = False,
) -> dict:
    """Return a dict with naive-LLM and RAG answers side by side.

    llm_backend:
      - "claude-code" : uses local `claude -p` (subscription OAuth)
      - "openai"      : uses OPENAI_API_KEY / LLM_API_KEY
      - "extractive"  : no LLM call; ranks sentences from retrieved chunks
    """
    retrieved = index.search(question, top_k=top_k)
    prompt = build_prompt(question, retrieved)

    if llm_backend == "extractive":
        rag_answer = generate_extractive_answer(question, retrieved)
        naive = "(skipped — extractive mode)"
    else:
        rag_answer = generate_rag_answer(prompt, llm_backend=llm_backend, model=llm_model)
        if skip_naive:
            naive = "(skipped)"
        else:
            naive = generate_naive_answer(question, llm_backend=llm_backend, model=llm_model)

    return {
        "question": question,
        "naive": naive,
        "rag": rag_answer,
        "retrieved": retrieved,
        "prompt": prompt,
    }


def hits_anchor(retrieved: list[RetrievedChunk], anchors: list[str], top_n: int = 6) -> list[str]:
    """Return list of anchor patterns that any of the top-N retrieved chunks satisfy.

    An anchor 'matches' if it appears as a substring of the source_file. Anchor
    patterns can be partial filenames (e.g. '20190624') or path segments
    (e.g. '2019/').
    """
    hit: list[str] = []
    files = [r.chunk.source_file for r in retrieved[:top_n]]
    for a in anchors:
        if any(a in f for f in files):
            hit.append(a)
    return hit


def retrieval_diagnostics(index, questions: list[dict], top_k: int = 6) -> list[dict]:
    """Run lightweight retrieval diagnostics over the curated question set."""
    rows: list[dict] = []
    for item in questions:
        retrieved = index.search(item["q"], top_k=top_k)
        anchors = item.get("anchors", [])
        hit = hits_anchor(retrieved, anchors, top_n=top_k)
        rows.append(
            {
                "id": item.get("id"),
                "theme": item.get("theme", ""),
                "question": item["q"],
                "top_score": retrieved[0].score if retrieved else 0.0,
                "anchor_hits": hit,
                "anchor_hit_count": len(hit),
                "anchor_count": len(anchors),
                "top_sources": [r.chunk.source_file for r in retrieved[:top_k]],
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG demo on the Ren Zhengfei corpus.")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_DIR)
    p.add_argument("--question", default="什么是灰度？任正非如何用它来选拔干部？")
    p.add_argument("--backend", choices=["tfidf", "embed"], default="tfidf")
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--overlap", type=int, default=80)
    p.add_argument(
        "--embedding-model",
        default=os.environ.get("RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    p.add_argument(
        "--llm-backend",
        choices=["extractive", "claude-code", "openai"],
        default="extractive",
        help="extractive uses no LLM; claude-code uses local `claude -p` subscription; "
             "openai needs an API key.",
    )
    p.add_argument(
        "--llm-model",
        default=None,
        help="Model name. Defaults: 'sonnet' for claude-code, 'gpt-4o-mini' for openai.",
    )
    p.add_argument("--skip-naive", action="store_true",
                   help="Skip the naive-LLM baseline call.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("\n========== RAG-on-Ren ==========")
    print(f"corpus      : {args.corpus}")
    print(f"retrieval   : {args.backend}")
    print(f"llm-backend : {args.llm_backend}")
    print(f"question    : {args.question}\n")

    if args.llm_backend == "claude-code":
        ok, msg = claude_code_auth_check()
        print(f"[auth] {msg}")
        if not ok:
            return

    index, chunks, docs = build_index(
        corpus_dir=args.corpus,
        backend=args.backend,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    result = compare(
        args.question,
        index,
        top_k=args.top_k,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        skip_naive=args.skip_naive,
    )

    print("\n----- Top retrieved chunks -----")
    for r in result["retrieved"]:
        preview = r.chunk.text.replace("\n", " ")[:100]
        print(f"  {r.score:+.3f}  [{r.chunk.year}] {r.chunk.title[:40]}")
        print(f"         {preview}...")

    print("\n----- Naive LLM (no retrieval) -----")
    print(result["naive"])

    print("\n----- RAG answer -----")
    print(result["rag"])


if __name__ == "__main__":
    main()
