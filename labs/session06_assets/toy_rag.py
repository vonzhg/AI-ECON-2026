"""Deck 6.2's toy RAG in one file: parse a folder, encode it once, keep the matrix, search it, write the prompt.

The Session 6 lab builds the toy cell by cell in block B1; this file is the same steps to take away. Call the
functions from a notebook, or use the shell:

    python toy_rag.py build FOLDER INDEX                  # Steps 1-2: cut every .txt file in FOLDER, encode, save
    python toy_rag.py ask INDEX "a question"              # Steps 3-4: the five nearest passages, then the prompt
    python toy_rag.py ask INDEX "a question" --where fomc_202203     # only addresses that start with this

Needs numpy and sentence-transformers (pip install sentence-transformers). The encoder, about 90 MB, is fetched
the first time and read from disk after that; in mainland China, set HF_ENDPOINT=https://hf-mirror.com first.
Your own library: point FOLDER at plain-text files saved as UTF-8. Another encoder: pass its name to
load_encoder. An answer: send the prompt to any chat model; nothing here calls one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"   # the weights Deck 6.2 used: pin the file, not the name


def parse(folder, min_chars=40):
    """Step 1: read every .txt file as UTF-8 and cut it at blank lines; each piece keeps its address, file#n."""
    docs, ids = [], []
    for f in sorted(Path(folder).glob("*.txt")):
        for n, p in enumerate(f.read_text("utf-8").split("\n\n")):
            if len(p) > min_chars and p[0] != "#":
                docs.append(p)
                ids.append(f"{f.stem}#{n}")
    return docs, ids


def load_encoder(model=MODEL, revision=REVISION):
    """The encoder: fetched once, then read from disk. Any sentence-transformers model can stand in."""
    import sentence_transformers as st
    return st.SentenceTransformer(model, revision=revision)


def encode(enc, texts):
    """Step 2: one row of length one per text, so that a dot product is a cosine."""
    return np.asarray(enc.encode(list(texts), normalize_embeddings=True), dtype=np.float32)


def save(index, V, ids, docs, model=MODEL, revision=REVISION):
    """The index: the matrix and its addresses in one order, the passages, and which weights wrote them."""
    index = Path(index)
    index.mkdir(parents=True, exist_ok=True)
    np.save(index / "vectors.npy", V)
    (index / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
    (index / "docs.json").write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    meta = {"model": model, "revision": revision, "rows": len(ids), "dimension": int(V.shape[1])}
    (index / "model.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def load(index):
    """Read an index back; refuses one whose matrix and addresses disagree."""
    index = Path(index)
    V = np.load(index / "vectors.npy")
    ids = json.loads((index / "ids.json").read_text(encoding="utf-8"))
    docs = json.loads((index / "docs.json").read_text(encoding="utf-8"))
    meta = json.loads((index / "model.json").read_text(encoding="utf-8"))
    if not len(V) == len(ids) == len(docs) == meta["rows"]:
        raise RuntimeError(f"{index}: the matrix and its addresses disagree; rebuild them together")
    return V, ids, docs, meta


def search(V, q, k=5, keep=None):
    """Step 3: every cosine from one product, best first; keep is an optional True/False per address."""
    s = V @ q
    order = np.argsort(-s)
    if keep is not None:
        order = [i for i in order if keep[i]]
    return [(int(i), float(s[i])) for i in order[:k]]


def make_prompt(question, hits, ids, docs):
    """Step 4: the rule, each passage with its address, then the question."""
    context = "\n\n".join(f"[{ids[i]}] {docs[i]}" for i, _ in hits)
    return f"""Answer only from these passages.
Tag every claim with its [address].
If the answer is not in them, say so.

{context}

Question: {question}"""


def main(argv=None):
    ap = argparse.ArgumentParser(description="Deck 6.2's toy RAG: build an index from a folder, then ask it.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="Steps 1-2: parse FOLDER and encode it into INDEX")
    b.add_argument("folder")
    b.add_argument("index")
    a = sub.add_parser("ask", help="Steps 3-4: the nearest passages, then the prompt to send")
    a.add_argument("index")
    a.add_argument("question")
    a.add_argument("--k", type=int, default=5)
    a.add_argument("--where", help="keep only addresses that start with this, e.g. a file name or a date")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        docs, ids = parse(args.folder)
        V = encode(load_encoder(), docs)
        save(args.index, V, ids, docs)
        print(f"{len(ids):,} passages from {args.folder}: a {V.shape[0]:,} x {V.shape[1]} matrix in {args.index}")
        return
    V, ids, docs, meta = load(args.index)
    q = encode(load_encoder(meta["model"], meta["revision"]), [args.question])[0]
    keep = None if args.where is None else np.array([a.startswith(args.where) for a in ids])
    hits = search(V, q, args.k, keep)
    for i, score in hits:
        print(f"{ids[i]}  {score:.2f}  {docs[i][:60]}")
    print()
    print(make_prompt(args.question, hits, ids, docs))


if __name__ == "__main__":
    main()
