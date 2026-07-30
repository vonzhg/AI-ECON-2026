from __future__ import annotations

import argparse
from pathlib import Path

from .chunking import load_chunks, save_chunks, split_markdown_into_chunks
from .generation import build_prompt, extractive_answer
from .retrieval import build_tfidf_index, load_index, save_index, search


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data" / "source" / "Spear-Young_OLG_final_preprint.md"
BUILD_DIR = ROOT / "build"
CHUNKS_PATH = BUILD_DIR / "chunks.json"
INDEX_PATH = BUILD_DIR / "tfidf_index.json"


def ingest(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source markdown not found: {source}")

    markdown = source.read_text(encoding="utf-8")
    chunks = split_markdown_into_chunks(
        markdown,
        source_name=str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else source.name,
        max_chars=args.chunk_size,
        overlap_lines=args.overlap_lines,
    )
    index = build_tfidf_index(chunks)

    save_chunks(chunks, CHUNKS_PATH)
    save_index(index, INDEX_PATH)

    print("========== Offline Ingestion ==========")
    print(f"Source: {source}")
    print(f"Characters: {len(markdown):,}")
    print(f"Chunks: {len(chunks):,}")
    print(f"Chunk file: {CHUNKS_PATH.relative_to(ROOT)}")
    print(f"TF-IDF index: {INDEX_PATH.relative_to(ROOT)}")
    print("Retriever: local sparse TF-IDF vectors, no API key, no neural embedding model")


def ensure_index(args: argparse.Namespace) -> None:
    if args.rebuild or not CHUNKS_PATH.exists() or not INDEX_PATH.exists():
        ingest(
            argparse.Namespace(
                source=args.source,
                chunk_size=args.chunk_size,
                overlap_lines=args.overlap_lines,
            )
        )


def ask(args: argparse.Namespace) -> None:
    ensure_index(args)
    chunks = load_chunks(CHUNKS_PATH)
    index = load_index(INDEX_PATH)
    retrieved = search(chunks, index, args.question, top_k=args.top_k)
    prompt = build_prompt(args.question, retrieved)

    print("\n========== RAG Query ==========")
    print(f"Question: {args.question}")
    print("Retriever: TF-IDF sparse-vector cosine search")

    print("\n[1] Retrieved evidence")
    for idx, item in enumerate(retrieved, start=1):
        chunk = item.chunk
        preview = " ".join(chunk.text.split())[:180]
        print(
            f"S{idx}: score={item.score:.3f} | {chunk.citation} | "
            f"{chunk.section}\n    {preview}..."
        )

    if args.show_context:
        print("\n========== Retrieved Context ==========")
        for idx, item in enumerate(retrieved, start=1):
            print(f"\n[S{idx}] {item.chunk.citation} | {item.chunk.section}")
            print(item.chunk.text)

    if args.show_prompt:
        print("\n========== Prompt To Give Claude/Codex ==========")
        print(prompt)

    if not args.prompt_only:
        print("\n========== Local Extractive Answer ==========")
        print(extractive_answer(args.question, retrieved))


def inspect(args: argparse.Namespace) -> None:
    ensure_index(args)
    chunks = load_chunks(CHUNKS_PATH)
    print("========== Chunk Inventory ==========")
    print(f"Chunks: {len(chunks):,}")
    for chunk in chunks[: args.limit]:
        print(f"{chunk.chunk_id} | {chunk.citation} | {chunk.section}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Portable classroom RAG demo over Spear and Young's OLG paper."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    common.add_argument("--chunk-size", type=int, default=2200)
    common.add_argument("--overlap-lines", type=int, default=8)

    ingest_parser = subparsers.add_parser("ingest", parents=[common])
    ingest_parser.set_defaults(func=ingest)

    ask_parser = subparsers.add_parser("ask", parents=[common])
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=5)
    ask_parser.add_argument("--rebuild", action="store_true")
    ask_parser.add_argument("--show-context", action="store_true")
    ask_parser.add_argument("--show-prompt", action="store_true")
    ask_parser.add_argument("--prompt-only", action="store_true")
    ask_parser.set_defaults(func=ask)

    inspect_parser = subparsers.add_parser("inspect", parents=[common])
    inspect_parser.add_argument("--rebuild", action="store_true")
    inspect_parser.add_argument("--limit", type=int, default=12)
    inspect_parser.set_defaults(func=inspect)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

