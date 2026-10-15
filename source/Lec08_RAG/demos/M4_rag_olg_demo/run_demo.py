#!/usr/bin/env python3
from __future__ import annotations

"""Terminal runner for the OLG RAG notebook flow.

Use this when Jupyter is not open or when the machine's default `python`
points to a broken pyenv shim. It mirrors the default offline path in
demo.ipynb and uses only the standard library.
"""

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from questions import QUESTIONS  # noqa: E402
from rag_olg.chunking import split_markdown_into_chunks  # noqa: E402
from rag_olg.generation import build_prompt, extractive_answer  # noqa: E402
from rag_olg.llm import hits_anchor  # noqa: E402
from rag_olg.retrieval import build_tfidf_index, search  # noqa: E402


SOURCE = ROOT / "data" / "source" / "Spear-Young_OLG_final_preprint.md"
BUILD_DIR = ROOT / "build"


def section(title: str) -> None:
    print("")
    print("=" * 10, title, "=" * 10)


def main() -> None:
    BUILD_DIR.mkdir(exist_ok=True)

    section("Step 0: Setup")
    print(f"Demo root: {ROOT}")
    print(f"Source:    {SOURCE.relative_to(ROOT)}")
    print(f"Exists:    {SOURCE.exists()}")

    section("Step 1: Source Audit")
    markdown = SOURCE.read_text(encoding="utf-8")
    lines = markdown.splitlines()
    headings = [line for line in lines if line.startswith("#")]
    print(f"Characters: {len(markdown):,}")
    print(f"Lines:      {len(lines):,}")
    print(f"Headings:   {len(headings):,}")
    print("First five headings:")
    for heading in headings[:5]:
        print(f"  {heading[:110]}")

    section("Step 2: Chunking")
    chunks = split_markdown_into_chunks(
        markdown,
        source_name="data/source/Spear-Young_OLG_final_preprint.md",
        max_chars=2200,
        overlap_lines=8,
    )
    print(f"Chunks: {len(chunks)}")
    for chunk in chunks[:3]:
        print(f"  {chunk.chunk_id} | {chunk.citation} | {chunk.section}")

    section("Step 3: TF-IDF Index")
    index = build_tfidf_index(chunks)
    print(f"TF-IDF vocabulary size: {len(index.idf):,}")
    print(f"Indexed vectors:        {len(index.vectors):,}")

    section("Step 4: Retrieval Diagnostics")
    diagnostics = []
    for item in QUESTIONS:
        retrieved = search(chunks, index, item["q"], top_k=5)
        answer = extractive_answer(item["q"], retrieved)
        anchor_hits = hits_anchor(retrieved, item.get("anchors", []), top_n=5)
        refused = "do not have enough retrieved evidence" in answer
        ok = refused if item.get("expect_refusal") else (
            retrieved[0].score > 0.035 and len(anchor_hits) >= 1
        )
        diagnostics.append(
            {
                "id": item["id"],
                "theme": item["theme"],
                "top_score": retrieved[0].score,
                "anchor_hits": anchor_hits,
                "expected_refusal": item.get("expect_refusal", False),
                "refused": refused,
                "ok": ok,
                "top_sources": [result.chunk.citation for result in retrieved],
            }
        )
        status = "PASS" if ok else "FAIL"
        if item.get("expect_refusal"):
            print(f"{status} | Q{item['id']:02d} | refusal={refused} | {item['theme']}")
        else:
            print(
                f"{status} | Q{item['id']:02d} | score={retrieved[0].score:.3f} | "
                f"anchors={len(anchor_hits)}/{len(item.get('anchors', []))} | {item['theme']}"
            )

    if not all(row["ok"] for row in diagnostics):
        raise SystemExit(1)

    section("Step 5: Single-Question Walkthrough")
    question_item = next(item for item in QUESTIONS if item["id"] == 3)
    retrieved = search(chunks, index, question_item["q"], top_k=5)
    print(f"Q3: {question_item['theme']}")
    print(question_item["q"])
    for rank, result in enumerate(retrieved, 1):
        chunk = result.chunk
        preview = " ".join(chunk.text.split())[:260]
        print(f"S{rank}: score={result.score:.3f} | {chunk.citation}")
        print(f"    Section: {chunk.section}")
        print(f"    {preview}...")

    section("Step 6: Grounded Prompt Preview")
    prompt = build_prompt(question_item["q"], retrieved)
    print(prompt[:2200])
    print("... prompt truncated ...")

    section("Step 7: Local Extractive Answer")
    print(extractive_answer(question_item["q"], retrieved))

    section("Step 8: Refusal Demo")
    unsupported = next(item for item in QUESTIONS if item.get("expect_refusal"))
    unsupported_results = search(chunks, index, unsupported["q"], top_k=5)
    print(unsupported["q"])
    print(f"Top score: {unsupported_results[0].score:.3f}")
    print(extractive_answer(unsupported["q"], unsupported_results))

    section("Step 9: Write Report")
    report_path = BUILD_DIR / "terminal_retrieval_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "theme",
                "top_score",
                "anchor_hits",
                "expected_refusal",
                "refused",
                "ok",
                "top_sources",
            ],
        )
        writer.writeheader()
        for row in diagnostics:
            writer.writerow(
                {
                    "id": row["id"],
                    "theme": row["theme"],
                    "top_score": f"{row['top_score']:.6f}",
                    "anchor_hits": "; ".join(row["anchor_hits"]),
                    "expected_refusal": row["expected_refusal"],
                    "refused": row["refused"],
                    "ok": row["ok"],
                    "top_sources": "; ".join(row["top_sources"]),
                }
            )
    print(f"Wrote: {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
