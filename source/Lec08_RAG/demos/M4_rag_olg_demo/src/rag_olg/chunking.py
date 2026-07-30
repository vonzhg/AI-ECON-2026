from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class Chunk:
    """A line-addressable document chunk."""

    chunk_id: str
    source: str
    title_path: list[str]
    start_line: int
    end_line: int
    text: str

    @property
    def citation(self) -> str:
        return f"{self.source}:L{self.start_line}-L{self.end_line}"

    @property
    def section(self) -> str:
        return " > ".join(self.title_path[-3:]) if self.title_path else "Untitled"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, object]) -> "Chunk":
        return cls(
            chunk_id=str(row["chunk_id"]),
            source=str(row["source"]),
            title_path=list(row.get("title_path", [])),
            start_line=int(row["start_line"]),
            end_line=int(row["end_line"]),
            text=str(row["text"]),
        )


def clean_heading(raw: str) -> str:
    """Clean marker/OCR markdown headings into readable section labels."""

    text = re.sub(r"<[^>]+>", "", raw)
    text = text.replace("**", "").replace("<br>", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" #*-")


def heading_level_and_text(line: str) -> Optional[Tuple[int, str]]:
    match = HEADING_RE.match(line.strip())
    if not match:
        return None
    title = clean_heading(match.group(2))
    if not title:
        return None
    return len(match.group(1)), title


def split_markdown_into_chunks(
    markdown: str,
    *,
    source_name: str,
    max_chars: int = 2200,
    overlap_lines: int = 8,
) -> list[Chunk]:
    """Split markdown into overlapping, line-addressable chunks.

    The chunker is deliberately simple for teaching: it preserves headings and
    line numbers, so students can see how retrieved evidence maps back to a
    source document.
    """

    if max_chars <= 400:
        raise ValueError("max_chars should be larger than 400 for this demo")
    if overlap_lines < 0:
        raise ValueError("overlap_lines must be non-negative")

    lines = markdown.splitlines()
    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, str]] = []
    chunk_lines: list[tuple[int, str]] = []
    chunk_titles: list[str] = []
    char_count = 0

    def flush() -> None:
        nonlocal chunk_lines, chunk_titles, char_count
        if not chunk_lines:
            return
        text = "\n".join(line for _, line in chunk_lines).strip()
        if text:
            chunk_no = len(chunks) + 1
            chunks.append(
                Chunk(
                    chunk_id=f"olg-{chunk_no:04d}",
                    source=source_name,
                    title_path=chunk_titles.copy(),
                    start_line=chunk_lines[0][0],
                    end_line=chunk_lines[-1][0],
                    text=text,
                )
            )

        if overlap_lines:
            max_overlap_chars = max_chars // 3
            overlap: list[tuple[int, str]] = []
            overlap_count = 0
            for line_no, line in reversed(chunk_lines[-overlap_lines:]):
                line_len = len(line) + 1
                if line_len > max_overlap_chars:
                    continue
                if overlap and overlap_count + line_len > max_overlap_chars:
                    break
                overlap.append((line_no, line))
                overlap_count += line_len
            chunk_lines = list(reversed(overlap))
            char_count = overlap_count
            chunk_titles = [title for _, title in heading_stack]
        else:
            chunk_lines = []
            char_count = 0
            chunk_titles = [title for _, title in heading_stack]

    for line_no, line in enumerate(lines, start=1):
        heading = heading_level_and_text(line)
        if heading:
            level, title = heading
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))

        if not chunk_lines:
            chunk_titles = [title for _, title in heading_stack]

        line_len = len(line) + 1
        if chunk_lines and char_count + line_len > max_chars:
            flush()

        chunk_lines.append((line_no, line))
        char_count += line_len

    flush()
    return chunks


def load_chunks(path: Path) -> list[Chunk]:
    import json

    rows = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk.from_dict(row) for row in rows]


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
