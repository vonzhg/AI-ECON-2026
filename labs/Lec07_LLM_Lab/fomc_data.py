"""Shared loader for the bundled FOMC statement corpus (Lecture 7 labs).

Pure standard library — no numpy, no network. Both Lab 7A and Lab 7B import
from this module so the corpus is loaded the same way everywhere.

Corpus layout: one plain-text file per post-meeting statement,
``data/fomc/fomc_YYYYMMDD.txt``, each starting with a 2-line provenance
header (``# FOMC statement, YYYY-MM-DD`` + ``# source: <url>``).
See ``data/README.md`` for provenance and regeneration notes.
"""
from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def find_fomc_dir() -> Path:
    """Locate the FOMC statement folder.

    Search order (first hit wins):
      1. ``$FOMC_DATA_DIR`` environment variable
      2. ``data/fomc/`` next to this file (the bundled default)
    """
    env = os.environ.get("FOMC_DATA_DIR")
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            return p
    bundled = _HERE / "data" / "fomc"
    if bundled.is_dir():
        return bundled
    raise FileNotFoundError(
        "FOMC corpus not found. Expected the bundled folder "
        f"{bundled} (start the notebook from Lec07_LLM_Lab/), "
        "or set $FOMC_DATA_DIR to a folder of fomc_YYYYMMDD.txt files."
    )


def load_statements(data_dir: Path | None = None) -> list[tuple[str, str]]:
    """Return the corpus as a date-sorted list of ``(date, text)`` pairs.

    ``date`` is ``"YYYY-MM-DD"``; ``text`` is the statement body with the
    2-line provenance header stripped.
    """
    data_dir = Path(data_dir) if data_dir else find_fomc_dir()
    out: list[tuple[str, str]] = []
    for path in sorted(data_dir.glob("fomc_*.txt")):
        stamp = path.stem.split("_")[1]
        date = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}"
        lines = path.read_text(encoding="utf-8").splitlines()
        body = [ln for ln in lines if not ln.startswith("#")]
        text = "\n".join(body).strip()
        if text:
            out.append((date, text))
    if not out:
        raise FileNotFoundError(f"no fomc_*.txt files found in {data_dir}")
    return out


def corpus_summary(statements: list[tuple[str, str]]) -> dict:
    """Small summary dict for the notebooks' environment-check cells."""
    chars = sum(len(t) for _, t in statements)
    return {
        "n_statements": len(statements),
        "total_chars": chars,
        "first_date": statements[0][0],
        "last_date": statements[-1][0],
        "avg_chars": chars // len(statements),
    }


if __name__ == "__main__":
    stmts = load_statements()
    print("FOMC corpus:", corpus_summary(stmts))
    date, text = stmts[-1]
    print(f"\nMost recent statement ({date}), first 300 chars:\n{text[:300]}")
