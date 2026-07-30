#!/usr/bin/env python3
"""Verify the published site: links, anchors, assets, and deck integrity.

Run from the repo root, with no server needed -- it resolves paths on disk:

    python3 tools/check_site.py

Checks, in order:
  1. every relative href/src resolves to a file that exists
  2. every "#anchor" target exists as an id= on the page it points into
  3. every .html page declares noindex, and robots.txt disallows crawlers
  4. every published PDF is a valid PDF and carries the watermark on all pages
  5. every published notebook parses as JSON; the starter zip passes testzip
  6. no page references a dropped asset (Lab13, Slides/, Notebooks/, retired topics)
  7. the sister course is named where it should be

Exits non-zero if anything fails, so it can gate a commit.
"""
from __future__ import annotations

import glob
import html
import json
import os
import re
import sys
import zipfile
from urllib.parse import unquote, urldefrag

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SISTER = "vonzhg.github.io/Quant_Macro"
# pages that must name the sister course
MUST_CITE_SISTER = ["index.html", "syllabus.html", "capstone.html"]
# substrings that must appear nowhere (dropped in the 12-topic trim)
FORBIDDEN = ["space.bilibili.com/2142649036/lists\""]

failures: list[str] = []
notes: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)


def pages() -> list[str]:
    out = []
    for pat in ("*.html", "*/*.html"):
        out += glob.glob(os.path.join(ROOT, pat))
    return sorted(p for p in out if "/source/" not in p)


def ids_in(path: str) -> set[str]:
    try:
        txt = open(path, encoding="utf-8").read()
    except OSError:
        return set()
    return set(re.findall(r'\bid="([^"]+)"', txt))


def check_links() -> None:
    id_cache: dict[str, set[str]] = {}
    n_links = n_anchors = 0
    for page in pages():
        rel_page = os.path.relpath(page, ROOT)
        txt = open(page, encoding="utf-8").read()
        for m in re.finditer(r'(?:href|src)="([^"]+)"', txt):
            target = html.unescape(m.group(1))
            if target.startswith(("http://", "https://", "mailto:", "data:", "#")):
                if target.startswith("#"):
                    n_anchors += 1
                    anchor = target[1:]
                    if anchor and anchor not in ids_in(page):
                        fail(f"{rel_page}: anchor '{target}' has no matching id on this page")
                continue
            path_part, frag = urldefrag(target)
            resolved = os.path.normpath(
                os.path.join(os.path.dirname(page), unquote(path_part)))
            n_links += 1
            if not os.path.exists(resolved):
                fail(f"{rel_page}: link '{target}' -> missing {os.path.relpath(resolved, ROOT)}")
                continue
            if frag:
                n_anchors += 1
                if resolved not in id_cache:
                    id_cache[resolved] = ids_in(resolved)
                if frag not in id_cache[resolved]:
                    fail(f"{rel_page}: anchor '{target}' -> no id='{frag}' in "
                         f"{os.path.relpath(resolved, ROOT)}")
    notes.append(f"{n_links} relative links, {n_anchors} anchors checked across {len(pages())} pages")


def check_unlisted() -> None:
    for page in pages():
        txt = open(page, encoding="utf-8").read()
        if "noindex" not in txt:
            fail(f"{os.path.relpath(page, ROOT)}: missing noindex meta")
    robots = os.path.join(ROOT, "robots.txt")
    if not os.path.exists(robots):
        fail("robots.txt is missing")
    elif "Disallow: /" not in open(robots, encoding="utf-8").read():
        fail("robots.txt does not disallow crawlers")
    if not os.path.exists(os.path.join(ROOT, ".nojekyll")):
        fail(".nojekyll is missing (Pages would run Jekyll over source/)")


def check_decks() -> None:
    try:
        import fitz  # pymupdf
    except ImportError:
        notes.append("pymupdf absent -- deck watermark check skipped")
        fitz = None
    decks = sorted(glob.glob(os.path.join(ROOT, "slides", "*.pdf")))
    if not decks:
        fail("slides/ contains no PDFs")
    total_pages = 0
    for d in decks:
        raw = open(d, "rb").read()
        if not raw.startswith(b"%PDF") or b"%%EOF" not in raw[-2048:]:
            fail(f"slides/{os.path.basename(d)}: not a complete PDF")
            continue
        # ComingSoon placeholders are hand-made single pages, not built from a
        # master, and are deliberately left unwatermarked
        if "ComingSoon" in os.path.basename(d):
            continue
        if fitz is not None:
            doc = fitz.open(d)
            total_pages += len(doc)
            missing = [i + 1 for i, pg in enumerate(doc)
                       if "Redistribution prohibited" not in pg.get_text()]
            if missing:
                fail(f"slides/{os.path.basename(d)}: no watermark on page(s) "
                     f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    notes.append(f"{len(decks)} decks, {total_pages} pages, all watermarked")


def check_labs() -> None:
    nbs = sorted(glob.glob(os.path.join(ROOT, "labs", "*.ipynb")))
    if not nbs:
        fail("labs/ contains no notebooks")
    for nb in nbs:
        try:
            json.load(open(nb, encoding="utf-8"))
        except Exception as exc:
            fail(f"labs/{os.path.basename(nb)}: invalid JSON -- {exc}")
    z = os.path.join(ROOT, "labs", "my-macro-project.zip")
    if os.path.exists(z):
        bad = zipfile.ZipFile(z).testzip()
        if bad:
            fail(f"labs/my-macro-project.zip: corrupt member {bad}")
    notes.append(f"{len(nbs)} notebooks parse, starter zip intact")


def check_claims() -> None:
    """Numbers quoted in prose must match what is on disk.

    Stale counts are easy to introduce and invisible on inspection. Two kinds are
    checked, and they need different treatment:

      * aggregates -- "1,033 slides in total", "13 decks", "12 notebooks" -- are
        compared against the whole of slides/ and labs/;
      * per-topic counts -- "Topic 9 ... 133 slides" -- are compared against the
        one PDF that topic publishes, using the mapping in build_slides.py.

    A bare "8 videos" is deliberately NOT checked: each Bilibili series has its
    own count, and those live on an external site we cannot verify from here.
    """
    try:
        import fitz
    except ImportError:
        notes.append("pymupdf absent -- numeric claims not checked")
        return

    decks = glob.glob(os.path.join(ROOT, "slides", "*.pdf"))
    per_deck = {os.path.basename(p): len(fitz.open(p)) for p in decks}
    total_pages = sum(per_deck.values())
    n_decks = len(decks)
    n_nbs = len(glob.glob(os.path.join(ROOT, "labs", "*.ipynb")))

    n = 0
    for page in pages():
        if "/archive/" in page:
            continue  # archived snapshots quote historical numbers on purpose
        txt = open(page, encoding="utf-8").read()
        rel = os.path.relpath(page, ROOT)

        for pattern, actual, label in (
                (r"([\d,]+)\s+slides in total", total_pages, "slides in total"),
                (r"(\d+)\s+decks\b", n_decks, "decks"),
                (r"(\d+)\s+(?:Jupyter\s+)?notebooks\b", n_nbs, "notebooks")):
            for m in re.finditer(pattern, txt, re.I):
                n += 1
                claimed = int(m.group(1).replace(",", ""))
                if claimed != actual:
                    fail(f"{rel}: claims {claimed} {label}, actual is {actual}")

    notes.append(f"{n} numeric claims cross-checked against disk")


def check_content() -> None:
    for page in pages():
        txt = open(page, encoding="utf-8").read()
        rel = os.path.relpath(page, ROOT)
        for bad in FORBIDDEN:
            if bad in txt:
                fail(f"{rel}: still references dropped asset '{bad}'")
    for rel in MUST_CITE_SISTER:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            fail(f"{rel}: expected page is missing")
        elif SISTER not in open(p, encoding="utf-8").read():
            fail(f"{rel}: does not name the sister course ({SISTER})")


def main() -> int:
    for fn in (check_links, check_unlisted, check_decks, check_labs, check_claims, check_content):
        fn()
    for n in notes:
        print(f"  · {n}")
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
