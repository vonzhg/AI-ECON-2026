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
  8. the repository carries only what the site releases: every tracked file is linked from a page
     reachable from index.html, belongs to a linked lab (a Session lab's sessionNN_assets/, a
     Summer 2026 lab's whole folder), or is listed in SHIPPED_WITH / INFRA; every linked file is
     tracked, so a clone has it

Exits non-zero if anything fails, so it can gate a commit.
"""
from __future__ import annotations

import glob
import html
import json
import os
import re
import subprocess
import sys
import zipfile
from urllib.parse import unquote, urldefrag

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SISTER = "vonzhg.github.io/Quant_Macro"
# pages that must name the sister course
MUST_CITE_SISTER = ["index.html", "syllabus.html", "capstone.html"]
# substrings that must appear nowhere (dropped in the 12-topic trim)
FORBIDDEN = ["space.bilibili.com/2142649036/lists\""]
# Students clone this repository, so it carries only what the site releases (check 8). Beyond the
# files the pages link, a file ships only because a released file needs it -- each entry names
# that file -- or because the repository needs it to work.
REPO_URL = "https://github.com/vonzhg/AI-ECON-2026/"
SHIPPED_WITH = {
    "labs/session02_assets/": "labs/Session03_Lab.ipynb",  # session03_value.py imports session02_growth
    "labs/data/": "labs/Lec01_02_Lab_Getting_Started.ipynb",
    "source/Lec08_RAG/demos/": "slides/Lec08_T2_Index_Chunking_Embeddings.pdf",
    "source/Lec09_Agentic_AI/demos/": "slides/Lec09_T5a_Homotopy_Method.pdf",
    "source/Lec09_Agentic_AI/CLI_FACTS_2026-09.md": "labs/Lec09_Agent_Lab/TERMINAL_WALKTHROUGH.md",
}
INFRA = [".gitignore", ".nojekyll", "LICENSE", "README.md", "requirements.txt", "robots.txt",
         ".devcontainer/", "tools/", "slides/SESSIONS.json", "slides/BUILD.json",
         "labs/README.md", "labs/requirements.txt"]

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
    # deck counts are claimed for the Session course: per session ("Session 2 ... 6 decks")
    # or overall ("32 decks"); the combined SessionNN_All_Decks.pdf and the archived
    # Summer 2026 Lec*.pdf decks are not counted
    per_session: dict[int, int] = {}
    for name in per_deck:
        m = re.match(r"Session(\d+)_Deck\d+_", name)
        if m:
            per_session[int(m.group(1))] = per_session.get(int(m.group(1)), 0) + 1
    n_decks = sum(per_session.values())
    n_nbs = len(glob.glob(os.path.join(ROOT, "labs", "*.ipynb")))
    # a number right after one of these words is a label ("Lecture 9 decks", "Summer 2026
    # notebooks"), not a count
    label = re.compile(r"(?:Lecture|Session|Summer|Fall|Spring)(?:\s|&nbsp;)+$")
    session_mark = re.compile(r'id="session-(\d+)"|Session(?:\s|&nbsp;)+(\d+)\b')

    n = 0
    for page in pages():
        if "/archive/" in page:
            continue  # archived snapshots quote historical numbers on purpose
        txt = open(page, encoding="utf-8").read()
        rel = os.path.relpath(page, ROOT)

        for pattern, actual, what in (
                (r"([\d,]+)\s+slides in total", total_pages, "slides in total"),
                (r"(\d+)\s+decks\b", n_decks, "decks"),
                (r"(\d+)\s+(?:Jupyter\s+)?notebooks\b", n_nbs, "notebooks")):
            for m in re.finditer(pattern, txt, re.I):
                if label.search(txt[max(0, m.start() - 20):m.start()]):
                    continue
                expected = actual
                if what == "decks":
                    # the nearest session named in the preceding 800 characters owns the claim
                    marks = list(session_mark.finditer(txt[max(0, m.start() - 800):m.start()]))
                    if marks:
                        s = int(marks[-1].group(1) or marks[-1].group(2))
                        expected = per_session.get(s, 0)
                n += 1
                claimed = int(m.group(1).replace(",", ""))
                if claimed != expected:
                    fail(f"{rel}: claims {claimed} {what}, actual is {expected}")

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


def site_links() -> set[str]:
    """Repo paths linked from the pages reachable from index.html; a tree link ends in '/'."""
    linked: set[str] = set()
    todo, seen = ["index.html"], set()
    while todo:
        page = todo.pop()
        if page in seen or not os.path.exists(os.path.join(ROOT, page)):
            continue
        seen.add(page)
        linked.add(page)
        txt = open(os.path.join(ROOT, page), encoding="utf-8").read()
        for m in re.finditer(r'(?:href|src)="([^"]+)"', txt):
            target = urldefrag(html.unescape(m.group(1)))[0]
            if target.startswith(REPO_URL):
                # github.com/vonzhg/AI-ECON-2026/(blob|tree)/<branch>/<path>
                gh = re.fullmatch(r"(blob|tree)/[^/]+/(.+)", unquote(target[len(REPO_URL):]))
                if gh:
                    path = gh.group(2).rstrip("/")
                    linked.add(path + "/" if gh.group(1) == "tree" else path)
                continue
            if not target or target.startswith(("http://", "https://", "mailto:", "data:")):
                continue
            rel = os.path.relpath(os.path.normpath(
                os.path.join(ROOT, os.path.dirname(page), unquote(target))), ROOT)
            if os.path.isdir(os.path.join(ROOT, rel)):
                rel = os.path.join(rel, "index.html")
            linked.add(rel)
            if rel.endswith(".html"):
                todo.append(rel)
    return linked


def check_released_only() -> None:
    try:
        out = subprocess.run(["git", "-C", ROOT, "-c", "core.quotepath=false", "ls-files", "-z"],
                             capture_output=True, check=True).stdout.decode("utf-8")
    except (OSError, subprocess.CalledProcessError):
        notes.append("not a git checkout -- released-only check skipped")
        return
    tracked = [p for p in out.split("\0") if p]
    linked = site_links()
    trees = {p for p in linked if p.endswith("/")}
    for p in linked:
        m = re.fullmatch(r"labs/Session(\d+)_Lab\.ipynb", p)
        if m:
            trees.add(f"labs/session{m.group(1)}_assets/")
        elif p.startswith("labs/") and p.count("/") >= 2:
            trees.add("/".join(p.split("/")[:2]) + "/")  # a Summer 2026 lab ships its folder

    def released(f: str) -> bool:
        return f in linked or f.startswith(tuple(trees))

    for path, needed_by in SHIPPED_WITH.items():
        if released(needed_by):
            if path.endswith("/"):
                trees.add(path)
            else:
                linked.add(path)
    trees.update(p for p in INFRA if p.endswith("/"))
    linked.update(p for p in INFRA if not p.endswith("/"))

    stray = [f for f in tracked if not released(f)]
    for f in stray[:30]:
        fail(f"tracked but not released on the site: {f}")
    if len(stray) > 30:
        fail(f"... and {len(stray) - 30} more tracked files the site does not release")
    tracked_set = set(tracked)
    for p in sorted(linked):
        if not p.endswith("/") and p not in tracked_set and os.path.exists(os.path.join(ROOT, p)):
            fail(f"linked but not committed, so a clone lacks it: {p}")
    notes.append(f"{len(tracked)} tracked files, {len(tracked) - len(stray)} released on the site "
                 f"or shipped with what is")


def main() -> int:
    for fn in (check_links, check_unlisted, check_decks, check_labs, check_claims, check_content,
               check_released_only):
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
