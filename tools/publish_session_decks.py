#!/usr/bin/env python3
"""Stamp the Session decks with the copyright watermark and write them into a site's slides/ folder.

Instructor tool, run from the private working repository (where source/SessionNN lives) with the
release worktree as the destination:

    python3 tools/publish_session_decks.py ../AI-ECON-2026.release/slides
    python3 tools/publish_session_decks.py ../AI-ECON-2026.release/slides --sessions 1 5

For every source/SessionNN/SessionNN_DeckK_*.pdf it writes slides/SessionNN_DeckK_*.pdf with the same
overlay tools/build_slides.py stamps on the Lecture decks (tools/watermark.tex), and one combined
slides/SessionNN_All_Decks.pdf with a bookmark per deck. slides/SESSIONS.json records, per published
file, the master it came from, the master's md5 and the page count, so a stale deck is easy to spot.
Only PDFs are published: the .tex sources, figures and notes never leave the private repository.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from build_slides import build_watermark, stamp  # noqa: E402  (same overlay as the Lecture decks)


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def deck_title(pdf: str) -> str:
    tex = open(pdf[:-4] + ".tex", encoding="utf-8").read()
    m = re.search(r"\\title\{((?:[^{}]|\{[^{}]*\})*)\}", tex)
    s = m.group(1) if m else os.path.basename(pdf)
    s = re.sub(r"\\(emph|textbf|textit)\{([^{}]*)\}", r"\2", s).replace(r"\&", "&").replace("--", "\u2013")
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    return re.sub(r"\s+", " ", re.sub(r"\\[a-zA-Z]+\s*", "", s).replace("{", "").replace("}", "")).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dest", help="the site's slides/ folder (in the release worktree)")
    ap.add_argument("--sessions", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6])
    args = ap.parse_args()
    import fitz
    import pikepdf

    dest = os.path.abspath(args.dest)
    if not os.path.isdir(dest):
        sys.exit(f"{dest} is not a folder")
    manifest_path = os.path.join(dest, "SESSIONS.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    with tempfile.TemporaryDirectory() as work:
        wm = pikepdf.open(build_watermark(work))     # keep the file open while its page is used
        overlay = wm.pages[0]
        for n in args.sessions:
            sess = f"Session{n:02d}"
            decks = sorted(glob.glob(os.path.join(ROOT, "source", sess, f"{sess}_Deck*.pdf")),
                           key=lambda p: int(re.search(r"_Deck(\d+)_", p).group(1)))
            if not decks:
                sys.exit(f"no decks in source/{sess}")
            combined, toc = fitz.open(), []
            for p in decks:
                k = int(re.search(r"_Deck(\d+)_", p).group(1))
                out = os.path.join(dest, os.path.basename(p))
                stamp(p, overlay, out)
                pages = fitz.open(p).page_count
                manifest[os.path.basename(p)] = {"master": os.path.relpath(p, ROOT), "md5": md5(p),
                                                 "pages": pages, "title": deck_title(p),
                                                 "published": datetime.date.today().isoformat()}
                toc.append([1, f"{n}.{k}  {deck_title(p)}", combined.page_count + 1])
                src = fitz.open(p)
                combined.insert_pdf(src)
                src.close()
                print(f"  {os.path.basename(out)}  ({pages} pp.)")
            combined.set_toc(toc)
            combined.set_metadata({"title": f"Session {n}: all decks", "author": "Zhigang Feng"})
            merged = os.path.join(work, f"{sess}_merged.pdf")
            combined.save(merged, garbage=3, deflate=True)
            out = os.path.join(dest, f"{sess}_All_Decks.pdf")
            stamp(merged, overlay, out)
            manifest[os.path.basename(out)] = {"master": f"source/{sess} (all decks, in order)",
                                               "pages": combined.page_count, "decks": len(decks),
                                               "published": datetime.date.today().isoformat()}
            print(f"  {os.path.basename(out)}  ({combined.page_count} pp., {len(decks)} decks)")
    json.dump(dict(sorted(manifest.items())), open(manifest_path, "w"), indent=1)
    print(f"wrote {os.path.relpath(manifest_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
