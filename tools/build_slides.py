#!/usr/bin/env python3
"""Build the published slide decks from the masters in source/.

Mirrors the sister course's tools/build_slides.py (Quant_Macro) so both sites are
maintained the same way: masters live in `source/`, this stamps the copyright
watermark on every page and writes to `slides/`, and a manifest records which
master each published deck came from.

    python3 tools/build_slides.py                     # build every deck
    python3 tools/build_slides.py --check             # report staleness, write nothing
    python3 tools/build_slides.py Lec08_T1 Lec08_T2   # rebuild only these

Unlike the sister course, decks here are named by lecture and topic, and the master
lives beside the .tex that produces it. DECKS is derived by scanning source/, so a
newly added deck is picked up automatically -- but only decks already present in
slides/ are rebuilt by default, because the site deliberately publishes a subset
(Lec08_T4/T5 exist as masters and are intentionally not published; see commit
2b10bbc "Lec08 site: keep T1-T3, remove T4").

Use --adopt to start publishing a master that is not yet in slides/.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLIDES = os.path.join(ROOT, "slides")
SOURCE = os.path.join(ROOT, "source")
MANIFEST = os.path.join(SLIDES, "BUILD.json")
# published without a master: hand-made placeholders, left alone
PLACEHOLDERS = {"Lec06_Slides_ComingSoon.pdf", "Lec10_Slides_ComingSoon.pdf"}


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def masters() -> dict[str, str]:
    """published filename -> newest master path under source/."""
    import fitz
    out: dict[str, tuple[str, str]] = {}
    for p in glob.glob(os.path.join(SOURCE, "**", "*.pdf"), recursive=True):
        name = os.path.basename(p)
        try:
            meta = fitz.open(p).metadata or {}
            created = (meta.get("creationDate") or "")[2:16]
        except Exception:
            continue
        # keep the newest copy when a deck appears in more than one folder
        if name not in out or created > out[name][0]:
            out[name] = (created, p)
    return {k: v[1] for k, v in out.items()}


def build_watermark(workdir: str) -> str:
    src = os.path.join(ROOT, "tools", "watermark.tex")
    shutil.copy(src, workdir)
    proc = None
    for _ in range(2):  # remember-picture needs two passes
        proc = subprocess.run(["pdflatex", "-interaction=batchmode", "watermark.tex"],
                              cwd=workdir, capture_output=True)
    out = os.path.join(workdir, "watermark.pdf")
    if not os.path.exists(out):
        sys.exit("watermark.pdf was not produced -- is pdflatex on PATH?\n"
                 + (proc.stderr.decode("utf-8", "replace")[-2000:] if proc else ""))
    return out


def stamp(master: str, overlay_page, dest: str) -> None:
    import pikepdf
    pdf = pikepdf.open(master)
    for page in pdf.pages:
        page.add_overlay(overlay_page, pikepdf.Rectangle(page.mediabox))
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest), suffix=".pdf")
    os.close(fd)
    pdf.save(tmp)
    pdf.close()
    os.replace(tmp, dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("decks", nargs="*",
                    help="deck name prefixes to build, e.g. Lec08_T1 (default: all published)")
    ap.add_argument("--check", action="store_true", help="report staleness; write nothing")
    ap.add_argument("--seed", action="store_true",
                    help="record manifest entries for decks whose published copy already "
                         "matches its master, without rebuilding them")
    ap.add_argument("--adopt", action="store_true",
                    help="also publish masters not yet present in slides/")
    args = ap.parse_args()

    try:
        import fitz  # noqa: F401
        import pikepdf  # noqa: F401
    except ImportError as exc:
        sys.exit(f"needs pymupdf and pikepdf: {exc}")
    import pikepdf

    src = masters()
    published = {os.path.basename(p) for p in glob.glob(os.path.join(SLIDES, "*.pdf"))}
    targets = sorted((published | set(src)) if args.adopt else published)
    targets = [t for t in targets if t not in PLACEHOLDERS]
    if args.decks:
        targets = [t for t in targets if any(t.startswith(d) for d in args.decks)]
        if not targets:
            sys.exit(f"no published deck matches: {', '.join(args.decks)}")

    manifest = {}
    if os.path.exists(MANIFEST):
        try:
            manifest = json.load(open(MANIFEST, encoding="utf-8"))
        except ValueError:
            manifest = {}

    def deck_date(path: str) -> str:
        import fitz
        try:
            return ((fitz.open(path).metadata or {}).get("creationDate") or "")[2:16]
        except Exception:
            return ""

    if args.check or args.seed:
        stale = missing = ok = unknown = seeded = 0
        for name in targets:
            master = src.get(name)
            dest = os.path.join(SLIDES, name)
            if not master:
                missing += 1
                print(f"  {name:<48} NO MASTER under source/")
                continue
            if not os.path.exists(dest):
                missing += 1
                print(f"  {name:<48} master exists but not published")
                continue
            recorded = manifest.get(name, {})
            live = md5(master)
            if not recorded:
                # No baseline. Fall back to comparing the PDFs' own build dates,
                # which is what tells us whether the published deck came from
                # this master -- pikepdf preserves /Info through watermarking.
                same = deck_date(dest) == deck_date(master)
                if args.seed and same:
                    manifest[name] = {"master": os.path.relpath(master, ROOT),
                                      "master_md5": live,
                                      "pages": len(__import__("pikepdf").open(dest).pages)}
                    seeded += 1
                    print(f"  {name:<48} seeded (published matches this master)")
                elif same:
                    unknown += 1
                    print(f"  {name:<48} no manifest entry, but dates match -- run --seed")
                else:
                    stale += 1
                    print(f"  {name:<48} STALE -- master is a newer build than published")
            elif recorded.get("master_md5") != live:
                stale += 1
                print(f"  {name:<48} STALE -- master changed since last build")
            else:
                ok += 1
        if args.seed and seeded:
            with open(MANIFEST, "w", encoding="utf-8") as fh:
                json.dump(dict(sorted(manifest.items())), fh, indent=2, ensure_ascii=False)
                fh.write("\n")
        unpublished = sorted(set(src) - published - PLACEHOLDERS)
        print(f"\n{ok} up to date, {seeded} seeded, {unknown} unrecorded, "
              f"{stale} stale, {missing} unresolved, {len(targets)} checked")
        if unpublished:
            print(f"{len(unpublished)} master(s) under source/ are deliberately not published "
                  f"(older drafts, and Lec08_T4/T5 which were removed from the site on purpose)")
        return 1 if (stale or missing) else 0

    workdir = tempfile.mkdtemp(prefix="wm-")
    try:
        # keep the Pdf bound: if it is garbage-collected the page reference goes
        # stale and add_overlay fails with "called with a direct object"
        wm = pikepdf.open(build_watermark(workdir))
        overlay = wm.pages[0]
        built = 0
        for name in targets:
            master = src.get(name)
            if not master:
                print(f"  {name:<48} SKIPPED -- no master under source/")
                continue
            dest = os.path.join(SLIDES, name)
            existed = os.path.exists(dest)
            stamp(master, overlay, dest)
            pages = len(pikepdf.open(dest).pages)
            manifest[name] = {
                "master": os.path.relpath(master, ROOT),
                "master_md5": md5(master),
                "pages": pages,
            }
            built += 1
            print(f"  {name:<48} {pages:>4}p  {'rebuilt' if existed else 'new'}")
        with open(MANIFEST, "w", encoding="utf-8") as fh:
            json.dump(dict(sorted(manifest.items())), fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        print(f"\n{built} deck(s) built; manifest written to slides/BUILD.json")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
