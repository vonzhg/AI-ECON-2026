#!/usr/bin/env python3
"""Assemble the reorganized Lecture 9 decks from the five-act source decks.

Lecture 9 moves from nine decks (the July 2026 "five acts" plus the September
T2b/T2c interludes) to eleven decks in five parts.  Every frame's new home,
tier, and reference edits live in source/Lec09_Agentic_AI/reorg_manifest.yaml;
this script is the only reader of that file, so the inventory cannot drift from
the decks it describes.

  python3 tools/lec09_assemble.py inventory [--deck T2b]  # every source frame
  python3 tools/lec09_assemble.py refs      [--deck T2b]  # lines citing decks
  python3 tools/lec09_assemble.py check     [--strict]    # manifest integrity
  python3 tools/lec09_assemble.py assemble  [--deck T1] [--force]
  python3 tools/lec09_assemble.py summary                 # tiers per deck
  python3 tools/lec09_assemble.py runsheet                # the CORE path, in order

`check` verifies that every source frame is used exactly once (as a frame, a
merge source, or a cut) and that every listed edit applies exactly as written.
With --strict it also fails when an assembled frame no longer matches its source
plus edits -- the Phase A acceptance test.  Phase B rewrites frames on purpose,
so after Phase A use plain `check`, which only reports drift.
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import sys
from dataclasses import dataclass, field

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L9 = os.path.join(ROOT, "source", "Lec09_Agentic_AI")
MANIFEST = os.path.join(L9, "reorg_manifest.yaml")
ARCHIVE = os.path.join(L9, "_archive", "five_acts_2026-07")

SOURCES = {
    "T1": "Lec09_T1_Framing_Setup.tex",
    "T1b": "Lec09_T1b_Agentic_Landscape.tex",
    "T2": "Lec09_T2_Under_the_Hood_FirstSession.tex",
    "T2b": "Lec09_T2b_Agents_Sessions_Skills.tex",
    "T2c": "Lec09_T2c_Invoking_Agents.tex",
    "T3": "Lec09_T3_Homotopy_Tools.tex",
    "T4": "Lec09_T4_Project_Org.tex",
    "T5": "Lec09_T5_Tracing_Mistakes_Context.tex",
    "T6": "Lec09_T6_Preview_Wrap.tex",
}
TIERS = ("CORE", "LAB", "READ")

BEGIN = re.compile(r"^\\begin\{frame\}")
END = re.compile(r"^\\end\{frame\}\s*$")
MARK = re.compile(r"^% ---- (?P<id>T[0-9][abc]?-F\d\d|NEW) · (?P<rest>.*)$")
REF = re.compile(r"\b(T[1-6][abc]?|Act\s*[1-6]|Topic\s+9\.\d[abc]?|9\.[1-6][abc]?)\b|\\S\s*\d")

PREAMBLE = r"""\input{../shared_preamble.tex}

% Suppress redundant auto-generated metropolis section page; \sectiondivider frames are the dividers we keep.
\metroset{sectionpage=none}

\graphicspath{{./pic/}{./figures/}{../}}

\usetikzlibrary{positioning,arrows.meta,shapes,calc,fit,backgrounds}
@@MAP_INPUT@@
\title[AI for Econ Research]{AI for Economic Research: Dynamic Models, Language, and Agents\\
@@SUBTITLE@@}

\begin{document}

\makebeamertitle
"""


@dataclass
class Frame:
    id: str
    deck: str
    start: int  # 1-based line of \begin{frame}
    end: int    # 1-based line of \end{frame}
    title: str
    text: str   # from \begin{frame} through \end{frame}, newline-terminated
    lead: list[str] = field(default_factory=list)  # comments directly above


def source_path(key: str) -> str:
    for base in (ARCHIVE, L9):
        path = os.path.join(base, SOURCES[key])
        if os.path.exists(path):
            return path
    sys.exit(f"source deck for {key} not found in {ARCHIVE} or {L9}")


def frame_title(line: str) -> str:
    i = len("\\begin{frame}")
    if line[i:i + 1] == "[":
        i = line.index("]", i) + 1
    if line[i:i + 1] != "{":
        return ""
    depth = 0
    for j in range(i, len(line)):
        if line[j] == "{":
            depth += 1
        elif line[j] == "}":
            depth -= 1
            if depth == 0:
                return line[i + 1:j].strip()
    return line[i + 1:].strip()


def load_frames(key: str) -> list[Frame]:
    path = source_path(key)
    lines = open(path, encoding="utf-8").read().splitlines(keepends=True)
    frames: list[Frame] = []
    i = 0
    while i < len(lines):
        if not BEGIN.match(lines[i]):
            i += 1
            continue
        j = i
        while not END.match(lines[j]):
            j += 1
            if j == len(lines):
                sys.exit(f"{path}:{i + 1}: frame never ends")
        lead = []
        p = i - 1
        while p >= 0 and lines[p].startswith("%") and not lines[p].startswith("%="):
            lead.insert(0, lines[p])
            p -= 1
        frames.append(Frame(f"{key}-F{len(frames) + 1:02d}", key, i + 1, j + 1,
                            frame_title(lines[i]), "".join(lines[i:j + 1]), lead))
        i = j + 1
    return frames


def all_frames() -> dict[str, Frame]:
    return {f.id: f for key in SOURCES for f in load_frames(key)}


def load_manifest() -> dict:
    with open(MANIFEST, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def apply_edits(fid: str, text: str, edits) -> tuple[str, list[str]]:
    """Apply exact-string edits; each must match once unless marked all: true."""
    errors = []
    for edit in edits or []:
        if isinstance(edit, dict):
            old, new, every = edit["old"], edit["new"], bool(edit.get("all"))
        else:
            (old, new), every = edit, False
        hits = text.count(old)
        if hits == 0:
            errors.append(f"{fid}: edit target not found: {old!r}")
        elif hits > 1 and not every:
            errors.append(f"{fid}: edit target matches {hits} times (mark all: true): {old!r}")
        else:
            text = text.replace(old, new)
    return text, errors


def item_edits(item: dict, fid: str):
    edits = item.get("edits")
    if isinstance(edits, dict):  # merge items key their edits by source frame
        return edits.get(fid)
    return edits


def render_item(item: dict, frames: dict[str, Frame]) -> tuple[str, list[str]]:
    tier = item.get("tier")
    todo = item.get("todo")
    todo_lines = "".join(f"% TODO(Phase B): {line}\n"
                         for line in (todo.strip().splitlines() if todo else []))
    if "section" in item:
        title = item["section"]
        return ("\n%=============================================================================\n"
                f"\\section{{{title}}}\n"
                "%===========================================================\n\n"
                f"\\sectiondivider{{{title}}}{{{item.get('divider', '')}}}\n\n"), []
    if "new" in item:
        body = item.get("body")
        if body is None:
            body = ("\\textbf{Placeholder --- written in Phase B.} "
                    "The specification is in the source comment above this frame.")
        return (f"% ---- NEW · TIER: {tier}\n{todo_lines}"
                f"\\begin{{frame}}{{{item['new']}}}\n{body.rstrip()}\n\\end{{frame}}\n\n"), []
    ids = [item["frame"]] if "frame" in item else list(item["merge"])
    out, errors = [], []
    for n, fid in enumerate(ids):
        if fid not in frames:
            errors.append(f"unknown source frame {fid}")
            continue
        fr = frames[fid]
        text, errs = apply_edits(fid, fr.text, item_edits(item, fid))
        errors += errs
        tag = item.get("tag", "KEEP") if "frame" in item else "MERGE-PENDING(" + "+".join(ids) + ")"
        src = f"{fr.deck}:{fr.start}"
        out.append(f"% ---- {fid} · {tag} · TIER: {tier} · from {src}\n")
        if n == 0:
            out.append(todo_lines)
        out.append("".join(fr.lead) + text + "\n")
    return "".join(out), errors


def render_deck(deck: dict, frames: dict[str, Frame]) -> tuple[str, list[str]]:
    part, sub = (deck.get("map") or [None, None])
    header = [f"%% FullCourse_10Lecs -- Lecture 9, {deck['subtitle'].split(':')[0].replace('Lecture ', 'Part ')}",
              "%% Assembled by tools/lec09_assemble.py from reorg_manifest.yaml: Phase A of",
              "%% Lec09_Reorg_Plan_2026-09-12.md as amended by Lec09_Reorg_Plan_Review_2026-09-12.md.",
              "%% Each \"% ---- <ID> · <TAG> · TIER\" marker names a frame's source deck and line;",
              "%% keep the markers until Phase B is done (lec09_assemble.py check reads them)."]
    for label in ("owns", "defers"):
        for line in deck.get(label) or []:
            header.append(f"%% {label.capitalize()}: {line}")
    map_input = "\n\\input{lec09_map.tex}\n" if part is not None else "\n"
    out = ["\n".join(header) + "\n\n",
           PREAMBLE.replace("@@MAP_INPUT@@", map_input).replace("@@SUBTITLE@@", deck["subtitle"]), "\n"]
    errors = []
    for item in deck["items"]:
        if "section" not in item and item.get("tier") not in TIERS:
            errors.append(f"{deck['key']}: item {item.get('frame') or item.get('merge') or item.get('new')} has no valid tier")
        text, errs = render_item(item, frames)
        out.append(text)
        errors += errs
    out.append("%===========================================================\n\\end{document}\n")
    return "".join(out), errors


def parse_output(path: str) -> dict[str, str]:
    """Map source frame ID -> frame text as it currently stands in an assembled deck."""
    found, lines, i = {}, open(path, encoding="utf-8").read().splitlines(keepends=True), 0
    while i < len(lines):
        m = MARK.match(lines[i])
        if not m or m.group("id") == "NEW":
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not BEGIN.match(lines[j]):
            j += 1
        k = j
        while k < len(lines) and not END.match(lines[k]):
            k += 1
        found[m.group("id")] = "".join(lines[j:k + 1])
        i = k + 1
    return found


def cmd_inventory(args):
    for key in SOURCES:
        if args.deck and key not in args.deck:
            continue
        frames = load_frames(key)
        print(f"== {key}  {SOURCES[key]}  ({len(frames)} frames)")
        for fr in frames:
            print(f"  {fr.id:8s} {fr.start:5d}-{fr.end:<5d} {fr.title}")
    if not args.deck:
        print(f"total frames: {sum(len(load_frames(k)) for k in SOURCES)}")


def cmd_refs(args):
    for key in SOURCES:
        if args.deck and key not in args.deck:
            continue
        for fr in load_frames(key):
            for n, line in enumerate(fr.text.splitlines(), start=fr.start):
                if not line.lstrip().startswith("%") and REF.search(line):
                    print(f"{fr.id}:{n}: {line.strip()[:220]}")


def cmd_check(args) -> int:
    manifest, frames = load_manifest(), all_frames()
    uses = collections.Counter()
    errors = []
    for deck in manifest["decks"]:
        _, errs = render_deck(deck, frames)
        errors += errs
        for item in deck["items"]:
            for fid in ([item["frame"]] if "frame" in item else item.get("merge", [])):
                uses[fid] += 1
    for fid in manifest.get("cut") or {}:
        uses[fid] += 1
    for fid in frames:
        if uses[fid] != 1:
            errors.append(f"{fid} ({frames[fid].title}) is used {uses[fid]} times")
    errors += [f"manifest names unknown frame {fid}" for fid in uses if fid not in frames]
    drift, rewritten = [], 0
    for deck in manifest["decks"]:
        path = os.path.join(L9, deck["file"])
        if not os.path.exists(path):
            continue
        current = parse_output(path)
        for item in deck["items"]:
            for fid in ([item["frame"]] if "frame" in item else item.get("merge", [])):
                if fid not in frames:
                    continue
                expected, _ = apply_edits(fid, frames[fid].text, item_edits(item, fid))
                if fid not in current:
                    # Phase B deletes a frame's marker once its rewrite is done.
                    if args.strict:
                        drift.append(f"{deck['file']}: marker for {fid} missing")
                    else:
                        rewritten += 1
                elif current[fid] != expected:
                    drift.append(f"{deck['file']}: {fid} differs from source + edits")
    for e in errors:
        print("ERROR", e)
    for d in drift:
        print("DRIFT" if not args.strict else "ERROR", d)
    status = 1 if errors or (args.strict and drift) else 0
    print(f"check: {len(frames)} source frames, {len(errors)} errors, {len(drift)} drifted, "
          f"{rewritten} rewritten (marker removed){' (strict)' if args.strict else ''}")
    return status


def cmd_assemble(args) -> int:
    manifest, frames = load_manifest(), all_frames()
    status = 0
    for deck in manifest["decks"]:
        if args.deck and deck["key"] not in args.deck:
            continue
        text, errors = render_deck(deck, frames)
        if errors:
            status = 1
            for e in errors:
                print("ERROR", e)
            continue
        path = os.path.join(L9, deck["file"])
        if os.path.exists(path) and not args.force:
            if open(path, encoding="utf-8").read() != text:
                print(f"SKIP {deck['file']}: exists and differs (Phase B edits?); use --force to overwrite")
            continue
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {deck['file']}")
    return status


def cmd_summary(args):
    manifest = load_manifest()
    totals = collections.Counter()
    print(f"{'deck':5s} {'file':42s} {'items':>5s} {'CORE':>5s} {'LAB':>4s} {'READ':>5s} {'NEW':>4s} {'MERGE':>6s}")
    for deck in manifest["decks"]:
        c = collections.Counter()
        for item in deck["items"]:
            if "section" in item:
                continue
            c["items"] += 1
            c[item.get("tier")] += 1
            c["NEW"] += "new" in item
            c["MERGE"] += "merge" in item
        totals.update(c)
        print(f"{deck['key']:5s} {deck['file']:42s} {c['items']:5d} {c['CORE']:5d} {c['LAB']:4d} {c['READ']:5d} {c['NEW']:4d} {c['MERGE']:6d}")
    print(f"{'all':5s} {'':42s} {totals['items']:5d} {totals['CORE']:5d} {totals['LAB']:4d} {totals['READ']:5d} {totals['NEW']:4d} {totals['MERGE']:6d}")
    print(f"cut: {len(manifest.get('cut') or {})}")


def cmd_runsheet(args):
    """CORE path in teaching order, as a Markdown table body."""
    frames = all_frames()
    for deck in load_manifest()["decks"]:
        core = [i for i in deck["items"] if i.get("tier") == "CORE"]
        if not core:
            continue
        print(f"\n**{deck['key']}** · {deck['subtitle'].split(': ', 1)[1]}\n")
        print("| # | Frame | Source |\n|---|---|---|")
        for n, item in enumerate(core, 1):
            if "new" in item:
                title, src = item["new"], "NEW"
            else:
                ids = [item["frame"]] if "frame" in item else item["merge"]
                text, _ = apply_edits(ids[0], frames[ids[0]].text, item_edits(item, ids[0]))
                title = frame_title(text.splitlines()[0])
                src = " + ".join(ids)
            print(f"| {n} | {title} | {src} |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("inventory", "refs", "assemble"):
        p = sub.add_parser(name)
        p.add_argument("--deck", nargs="+")
    sub.choices["assemble"].add_argument("--force", action="store_true")
    sub.add_parser("check").add_argument("--strict", action="store_true")
    sub.add_parser("summary")
    sub.add_parser("runsheet")
    args = ap.parse_args()
    return {"inventory": cmd_inventory, "refs": cmd_refs, "check": cmd_check,
            "assemble": cmd_assemble, "summary": cmd_summary,
            "runsheet": cmd_runsheet}[args.cmd](args) or 0


if __name__ == "__main__":
    sys.exit(main())
