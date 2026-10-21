"""Session 6 lab -- retrieval written from scratch, and the reader that cannot read (blocks B, C and D).

Block B1  Deck 6.2's toy index: the deck's own matrix and the vectors of its question and the graded questions,
          every file hash-checked before a row is read (load_toy_index), so the slide's four listings run in the
          notebook with no download.
Block B2  The corpus. The 245 FOMC statements Session 1 trained on, copied in under the hashes Session 5
          recorded, then cut into the units a retriever actually searches. Nineteen of the older statements are
          hard-wrapped with a blank line inside the sentence, so the obvious paragraph split returns fragments;
          the loader reflows them and says how many. Eighteen sentences in the corpus were cut off by the scrape
          and stay cut off; that is counted too, because a citation can only be as complete as the document it
          points at.
Block B3  Chunking, two ways. The document-aware chunker follows the shape an FOMC statement actually has --
          the decision, the assessment of conditions, the forward guidance, the vote -- and never ends a chunk in
          the middle of a sentence. The fixed-size chunker cuts every N characters, which is what most tutorials
          do, and `boundary_damage` finds the sentences it cuts in half so the notebook can print one.
Block B4  Retrieval, in numpy. TF-IDF with the vocabulary fitted on the corpus and cosine scoring; BM25 in the
-B6       Robertson-Sparck Jones form, k1 and b exposed; reciprocal rank fusion of the two, which needs no score
          calibration because it only reads ranks; and grading: recall@k and the mean reciprocal rank against a
          gold set of dated questions -- including questions this corpus cannot answer, because a refusal that
          never fires is not a refusal.
Block C   A refusal that is a number, calibrated on those questions rather than written into a prompt; three
          failures, reproduced on purpose, each returning its own before and after (an index built before the
          answer was published, an answer cut in half by a chunk boundary, a query whose wording matches two
          meetings equally well); and the reader. Session 1's Machine B has a context of 128 characters and the
          median chunk here is longer than that, so a second character-level model is trained once, here, with a
          context of 256 -- the same classes as Session 5, imported rather than copied again. The notebook then
          measures one thing: the model's loss on the passage retrieval handed it, against its loss on a passage
          about something else. The two are nearly the same number, which is what "retrieval without a reader is
          not grounding" means arithmetically.
Block D   The grounded answer, extractive and cited, whose claims the notebook then marks by hand.

Descended from `labs/Lec08_RAG_Lab/rag_ren.py`, but written again rather than adapted. Three
things in the ancestor do not survive the move to an English corpus: its sentence splitter has no
"." among its terminators, so an English paragraph comes back as one sentence; its refusal is
either an instruction to a live model or a zero-overlap test that English function words never
trigger; and its loader expects `YYYY/*.md`. What is kept is the shape of the pipeline and the
insistence that every quoted sentence carries the document it came from.

Imports: numpy, torch, the standard library, and Session 5's model classes. No network. CPU only.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import sys
import time

if __name__ == "__main__":      # instructor runs only: keep the linear-algebra library polite
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "8")
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
FOMC_DIR = HERE / "fomc"
GOLD = HERE / "gold_questions.json"
READER_DIR = HERE / "reader"
READER_MANIFEST = READER_DIR / "MANIFEST.json"
RECORDED = HERE / "recorded_rag.json"
TOY_INDEX = HERE / "toy_index"

# The corpus and its hashes live in Session 5's folder; this folder holds a byte-identical copy
# and asserts every hash before reading a single statement. Session 5's manifest is the authority
# because Session 5 is where the copy out of the read-only 2025 lab folder was made and recorded.
SESSION05 = HERE.parent / "session05_assets"
CORPUS_MANIFEST = SESSION05 / "fomc_manifest.json"

# Session 5 reproduced Session 1's model classes verbatim so the 2026 checkpoint would load key
# for key. A third copy would be a third thing to keep in step, so they are imported. The import
# is by path because a lab folder is a folder, not a package; nothing in Session 5 runs on import.
if str(SESSION05) not in sys.path:
    sys.path.insert(0, str(SESSION05))
try:
    from session05_attention import (MiniGPT, character_split, corpus_statements,  # noqa: E402
                                     parameter_count)
except ImportError as exc:                                                          # pragma: no cover
    raise ImportError(f"session06_rag needs Session 5's model classes at {SESSION05}: {exc}") from exc

SEED = 2026                     # Session 1's training seed, kept so the two runs are comparable
RRF_K = 60                      # the constant Cormack, Clarke and Büttcher report; see below
BM25_K1, BM25_B = 1.5, 0.75     # the usual defaults; both are arguments, not constants, in the code
TOP_K = 5                       # how many chunks an answer is allowed to look at


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# Block B1 -- Deck 6.2's toy index, as the deck's own run wrote it
# --------------------------------------------------------------------------------------
def load_toy_index(folder: str | Path = TOY_INDEX) -> dict:
    """Deck 6.2's toy index: every file's sha256 checked against toy_index/MANIFEST.json before it is read.

    Returns the matrix V (one unit-length row per passage, in the order of the addresses), the addresses, the
    question vectors by id ("toy" for the slide's question, "q01".."q26" for the graded ones) with their text,
    and the manifest, which carries the rows and the prompt length the slide printed.
    """
    folder = Path(folder)
    manifest = json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        if file_sha256(folder / row["file"]) != row["sha256"]:
            raise RuntimeError(f"{row['file']} does not match toy_index/MANIFEST.json; refusing to read it")
    V = np.load(folder / "vectors.npy")
    ids = json.loads((folder / "ids.json").read_text(encoding="utf-8"))
    Q = np.load(folder / "questions.npy")
    meta = json.loads((folder / "questions.json").read_text(encoding="utf-8"))
    if V.shape != (manifest["passages"], manifest["dimension"]) or len(ids) != len(V) or len(meta) != len(Q):
        raise RuntimeError("toy_index: the files disagree with their manifest about their shapes")
    return {"V": V, "ids": ids, "manifest": manifest, "hashes_checked": len(manifest["files"]),
            "questions": {q["id"]: Q[k] for k, q in enumerate(meta)},
            "question_text": {q["id"]: q["question"] for q in meta}}


# --------------------------------------------------------------------------------------
# Block B2 -- the corpus: copy it, prove it, cut it into paragraphs
# --------------------------------------------------------------------------------------
def corpus_manifest(path: Path = CORPUS_MANIFEST) -> dict:
    """Session 5's record of the 245 statements: filename, date, character count, sha256."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def copy_corpus(destination: Path = FOMC_DIR, source: Path = SESSION05 / "fomc") -> dict:
    """Copy the statements here and assert every recorded hash. Run once; idempotent.

    The 2025 lab folder `labs/Lec07_LLM_Lab/data/fomc` is read-only -- Session 1 hash-checks the
    corpus where it lives -- so nothing is ever written back to it. This copies from Session 5's
    copy, which carries the same bytes and the manifest that fixes them.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = corpus_manifest()
    copied, checked = 0, []
    for row in manifest["files"]:
        target = destination / row["file"]
        if not target.exists() or file_sha256(target) != row["sha256"]:
            shutil.copy2(Path(source) / row["file"], target)
            copied += 1
        got = file_sha256(target)
        if got != row["sha256"]:
            raise RuntimeError(f"{row['file']} does not match the corpus manifest "
                               f"({got[:12]} against {row['sha256'][:12]})")
        checked.append(row["file"])
    extra = sorted(p.name for p in destination.glob("fomc_*.txt"))
    missing = sorted(set(checked) - set(extra))
    unexpected = sorted(set(extra) - set(checked))
    if missing or unexpected:
        raise RuntimeError(f"corpus folder disagrees with the manifest; missing {missing}, extra {unexpected}")
    return dict(n=len(checked), copied=copied, source=str(source), destination=str(destination),
                manifest=str(CORPUS_MANIFEST), manifest_sha256=file_sha256(CORPUS_MANIFEST),
                first=manifest["first"], last=manifest["last"])


# A blank line separates paragraphs -- except in nineteen statements from 1996 to 2010, where the
# scrape left a blank line between every wrapped line of the same sentence. A block that does not
# end at a sentence boundary is therefore joined to the next one before anything else happens.
# Only ".", "!" and "?" close a block. A colon or a semicolon looks like an ending and is not:
# the vote lists of the early 2000s wrap after "Alan Greenspan, Chairman;", and treating that
# as the end of a paragraph cuts the vote into a dozen fragments, none of which can be cited.
# The price of the stricter rule is that a lead-in ending in a colon stays joined to the item
# that follows it, which costs a paragraph break in five statements and no citation anywhere.
_CLOSES_A_SENTENCE = re.compile(r"""[.!?]["')\]’”]*$""")


def statement_paragraphs(body: str) -> list[str]:
    """The paragraphs of one statement, with hard-wrapped blocks reflowed and spaces normalised."""
    blocks = [b.strip() for b in body.split("\n\n") if b.strip()]
    paragraphs, buffer = [], ""
    for block in blocks:
        buffer = f"{buffer} {block}".strip() if buffer else block
        if _CLOSES_A_SENTENCE.search(buffer):
            paragraphs.append(" ".join(buffer.split()))
            buffer = ""
    if buffer:
        paragraphs.append(" ".join(buffer.split()))
    return paragraphs


@dataclass
class Statement:
    """One FOMC statement: its date, its body, and the paragraphs the body is made of."""
    index: int                  # position in date order, 0-based
    date: str                   # ISO, e.g. "2015-12-16"
    file: str
    body: str
    paragraphs: list[str]
    source_url: str = ""

    @property
    def year(self) -> int:
        return int(self.date[:4])

    def pretty_date(self) -> str:
        return f"{_MONTHS[int(self.date[5:7]) - 1]} {int(self.date[8:10])}, {self.date[:4]}"


_MONTHS = ("January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December")


def load_statements(directory: Path = FOMC_DIR, check_hashes: bool = True) -> list[Statement]:
    """The 245 statements in date order, headers stripped, hashes checked, paragraphs reflowed.

    The two header lines carry the date and the Federal Reserve URL; they are read for the
    metadata and then dropped, so no retrieved sentence is ever a comment line.
    """
    manifest = corpus_manifest()
    recorded = {row["file"]: row for row in manifest["files"]}
    statements = []
    for i, path in enumerate(sorted(Path(directory).glob("fomc_*.txt"))):
        row = recorded.get(path.name)
        if row is None:
            raise RuntimeError(f"{path.name} is not in the corpus manifest")
        if check_hashes and file_sha256(path) != row["sha256"]:
            raise RuntimeError(f"{path.name} does not match the corpus manifest")
        lines = path.read_text(encoding="utf-8").splitlines()
        url = next((ln.split("source:", 1)[1].strip() for ln in lines if ln.startswith("# source:")), "")
        body = "\n".join(ln for ln in lines if not ln.startswith("#")).strip()
        statements.append(Statement(index=i, date=row["date"], file=path.name, body=body,
                                    paragraphs=statement_paragraphs(body), source_url=url))
    if len(statements) != manifest["n"]:
        raise RuntimeError(f"found {len(statements)} statements, the manifest records {manifest['n']}")
    return statements


def reflow_report(statements: Sequence[Statement]) -> dict:
    """What the blank-line split would have done, and what the source documents do to themselves.

    Two different problems, both worth a number. The first is ours to fix: a blank line inside a
    sentence, which reflowing repairs. The second is not: a handful of statements were scraped
    with the last sentence cut off mid-clause, and no chunker or retriever can put those words
    back. They are counted here so that a citation ending in a comma is recognised as the corpus
    speaking, not the pipeline failing.
    """
    naive = repaired = truncated = 0
    worst = None
    cut_off = []
    for s in statements:
        blocks = len([b for b in s.body.split("\n\n") if b.strip()])
        naive += blocks
        if blocks != len(s.paragraphs):
            repaired += 1
            if worst is None or blocks - len(s.paragraphs) > worst[1]:
                worst = (s.date, blocks - len(s.paragraphs), blocks, len(s.paragraphs))
        for paragraph in s.paragraphs:
            for sentence in split_sentences(paragraph):
                if sentence.rstrip()[-1:] not in ".!?\"')":
                    truncated += 1
                    if len(cut_off) < 5:
                        cut_off.append(dict(date=s.date, sentence=sentence))
    return dict(statements=len(statements), paragraphs=sum(len(s.paragraphs) for s in statements),
                blocks_before_reflow=naive, statements_repaired=repaired,
                worst_case=(dict(zip(("date", "blocks_removed", "blocks", "paragraphs"), worst))
                            if worst else None),
                sentences_truncated_in_the_source=truncated, truncated_examples=cut_off)


# --------------------------------------------------------------------------------------
# Sentences. The ancestor's splitter had no "." among its terminators, which on English prose
# returns the whole paragraph as one sentence; every citation below depends on getting this right.
# --------------------------------------------------------------------------------------
ABBREVIATIONS = frozenset((
    "u.s", "u.k", "mr", "mrs", "ms", "dr", "jr", "sr", "st", "no", "vs", "inc", "corp", "co",
    "approx", "e.g", "i.e", "a.m", "p.m", "fig", "sec", "dept", "govs",
))
_SENTENCE_END = re.compile(r"""([.!?])(["')\]’”]*)(\s+)""")
# A single letter ("Ben S. Bernanke") or a run of them ("U.S.", "a.m."): the period that follows
# is part of a name, not the end of a sentence. Vote paragraphs are a dozen of these in a row.
_INITIAL = re.compile(r"(?:[a-z]\.)*[a-z]")


def split_sentences(text: str) -> list[str]:
    """Split English prose into sentences, keeping the terminator on the sentence.

    Three things stop a full stop from ending a sentence, and all three are everywhere in this
    corpus: a known abbreviation ("U.S. economy"), a middle initial ("Ben S. Bernanke, Chairman"),
    and a following lower-case letter. Vote paragraphs are the hard case -- a single sentence can
    carry a dozen initials -- so they are the ones to check a change against.
    """
    text = " ".join(text.split())
    sentences, start = [], 0
    for match in _SENTENCE_END.finditer(text):
        head = text[start:match.end(2)]
        word = re.split(r"[\s(\[]", head.rstrip(".!?\"')]’”"))[-1].lower()
        if word in ABBREVIATIONS or _INITIAL.fullmatch(word):
            continue
        following = text[match.end():match.end() + 1]
        if following and following.islower():
            continue
        if head.strip():
            sentences.append(head.strip())
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


# --------------------------------------------------------------------------------------
# Block B3 -- chunking: the statement's own structure, and the fixed-size chunker to compare it to
# --------------------------------------------------------------------------------------
SECTIONS = ("decision", "assessment", "guidance", "implementation", "objectives", "vote", "other")

# An FOMC statement is not free prose. It has parts, and after 1994 the parts announce themselves
# in a small, stable vocabulary. These are the cues, in priority order: the first list that matches
# names the paragraph. A paragraph that genuinely does both -- "the Committee seeks ... decided to
# raise" is one paragraph after 2020 -- is labelled by whichever cue comes first here, and the
# recorded run prints how much falls through to "other" so the reader can judge the cue list.
SECTION_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("vote", (r"^voting (for|against)\b", r"\bvoted as (an|the) alternate\b",
              r"\bvoting for the fomc\b", r"\bvoted against\b",
              r"^(mr|ms|mrs)\.\s+\w+ (voted|abstained)")),
    ("decision", (r"\bdecided (today )?to (raise|lower|maintain|keep|increase|reduce"
                  r"|decrease|leave|hold|set|establish)\b",
                  r"\bdecided today to\b", r"\bdecided to (increase|decrease) slightly the degree of pressure\b",
                  r"\bapproved a (reduction|decrease|increase) in the discount rate\b",
                  r"\bvoted (today )?to (raise|lower|maintain)\b",
                  r"\bwill (raise|lower|maintain) the target range\b",
                  r"\bdecided to (begin|continue) reducing\b", r"\btoday's action will be reflected\b",
                  r"\bthe federal funds rate is expected to (fall|rise)\b")),
    ("guidance", (r"\bin determining the (timing|size|extent)\b", r"\bin assessing the appropriate stance\b",
                  r"\bwill continue to monitor\b", r"\bwill closely monitor\b", r"\bwill monitor\b",
                  r"\bwould be prepared to adjust\b", r"\bis prepared to adjust\b", r"\bis prepared to use\b",
                  r"\bwhen the committee decides\b", r"\bin view of realized and expected\b",
                  r"\bthe committee (expects|anticipates|judges|believes|perceives|continues to believe)\b",
                  r"\bfuture adjustments\b", r"\bthe path of the (federal funds rate|economy)\b",
                  r"\brisks are (weighted|balanced|roughly)\b", r"\bwill act as needed\b",
                  r"\bcommitted to using its full range of tools\b", r"\bpolicy concern remains\b",
                  r"\bstands ready\b", r"\bgoing forward\b", r"\blikely to (be|remain|warrant|weigh|create)\b")),
    ("assessment", (r"^information received since\b", r"^recent indicators\b",
                    r"^indicators of economic activity\b",
                    r"^the information that has become available\b", r"^economic (growth|activity|conditions)\b",
                    r"\bjob gains\b", r"\bunemployment rate\b", r"\breadings on core inflation\b",
                    r"\beconomic activity (has|is|expanded|appears|rose|slowed|picked)\b",
                    r"\binflation (has|remains|is)\b", r"\blabor market(s)? (conditions|have|has|continued)\b",
                    r"\bhousehold spending\b", r"\bfinancial market(s)? (conditions|have)\b",
                    r"\bproductivity growth\b", r"\bthe patterns evident\b")),
    ("implementation", (r"\breinvest", r"\bagency mortgage-backed securities\b",
                        r"\bholdings of treasury securities\b", r"^in taking the discount\b",
                        r"\bthe board of governors\b", r"\bdiscount rate\b", r"\bample reserves\b",
                        r"\bsystem open market account\b", r"\bopen market desk\b", r"\breserve requirements\b",
                        r"\bbalance sheet\b", r"\bswap line\b", r"\brepurchase agreement\b",
                        r"\bliquidity facilit", r"\bbasis point (decline|increase|reduction)\b")),
    ("objectives", (r"\bconsistent with its statutory mandate\b",
                    r"\bseeks to (achieve|foster) maximum employment\b",
                    r"\bdual mandate\b", r"\blonger-run goals\b")),
)
_COMPILED_CUES = tuple((name, tuple(re.compile(p) for p in pats)) for name, pats in SECTION_CUES)


def classify_section(paragraph: str) -> str:
    """Which part of the statement this paragraph is. Cue-based; `SECTION_CUES` is the whole rule."""
    low = paragraph.lower()
    for name, patterns in _COMPILED_CUES:
        if any(p.search(low) for p in patterns):
            return name
    return "other"


@dataclass
class Chunk:
    """One retrievable unit, and enough provenance to cite it without going back to the file."""
    chunk_id: int
    statement_index: int
    date: str
    section: str
    text: str
    start_char: int             # offset into `flatten(statement)`, not into the raw file
    end_char: int
    strategy: str               # "document" or "fixed"
    opens_mid_sentence: bool = False
    closes_mid_sentence: bool = False

    def citation(self) -> str:
        return f"[FOMC statement of {self.date}]"

    def indexed_text(self) -> str:
        """What the retriever actually sees: the date, in two spellings, then the section, then the text.

        Indexing the metadata with the body is the difference between a corpus that can answer
        "at the meeting of December 16, 2015" and one that can only answer "about inflation". It
        is also what lets an out-of-corpus question about a meeting that never happened score low
        instead of scoring on the word "inflation" alone.
        """
        return f"{self.date} {_pretty(self.date)} {self.date[:4]} {self.section} {self.text}"


def _pretty(date: str) -> str:
    return f"{_MONTHS[int(date[5:7]) - 1]} {int(date[8:10])}, {date[:4]}"


def flatten(statement: Statement) -> str:
    """The statement as one line: paragraphs reflowed, joined by a single space.

    Both chunkers and `boundary_damage` measure offsets against this string, so a chunk's
    `start_char` means the same thing whichever chunker produced it.
    """
    return " ".join(" ".join(statement.paragraphs).split())


# 800 characters, chosen on the gold set rather than guessed: `chunk_size_sweep` runs the whole
# grading table at 600, 800, 1000 and 1400, and 800 is best in every column. Both directions cost
# something. Longer chunks dilute a cosine -- the same handful of matching terms divided by a
# longer vector -- so at 1000 and 1400 the refusal threshold starts turning away a question the
# corpus can answer. Shorter chunks split a paragraph's argument across two of them, and at 600
# every retriever loses rank-1 accuracy. The sweep is recorded with the results.
DOCUMENT_MAX_CHARS = 800        # a merged run of same-section paragraphs stops growing here
DOCUMENT_MIN_CHARS = 40         # anything shorter is a fragment and is folded into its neighbour


def document_chunks(statements: Sequence[Statement], max_chars: int = DOCUMENT_MAX_CHARS,
                    min_chars: int = DOCUMENT_MIN_CHARS) -> list[Chunk]:
    """Chunk on what the document is made of, not on a character count.

    Three rules, in order. Consecutive paragraphs with the same section label are one chunk, so a
    two-paragraph assessment is retrieved whole. A run stops growing at `max_chars`. A single
    paragraph longer than `max_chars` is split at sentence boundaries and never inside one -- which
    is the property `fixed_size_chunks` gives up, and the reason a chunk here can be quoted
    verbatim into an answer.
    """
    chunks: list[Chunk] = []
    for statement in statements:
        labelled = [(p, classify_section(p)) for p in statement.paragraphs]
        merged: list[tuple[str, str]] = []
        for paragraph, section in labelled:
            if merged and merged[-1][1] == section and len(merged[-1][0]) + len(paragraph) + 1 <= max_chars:
                merged[-1] = (f"{merged[-1][0]} {paragraph}", section)
            elif merged and len(paragraph) < min_chars:
                merged[-1] = (f"{merged[-1][0]} {paragraph}", merged[-1][1])
            else:
                merged.append((paragraph, section))
        flat = flatten(statement)
        cursor = 0
        for text, section in merged:
            for piece in _split_at_sentences(text, max_chars):
                start = flat.find(piece, cursor)
                start = cursor if start < 0 else start
                chunks.append(Chunk(chunk_id=len(chunks), statement_index=statement.index,
                                    date=statement.date, section=section, text=piece,
                                    start_char=start, end_char=start + len(piece), strategy="document"))
                cursor = start + len(piece)
    return chunks


def _split_at_sentences(text: str, max_chars: int) -> list[str]:
    """Break a long paragraph into pieces of at most `max_chars`, always at a sentence boundary."""
    if len(text) <= max_chars:
        return [text]
    pieces, current = [], ""
    for sentence in split_sentences(text):
        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


FIXED_CHUNK_CHARS = 500
FIXED_OVERLAP = 0


def fixed_size_chunks(statements: Sequence[Statement], chunk_size: int = FIXED_CHUNK_CHARS,
                      overlap: int = FIXED_OVERLAP) -> list[Chunk]:
    """The chunker most tutorials reach for: cut every `chunk_size` characters, then move on.

    It knows nothing about the document, so it will end a chunk in the middle of "the Committee
    decided to raise the target range for the federal" and open the next one on "funds rate to
    1/4 to 1/2 percent". `opens_mid_sentence` and `closes_mid_sentence` mark exactly that, and
    `boundary_damage` collects the sentences it happened to.
    """
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least zero and smaller than chunk_size")
    chunks: list[Chunk] = []
    for statement in statements:
        flat = flatten(statement)
        start = 0
        while start < len(flat):
            end = min(start + chunk_size, len(flat))
            text = flat[start:end]
            chunks.append(Chunk(chunk_id=len(chunks), statement_index=statement.index,
                                date=statement.date, section="fixed", text=text,
                                start_char=start, end_char=end, strategy="fixed",
                                opens_mid_sentence=start > 0 and not _at_boundary(flat, start),
                                closes_mid_sentence=end < len(flat) and not _at_boundary(flat, end)))
            if end >= len(flat):
                break
            start = end - overlap
    return chunks


def _at_boundary(text: str, position: int) -> bool:
    """Is `position` the start of a sentence? True only just after a terminator and a space."""
    before = text[max(0, position - 2):position].strip()
    return bool(before) and before[-1] in ".!?"


def boundary_damage(chunks: Sequence[Chunk], statements: Sequence[Statement], limit: int = 5) -> dict:
    """Sentences the fixed-size chunker cut in half, with both halves, ready to print.

    A sentence is counted as cut when a chunk boundary falls strictly inside it. The two halves
    are what each chunk would offer a reader: neither one is the sentence.
    """
    by_statement: dict[int, list[Chunk]] = {}
    for chunk in chunks:
        by_statement.setdefault(chunk.statement_index, []).append(chunk)
    cut, examples = 0, []
    total_sentences = 0
    for statement in statements:
        flat = flatten(statement)
        boundaries = sorted({c.end_char for c in by_statement.get(statement.index, ()) if c.end_char < len(flat)})
        position = 0
        for sentence in split_sentences(flat):
            found = flat.find(sentence, position)
            if found < 0:
                continue
            total_sentences += 1
            lo, hi = found, found + len(sentence)
            inside = [b for b in boundaries if lo < b < hi]
            if inside:
                cut += 1
                if len(examples) < limit:
                    b = inside[0]
                    examples.append(dict(date=statement.date, sentence=sentence, boundary=b,
                                         first_half=flat[lo:b], second_half=flat[b:hi]))
            position = hi
    return dict(sentences=total_sentences, sentences_cut=cut,
                share_cut=round(cut / max(total_sentences, 1), 4), examples=examples)


def chunk_report(chunks: Sequence[Chunk], reader_context: int = 256) -> dict:
    """Counts, lengths, sections, and how many chunks are longer than the reader's window."""
    lengths = np.array([len(c.text) for c in chunks], dtype=float)
    sections: dict[str, int] = {}
    for c in chunks:
        sections[c.section] = sections.get(c.section, 0) + 1
    return dict(
        strategy=chunks[0].strategy if chunks else "",
        n_chunks=len(chunks),
        characters=dict(total=int(lengths.sum()), mean=round(float(lengths.mean()), 1),
                        median=float(np.median(lengths)), minimum=float(lengths.min()),
                        maximum=float(lengths.max())),
        sections=dict(sorted(sections.items(), key=lambda kv: -kv[1])),
        opens_mid_sentence=int(sum(c.opens_mid_sentence for c in chunks)),
        closes_mid_sentence=int(sum(c.closes_mid_sentence for c in chunks)),
        longer_than_reader_context=dict(context=reader_context,
                                        n=int((lengths > reader_context).sum()),
                                        share=round(float((lengths > reader_context).mean()), 4)),
    )



# --------------------------------------------------------------------------------------
# Blocks B4-B5 -- retrieval in numpy: TF-IDF, BM25, and the fusion of the two
# --------------------------------------------------------------------------------------
# Curly punctuation is in the corpus -- the 78-character vocabulary Session 1 fitted contains a
# right single quote, an en dash, an em dash, a non-breaking hyphen and one replacement character
# -- so it is folded to ASCII before tokenising. Without the fold, a statement that spells the
# possessive with the typographic apostrophe and one that spells it with the typewriter apostrophe
# put "committee's" in the vocabulary twice, and a query matches only one of them.
_FOLD = {"’": "'", "‘": "'", "“": '"', "”": '"',
         "‑": "-", "–": " ", "—": " ", "�": " "}
_TOKEN = re.compile(r"[a-z0-9]+(?:[-/'][a-z0-9]+)*")


def normalise(text: str) -> str:
    for bad, good in _FOLD.items():
        text = text.replace(bad, good)
    return text.lower()


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens, keeping the hyphen, the slash and the apostrophe inside a word.

    The slash matters here and nowhere else: this corpus writes rates as "1/4 to 1/2 percent" and
    "3-1/2 percent", and a tokenizer that throws the slash away turns every rate into the same
    bag of small integers. No stop-word list: inverse document frequency already prices "the" at
    almost nothing, and a hand-written list is one more thing that can be wrong.
    """
    return _TOKEN.findall(normalise(text))


@dataclass
class TermSpace:
    """The vocabulary fitted on the corpus, and the term counts of every chunk in it."""
    vocabulary: dict[str, int]
    counts: np.ndarray                  # (n_documents, n_terms), raw term counts
    lengths: np.ndarray                 # (n_documents,), tokens per document
    document_frequency: np.ndarray      # (n_terms,), documents containing the term

    @property
    def n_documents(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n_terms(self) -> int:
        return int(self.counts.shape[1])

    def query_counts(self, query: str) -> np.ndarray:
        """Query terms outside the fitted vocabulary contribute nothing -- which is the point."""
        vector = np.zeros(self.n_terms, dtype=np.float64)
        for token in tokenize(query):
            index = self.vocabulary.get(token)
            if index is not None:
                vector[index] += 1.0
        return vector

    def unseen_terms(self, query: str) -> list[str]:
        return sorted({t for t in tokenize(query) if t not in self.vocabulary})


def fit_term_space(texts: Sequence[str]) -> TermSpace:
    """One pass to collect the vocabulary, a second to fill the count matrix. Dense on purpose.

    A thousand chunks by five thousand terms is forty megabytes, which fits, and a dense matrix
    turns the whole of retrieval into one matrix-vector product the notebook can print.
    """
    tokenised = [tokenize(t) for t in texts]
    vocabulary: dict[str, int] = {}
    for tokens in tokenised:
        for token in tokens:
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)
    counts = np.zeros((len(texts), len(vocabulary)), dtype=np.float64)
    for row, tokens in enumerate(tokenised):
        for token in tokens:
            counts[row, vocabulary[token]] += 1.0
    return TermSpace(vocabulary=vocabulary, counts=counts,
                     lengths=counts.sum(axis=1),
                     document_frequency=(counts > 0).sum(axis=0).astype(np.float64))


class TfidfIndex:
    """TF-IDF with the vocabulary fitted on the corpus, scored by cosine similarity.

    tf is 1 + log(count) for a term that occurs, and zero otherwise -- sublinear, because a
    paragraph that says "inflation" six times is not six times as much about inflation.
    idf is log((N + 1) / (df + 1)) + 1, the smoothed form, so a term in every document still
    carries a little weight rather than exactly none and no term can divide by zero.
    Rows are L2-normalised, so a score is a cosine in [0, 1] and scores from different queries
    are on the same scale. That last property is what makes a single refusal threshold possible.
    """

    def __init__(self, texts: Sequence[str], space: TermSpace | None = None) -> None:
        self.space = space if space is not None else fit_term_space(texts)
        n = self.space.n_documents
        self.idf = np.log((n + 1.0) / (self.space.document_frequency + 1.0)) + 1.0
        weights = np.where(self.space.counts > 0, 1.0 + np.log(np.maximum(self.space.counts, 1.0)), 0.0)
        weights = weights * self.idf
        self.norms = np.linalg.norm(weights, axis=1)
        self.matrix = weights / np.maximum(self.norms, 1e-12)[:, None]

    def vector(self, query: str) -> np.ndarray:
        counts = self.space.query_counts(query)
        weights = np.where(counts > 0, 1.0 + np.log(np.maximum(counts, 1.0)), 0.0) * self.idf
        return weights / max(float(np.linalg.norm(weights)), 1e-12)

    def scores(self, query: str) -> np.ndarray:
        return self.matrix @ self.vector(query)

    def rank(self, query: str) -> np.ndarray:
        return rank_of(self.scores(query))

    def search(self, query: str, top_k: int = TOP_K) -> list[tuple[int, float]]:
        scores = self.scores(query)
        order = rank_of(scores)[:top_k]
        return [(int(i), float(scores[i])) for i in order]


class BM25Index:
    """BM25 in the Robertson-Sparck Jones form, written out in numpy.

        score(q, d) = sum over query terms t of  idf(t) * f(t,d) * (k1 + 1)
                                                 -------------------------------------
                                                 f(t,d) + k1 * (1 - b + b * |d| / avgdl)

        idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

    `k1` controls how fast term frequency saturates: at k1 = 0 a term counts once however often it
    appears, and as k1 grows the score approaches raw frequency. `b` controls length normalisation:
    b = 0 ignores document length, b = 1 divides through by it in full. The defaults here are
    k1 = 1.5 and b = 0.75, the values Robertson and Walker reported working across TREC
    collections and the ones nearly every implementation ships with.

    The half added to the numerator and denominator of the idf is the Robertson-Sparck Jones
    smoothing; the outer 1 + is the standard guard, because without it a term appearing in more
    than half the documents earns a negative weight and can be used to push a document down.
    Unlike the cosine above, a BM25 score has no ceiling and is not comparable across queries --
    which is exactly why the fusion below reads ranks and not scores.
    """

    def __init__(self, texts: Sequence[str], k1: float = BM25_K1, b: float = BM25_B,
                 space: TermSpace | None = None) -> None:
        self.space = space if space is not None else fit_term_space(texts)
        self.k1, self.b = float(k1), float(b)
        n = self.space.n_documents
        df = self.space.document_frequency
        self.idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        self.average_length = float(self.space.lengths.mean())
        denominator_length = self.k1 * (1.0 - self.b + self.b * self.space.lengths / self.average_length)
        f = self.space.counts
        self.matrix = self.idf * (f * (self.k1 + 1.0)) / (f + denominator_length[:, None])

    def scores(self, query: str) -> np.ndarray:
        # Query-term frequency is clipped to one, as Lucene and most implementations do: asking
        # the same question with a word repeated should not change which document comes back.
        return self.matrix @ np.minimum(self.space.query_counts(query), 1.0)

    def rank(self, query: str) -> np.ndarray:
        return rank_of(self.scores(query))

    def search(self, query: str, top_k: int = TOP_K) -> list[tuple[int, float]]:
        scores = self.scores(query)
        order = rank_of(scores)[:top_k]
        return [(int(i), float(scores[i])) for i in order]


def rank_of(scores: np.ndarray) -> np.ndarray:
    """Document indices from best to worst, ties broken by index so a run is reproducible."""
    return np.lexsort((np.arange(scores.size), -scores))


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], n_documents: int,
                           k: int = RRF_K) -> np.ndarray:
    """Combine several rankings by their ranks alone: RRF(d) = sum over rankings of 1 / (k + rank(d)).

    Cormack, Clarke and Büttcher (2009), "Reciprocal Rank Fusion outranks Condorcet and individual
    rank learning methods", SIGIR '09. Their point is that the arithmetic needs neither training
    nor score calibration: a cosine in [0, 1] and an unbounded BM25 score cannot be added, but
    their ranks can. k = 60 is the constant they report, and it does one job -- it flattens the
    top of the curve, so that the gap between rank 1 and rank 2 (1/61 against 1/62) is small and a
    document has to do well in more than one ranking to rise. Returns a fused score per document,
    higher is better.
    """
    fused = np.zeros(n_documents, dtype=np.float64)
    for ranking in rankings:
        for position, document in enumerate(ranking, start=1):
            fused[int(document)] += 1.0 / (k + position)
    return fused


class FusionRetriever:
    """TF-IDF and BM25 over the same chunks, fused by reciprocal rank.

    `depth` is how far down each ranking the fusion looks. Fusing the full ranking would let a
    document that both retrievers put near the bottom accumulate weight; 100 is deep enough that
    nothing a retriever considers plausible is dropped.
    """

    def __init__(self, chunks: Sequence[Chunk], k: int = RRF_K, depth: int = 100,
                 k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.chunks = list(chunks)
        texts = [c.indexed_text() for c in self.chunks]
        self.space = fit_term_space(texts)
        self.tfidf = TfidfIndex(texts, space=self.space)
        self.bm25 = BM25Index(texts, k1=k1, b=b, space=self.space)
        self.k, self.depth = int(k), int(depth)

    def scores(self, query: str) -> np.ndarray:
        rankings = (self.tfidf.rank(query)[:self.depth], self.bm25.rank(query)[:self.depth])
        return reciprocal_rank_fusion(rankings, len(self.chunks), k=self.k)

    def rank(self, query: str) -> np.ndarray:
        return rank_of(self.scores(query))

    def search(self, query: str, top_k: int = TOP_K) -> list[tuple[int, float]]:
        scores = self.scores(query)
        order = rank_of(scores)[:top_k]
        return [(int(i), float(scores[i])) for i in order]

    def named(self) -> dict[str, object]:
        """The three retrievers under the names the notebook and the slides use."""
        return {"tf-idf": self.tfidf, "bm25": self.bm25, "fusion": self}


# --------------------------------------------------------------------------------------
# Blocks B2, B6 -- the gold set, recall@k, the mean reciprocal rank
# --------------------------------------------------------------------------------------
@dataclass
class Question:
    """One graded question. `anchors` is empty exactly when the corpus cannot answer it."""
    id: str
    question: str
    kind: str                   # "in_corpus" or "out_of_corpus"
    anchors: list[str] = field(default_factory=list)      # ISO dates of the statements that answer it
    topic: str = ""
    note: str = ""

    @property
    def answerable(self) -> bool:
        return self.kind == "in_corpus"


def load_gold(path: Path = GOLD) -> list[Question]:
    """The graded questions, with the two classes checked against each other.

    An out-of-corpus question with an anchor, or an in-corpus question without one, is a bug in
    the file rather than a hard case, so it raises here instead of quietly distorting a threshold.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = [Question(**{k: v for k, v in row.items() if k in Question.__annotations__})
                 for row in payload["questions"]]
    for q in questions:
        if q.kind not in ("in_corpus", "out_of_corpus"):
            raise RuntimeError(f"{q.id}: kind must be in_corpus or out_of_corpus, not {q.kind!r}")
        if q.answerable and not q.anchors:
            raise RuntimeError(f"{q.id} is marked in_corpus but names no anchor statement")
        if not q.answerable and q.anchors:
            raise RuntimeError(f"{q.id} is marked out_of_corpus but names anchor {q.anchors}")
    if not any(q.answerable for q in questions) or not any(not q.answerable for q in questions):
        raise RuntimeError("the gold set needs both kinds: a threshold fitted on one class is not a threshold")
    return questions


def check_gold_against_corpus(questions: Sequence[Question], statements: Sequence[Statement]) -> dict:
    """Every anchor names a statement that is actually here, and no unanswerable question does.

    Cheap, and it catches the failure that would quietly wreck every number below: an anchor
    date with a typo scores as a permanent miss, and recall falls without anything looking wrong.
    """
    dates = {s.date for s in statements}
    missing = sorted({a for q in questions for a in q.anchors if a not in dates})
    if missing:
        raise RuntimeError(f"gold anchors name statements this corpus does not have: {missing}")
    return dict(questions=len(questions),
                in_corpus=sum(1 for q in questions if q.answerable),
                out_of_corpus=sum(1 for q in questions if not q.answerable),
                anchors=sorted({a for q in questions for a in q.anchors}),
                anchors_checked=True)


def first_anchor_rank(retriever, chunks: Sequence[Chunk], question: Question,
                      depth: int = 50) -> int | None:
    """One-based rank of the first chunk from an anchor statement, or None within `depth`."""
    anchors = set(question.anchors)
    for position, index in enumerate(retriever.rank(question.question)[:depth], start=1):
        if chunks[int(index)].date in anchors:
            return position
    return None


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Share of answerable questions with an anchor chunk somewhere in the top k.

    Recall, not precision, because a question here has one right document and the retriever is
    allowed several guesses; what matters is whether the reader is ever shown the right one.
    """
    if not ranks:
        return 0.0
    return float(np.mean([1.0 if (r is not None and r <= k) else 0.0 for r in ranks]))


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float:
    """Mean of 1/rank of the first correct document; a miss contributes zero.

    It is the recall@k curve summarised in one number, and it is harsh in the right way: moving
    the answer from rank 3 to rank 1 is worth more than moving it from rank 30 to rank 10.
    """
    if not ranks:
        return 0.0
    return float(np.mean([0.0 if r is None else 1.0 / r for r in ranks]))


def evaluate_retriever(retriever, chunks: Sequence[Chunk], questions: Sequence[Question],
                       ks: Sequence[int] = (1, 3, 5, 10), depth: int = 50) -> dict:
    """recall@k and MRR over the answerable questions, with the per-question ranks kept."""
    answerable = [q for q in questions if q.answerable]
    ranks = [first_anchor_rank(retriever, chunks, q, depth=depth) for q in answerable]
    return dict(
        n_questions=len(answerable), depth=depth,
        recall_at_k={str(k): round(recall_at_k(ranks, k), 4) for k in ks},
        mean_reciprocal_rank=round(mean_reciprocal_rank(ranks), 4),
        misses=[q.id for q, r in zip(answerable, ranks) if r is None],
        ranks={q.id: r for q, r in zip(answerable, ranks)},
    )


# --------------------------------------------------------------------------------------
# Blocks C1-C4, D -- the grounded answer, and a refusal that is a number
# --------------------------------------------------------------------------------------
@dataclass
class Evidence:
    """One sentence, and where it came from. The pair is the whole point of the exercise."""
    sentence: str
    date: str
    section: str
    chunk_id: int
    score: float

    def cited(self) -> str:
        return f"{self.sentence} [FOMC statement of {self.date}]"


@dataclass
class Answer:
    """What the pipeline returns: either cited sentences, or a refusal with the number behind it."""
    question: str
    refused: bool
    score: float
    threshold: float
    evidence: list[Evidence]
    considered: list[dict]
    text: str

    def to_json(self) -> dict:
        return dict(question=self.question, refused=self.refused, score=round(self.score, 6),
                    threshold=round(self.threshold, 6), text=self.text,
                    evidence=[dict(sentence=e.sentence, date=e.date, section=e.section,
                                   chunk_id=e.chunk_id, score=round(e.score, 6)) for e in self.evidence],
                    considered=self.considered)


def evidence_score(retriever: FusionRetriever, question: str, top_k: int = TOP_K) -> float:
    """The refusal statistic: the strongest TF-IDF cosine among the chunks the answer may quote.

    A cosine and not the fused rank score, because the fused score depends only on ranks and so is
    almost the same number for every query -- it cannot tell a query the corpus answers from one
    it does not. A cosine can: it is bounded, it is comparable across queries, and it falls when
    the query's distinctive terms are missing from the corpus altogether.
    """
    tfidf = retriever.tfidf.scores(question)
    top = retriever.rank(question)[:top_k]
    return float(max(tfidf[int(i)] for i in top)) if len(top) else 0.0


REFUSAL_TEMPLATE = (
    "No statement in this corpus supports an answer. The best passage retrieved scored {score:.3f} "
    "against an evidence threshold of {threshold:.3f} calibrated on the graded question set, and "
    "the corpus runs from {first} to {last}. Answering would mean writing a citation that the "
    "corpus does not carry."
)


def grounded_answer(question: str, retriever: FusionRetriever, threshold: float,
                    top_k: int = TOP_K, max_sentences: int = 3,
                    first: str = "", last: str = "") -> Answer:
    """Return the sentences actually used, each carrying the statement it came from.

    Extractive on purpose: nothing is paraphrased, so every word in the answer can be found in a
    statement by searching for it. Sentences are scored in the same TF-IDF space the chunks are
    indexed in, date and section included (see `_index_sentence`), plus a quarter of their chunk's
    own score -- so a strong sentence in a weak chunk can still be quoted, and a weak sentence in
    a strong chunk cannot ride in on the chunk alone.

    The refusal is the same arithmetic with nothing to show for it: when the best evidence scores
    below `threshold`, the function says so and names the number. It is not an instruction to a
    model to be careful, which is what the 2025 version of this lab used and what a model is free
    to ignore; it is a comparison, and it either holds or it does not.
    """
    hits = retriever.search(question, top_k=top_k)
    tfidf_scores = retriever.tfidf.scores(question)
    considered = [dict(chunk_id=int(i), date=retriever.chunks[i].date,
                       section=retriever.chunks[i].section, fused_score=round(float(s), 6),
                       cosine=round(float(tfidf_scores[i]), 6)) for i, s in hits]
    score = float(max((tfidf_scores[i] for i, _ in hits), default=0.0))
    if score < threshold:
        return Answer(question=question, refused=True, score=score, threshold=threshold,
                      evidence=[], considered=considered,
                      text=REFUSAL_TEMPLATE.format(score=score, threshold=threshold,
                                                   first=first or "1994-02-04", last=last or "2026-06-17"))
    scored: list[tuple[float, Evidence]] = []
    seen: set[str] = set()
    question_vector = retriever.tfidf.vector(question)
    for index, _ in hits:
        chunk = retriever.chunks[index]
        for sentence in split_sentences(chunk.text):
            key = re.sub(r"\W+", "", sentence.lower())
            if len(sentence) < 25 or key in seen:
                continue
            seen.add(key)
            similarity = float(retriever.tfidf.vector(_index_sentence(chunk, sentence))
                               @ question_vector)
            total = similarity + 0.25 * float(tfidf_scores[index])
            scored.append((total, Evidence(sentence=sentence, date=chunk.date, section=chunk.section,
                                           chunk_id=int(index), score=round(similarity, 6))))
    scored.sort(key=lambda pair: -pair[0])
    evidence = [e for _, e in scored[:max_sentences]]
    body = "\n".join(f"- {e.cited()}" for e in evidence)
    return Answer(question=question, refused=False, score=score, threshold=threshold,
                  evidence=evidence, considered=considered,
                  text=f"From the statements retrieved, the sentences that bear on the question are:\n{body}")


def _index_sentence(chunk: Chunk, sentence: str) -> str:
    """A sentence in the same space its chunk is indexed in: the date and the section come along.

    Without this a dated question ranks sentences on boilerplate alone -- every meeting since 2015
    contains "the target range for the federal funds rate" -- and the sentence from the meeting the
    question names loses to an identical sentence from a meeting three years away.
    """
    return f"{chunk.date} {_pretty(chunk.date)} {chunk.date[:4]} {chunk.section} {sentence}"


def calibrate_refusal(retriever: FusionRetriever, questions: Sequence[Question],
                      top_k: int = TOP_K) -> dict:
    """Choose the evidence threshold on the gold set, and report what it costs.

    Refusing is the positive class. Precision is the share of refusals that were right -- of the
    questions turned away, how many the corpus really could not answer. Recall is the share of
    unanswerable questions that were turned away. The threshold maximises F1, with ties broken in
    favour of answering more of the answerable questions, because a retriever that refuses
    everything scores perfect recall and is useless.

    The candidate thresholds are the midpoints between neighbouring observed scores; there is no
    other place a threshold can change a decision, so this is the whole of the search space.
    """
    scores = {q.id: evidence_score(retriever, q.question, top_k=top_k) for q in questions}
    answerable = [q for q in questions if q.answerable]
    unanswerable = [q for q in questions if not q.answerable]
    values = sorted(scores.values())
    candidates = sorted({0.0, *(round((a + b) / 2.0, 6) for a, b in zip(values, values[1:])),
                         round(values[-1] + 1e-6, 6)})
    rows = []
    for tau in candidates:
        refused_unanswerable = sum(1 for q in unanswerable if scores[q.id] < tau)
        refused_answerable = sum(1 for q in answerable if scores[q.id] < tau)
        refused = refused_unanswerable + refused_answerable
        precision = refused_unanswerable / refused if refused else 1.0
        recall = refused_unanswerable / len(unanswerable) if unanswerable else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        rows.append(dict(threshold=tau, refusals=refused, precision=round(precision, 4),
                         recall=round(recall, 4), f1=round(f1, 4),
                         answerable_kept=len(answerable) - refused_answerable,
                         answerable_refused=refused_answerable))
    best = max(rows, key=lambda r: (r["f1"], r["answerable_kept"], -r["threshold"]))
    return dict(threshold=best["threshold"], chosen=best, grid=rows,
                scores={k: round(v, 6) for k, v in scores.items()},
                score_by_kind=dict(
                    in_corpus=_describe([scores[q.id] for q in answerable]),
                    out_of_corpus=_describe([scores[q.id] for q in unanswerable])),
                n_in_corpus=len(answerable), n_out_of_corpus=len(unanswerable),
                statistic="maximum TF-IDF cosine among the top-k fused chunks", top_k=top_k)


def _describe(values: Sequence[float]) -> dict:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return dict(n=0)
    return dict(n=int(array.size), minimum=round(float(array.min()), 6),
                median=round(float(np.median(array)), 6), mean=round(float(array.mean()), 6),
                maximum=round(float(array.max()), 6))


def refusal_cost(retriever: FusionRetriever, chunks: Sequence[Chunk], questions: Sequence[Question],
                 threshold: float, ks: Sequence[int] = (1, 3, 5, 10), top_k: int = TOP_K) -> dict:
    """recall@k over the answerable questions before and after the threshold turns some away.

    An answerable question that is refused counts as a miss at every k. That is the honest way to
    price a refusal: the corpus had the answer, the pipeline declined to give it.
    """
    answerable = [q for q in questions if q.answerable]
    ranks = [first_anchor_rank(retriever, chunks, q) for q in answerable]
    kept = [evidence_score(retriever, q.question, top_k=top_k) >= threshold for q in answerable]
    after = [r if keep else None for r, keep in zip(ranks, kept)]
    return dict(
        threshold=threshold,
        recall_before={str(k): round(recall_at_k(ranks, k), 4) for k in ks},
        recall_after={str(k): round(recall_at_k(after, k), 4) for k in ks},
        mrr_before=round(mean_reciprocal_rank(ranks), 4),
        mrr_after=round(mean_reciprocal_rank(after), 4),
        answerable=len(answerable), answerable_refused=int(sum(1 for k in kept if not k)),
        refused_ids=[q.id for q, keep in zip(answerable, kept) if not keep],
    )

def chunk_size_sweep(statements: Sequence[Statement], sizes: Sequence[int] = (600, 800, 1000, 1400),
                     questions: Sequence[Question] | None = None) -> list[dict]:
    """Grade the whole pipeline at several `max_chars`, so the chosen value is a result, not a taste."""
    questions = list(questions) if questions is not None else load_gold()
    rows = []
    for size in sizes:
        chunks = document_chunks(statements, max_chars=size)
        retriever = FusionRetriever(chunks)
        graded = {name: evaluate_retriever(index, chunks, questions)
                  for name, index in retriever.named().items()}
        chosen = calibrate_refusal(retriever, questions)["chosen"]
        lengths = np.array([len(c.text) for c in chunks], dtype=float)
        rows.append(dict(max_chars=size, chunks=len(chunks),
                         median_characters=float(np.median(lengths)),
                         recall_at_1={k: v["recall_at_k"]["1"] for k, v in graded.items()},
                         recall_at_5={k: v["recall_at_k"]["5"] for k, v in graded.items()},
                         mean_reciprocal_rank={k: v["mean_reciprocal_rank"] for k, v in graded.items()},
                         refusal_f1=chosen["f1"], refusal_precision=chosen["precision"],
                         refusal_recall=chosen["recall"],
                         answerable_refused=chosen["answerable_refused"]))
    return rows


# --------------------------------------------------------------------------------------
# Block C5 -- three failures, reproduced on purpose
# --------------------------------------------------------------------------------------
# Each of these returns a dict with "before" and "after". The failure is not a bug to be fixed in
# the code; it is a property of the pipeline that the lab wants visible, so each one is staged
# with the smallest change that makes it go away, and both sides are recorded.

STALE_CUTOFF = "2026-01-01"
STALE_QUESTION = ("what did the Committee say about maintaining ample reserves in the banking "
                  "system at the meeting of June 17, 2026")
STALE_ANCHOR = "2026-06-17"


def failure_stale_index(statements: Sequence[Statement], question: str = STALE_QUESTION,
                        anchor: str = STALE_ANCHOR, cutoff: str = STALE_CUTOFF,
                        top_k: int = TOP_K, threshold: float = 0.0) -> dict:
    """An index built before the answer was published. The retriever does not know it is missing.

    This is the failure with no symptom. The 2026-06-17 statement is the first in the corpus to
    speak of maintaining ample reserves; an index built on everything before 2026 cannot return
    it, and does not say so -- it returns its best match from the documents it does have, with a
    perfectly ordinary-looking score. Rebuilding the index is the whole fix, which is why a RAG
    system's refresh schedule is part of its accuracy and not an operational detail.
    """
    before_statements = [s for s in statements if s.date < cutoff]
    out = {}
    for label, subset in (("before", before_statements), ("after", list(statements))):
        chunks = document_chunks(subset)
        retriever = FusionRetriever(chunks)
        hits = retriever.search(question, top_k=top_k)
        answer = grounded_answer(question, retriever, threshold, top_k=top_k,
                                 first=subset[0].date, last=subset[-1].date)
        out[label] = dict(
            statements_indexed=len(subset), chunks=len(chunks),
            newest_document=subset[-1].date,
            anchor_in_index=any(s.date == anchor for s in subset),
            top_dates=[retriever.chunks[i].date for i, _ in hits],
            top_score=round(float(hits[0][1]), 6) if hits else 0.0,
            top_cosine=round(evidence_score(retriever, question, top_k=top_k), 6),
            anchor_rank=first_anchor_rank(retriever, chunks,
                                          Question(id="stale", question=question, kind="in_corpus",
                                                   anchors=[anchor])),
            answer=answer.text,
        )
    out["question"], out["anchor"], out["cutoff"] = question, anchor, cutoff
    out["reading"] = ("The stale index answers with the wrong meeting and gives no sign of it; the "
                      "score it returns is in the ordinary range. Only the rebuilt index can reach "
                      f"the statement of {anchor}.")
    return out


def failure_chunk_boundary(statements: Sequence[Statement], chunk_size: int = FIXED_CHUNK_CHARS,
                           top_k: int = TOP_K, threshold: float = 0.0) -> dict:
    """An answer split across two chunks, so that no single chunk can be quoted for it.

    The fixed-size chunker is given no reason to stop at a sentence, so sooner or later it stops
    in the middle of one. When that sentence is the decision sentence, the retriever can still
    find the document -- both halves share its vocabulary -- but every chunk it can hand a reader
    is a fragment. The document-aware chunker never creates the situation, because a sentence is
    the smallest thing it is willing to cut at.
    """
    fixed = fixed_size_chunks(statements, chunk_size=chunk_size)
    damage = boundary_damage(fixed, statements, limit=10 ** 6)
    picked = _worst_cut(damage["examples"], chunk_size)
    if picked is None:
        raise RuntimeError(f"no sentence is cut at chunk_size={chunk_size}; choose another size")
    question = (f"what did the Committee decide about the target range for the federal funds rate "
                f"at the meeting of {_pretty(picked['date'])}")
    out = {}
    for label, chunks in (("before", fixed), ("after", document_chunks(statements))):
        retriever = FusionRetriever(chunks)
        hits = retriever.search(question, top_k=top_k)
        whole = [int(i) for i, _ in hits if picked["sentence"] in retriever.chunks[int(i)].text]
        coverage = max((_coverage(picked["sentence"], retriever.chunks[int(i)].text) for i, _ in hits),
                       default=0.0)
        answer = grounded_answer(question, retriever, threshold, top_k=top_k,
                                 first=statements[0].date, last=statements[-1].date)
        out[label] = dict(
            strategy=chunks[0].strategy, chunks=len(chunks),
            top_dates=[retriever.chunks[int(i)].date for i, _ in hits],
            anchor_retrieved=picked["date"] in [retriever.chunks[int(i)].date for i, _ in hits],
            whole_sentence_in_a_retrieved_chunk=bool(whole),
            best_coverage_of_the_sentence=round(coverage, 4),
            answer=answer.text,
        )
    out.update(question=question, date=picked["date"], sentence=picked["sentence"],
               first_half=picked["first_half"], second_half=picked["second_half"],
               cut_after_characters=len(picked["first_half"]), sentence_characters=len(picked["sentence"]),
               cut_inside_a_word=bool(picked["first_half"][-1:].strip() and picked["second_half"][:1].strip()),
               chunk_size=chunk_size, sentences=damage["sentences"],
               sentences_cut=damage["sentences_cut"], share_cut=damage["share_cut"],
               decision_sentences_cut=sum(1 for e in damage["examples"] if _DECIDED.search(e["sentence"])))
    out["reading"] = ("Both chunkers retrieve the right statement. Only the document-aware one can "
                      "hand back the decision sentence whole; the fixed-size one offers a fragment "
                      "that stops mid-clause, and a citation attached to a fragment is a citation "
                      "to something the statement never said.")
    return out


_DECIDED = re.compile(r"\bdecided (today )?to\b", re.I)


def _worst_cut(examples: Sequence[dict], chunk_size: int) -> dict | None:
    """Pick the cut sentence the lab should print: a decision, cut nearest its middle.

    Sentences longer than a whole chunk are excluded -- those are cut because they could not fit,
    which is a different and less interesting complaint. What is wanted is a sentence that would
    have fitted comfortably and was cut anyway, because the chunker was counting characters.
    """
    candidates = [e for e in examples
                  if _DECIDED.search(e["sentence"]) and len(e["sentence"]) < chunk_size]
    candidates = candidates or [e for e in examples if len(e["sentence"]) < chunk_size] or list(examples)
    if not candidates:
        return None
    return min(candidates, key=lambda e: (abs(len(e["first_half"]) / len(e["sentence"]) - 0.5), e["date"]))


def _coverage(sentence: str, text: str) -> float:
    """Longest run of the sentence's words that appears contiguously in the text, as a fraction."""
    words = sentence.split()
    for length in range(len(words), 0, -1):
        for start in range(0, len(words) - length + 1):
            if " ".join(words[start:start + length]) in text:
                return length / len(words)
    return 0.0


AMBIGUOUS_QUESTION = ("what did the Committee say when it raised the target range for the federal "
                      "funds rate to 1/4 to 1/2 percent")
AMBIGUOUS_MEETINGS = ("2015-12-16", "2022-03-16")


def failure_ambiguous_query(statements: Sequence[Statement], question: str = AMBIGUOUS_QUESTION,
                            meetings: Sequence[str] = AMBIGUOUS_MEETINGS, top_k: int = TOP_K,
                            threshold: float = 0.0) -> dict:
    """A query that retrieves confidently from the wrong meeting, because two meetings match it.

    The Committee raised the target range to 1/4 to 1/2 percent twice -- lifting off the zero
    bound on 2015-12-16, and again on 2022-03-16 after the pandemic cuts. The wording of the
    question fits both perfectly, so the retriever picks one on a margin that has nothing to do
    with the user's intent, and reports no doubt about it. The fix is not a better retriever: it
    is a question that says which meeting, which is why dated questions are the gold set's style.
    """
    chunks = document_chunks(statements)
    retriever = FusionRetriever(chunks)
    phrase = "1/4 to 1/2 percent"
    carriers = sorted({s.date for s in statements if phrase in " ".join(s.paragraphs)})
    intended = meetings[0]
    out = {}
    for label, query in (("before", question),
                         ("after", f"{question} at the meeting of {_pretty(intended)}")):
        hits = retriever.search(query, top_k=top_k)
        cosine = retriever.tfidf.scores(query)
        best = {date: round(max((float(cosine[i]) for i in range(len(chunks)) if chunks[i].date == date),
                                default=0.0), 6) for date in carriers}
        top_date = retriever.chunks[int(hits[0][0])].date if hits else ""
        answer = grounded_answer(query, retriever, threshold, top_k=top_k,
                                 first=statements[0].date, last=statements[-1].date)
        out[label] = dict(
            query=query, top_dates=[retriever.chunks[int(i)].date for i, _ in hits],
            top_date=top_date, top_score=round(float(hits[0][1]), 6) if hits else 0.0,
            top_cosine=round(evidence_score(retriever, query, top_k=top_k), 6),
            best_cosine_by_meeting=best,
            margin=round(abs(best.get(top_date, 0.0) - best.get(intended, 0.0)), 6),
            intended_meeting_first=bool(top_date == intended),
            intended_meeting_rank=first_anchor_rank(
                retriever, chunks, Question(id="ambiguous", question=query, kind="in_corpus",
                                            anchors=[intended])),
            answer=answer.text,
        )
    out.update(question=question, meetings=list(meetings), intended=intended, phrase=phrase,
               statements_containing_the_phrase=carriers,
               n_statements_containing_the_phrase=len(carriers))
    out["reading"] = (
        f"The phrase {phrase!r} appears in {len(carriers)} statements -- the Committee arrived at "
        f"that target range in December 2015, stayed there through 2016, and returned to it in "
        f"March 2022. Undated, the question retrieves {out['before']['top_date']} at a cosine "
        f"{out['before']['margin']:.3f} above the meeting the user meant, and reports nothing "
        f"unusual while doing it. Adding the date puts {intended} first. A retriever's confidence "
        f"is a statement about vocabulary overlap, not about whether the right document was found.")
    return out


# --------------------------------------------------------------------------------------
# Block C6 -- the reader: a second character model, with a window twice as wide
# --------------------------------------------------------------------------------------
# Session 1's Machine B reads 128 characters at a time. The experiment below needs an answer
# sentence and a passage of matching length in the window together -- about 240 characters for
# this corpus, where the sentences run to roughly 120 and the chunks to roughly 400. At a context
# of 128 the passage is cut away entirely and what is measured is truncation, not reading. So this
# model is the same architecture family with `context` doubled to 256 and nothing else changed,
# trained here once so the notebook only ever measures it.
READER_CONFIG = {"d_model": 128, "n_heads": 4, "n_layers": 3, "context": 256, "dropout": 0.1}
READER_FILES = ("reader_fomc.pt", "reader_config.json", "reader_vocab.json", "loss_table.json")
READER_STEPS = 1800
READER_BATCH = 32
READER_LEARNING_RATE = 3e-4
READER_WARMUP = 100
READER_CLIP = 1.0
READER_HOLD_OUT_EVERY = 20
READER_EVALUATE_EVERY = 50
READER_THREADS = 8


@dataclass
class Reader:
    """The trained 256-character model and everything needed to speak to it in characters."""
    model: MiniGPT
    vocabulary: list[str]
    stoi: dict[str, int]
    config: dict
    manifest: dict
    loss_table: dict
    parameters: int

    def encode(self, text: str) -> torch.Tensor:
        unknown = sorted(set(text) - set(self.stoi))
        if unknown:
            raise ValueError(f"characters outside the {len(self.vocabulary)}-character vocabulary: {unknown}")
        return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)


def reader_data(statements: Sequence[Statement] | None = None,
                every: int = READER_HOLD_OUT_EVERY) -> dict:
    """Session 1's split: every `every`-th whole statement held out, never a random window.

    The unit is the statement. Joining the corpus into one string and splitting it on blank lines
    gives paragraphs, not statements -- Session 5 hit that and fixed it -- and a paragraph split
    leaks, because the other four paragraphs of a held-out meeting stay in training and say much
    the same thing in much the same words. The split is done by Session 5's `character_split`, on
    the statement bodies this module loads, so there is one implementation of it in the course.
    """
    bodies = [s.body for s in statements] if statements is not None else corpus_statements()
    text = "\n\n".join(bodies)
    vocabulary = sorted(set(text))
    split = character_split(bodies, vocabulary, every=every)
    split["vocabulary"] = vocabulary
    split["blank_line_blocks"] = sum(len([b for b in body.split("\n\n") if b.strip()]) for body in bodies)
    split["corpus_sha256"] = text_sha256(text)
    return split


def train_reader(steps: int = READER_STEPS, batch: int = READER_BATCH,
                 learning_rate: float = READER_LEARNING_RATE, warmup: int = READER_WARMUP,
                 clip: float = READER_CLIP, seed: int = SEED,
                 evaluate_every: int = READER_EVALUATE_EVERY, every: int = READER_HOLD_OUT_EVERY,
                 destination: Path = READER_DIR, threads: int = READER_THREADS,
                 statements: Sequence[Statement] | None = None) -> dict:
    """Train the 256-character reader once, write it out with its loss table and a manifest.

    The recipe is Session 1's -- AdamW at 3e-4, linear warm-up over 100 steps, gradients clipped
    at 1.0, batches of 32 windows -- with the window doubled from 128 to 256 characters. Doubling
    the window roughly doubles the cost of a step, so this is 1,800 steps rather than Session 1's
    3,000, which is about a quarter of an hour of one CPU. The loss table is written at every
    `evaluate_every` steps and is what the notebook plots; nothing is trained in the notebook.
    """
    torch.set_num_threads(max(1, int(threads)))
    started = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    statements = list(statements) if statements is not None else load_statements()
    data = reader_data(statements, every=every)
    vocabulary = data["vocabulary"]
    size = len(vocabulary)
    model = MiniGPT(size, READER_CONFIG)
    context = model.context
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: min(1.0, (s + 1) / warmup))

    def batch_from(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(0, len(ids) - context - 1, (batch,))
        return (torch.stack([ids[s: s + context] for s in starts]),
                torch.stack([ids[s + 1: s + context + 1] for s in starts]))

    @torch.no_grad()
    def validation_loss(n_batches: int = 8) -> float:
        model.eval()
        losses = [F.cross_entropy(model(x).view(-1, size), y.view(-1)).item()
                  for x, y in (batch_from(data["validation"]) for _ in range(n_batches))]
        model.train()
        return float(np.mean(losses))

    model.train()
    train_curve, validation_curve = [], []
    for step in range(1, steps + 1):
        x, y = batch_from(data["train"])
        loss = F.cross_entropy(model(x).view(-1, size), y.view(-1))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimiser.step()
        schedule.step()
        train_curve.append((step, round(float(loss.item()), 4)))
        if step == 1 or step % evaluate_every == 0 or step == steps:
            validation_curve.append((step, round(validation_loss(), 4)))
    seconds = time.perf_counter() - started

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.save(model.state_dict(), destination / "reader_fomc.pt")
    (destination / "reader_config.json").write_text(json.dumps(READER_CONFIG, indent=1), encoding="utf-8")
    (destination / "reader_vocab.json").write_text(json.dumps(vocabulary), encoding="utf-8")
    loss_table = dict(
        uniform_loss=round(float(np.log(size)), 4),
        evaluate_every=evaluate_every, steps=steps,
        train=[[s, v] for s, v in train_curve if s == 1 or s % evaluate_every == 0 or s == steps],
        validation=[[s, v] for s, v in validation_curve],
        final_train=train_curve[-1][1], final_validation=validation_curve[-1][1],
        first_train=train_curve[0][1], first_validation=validation_curve[0][1],
        train_every_step=[[s, v] for s, v in train_curve],
    )
    (destination / "loss_table.json").write_text(json.dumps(loss_table, indent=1), encoding="utf-8")
    manifest = dict(
        written=time.strftime("%Y-%m-%d"),
        what=("A character-level MiniGPT with a 256-character context, trained here once so the "
              "notebook can measure a reader whose window is wide enough to hold a retrieved "
              "passage. Session 1's Machine B is the same architecture at context 128."),
        architecture=dict(vocabulary_size=size, **READER_CONFIG),
        parameters=parameter_count(model),
        classes=("session05_attention.MiniGPT, which reproduces Session 1's classes verbatim; "
                 "imported, not copied a third time"),
        seed=seed,
        training=dict(steps=steps, batch=batch, optimizer="AdamW", learning_rate=learning_rate,
                      warmup=warmup, gradient_clip=clip, context=context),
        hold_out=dict(rule=f"every {every}th statement, by whole statement",
                      n_statements=data["n_statements"], n_held_out=data["n_held_out"],
                      train_characters=data["train_characters"],
                      validation_characters=data["validation_characters"],
                      blank_line_blocks_if_split_naively=data["blank_line_blocks"],
                      note=("Splitting the joined corpus on blank lines would give "
                            f"{data['blank_line_blocks']} paragraphs rather than "
                            f"{data['n_statements']} statements, and would leave four fifths of "
                            "every held-out meeting in the training text.")),
        corpus=dict(statements=data["n_statements"], characters=data["characters"],
                    sha256=data["corpus_sha256"], manifest_sha256=file_sha256(CORPUS_MANIFEST)),
        loss=dict(final_train=loss_table["final_train"], final_validation=loss_table["final_validation"],
                  uniform=loss_table["uniform_loss"]),
        cost=dict(seconds=round(seconds, 1), minutes=round(seconds / 60.0, 2),
                  seconds_per_step=round(seconds / max(steps, 1), 4),
                  torch_threads=int(threads), device="cpu",
                  machine=platform.platform(), processor=platform.processor() or "unknown",
                  note=("Wall-clock for the training run on one CPU at "
                        f"{threads} threads; the notebook does not repeat it.")),
        versions=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__),
        files=[dict(file=name, bytes=(destination / name).stat().st_size,
                    sha256=file_sha256(destination / name)) for name in READER_FILES],
    )
    (destination / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return dict(manifest=manifest, loss_table=loss_table, seconds=seconds)


def load_reader(directory: Path = READER_DIR, check: bool = True) -> Reader:
    """Load the reader after proving the four files are the ones the manifest records."""
    directory = Path(directory)
    manifest = json.loads((directory / "MANIFEST.json").read_text(encoding="utf-8"))
    if check:
        for row in manifest["files"]:
            got = file_sha256(directory / row["file"])
            if got != row["sha256"]:
                raise RuntimeError(f"{row['file']} does not match reader/MANIFEST.json "
                                   f"({got[:12]} against {row['sha256'][:12]})")
    config = json.loads((directory / "reader_config.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((directory / "reader_vocab.json").read_text(encoding="utf-8"))
    loss_table = json.loads((directory / "loss_table.json").read_text(encoding="utf-8"))
    model = MiniGPT(len(vocabulary), config)
    state = torch.load(directory / "reader_fomc.pt", map_location="cpu", weights_only=True)
    mine = model.state_dict()
    if set(mine) != set(state):
        raise RuntimeError("the checkpoint's state-dict keys are not this architecture's")
    model.load_state_dict(state)
    model.eval()
    n = parameter_count(model)
    if n != manifest["parameters"]:
        raise RuntimeError(f"parameter count is {n:,}, the manifest records {manifest['parameters']:,}")
    return Reader(model=model, vocabulary=vocabulary, stoi={c: i for i, c in enumerate(vocabulary)},
                  config=config, manifest=manifest, loss_table=loss_table, parameters=n)


@torch.no_grad()
def window_loss(reader: Reader, prefix: str, target: str) -> float:
    """Mean cross-entropy, in nats per character, over `target` alone, given `prefix` before it.

    Everything before the target is context the model may use and is not scored; only the
    characters of the target are. Both conditions in the experiments below put the target at
    exactly the same positions in the window, so the numbers differ only in what the model was
    given to read, never in where it had to read it.
    """
    text = prefix + target
    if len(text) - 1 > reader.model.context:
        raise ValueError(f"window of {len(text)} characters exceeds the model's context of "
                         f"{reader.model.context}")
    ids = reader.encode(text)
    logits = reader.model(ids[None, :-1])
    losses = F.cross_entropy(logits.view(-1, logits.shape[-1]), ids[1:].view(-1), reduction="none")
    return float(losses[len(prefix) - 1:].mean())


READER_TARGET_CHARS = 120       # so that target + separator + target fits a 256-character window
READER_UNRELATED_YEARS = 10     # "unrelated" means a meeting at least this far away


def _truncate_at_word(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return cut[:space] if space > limit // 2 else cut


def _best_sentence(retriever: FusionRetriever, question: str, statement: Statement) -> str:
    """The sentence in one statement that best answers the question, in the retriever's own space."""
    query = retriever.tfidf.vector(question)
    sentences = [s for p in statement.paragraphs for s in split_sentences(p) if len(s) >= 40]
    if not sentences:
        sentences = [statement.body[:200]]
    return max(sentences, key=lambda s: float(retriever.tfidf.vector(s) @ query))


def _filler(statements: Sequence[Statement], anchor: str, length: int, rng: np.random.Generator) -> str:
    """Text of exactly `length` characters from a meeting at least ten years from the anchor."""
    far = [s for s in statements if abs(int(s.date[:4]) - int(anchor[:4])) >= READER_UNRELATED_YEARS]
    far = far or list(statements)
    pool = ""
    while len(pool) < length:
        pool += " " + " ".join(far[int(rng.integers(len(far)))].paragraphs)
    return pool[-length:]


def reader_experiment(reader: Reader, retriever: FusionRetriever, statements: Sequence[Statement],
                      questions: Sequence[Question], target_chars: int = READER_TARGET_CHARS,
                      seed: int = SEED, top_k: int = TOP_K) -> dict:
    """Retrieval without a reader is not grounding, measured in nats per character.

    Two experiments, both scoring the same answer sentence and both putting it at exactly the same
    positions in the window, so that only the context differs.

    `passage`   the model is given the passage retrieval actually returned for the question, with
                the answer sentence itself removed, and then has to predict the answer sentence.
                The control is a passage from a meeting at least ten years away. If retrieval were
                grounding, the right passage would make the answer much cheaper to predict.
    `verbatim`  the answer sentence is placed in the window twice, and only the second copy is
                scored. The control puts unrelated text in the first half. This is the easiest
                possible use of a context -- copy what you were just shown -- and the gap between
                the two conditions is the clearest measure of whether the model reads its window
                at all.

    A model that cannot read returns nearly the same number in both conditions of both
    experiments. That is not a failure of retrieval: the right passage was found and handed over.
    It is what the phrase "the reader is the other half of the system" costs when the reader is
    a three-layer character model.
    """
    rng = np.random.default_rng(seed)
    by_date = {s.date: s for s in statements}
    context = reader.model.context
    rows = []
    for q in questions:
        if not q.answerable or q.anchors[0] not in by_date:
            continue
        anchor = by_date[q.anchors[0]]
        sentence = _best_sentence(retriever, q.question, anchor)
        target = _truncate_at_word(sentence, target_chars)
        hits = retriever.search(q.question, top_k=top_k)
        # The whole source sentence comes out of the passage, not just the part being scored, so
        # that no continuation of the target survives in the context and makes the task easier.
        body = retriever.chunks[int(hits[0][0])].text.replace(sentence, " ").strip() if hits else ""
        passage_prefix_length = context - len(target)
        padded = len(body) < passage_prefix_length
        # Padded at the front with distant text only when the passage is shorter than the window
        # allows; at 800-character chunks and a 136-character prefix it rarely is.
        prefix_retrieved = (_filler(statements, anchor.date, passage_prefix_length, rng) + " " + body
                            )[-passage_prefix_length:]
        prefix_unrelated = _filler(statements, anchor.date, passage_prefix_length, rng)
        copy_prefix_length = len(target) + 1
        rows.append(dict(
            id=q.id, date=anchor.date, retrieved_date=retriever.chunks[int(hits[0][0])].date if hits else "",
            target_characters=len(target), target=target,
            source_sentence_characters=len(sentence), passage_padded=bool(padded),
            passage_retrieved=round(window_loss(reader, prefix_retrieved, target), 4),
            passage_unrelated=round(window_loss(reader, prefix_unrelated, target), 4),
            verbatim_copy=round(window_loss(reader, target + " ", target), 4),
            verbatim_control=round(window_loss(
                reader, _filler(statements, anchor.date, copy_prefix_length, rng), target), 4),
        ))
    if not rows:
        raise RuntimeError("no answerable gold question has an anchor statement in the corpus")

    def column(name: str) -> np.ndarray:
        return np.array([r[name] for r in rows], dtype=float)

    passage_gap = column("passage_unrelated") - column("passage_retrieved")
    verbatim_gap = column("verbatim_control") - column("verbatim_copy")
    chunk_lengths = np.array([len(c.text) for c in retriever.chunks], dtype=float)
    return dict(
        n_questions=len(rows), target_characters=target_chars, context=context,
        rows=rows,
        passage=dict(
            retrieved=round(float(column("passage_retrieved").mean()), 4),
            unrelated=round(float(column("passage_unrelated").mean()), 4),
            gap=round(float(passage_gap.mean()), 4),
            gap_share_of_unrelated=round(float(passage_gap.mean() / column("passage_unrelated").mean()), 4),
            helped=int((passage_gap > 0).sum()),
            reading=("The passage retrieval returned makes the answer sentence this much cheaper "
                     "to predict than a passage from a meeting ten years away.")),
        verbatim=dict(
            copy=round(float(column("verbatim_copy").mean()), 4),
            control=round(float(column("verbatim_control").mean()), 4),
            gap=round(float(verbatim_gap.mean()), 4),
            gap_share_of_control=round(float(verbatim_gap.mean() / column("verbatim_control").mean()), 4),
            helped=int((verbatim_gap > 0).sum()),
            reading=("The answer sentence is already in the window, character for character. A "
                     "reader that used its context would score near zero on the second copy.")),
        reference=dict(
            validation_loss=reader.loss_table["final_validation"],
            uniform_loss=reader.loss_table["uniform_loss"],
            note="ln(vocabulary size) is the loss of a model that has learned nothing."),
        window=dict(
            reader_context=context, machine_b_context=128,
            chunks=len(retriever.chunks),
            longer_than_128=int((chunk_lengths > 128).sum()),
            longer_than_256=int((chunk_lengths > 256).sum()),
            median_chunk_characters=float(np.median(chunk_lengths)),
            median_answer_sentence_characters=float(np.median(column("target_characters"))),
            note=("Why a second model had to be trained. Each condition needs an answer sentence "
                  "and a passage of the same length in the window at once. The answer sentences "
                  "here run to about 120 characters, so the pair needs roughly 240 -- more than "
                  "Session 1's Machine B has. At a context of 128 the passage would be cut away "
                  "entirely and the experiment would measure truncation instead of reading. At 256 "
                  "both fit. Retrieved chunks are longer still, which is a separate point the "
                  "chunking table makes: a reader small enough to train in a lab cannot hold a "
                  "whole retrieved paragraph either.")),
    )


# --------------------------------------------------------------------------------------
# The recorded run (the instructor's record)
# --------------------------------------------------------------------------------------
EXAMPLE_QUESTIONS = (
    "what did the Committee say about inflation at the meeting of December 16, 2015",
    "what did the Committee decide about the target range for the federal funds rate "
    "at the meeting of March 16, 2022",
    "what did the Bank of Japan decide about its yield curve control policy in 2016",
)


def build_records(path: Path = RECORDED, train: bool = True,
                  statements: Sequence[Statement] | None = None) -> dict:
    """Everything the notebook or a slide might print, computed once and written to JSON."""
    started = time.perf_counter()
    corpus = copy_corpus()
    statements = list(statements) if statements is not None else load_statements()
    first, last = statements[0].date, statements[-1].date
    documents = document_chunks(statements)
    fixed = fixed_size_chunks(statements)
    retriever = FusionRetriever(documents)
    questions = load_gold()
    gold_check = check_gold_against_corpus(questions, statements)

    graded = {name: evaluate_retriever(index, documents, questions)
              for name, index in retriever.named().items()}
    refusal = calibrate_refusal(retriever, questions)
    threshold = refusal["threshold"]
    cost = refusal_cost(retriever, documents, questions, threshold)
    fixed_retriever = FusionRetriever(fixed)
    graded_fixed = {name: evaluate_retriever(index, fixed, questions)
                    for name, index in fixed_retriever.named().items()}

    answers = [grounded_answer(q, retriever, threshold, first=first, last=last).to_json()
               for q in EXAMPLE_QUESTIONS]
    gold_answers = {q.id: grounded_answer(q.question, retriever, threshold, first=first, last=last).to_json()
                    for q in questions}

    failures = dict(
        stale_index=failure_stale_index(statements, threshold=threshold),
        chunk_boundary=failure_chunk_boundary(statements, threshold=threshold),
        ambiguous_query=failure_ambiguous_query(statements, threshold=threshold),
    )

    if train or not READER_MANIFEST.exists():
        train_reader(statements=statements)
    reader = load_reader()
    experiment = reader_experiment(reader, retriever, statements, questions)

    record = dict(
        _meta=dict(
            helper=Path(__file__).name,
            helper_sha256=file_sha256(__file__),
            written=time.strftime("%Y-%m-%d"), run_seconds=0.0, seed=SEED,
            corpus=corpus,
            gold_questions=dict(path=GOLD.name, sha256=file_sha256(GOLD),
                                n=len(questions),
                                n_in_corpus=sum(1 for q in questions if q.answerable),
                                n_out_of_corpus=sum(1 for q in questions if not q.answerable),
                                anchors=gold_check["anchors"]),
            reader=dict(path=str(READER_DIR.name), manifest_sha256=file_sha256(READER_MANIFEST),
                        checkpoint_sha256=file_sha256(READER_DIR / "reader_fomc.pt")),
            versions=dict(python=platform.python_version(), numpy=np.__version__,
                          torch=torch.__version__, platform=platform.platform()),
            settings=dict(rrf_k=RRF_K, bm25_k1=BM25_K1, bm25_b=BM25_B, top_k=TOP_K,
                          document_max_chars=DOCUMENT_MAX_CHARS, fixed_chunk_chars=FIXED_CHUNK_CHARS,
                          fixed_overlap=FIXED_OVERLAP),
        ),
        corpus=dict(reflow=reflow_report(statements),
                    statements=[dict(date=s.date, paragraphs=len(s.paragraphs), characters=len(s.body))
                                for s in statements]),
        chunking=dict(document=chunk_report(documents), fixed=chunk_report(fixed),
                      boundary_damage=boundary_damage(fixed, statements, limit=5),
                      size_sweep=chunk_size_sweep(statements, questions=questions),
                      size_sweep_note=("Why DOCUMENT_MAX_CHARS is 800: every column is best there. "
                                       "At 1000 and 1400 the cosines dilute and the refusal "
                                       "threshold turns away an answerable question; at 600 every "
                                       "retriever loses rank-1 accuracy.")),
        retrieval=dict(document_aware=graded, fixed_size=graded_fixed,
                       note=("Both tables are the same questions and the same three retrievers; "
                             "only the chunking differs.")),
        refusal=dict(calibration=refusal, cost=cost,
                     refusal_template=REFUSAL_TEMPLATE,
                     note=("The threshold is a number fitted on the gold set, not a sentence in a "
                           "prompt. The gold set has to contain questions the corpus cannot answer "
                           "or there is nothing to fit it against.")),
        answers=dict(examples=answers, gold=gold_answers),
        failures=failures,
        reader=dict(manifest=reader.manifest, loss_table=reader.loss_table, experiment=experiment),
    )
    record["_meta"]["run_seconds"] = round(time.perf_counter() - started, 1)
    Path(path).write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def self_test() -> None:
    """A short end-to-end run that prints the numbers a slide would quote."""
    t0 = time.perf_counter()
    corpus = copy_corpus()
    statements = load_statements()
    print(f"corpus: {corpus['n']} statements, {corpus['first']} to {corpus['last']}, hashes checked")
    flow = reflow_report(statements)
    print(f"  paragraphs {flow['paragraphs']} (blank-line blocks {flow['blocks_before_reflow']}; "
          f"{flow['statements_repaired']} statements reflowed; "
          f"{flow['sentences_truncated_in_the_source']} sentences cut off in the source)")
    documents, fixed = document_chunks(statements), fixed_size_chunks(statements)
    for chunks in (documents, fixed):
        r = chunk_report(chunks)
        print(f"  {r['strategy']:<9} {r['n_chunks']:>5} chunks, median {r['characters']['median']:.0f} chars, "
              f"{r['closes_mid_sentence']} of them end mid-sentence")
    damage = boundary_damage(fixed, statements, limit=1)
    print(f"  fixed-size chunking cuts {damage['sentences_cut']} of {damage['sentences']} sentences "
          f"({100 * damage['share_cut']:.1f}%)")

    retriever = FusionRetriever(documents)
    questions = load_gold()
    check_gold_against_corpus(questions, statements)
    print(f"\ngold set: {len(questions)} questions, "
          f"{sum(1 for q in questions if not q.answerable)} of them unanswerable from this corpus")
    for name, index in retriever.named().items():
        g = evaluate_retriever(index, documents, questions)
        print(f"  {name:<8} recall@1 {g['recall_at_k']['1']:.3f}  recall@5 {g['recall_at_k']['5']:.3f}  "
              f"recall@10 {g['recall_at_k']['10']:.3f}  MRR {g['mean_reciprocal_rank']:.3f}")

    refusal = calibrate_refusal(retriever, questions)
    chosen, tau = refusal["chosen"], refusal["threshold"]
    print(f"\nrefusal threshold {tau:.4f}: precision {chosen['precision']:.3f}, recall {chosen['recall']:.3f}, "
          f"{chosen['answerable_refused']} answerable question(s) turned away")
    cost = refusal_cost(retriever, documents, questions, tau)
    print(f"  recall@5 {cost['recall_before']['5']:.3f} before the threshold, "
          f"{cost['recall_after']['5']:.3f} after")

    for name, demo in (("stale index", failure_stale_index(statements, threshold=tau)),
                       ("chunk boundary", failure_chunk_boundary(statements, threshold=tau)),
                       ("ambiguous query", failure_ambiguous_query(statements, threshold=tau))):
        print(f"\n{name}:")
        for side in ("before", "after"):
            row = demo[side]
            print(f"  {side:<7} " + ", ".join(f"{k}={v}" for k, v in row.items()
                                              if k in ("statements_indexed", "anchor_rank", "top_date",
                                                       "whole_sentence_in_a_retrieved_chunk",
                                                       "best_coverage_of_the_sentence", "margin",
                                                       "intended_meeting_first")))

    if READER_MANIFEST.exists():
        reader = load_reader()
        experiment = reader_experiment(reader, retriever, statements, questions)
        print(f"\nreader ({reader.parameters:,} parameters, context {reader.model.context}, "
              f"validation loss {reader.loss_table['final_validation']:.3f}):")
        print(f"  answer sentence given the retrieved passage {experiment['passage']['retrieved']:.3f} "
              f"nats/char, given an unrelated passage {experiment['passage']['unrelated']:.3f}")
        print(f"  the same sentence already in the window {experiment['verbatim']['copy']:.3f}, "
              f"against {experiment['verbatim']['control']:.3f} with unrelated text there")
    else:
        print(f"\nreader: not trained yet; run `python session06_rag.py train` ({READER_STEPS} steps).")
    print(f"\n{time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    if sys.argv[1:2] == ["train"]:
        out = train_reader()
        print(f"reader trained in {out['seconds'] / 60:.1f} minutes, "
              f"final validation loss {out['loss_table']['final_validation']:.4f}")
    elif sys.argv[1:2] == ["record"]:
        rec = build_records(train="--no-train" not in sys.argv)
        print(f"wrote {RECORDED.name} in {rec['_meta']['run_seconds']:.1f} s")
        for name, table in rec["retrieval"]["document_aware"].items():
            print(f"  {name:<8} recall@5 {table['recall_at_k']['5']:.3f}  "
                  f"MRR {table['mean_reciprocal_rank']:.3f}")
    else:
        self_test()
