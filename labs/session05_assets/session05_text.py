"""Session 5 lab -- text as data: counts, vectors, and four candidate signals.

Part A  Preprocessing as a chain of switchable steps. Lower-casing, punctuation, numbers,
        stop-words and stemming are researcher degrees of freedom, so each is its own function
        and the table in `degrees_of_freedom` shows what each switch costs in vocabulary.
Part B  A toy byte-pair encoder learned on the training window: `learn_bpe` returns the merge
        list in the order it was learned, `apply_bpe` segments a word with it.
Part C  Document-term matrix and TF-IDF. Vocabulary and IDF are fitted on a caller-supplied
        subset -- by default the 1994-2014 training window -- and only then applied to the rest,
        so the representation never peeks at documents it would not have had.
Part C2 Latent Dirichlet allocation by collapsed Gibbs sampling, on the same training window.
Part D  PPMI-SVD word vectors from a co-occurrence matrix, fitted on the same subset.
Part E  Four signals over the same statements: the TF-IDF representation, a seed-word hawk-dove
        axis with a novelty index, a regex on the decision sentence, and the interface the
        separately recorded language-model stance score has to satisfy.
Part F  Validation against the Fed's own hiking and cutting cycles, per cycle, so the caller can
        see where a signal works and where it fails.

The headline result of the lab is a failure and a success side by side. The embedding hawk-dove
score cannot separate hiking from cutting meetings -- the 2004-06 hiking cycle reads as the most
dovish stretch in the sample, because a Committee that is removing accommodation says
"accommodation" in every paragraph. A ten-line regex aimed at the one sentence that announces the
decision separates them cleanly. Identification precedes measurement.

Descended from Lab 7A of the 2025 Lecture 7 lab (`labs/Lec07_LLM_Lab/Lab7A_Text_as_Data.ipynb`),
rewritten as a library: no module-level computation, fitting confined to a caller-supplied subset,
and both validation outcomes recorded rather than narrated.
Imports: numpy, pandas and the standard library. No network, no third-party NLP package.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import platform
import re
import time

if __name__ == "__main__":      # instructor runs only: keep the linear-algebra library polite
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "4")
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FOMC_DIR = HERE / "fomc"
MANIFEST = HERE / "fomc_manifest.json"
RECORDED = HERE / "recorded_text.json"
LLM_SCORES = HERE / "llm_stance" / "scores.csv"

SEED = 42
TRAIN_END = "2014-12-31"        # the text training window ends here; 2015 onwards is held back


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------------------
# The corpus
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Statement:
    """One post-meeting FOMC statement: its date, its body, and the file it came from."""
    date: str
    text: str
    file: str


def load_corpus(directory: Path = FOMC_DIR, check_hashes: bool = True) -> list[Statement]:
    """Date-sorted statements with the two-line provenance header removed.

    With `check_hashes` the sha256 of every file is compared against `fomc_manifest.json`, so a
    silently edited corpus stops the lab instead of quietly changing every number downstream.
    """
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    recorded = {row["file"]: row["sha256"] for row in manifest["files"]}
    out: list[Statement] = []
    for path in sorted(Path(directory).glob("fomc_*.txt")):
        if check_hashes:
            want = recorded.get(path.name)
            if want is None:
                raise RuntimeError(f"{path.name} is not in the corpus manifest")
            if file_sha256(path) != want:
                raise RuntimeError(f"{path.name} does not match its manifest hash")
        stamp = path.stem.split("_")[1]
        body = "\n".join(ln for ln in path.read_text(encoding="utf-8").splitlines()
                         if not ln.startswith("#")).strip()
        if body:
            out.append(Statement(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}", body, path.name))
    if check_hashes and len(out) != manifest["n"]:
        raise RuntimeError(f"corpus has {len(out)} statements, the manifest records {manifest['n']}")
    return out


def training_window(statements: Sequence[Statement], end: str = TRAIN_END) -> list[Statement]:
    """The subset every fitted object in this module is allowed to see."""
    return [s for s in statements if s.date <= end]


def corpus_summary(statements: Sequence[Statement]) -> dict:
    chars = [len(s.text) for s in statements]
    return dict(n=len(statements), first=statements[0].date, last=statements[-1].date,
                characters=int(sum(chars)), words=int(sum(len(s.text.split()) for s in statements)),
                median_characters=int(np.median(chars)))


# --------------------------------------------------------------------------------------
# Part A -- preprocessing, one switch at a time
# --------------------------------------------------------------------------------------
# The slide's point: none of these five choices is forced by the data, every one of them moves the
# vocabulary, and a paper that does not report them cannot be replicated. So each is a function.

TOKEN_RE = re.compile(r"[a-z']+|\d+(?:\.\d+)?|%")
PUNCT_RE = re.compile(r"[^\w\s%']+")
NUMBER_RE = re.compile(r"\d+(?:[.,/\-]\d+)*")

# A short, auditable function-word list. Which words belong on it is itself a researcher choice:
# "not", "no" and "less" are deliberately absent, because negation carries policy meaning.
STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being
below between both but by can did do does doing down during each few for from further had has have
having he her here hers him his how i if in into is it its itself just me more most my nor of off
on once only or other our ours out over own same she should so some such than that the their theirs
them then there these they this those through to too under until up very was we were what when
where which while who whom why will with would you your yours
""".split())

# A crude suffix stripper, longest match first. Porter's algorithm is the standard; this is a
# transparent stand-in with no dependency, and a step whose damage the table below makes visible:
# "housing" becomes "hous", "policies" becomes "polic", and "inflation" becomes "infl", which is
# why the stemmed row of `degrees_of_freedom` has a top term nobody would choose to print.
_SUFFIXES = ("ational", "iveness", "fulness", "ousness", "ization", "ation", "ition", "ingly",
             "ously", "ments", "ment", "ness", "ally", "ance", "ence", "able", "ible", "ical",
             "ing", "ers", "ies", "est", "ive", "ity", "ful", "ous", "ary", "ory",
             "ed", "es", "er", "ly", "s")
MIN_STEM = 4


def to_lower(text: str) -> str:
    """Step 1. Cheap, and it merges "Inflation" with "inflation" -- and "US" with "us"."""
    return text.lower()


def drop_punctuation(text: str) -> str:
    """Step 2. Removes everything that is not a word character, whitespace, apostrophe or percent."""
    return PUNCT_RE.sub(" ", text)


def drop_numbers(text: str) -> str:
    """Step 3. Deletes numerals, including "5-1/4" and "2.5" -- and with them every rate level."""
    return NUMBER_RE.sub(" ", text)


def split_tokens(text: str) -> list[str]:
    """Step 4. The tokenizer itself: words, bare numbers and the percent sign."""
    return TOKEN_RE.findall(text)


def drop_stopwords(tokens: Sequence[str], stopwords: frozenset = STOPWORDS) -> list[str]:
    """Step 5. Removes function words. Cheap for topics, destructive for syntax and negation."""
    return [t for t in tokens if t not in stopwords]


def crude_stem(tokens: Sequence[str]) -> list[str]:
    """Step 6. Strips the longest matching suffix, never below `MIN_STEM` characters."""
    out = []
    for token in tokens:
        for suffix in _SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= MIN_STEM:
                token = token[: -len(suffix)]
                break
        out.append(token)
    return out


def drop_short(tokens: Sequence[str], min_length: int = 2) -> list[str]:
    """Step 7. Drops one-character tokens, which in this corpus are mostly list markers."""
    return [t for t in tokens if len(t) >= min_length]


@dataclass(frozen=True)
class Pipeline:
    """A named combination of the switches above. `run` is the whole chain in order."""
    name: str
    lower: bool = True
    punctuation: bool = False
    numbers: bool = False
    stopwords: bool = False
    stem: bool = False
    min_length: int = 1

    def steps(self) -> list[str]:
        on = [("lower-case", self.lower), ("drop punctuation", self.punctuation),
              ("drop numbers", self.numbers), ("tokenize", True),
              ("drop stop-words", self.stopwords), ("stem", self.stem),
              (f"drop tokens shorter than {self.min_length}", self.min_length > 1)]
        return [label for label, active in on if active]

    def run(self, text: str) -> list[str]:
        if self.lower:
            text = to_lower(text)
        if self.punctuation:
            text = drop_punctuation(text)
        if self.numbers:
            text = drop_numbers(text)
        tokens = split_tokens(text)
        if self.stopwords:
            tokens = drop_stopwords(tokens)
        if self.stem:
            tokens = crude_stem(tokens)
        if self.min_length > 1:
            tokens = drop_short(tokens, self.min_length)
        return tokens


# The baseline is Lab 7A's tokenizer exactly: lower-case, then the regex. Everything else is a
# deliberate departure from it, so the table below reads as a ladder away from a known reference.
BASELINE = Pipeline("baseline: lower-case + tokenize")
PIPELINES: tuple[Pipeline, ...] = (
    BASELINE,
    Pipeline("+ drop numbers", numbers=True),
    Pipeline("+ drop stop-words", numbers=True, stopwords=True),
    Pipeline("+ stem", numbers=True, stopwords=True, stem=True),
    Pipeline("+ drop 1-character tokens", numbers=True, stopwords=True, stem=True, min_length=2),
    Pipeline("everything on", punctuation=True, numbers=True, stopwords=True, stem=True, min_length=2),
)


def degrees_of_freedom(statements: Sequence[Statement],
                       pipelines: Sequence[Pipeline] = PIPELINES) -> pd.DataFrame:
    """One row per preprocessing choice: tokens kept, vocabulary size, and the top five terms.

    Every row is the same corpus. The spread across rows is the size of the researcher's hand.
    """
    rows = []
    for pipe in pipelines:
        counter: collections.Counter = collections.Counter()
        for s in statements:
            counter.update(pipe.run(s.text))
        total = sum(counter.values())
        rows.append(dict(pipeline=pipe.name, steps=" -> ".join(pipe.steps()), tokens=total,
                         vocabulary=len(counter),
                         top_terms=", ".join(w for w, _ in counter.most_common(5))))
    frame = pd.DataFrame(rows).set_index("pipeline")
    base = frame["vocabulary"].iloc[0]
    frame["vocabulary_vs_baseline"] = (frame["vocabulary"] / base - 1.0).round(3)
    return frame


def word_frequencies(statements: Sequence[Statement], pipeline: Pipeline = BASELINE) -> collections.Counter:
    counter: collections.Counter = collections.Counter()
    for s in statements:
        counter.update(pipeline.run(s.text))
    return counter


# --------------------------------------------------------------------------------------
# Part B -- byte-pair encoding, learned here
# --------------------------------------------------------------------------------------
END = "</w>"


def learn_bpe(corpus: Iterable[str] | Iterable[Statement] | Mapping[str, int], n_merges: int = 200,
              pipeline: Pipeline = BASELINE) -> list[tuple[str, str]]:
    """Learn `n_merges` byte-pair merges, in the order they were learned.

    `corpus` is a word-frequency table, an iterable of raw texts, or an iterable of `Statement`s,
    which is tokenized with `pipeline` first. Every word starts as its characters plus an
    end-of-word marker; each round counts adjacent symbol pairs, weights them by word frequency,
    and glues the most frequent pair everywhere. Ties go to the first pair `max` meets, which is
    the insertion order of the counter, so the result is deterministic without a seed.
    """
    if isinstance(corpus, Mapping):
        freqs = dict(corpus)
    else:
        counter: collections.Counter = collections.Counter()
        for item in corpus:
            counter.update(pipeline.run(item.text if isinstance(item, Statement) else item))
        freqs = dict(counter)
    vocab = {tuple(word) + (END,): count for word, count in freqs.items()}
    merges: list[tuple[str, str]] = []
    for _ in range(n_merges):
        pairs: collections.Counter = collections.Counter()
        for symbols, count in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += count
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        vocab = {_merge_once(symbols, best): count for symbols, count in vocab.items()}
    return merges


def _merge_once(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    out, i = [], 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def apply_bpe(word: str, merges: Sequence[tuple[str, str]]) -> list[str]:
    """Segment one word by replaying the merge list in the order it was learned."""
    symbols = tuple(word.lower()) + (END,)
    for pair in merges:
        symbols = _merge_once(symbols, pair)
    return list(symbols)


def bpe_report(merges: Sequence[tuple[str, str]], words: Sequence[str],
               n_show: int = 20) -> dict:
    """The first `n_show` merges and a few segmentations, shaped for a slide."""
    return dict(
        n_merges=len(merges),
        first_merges=[f"{a}+{b}" for a, b in merges[:n_show]],
        first_merges_pairs=[[a, b] for a, b in merges[:n_show]],
        segmentation={w: apply_bpe(w, merges) for w in words},
        pieces={w: len(apply_bpe(w, merges)) for w in words},
    )


# --------------------------------------------------------------------------------------
# Part C -- document-term matrix and TF-IDF
# --------------------------------------------------------------------------------------
@dataclass
class Tfidf:
    """A vocabulary and an IDF vector, fitted on one subset and applied to any statement.

    `vocabulary` is sorted, so column order never depends on iteration order. `n_fit` is the number
    of documents the IDF was computed from -- the denominator that has to stay fixed when the model
    is applied out of sample.
    """
    vocabulary: list[str]
    idf: np.ndarray
    document_frequency: np.ndarray
    n_fit: int
    min_df: int
    alphabetic_only: bool
    pipeline: Pipeline
    column: dict[str, int] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.column = {w: j for j, w in enumerate(self.vocabulary)}

    def counts(self, statements: Sequence[Statement]) -> np.ndarray:
        matrix = np.zeros((len(statements), len(self.vocabulary)))
        col = self.column
        for i, s in enumerate(statements):
            for token in self.pipeline.run(s.text):
                j = col.get(token)
                if j is not None:
                    matrix[i, j] += 1.0
        return matrix

    def transform(self, statements: Sequence[Statement]) -> np.ndarray:
        """TF times IDF. TF is the within-document share, so length does not drive the norm."""
        counts = self.counts(statements)
        totals = counts.sum(axis=1, keepdims=True)
        return counts / np.where(totals == 0.0, 1.0, totals) * self.idf

    def coverage(self, statements: Sequence[Statement]) -> float:
        """Share of tokens that fall inside the fitted vocabulary. The cost of not peeking."""
        inside = total = 0
        col = self.column
        for s in statements:
            tokens = self.pipeline.run(s.text)
            total += len(tokens)
            inside += sum(1 for t in tokens if t in col)
        return inside / total if total else float("nan")


def fit_tfidf(statements: Sequence[Statement], min_df: int = 2, alphabetic_only: bool = True,
              pipeline: Pipeline = BASELINE) -> Tfidf:
    """Fit the vocabulary and IDF on exactly the statements handed in -- nothing else.

    `min_df` keeps a word that appears in at least that many fitting documents; `alphabetic_only`
    drops bare numerals, so meeting dates and vote tallies cannot masquerade as vocabulary.
    """
    document_frequency: collections.Counter = collections.Counter()
    for s in statements:
        document_frequency.update(set(pipeline.run(s.text)))
    vocabulary = sorted(w for w, df in document_frequency.items()
                        if df >= min_df and (w[0].isalpha() or not alphabetic_only))
    df = np.array([document_frequency[w] for w in vocabulary], dtype=float)
    n_fit = len(statements)
    return Tfidf(vocabulary=vocabulary, idf=np.log(n_fit / df), document_frequency=df,
                 n_fit=n_fit, min_df=min_df, alphabetic_only=alphabetic_only, pipeline=pipeline)


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu and nv else 0.0


def top_terms(matrix: np.ndarray, model: Tfidf, row: int, k: int = 10) -> list[str]:
    """The k highest-weighted terms of one statement: what this meeting said that others did not."""
    return [model.vocabulary[j] for j in np.argsort(matrix[row])[::-1][:k]]


# The worked example a slide prints verbatim: eight short sentences, one vocabulary, no corpus.
TOY_SENTENCES: tuple[str, ...] = (
    "the committee cuts rates to support growth",
    "the committee raises rates to curb inflation",
    "the committee holds rates steady and watches inflation",
    "inflation remains elevated and the labor market is strong",
    "growth slowed and the labor market cooled",
    "policy remains accommodative and the committee removes accommodation",
    "the committee will raise rates at a measured pace",
    "risks to growth and risks to inflation are balanced",
)
TOY_LABELS: tuple[str, ...] = tuple(f"d{i + 1}" for i in range(len(TOY_SENTENCES)))


def toy_tfidf(sentences: Sequence[str] = TOY_SENTENCES,
              labels: Sequence[str] = TOY_LABELS) -> dict[str, pd.DataFrame]:
    """Counts, document frequency, IDF and TF-IDF for the worked example, as printable tables.

    Every number here can be checked by hand, which is the point: the same two formulas that run
    over 245 statements in Part C run over eight sentences a reader can add up.
    """
    docs = [s.split() for s in sentences]
    vocabulary = sorted({w for d in docs for w in d})
    n = len(docs)
    counts = pd.DataFrame([[d.count(w) for w in vocabulary] for d in docs],
                          index=list(labels), columns=vocabulary)
    df = (counts > 0).sum(axis=0)
    idf = np.log(n / df)
    tf = counts.div(counts.sum(axis=1), axis=0)
    tfidf = tf.mul(idf, axis=1)
    weights = pd.DataFrame({"documents": df, "IDF": idf.round(4)})
    weights.index.name = "term"
    cos_counts = pd.DataFrame(
        [[cosine(counts.values[i].astype(float), counts.values[j].astype(float)) for j in range(n)]
         for i in range(n)], index=list(labels), columns=list(labels))
    cos_tfidf = pd.DataFrame(
        [[cosine(tfidf.values[i], tfidf.values[j]) for j in range(n)] for i in range(n)],
        index=list(labels), columns=list(labels))
    # The full matrix is 8 by |V| and too wide for a slide, so this is the printable view: each
    # sentence's three heaviest terms, which is what TF-IDF is for in the first place.
    top = pd.DataFrame(
        [[f"{tfidf.columns[j]} {tfidf.values[i, j]:.3f}"
          for j in np.argsort(tfidf.values[i])[::-1][:3]] for i in range(n)],
        index=list(labels), columns=["top 1", "top 2", "top 3"])
    top.insert(0, "sentence", list(sentences))
    return dict(sentences=pd.Series(list(sentences), index=list(labels), name="sentence"),
                counts=counts, weights=weights, tf=tf.round(4), tfidf=tfidf.round(4),
                top_terms=top, cosine_counts=cos_counts.round(3), cosine_tfidf=cos_tfidf.round(3))


# Lab 7A's two-document matrix, kept so the older T2a slide still validates against this module.
SLIDE_DOCS: tuple[str, str] = ("central bank cuts rates to support growth",
                               "central bank raises rates to curb inflation")
SLIDE_VOCAB: tuple[str, ...] = ("central", "bank", "cuts", "raises", "rates", "support", "curb",
                                "growth", "inflation")


def slide_document_term_matrix() -> pd.DataFrame:
    return pd.DataFrame([[d.split().count(w) for w in SLIDE_VOCAB] for d in SLIDE_DOCS],
                        index=["doc 1", "doc 2"], columns=list(SLIDE_VOCAB))


def idf_lookahead(statements: Sequence[Statement], terms: Sequence[str],
                  end: str = TRAIN_END, pipeline: Pipeline = BASELINE) -> list[dict]:
    """Each term's document frequency and IDF on the training window and on the whole sample.

    The gap between the two IDFs is what look-ahead puts into a pre-`end` statement's weights:
    under the full-sample fit, its TF-IDF row depends on statements written after `end`.
    """
    rows = []
    windows = (("window", [s for s in statements if s.date <= end]), ("full", list(statements)))
    token_sets = {id(s): set(pipeline.run(s.text)) for s in statements}
    for term in terms:
        row = dict(term=term)
        for name, subset in windows:
            df = sum(1 for s in subset if term in token_sets[id(s)])
            row[f"df_{name}"] = df
            row[f"n_{name}"] = len(subset)
            row[f"idf_{name}"] = float(np.log(len(subset) / df)) if df else None
        rows.append(row)
    return rows


# The terms the look-ahead slide and cell A4 print: one whose weight a full-sample fit cuts by a
# third, one it cuts by a similar share, and one that did not exist in the window at all.
LOOKAHEAD_TERMS: tuple[str, ...] = ("balance", "securities", "pandemic")


# --------------------------------------------------------------------------------------
# Part C2 -- a topic model on the same counts
# --------------------------------------------------------------------------------------
# Latent Dirichlet allocation fitted by collapsed Gibbs sampling, written out so the update is the
# one on the slide: Pr(z = k | rest) is proportional to (n_kv + b) / (n_k + V b) * (n_ik + a).
# Like every other fitted object here it sees the training window only.

LDA_PIPELINE = Pipeline("topic model: + drop punctuation, numbers, stop-words, short tokens",
                        punctuation=True, numbers=True, stopwords=True, min_length=3)


@dataclass
class TopicModel:
    vocabulary: list[str]
    phi: np.ndarray                 # K x V topic-word distributions
    theta: np.ndarray               # D x K topic shares of the fitting documents
    dates: list[str]
    K: int
    a: float
    b: float
    sweeps: int
    seed: int
    min_df: int
    max_df_share: float
    tokens: int
    loglik_trace: list[float] = field(default_factory=list)

    def top_words(self, k: int, n: int = 8) -> list[str]:
        return [self.vocabulary[j] for j in np.argsort(-self.phi[k])[:n]]

    def yearly_shares(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.theta, columns=[f"topic {k + 1}" for k in range(self.K)])
        frame["year"] = [d[:4] for d in self.dates]
        return frame.groupby("year").mean()


def fit_lda_gibbs(statements: Sequence[Statement], K: int = 4, a: float = 0.1, b: float = 0.01,
                  sweeps: int = 300, seed: int = SEED, min_df: int = 5, max_df_share: float = 0.8,
                  pipeline: Pipeline = LDA_PIPELINE) -> TopicModel:
    """Collapsed Gibbs sampling for LDA on exactly the statements handed in.

    The vocabulary keeps words in at least `min_df` statements and at most `max_df_share` of them.
    Each sweep visits every token once, removes it from the counts, draws its topic from the
    conditional above, and adds it back. phi and theta are read off the final counts.
    """
    docs = [pipeline.run(s.text) for s in statements]
    df: collections.Counter = collections.Counter(w for d in docs for w in set(d))
    n = len(docs)
    vocabulary = sorted(w for w, c in df.items() if min_df <= c <= max_df_share * n)
    col = {w: j for j, w in enumerate(vocabulary)}
    words = [np.array([col[w] for w in d if w in col], dtype=int) for d in docs]
    V = len(vocabulary)
    rng = np.random.default_rng(seed)
    z = [rng.integers(K, size=len(d)) for d in words]
    n_kv, n_k, n_ik = np.zeros((K, V)), np.zeros(K), np.zeros((n, K))
    for i, d in enumerate(words):
        np.add.at(n_kv, (z[i], d), 1.0)
        np.add.at(n_ik[i], z[i], 1.0)
    n_k[:] = n_kv.sum(axis=1)
    trace = []
    for sweep in range(sweeps):
        for i, d in enumerate(words):
            zi, row = z[i], n_ik[i]
            u = rng.random(len(d))
            for ell, v in enumerate(d):
                k = zi[ell]
                n_kv[k, v] -= 1.0; n_k[k] -= 1.0; row[k] -= 1.0
                p = (n_kv[:, v] + b) / (n_k + V * b) * (row + a)
                c = np.cumsum(p)
                k = int(np.searchsorted(c, u[ell] * c[-1], side="right"))
                k = min(k, K - 1)
                zi[ell] = k
                n_kv[k, v] += 1.0; n_k[k] += 1.0; row[k] += 1.0
        if sweep % 10 == 9 or sweep == sweeps - 1:
            phi = (n_kv + b) / (n_k[:, None] + V * b)
            theta = (n_ik + a) / (n_ik.sum(axis=1, keepdims=True) + K * a)
            trace.append(float(sum(np.log(theta[i] @ phi[:, d]).sum() for i, d in enumerate(words))))
    phi = (n_kv + b) / (n_k[:, None] + V * b)
    theta = (n_ik + a) / (n_ik.sum(axis=1, keepdims=True) + K * a)
    return TopicModel(vocabulary=vocabulary, phi=phi, theta=theta,
                      dates=[s.date for s in statements], K=K, a=a, b=b, sweeps=sweeps, seed=seed,
                      min_df=min_df, max_df_share=max_df_share,
                      tokens=int(sum(len(d) for d in words)), loglik_trace=trace)


# The three-document example of the counting slides, fitted by the same sampler. Three documents are far
# too few for a stable fit, which is the slide's second point: the run is repeated under three seeds.
TOY_LDA_DOCS: tuple[str, ...] = ("fed cut support growth", "fed hike curb inflation",
                                 "fed hike respond inflation inflation")
TOY_LDA_SEEDS: tuple[int, ...] = (42, 2, 3)


def toy_lda(K: int = 2, sweeps: int = 200, seeds: Sequence[int] = TOY_LDA_SEEDS) -> list[dict]:
    """LDA on the three toy documents: top words per topic and each document's shares, per seed."""
    docs = [Statement(date=f"2000-01-0{i + 1}", text=t, file=f"d{i + 1}")
            for i, t in enumerate(TOY_LDA_DOCS)]
    plain = Pipeline("toy: lower-case + tokenize")
    rows = []
    for seed in seeds:
        m = fit_lda_gibbs(docs, K=K, sweeps=sweeps, seed=seed, min_df=1, max_df_share=1.0,
                          pipeline=plain)
        rows.append(dict(seed=seed, K=K, sweeps=sweeps, a=m.a, b=m.b,
                         top_words=[m.top_words(k, 4) for k in range(K)],
                         phi=_round(m.phi, 4), theta=_round(m.theta, 4)))
    return rows


# --------------------------------------------------------------------------------------
# Part D -- PPMI-SVD word vectors
# --------------------------------------------------------------------------------------
@dataclass
class Embedding:
    """Unit-length word vectors, so a dot product is a cosine. Fitted on one subset only."""
    vocabulary: list[str]
    vectors: np.ndarray
    counts: dict[str, int]
    singular_values: np.ndarray
    window: int
    min_count: int
    dim: int
    n_fit: int
    ppmi_density: float
    index: dict[str, int] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.index = {w: i for i, w in enumerate(self.vocabulary)}

    def __contains__(self, word: str) -> bool:
        return word in self.index

    def __getitem__(self, word: str) -> np.ndarray:
        return self.vectors[self.index[word]]


def fit_embeddings(statements: Sequence[Statement], window: int = 5, min_count: int = 10,
                   dim: int = 100, pipeline: Pipeline = BASELINE) -> Embedding:
    """Co-occurrence, positive pointwise mutual information, truncated SVD -- in that order.

    The matrix is symmetric-by-construction over a +/- `window` word neighbourhood. PPMI zeroes
    every pair that co-occurs no more often than chance, which is what turns raw counts into
    something a singular value decomposition can compress. Rows are normalised at the end, so the
    geometry is angular and nothing depends on how often a word happens to appear.

    Deterministic: `np.linalg.svd` on a dense matrix, no randomised solver, no seed.
    """
    counter = word_frequencies(statements, pipeline)
    vocabulary = [w for w, c in counter.most_common() if c >= min_count]
    index = {w: i for i, w in enumerate(vocabulary)}
    size = len(vocabulary)
    cooc = np.zeros((size, size))
    for s in statements:
        ids = [index[t] for t in pipeline.run(s.text) if t in index]
        for pos, centre in enumerate(ids):
            for context in ids[max(0, pos - window): pos] + ids[pos + 1: pos + 1 + window]:
                cooc[centre, context] += 1.0
    total = cooc.sum()
    marginal = cooc.sum(axis=1) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((cooc / total) / np.outer(marginal, marginal))
    ppmi = np.where(np.isfinite(pmi) & (pmi > 0.0), pmi, 0.0)
    left, singular, _ = np.linalg.svd(ppmi, full_matrices=False)
    vectors = left[:, :dim] * np.sqrt(singular[:dim])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0.0, 1.0, norms)
    return Embedding(vocabulary=vocabulary, vectors=vectors,
                     counts={w: int(counter[w]) for w in vocabulary},
                     singular_values=singular[: min(dim, singular.size)], window=window,
                     min_count=min_count, dim=dim, n_fit=len(statements),
                     ppmi_density=float((ppmi > 0).mean()))


def neighbors(emb: Embedding, word: str, k: int = 8) -> list[tuple[str, float]]:
    """The k nearest words by cosine, the word itself excluded."""
    sims = emb.vectors @ emb[word]
    return [(emb.vocabulary[j], round(float(sims[j]), 3))
            for j in np.argsort(sims)[::-1][1: k + 1]]


def analogy(emb: Embedding, a: str, b: str, c: str, k: int = 5) -> list[tuple[str, float]]:
    """b - a + c, the three inputs excluded from the answer. Honest about how often it misses."""
    absent = [w for w in (a, b, c) if w not in emb]
    if absent:
        raise KeyError(f"outside the fitted embedding vocabulary: {absent} "
                       f"(vocabulary is {len(emb.vocabulary)} words at min_count {emb.min_count})")
    v = emb[b] - emb[a] + emb[c]
    v = v / np.linalg.norm(v)
    sims = emb.vectors @ v
    out = [(emb.vocabulary[j], round(float(sims[j]), 3)) for j in np.argsort(sims)[::-1]
           if emb.vocabulary[j] not in {a, b, c}]
    return out[:k]


# --------------------------------------------------------------------------------------
# Part E -- four signals over the same statements
# --------------------------------------------------------------------------------------
def signal_tfidf(statements: Sequence[Statement], model: Tfidf) -> dict:
    """Signal 1: the representation itself, plus the statement-to-statement change series.

    `delta[t] = 1 - cos(x_{t-1}, x_t)` is what a trader does by hand -- read the new statement as a
    difference from the last one -- written as one line of linear algebra.
    """
    matrix = model.transform(statements)
    dates = [s.date for s in statements]
    delta = np.array([1.0 - cosine(matrix[i - 1], matrix[i]) for i in range(1, len(statements))])
    order = np.argsort(delta)[::-1]
    return dict(dates=dates, matrix=matrix, delta=delta, delta_dates=dates[1:],
                coverage=model.coverage(statements),
                largest_changes=[(dates[1:][i], round(float(delta[i]), 4)) for i in order[:10]])


HAWK_SEEDS: tuple[str, ...] = ("raise", "increase", "increased", "increasing", "tightening", "firming")
DOVE_SEEDS: tuple[str, ...] = ("lower", "easing", "eased", "accommodative", "accommodation")


def signal_embedding_hawkdove(statements: Sequence[Statement], emb: Embedding, model: Tfidf,
                              hawk: Sequence[str] = HAWK_SEEDS, dove: Sequence[str] = DOVE_SEEDS,
                              novelty_window: int = 8) -> dict:
    """Signal 2: a seed-word axis in embedding space, and a novelty index beside it.

    The axis is `mean(hawk vectors) - mean(dove vectors)`, normalised; a statement is scored by the
    cosine between that axis and its TF-IDF-weighted mean word vector, so boilerplate carries little
    weight. The novelty index is `1 - cos(x_t, centroid of the previous `novelty_window` statements)`:
    it asks how far this meeting's language sits from what the Committee has been saying lately, a
    question the hawk-dove axis is not even trying to answer.

    Read the two together. The axis is the measurement that fails validation in Part F; the novelty
    index is a measurement of something else entirely, and is not offered as a stance proxy.
    """
    hawk_in = [w for w in hawk if w in emb]
    dove_in = [w for w in dove if w in emb]
    if not hawk_in or not dove_in:
        raise RuntimeError("no seed word survived the embedding vocabulary; lower `min_count`")
    axis = emb.vectors[[emb.index[w] for w in hawk_in]].mean(axis=0) \
        - emb.vectors[[emb.index[w] for w in dove_in]].mean(axis=0)
    axis = axis / np.linalg.norm(axis)

    tfidf = model.transform(statements)
    column, index = model.column, emb.index
    doc_vectors = np.zeros((len(statements), emb.dim))
    for i, s in enumerate(statements):
        weight = 0.0
        for token in set(model.pipeline.run(s.text)):
            if token in index and token in column:
                w = tfidf[i, column[token]]
                doc_vectors[i] += w * emb.vectors[index[token]]
                weight += w
        if weight:
            doc_vectors[i] /= weight

    score = np.array([cosine(doc_vectors[i], axis) for i in range(len(statements))])
    novelty = np.full(len(statements), np.nan)
    for i in range(novelty_window, len(statements)):
        novelty[i] = 1.0 - cosine(doc_vectors[i], doc_vectors[i - novelty_window:i].mean(axis=0))
    dates = [s.date for s in statements]
    finite = np.where(np.isfinite(novelty))[0]
    order = finite[np.argsort(novelty[finite])[::-1]]
    return dict(dates=dates, score=score, novelty=novelty, axis=axis, doc_vectors=doc_vectors,
                hawk_used=hawk_in, dove_used=dove_in,
                hawk_counts={w: emb.counts[w] for w in hawk_in},
                dove_counts={w: emb.counts[w] for w in dove_in},
                novelty_window=novelty_window,
                most_novel=[(dates[i], round(float(novelty[i]), 4)) for i in order[:10]])


def embedding_centred_check(statements: Sequence[Statement], emb: Embedding, model: Tfidf) -> dict:
    """Does the hawk-dove failure survive centring? Deck 5.2's worked example subtracts the mean vector
    before taking a cosine; this asks whether doing the same to the FOMC score changes its verdict.

    Three variants of one score, each validated as `validate_signal` validates the original: the
    cosine as recorded; the cosine after subtracting the mean word vector of the embedding; the
    cosine after subtracting the mean document vector. Added 2026-10-03 (Deck 5.2 revision).
    """
    hd = signal_embedding_hawkdove(statements, emb, model)
    dates, docs, axis = hd["dates"], hd["doc_vectors"], hd["axis"]
    unit = docs / np.where(np.linalg.norm(docs, axis=1, keepdims=True) == 0, 1.0,
                           np.linalg.norm(docs, axis=1, keepdims=True))
    variants = {"as recorded": docs,
                "minus the mean word vector": docs - emb.vectors.mean(axis=0),
                "minus the mean document vector": docs - docs.mean(axis=0)}
    rows = []
    for name, matrix in variants.items():
        score = np.array([cosine(matrix[i], axis) for i in range(len(matrix))])
        v = validate_signal(dates, score, name)
        rows.append(dict(variant=name, rank_correlation=round(float(v["rank_correlation_with_subsequent_path"]), 4),
                         passes=bool(v["passes"]), n_scored=int(v["n_scored"])))
    return dict(mean_pairwise_cosine_of_document_vectors=round(float((unit @ unit.T).mean()), 4),
                rows=rows)


NOISE_SENTENCE = "The central bank raised interest rates amid rising inflation"
NOISE_CENTRE, NOISE_WINDOW, NOISE_K = "raised", 2, 2


def noise_draws(emb: Embedding, sentence: str = NOISE_SENTENCE, centre: str = NOISE_CENTRE,
                window: int = NOISE_WINDOW, k: int = NOISE_K, seed: int = SEED) -> dict:
    """Deck 5.2's word2vec example on the lab's own vocabulary: f(w), the f^(3/4) noise distribution,
    and K seeded noise draws for each real (centre, context) pair of one sentence.

    f(w) is a word's share of all tokens of the embedding vocabulary (the B1 words, counted on the
    training window). The noise distribution raises it to the 3/4 power and renormalises, as in
    Mikolov et al. (2013). Added 2026-10-03 (Deck 5.2 revision).
    """
    words = list(emb.vocabulary)                      # most frequent first
    counts = np.array([emb.counts[w] for w in words], dtype=float)
    f = counts / counts.sum()
    g = f ** 0.75
    g = g / g.sum()

    def row(j: int) -> dict:
        return dict(word=words[j], count=int(counts[j]), f=round(float(f[j]), 6),
                    p_noise=round(float(g[j]), 6), ratio=round(float(g[j] / f[j]), 3))

    tokens = BASELINE.run(sentence)
    pos = tokens.index(centre)
    contexts = tokens[max(0, pos - window): pos] + tokens[pos + 1: pos + 1 + window]
    rng = np.random.default_rng(seed)
    pairs = [dict(context=c, label=1, noise=[words[j] for j in rng.choice(len(words), size=k, p=g)])
             for c in contexts]
    top = 10
    return dict(sentence=sentence, centre=centre, window=window, k=k, seed=seed,
                vocabulary_size=len(words), tokens=int(counts.sum()),
                rows=dict(most_frequent=row(0), median=row(len(words) // 2), least_frequent=row(len(words) - 1)),
                top10_share_f=round(float(f[:top].sum()), 4), top10_share_noise=round(float(g[:top].sum()), 4),
                pairs=pairs)


# Lab 7A exercise A3, unchanged: a decision verb within six words of the policy instrument.
HAWK_PATTERN = re.compile(
    r"\b(raise|raising|increase|increasing|firm|firming|tighten)\w*\W+(?:\w+\W+){0,6}?"
    r"(target|federal funds rate|pressure on reserve)", re.I)
DOVE_PATTERN = re.compile(
    r"\b(lower|lowering|reduce|reducing|cut|cutting|ease|easing|decrease)\w*\W+(?:\w+\W+){0,6}?"
    r"(target|federal funds rate|pressure on reserve)", re.I)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
INSTRUMENT = re.compile(r"target range|target for the federal funds rate|federal funds rate|"
                        r"pressure on reserve positions|discount rate", re.I)
DECISION_VERB = re.compile(r"\b(decided|voted|will|agreed|announced|maintain|keep|hold|raise|raising|"
                           r"increase|increasing|lower|lowering|reduce|reducing|cut|cutting|ease|"
                           r"easing|firm|firming|tighten)\w*", re.I)


def decision_sentence(text: str) -> str:
    """The one sentence that announces the decision, or the best available stand-in.

    Preference order: a sentence naming the policy instrument and carrying a decision verb; then any
    sentence naming the instrument; then the first sentence. The fallbacks are reported, not hidden:
    `signal_decision_sentence` counts how often each one fired.
    """
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text.replace("\n", " ")) if s.strip()]
    for sentence in sentences:
        if INSTRUMENT.search(sentence) and DECISION_VERB.search(sentence):
            return sentence
    for sentence in sentences:
        if INSTRUMENT.search(sentence):
            return sentence
    return sentences[0] if sentences else ""


def signal_decision_sentence(statements: Sequence[Statement], scope: str = "statement") -> dict:
    """Signal 3: hawkish minus dovish regex matches, counted where the decision is announced.

    `scope="statement"` runs the two patterns over the whole statement, which is exactly what Lab 7A
    exercise A3 did and is the number Part F validates. `scope="decision_sentence"` narrows the same
    two patterns to the sentence `decision_sentence` picks out: a sharper instrument, and a chance to
    see what the narrowing costs when the Committee announces a decision in more than one sentence.
    """
    if scope not in {"statement", "decision_sentence"}:
        raise ValueError("scope must be 'statement' or 'decision_sentence'")
    targets, fallback = [], collections.Counter()
    for s in statements:
        if scope == "statement":
            targets.append(s.text)
        else:
            sentence = decision_sentence(s.text)
            targets.append(sentence)
            fallback["instrument and verb" if INSTRUMENT.search(sentence) and DECISION_VERB.search(sentence)
                      else "instrument only" if INSTRUMENT.search(sentence) else "first sentence"] += 1
    hawk = np.array([len(HAWK_PATTERN.findall(t)) for t in targets], dtype=float)
    dove = np.array([len(DOVE_PATTERN.findall(t)) for t in targets], dtype=float)
    return dict(dates=[s.date for s in statements], score=hawk - dove, hawk_matches=hawk,
                dove_matches=dove, scope=scope, sentence=targets if scope != "statement" else None,
                sentence_source=dict(fallback), nonzero_share=float((hawk - dove != 0).mean()))


LABEL_TO_SCORE: dict[str, float] = {"HAWKISH": 1.0, "NEUTRAL": 0.0, "DOVISH": -1.0}

LLM_CONTRACT = dict(
    produced_by="session05_measure.py, outside this module and outside this process",
    file="labs/session05_assets/llm_stance/scores.csv",
    format="CSV with a header row and either the columns date,variant,label or the columns date,score",
    date="YYYY-MM-DD, matching a statement date in the corpus",
    variant="A is the frozen prompt and the one this interface reads by default; B, C and D exist "
            "on a subsample so prompt sensitivity can be measured",
    label="HAWKISH, NEUTRAL or DOVISH, mapped here to +1, 0 and -1",
    score="if the file carries a numeric column instead, it is used as is, on the same orientation "
          "as the other three signals: positive hawkish, negative dovish",
    missing="a date absent from the file scores as nan and is dropped by `validate_signal`",
    why_here="so the notebook can validate all four signals through one function; this module never "
             "calls a model, never touches the network, and never writes the score file",
)


def load_llm_scores(path: Path = LLM_SCORES, variant: str = "A") -> dict[str, float]:
    """Read the separately recorded language-model stance score. Returns {} when it is not there.

    Accepts either the recorded label file (date, variant, label) or a plain numeric file
    (date, score). Labels become +1 / 0 / -1 so the score shares an orientation with signals 2 and 3.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    columns = set(frame.columns)
    if {"date", "label"} <= columns:
        if "variant" in columns:
            frame = frame[frame["variant"].astype(str) == variant]
        unknown = sorted(set(frame["label"].astype(str)) - set(LABEL_TO_SCORE))
        if unknown:
            raise RuntimeError(f"{path} carries labels this module cannot map: {unknown}")
        return {str(d): LABEL_TO_SCORE[str(v)] for d, v in zip(frame["date"], frame["label"])}
    if {"date", "score"} <= columns:
        return {str(d): float(v) for d, v in zip(frame["date"], frame["score"]) if pd.notna(v)}
    raise RuntimeError(f"{path} must carry date,variant,label or date,score -- "
                       f"found {list(frame.columns)}")


def signal_llm_stance(statements: Sequence[Statement],
                      scores: Mapping[str, float] | None = None,
                      path: Path = LLM_SCORES, variant: str = "A") -> dict:
    """Signal 4: the interface only. The score itself is produced and recorded elsewhere.

    Hand in `scores` as {date: value}, or leave it out and the recorded file at `path` is read.
    The returned dictionary has the same shape as the other three signals, so `validate_signal`
    treats all four alike; `available` says whether any score was found.
    """
    table = dict(scores) if scores is not None else load_llm_scores(path, variant)
    dates = [s.date for s in statements]
    score = np.array([table.get(d, np.nan) for d in dates])
    return dict(dates=dates, score=score, contract=LLM_CONTRACT, variant=variant,
                available=bool(np.isfinite(score).any()),
                n_scored=int(np.isfinite(score).sum()), n_statements=len(dates))


# --------------------------------------------------------------------------------------
# Part F -- validation against the policy cycles
# --------------------------------------------------------------------------------------
# The Fed's own hiking and cutting cycles, as Lab 7A dated them. Ground truth we happen to have.
HIKING: tuple[tuple[str, str], ...] = (("1994-02", "1995-02"), ("1999-06", "2000-05"),
                                       ("2004-06", "2006-06"), ("2015-12", "2018-12"),
                                       ("2022-03", "2023-07"))
CUTTING: tuple[tuple[str, str], ...] = (("2001-01", "2003-06"), ("2007-09", "2008-12"),
                                        ("2019-07", "2020-03"), ("2024-09", "2025-12"))


def _month_end(stamp: str) -> str:
    """"2006-06" means the whole of June 2006, the way a pandas month slice reads it."""
    year, month = int(stamp[:4]), int(stamp[5:7])
    year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return f"{year:04d}-{month:02d}-01"


def cycle_labels(dates: Sequence[str], hiking: Sequence[tuple[str, str]] = HIKING,
                 cutting: Sequence[tuple[str, str]] = CUTTING) -> np.ndarray:
    """+1 inside a hiking band, -1 inside a cutting band, 0 between cycles."""
    labels = np.zeros(len(dates))
    for sign, bands in ((1.0, hiking), (-1.0, cutting)):
        for lo, hi in bands:
            hi_end = _month_end(hi)
            for i, d in enumerate(dates):
                if lo <= d[:7] and d < hi_end:
                    labels[i] = sign
    return labels


def policy_path(dates: Sequence[str], horizon: int = 3, **bands) -> np.ndarray:
    """The subsequent policy path: the mean cycle label over the next `horizon` meetings.

    A stance measure read at meeting t is a claim about where policy goes next, so this -- and not
    the label of meeting t itself -- is what the score has to line up with. The last `horizon`
    statements have no future inside the sample and score nan.
    """
    labels = cycle_labels(dates, **bands)
    out = np.full(len(dates), np.nan)
    for i in range(len(dates) - horizon):
        out[i] = labels[i + 1: i + 1 + horizon].mean()
    return out


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rho in numpy: Pearson on average ranks, ties shared. No scipy on this machine."""
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return float("nan")
    rx, ry = _average_ranks(x[keep]), _average_ranks(y[keep])
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float(rx @ ry / denominator) if denominator else float("nan")


def _average_ranks(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(v.size, dtype=float)
    ranks[order] = np.arange(v.size, dtype=float)
    for value in np.unique(v):
        tied = v == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def validate_signal(dates: Sequence[str], values: np.ndarray, name: str = "signal",
                    end: str | None = TRAIN_END, horizon: int = 3,
                    hiking: Sequence[tuple[str, str]] = HIKING,
                    cutting: Sequence[tuple[str, str]] = CUTTING) -> dict:
    """Score one signal against the policy cycles, per cycle, inside a window.

    `end` restricts the exercise to the training window; pass None for the whole sample. A signal
    passes only if every hiking band it covers has a positive mean and every cutting band a negative
    one -- a sign test, not a threshold, because the units of these four signals are not comparable.
    The per-cycle table is the output that matters: it shows *where* a signal works, and one
    aggregate gap can hide a cycle with the wrong sign, which is exactly what happens below.
    """
    values = np.asarray(values, dtype=float)
    keep = [i for i, d in enumerate(dates) if (end is None or d <= end) and np.isfinite(values[i])]
    kept_dates = [dates[i] for i in keep]
    kept = values[list(keep)]
    rows, means = [], {"hiking": [], "cutting": []}
    for kind, bands in (("hiking", hiking), ("cutting", cutting)):
        for lo, hi in bands:
            hi_end = _month_end(hi)
            inside = [j for j, d in enumerate(kept_dates) if lo <= d[:7] and d < hi_end]
            if not inside:
                rows.append(dict(kind=kind, start=lo, end=hi, n=0, mean=None, covered=False))
                continue
            mean = float(kept[inside].mean())
            rows.append(dict(kind=kind, start=lo, end=hi, n=len(inside), mean=mean, covered=True))
            means[kind].append(mean)
    hiking_mean = float(np.mean(means["hiking"])) if means["hiking"] else float("nan")
    cutting_mean = float(np.mean(means["cutting"])) if means["cutting"] else float("nan")
    all_hiking_positive = bool(means["hiking"]) and all(m > 0 for m in means["hiking"])
    all_cutting_negative = bool(means["cutting"]) and all(m < 0 for m in means["cutting"])
    path = policy_path(kept_dates, horizon=horizon, hiking=hiking, cutting=cutting)
    labels = cycle_labels(kept_dates, hiking=hiking, cutting=cutting)
    return dict(
        signal=name, window_end=end, n_scored=len(keep), n_statements=len(dates),
        per_cycle=rows, hiking_mean=hiking_mean, cutting_mean=cutting_mean,
        gap=hiking_mean - cutting_mean,
        all_hiking_bands_positive=all_hiking_positive,
        all_cutting_bands_negative=all_cutting_negative,
        passes=bool(all_hiking_positive and all_cutting_negative),
        worst_hiking_band=_extreme(rows, "hiking", worst=True),
        worst_cutting_band=_extreme(rows, "cutting", worst=True),
        rank_correlation_with_current_cycle=_rank_correlation(kept, labels),
        rank_correlation_with_subsequent_path=_rank_correlation(kept, path),
        subsequent_path_horizon=horizon,
        verdict=("every hiking band positive and every cutting band negative"
                 if all_hiking_positive and all_cutting_negative else
                 "at least one band carries the wrong sign"),
    )


def _extreme(rows: Sequence[dict], kind: str, worst: bool = True) -> dict | None:
    covered = [r for r in rows if r["kind"] == kind and r["covered"]]
    if not covered:
        return None
    # "worst" for a hiking band is the most negative mean; for a cutting band the most positive.
    return min(covered, key=lambda r: r["mean"]) if kind == "hiking" else max(covered, key=lambda r: r["mean"])


def validation_table(results: Sequence[dict]) -> pd.DataFrame:
    """One row per (signal, cycle), for a slide that wants the failure and the success side by side."""
    rows = []
    for result in results:
        for cycle in result["per_cycle"]:
            rows.append(dict(signal=result["signal"], kind=cycle["kind"],
                             cycle=f"{cycle['start']} to {cycle['end']}", n=cycle["n"],
                             mean=None if cycle["mean"] is None else round(cycle["mean"], 4)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# The recorded run
# --------------------------------------------------------------------------------------
def _round(x, k: int = 6):
    return np.round(np.asarray(x, dtype=float), k).tolist()


def _analogies(emb: Embedding, triples: Sequence[tuple[str, str, str]]) -> list[dict]:
    """Run each analogy that the fitted vocabulary can carry, and record the ones it cannot."""
    rows = []
    for a, b, c in triples:
        absent = [w for w in (a, b, c) if w not in emb]
        rows.append(dict(question=f"{a} : {b} :: {c} : ?",
                         answer=None if absent else analogy(emb, a, b, c),
                         outside_vocabulary=absent))
    return rows


def build_records(path: Path = RECORDED) -> dict:
    """Recompute every number the notebook or a slide might print, and write it to JSON."""
    t0 = time.perf_counter()
    np.random.seed(SEED)
    statements = load_corpus()
    train = training_window(statements)
    later = [s for s in statements if s.date > TRAIN_END]
    dates = [s.date for s in statements]

    # Part A
    dof = degrees_of_freedom(train)
    train_counts = word_frequencies(train)

    # Part B -- merges learned on the training window, and on the whole corpus as a reference
    merges_train = learn_bpe(train_counts, 200)
    merges_full = learn_bpe(word_frequencies(statements), 20)
    probe = ["raising", "unemployment", "disinflationary", "reflationary", "accommodation"]
    word_vocab = set(train_counts)

    # Part C
    model = fit_tfidf(train)
    matrix = model.transform(statements)
    toy = toy_tfidf()

    # Part D
    emb = fit_embeddings(train)

    # Part E
    s1 = signal_tfidf(statements, model)
    s2 = signal_embedding_hawkdove(statements, emb, model)
    s3 = signal_decision_sentence(statements, scope="statement")
    s3_narrow = signal_decision_sentence(statements, scope="decision_sentence")
    s4 = signal_llm_stance(statements)

    # Part F -- inside the training window, which is the honest exercise, and over the full sample
    checks = []
    for name, values in (("embedding hawk-dove", s2["score"]),
                         ("decision-sentence regex", s3["score"]),
                         ("decision-sentence regex, narrowed", s3_narrow["score"]),
                         ("novelty index", s2["novelty"])):
        checks.append((name, values))
    validation = {f"{name} | training window": validate_signal(dates, v, name, end=TRAIN_END)
                  for name, v in checks}
    validation.update({f"{name} | full sample": validate_signal(dates, v, name, end=None)
                       for name, v in checks})
    if s4["available"]:
        validation["language-model stance | training window"] = validate_signal(
            dates, s4["score"], "language-model stance", end=TRAIN_END)
        validation["language-model stance | full sample"] = validate_signal(
            dates, s4["score"], "language-model stance", end=None)

    # The reproduction of Lab 7A: same fitting set as the 2025 notebook, which was the whole corpus.
    full_model = fit_tfidf(statements)
    full_emb = fit_embeddings(statements)
    lab7a_hd = signal_embedding_hawkdove(statements, full_emb, full_model)
    lab7a_emb = validate_signal(dates, lab7a_hd["score"], "embedding hawk-dove (Lab 7A fitting)",
                                end=None)
    lab7a_regex = validate_signal(dates, s3["score"], "decision-sentence regex (Lab 7A fitting)",
                                  end=None)

    reproduction = dict(
        what="Lab 7A fitted IDF and the embedding on all 245 statements; this module fits on the "
             "1994-2014 training window. Both are recorded: the first to show the reimplementation "
             "is faithful, the second because it is the design a forecast can actually use.",
        lab7a_printed=dict(hiking_mean=0.029, cutting_mean=0.028, gap=0.001,
                           hiking_2004_06=-0.074, regex_hiking_mean=1.23, regex_cutting_mean=-0.68,
                           embedding_vocabulary=831, tfidf_vocabulary=1424),
        recomputed=dict(
            hiking_mean=round(lab7a_emb["hiking_mean"], 6),
            cutting_mean=round(lab7a_emb["cutting_mean"], 6),
            gap=round(lab7a_emb["gap"], 6),
            hiking_2004_06=round([r for r in lab7a_emb["per_cycle"]
                                  if r["start"] == "2004-06"][0]["mean"], 6),
            regex_hiking_mean=round(lab7a_regex["hiking_mean"], 6),
            regex_cutting_mean=round(lab7a_regex["cutting_mean"], 6),
            embedding_vocabulary=len(full_emb.vocabulary),
            tfidf_vocabulary=len(full_model.vocabulary)),
        bpe_first_20_merges_full_corpus=[f"{a}+{b}" for a, b in merges_full],
    )
    reproduction["matches_lab7a"] = dict(
        embedding_fails=bool(not lab7a_emb["passes"]),
        embedding_2004_06_negative=bool(reproduction["recomputed"]["hiking_2004_06"] < 0),
        embedding_gap_near_zero=bool(abs(reproduction["recomputed"]["gap"]) < 0.01),
        regex_passes=bool(lab7a_regex["passes"]),
        band_means_to_printed_precision=bool(
            round(lab7a_emb["hiking_mean"], 3) == 0.029 and round(lab7a_emb["cutting_mean"], 3) == 0.028
            and round(lab7a_emb["gap"], 3) == 0.001
            and round(reproduction["recomputed"]["hiking_2004_06"], 3) == -0.074
            and round(lab7a_regex["hiking_mean"], 2) == 1.23
            and round(lab7a_regex["cutting_mean"], 2) == -0.68),
        vocabularies_match=bool(len(full_emb.vocabulary) == 831 and len(full_model.vocabulary) == 1424),
    )

    train_validation = validation["embedding hawk-dove | training window"]
    regex_validation = validation["decision-sentence regex | training window"]
    headline = dict(
        embedding_hawkdove_reproduces_the_failure=bool(not train_validation["passes"]
                                                       and not lab7a_emb["passes"]),
        decision_regex_reproduces_the_success=bool(regex_validation["passes"] and lab7a_regex["passes"]),
        note=("Both outcomes are recorded as computed. Nothing in this module was tuned to make a "
              "signal pass: the seed lists, the window, the dimension and the two regexes are Lab "
              "7A's, unchanged."),
    )

    record = dict(
        _meta=dict(
            module=Path(__file__).name, run_date=time.strftime("%Y-%m-%d"), seed=SEED,
            python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
            corpus_manifest_sha256=file_sha256(MANIFEST),
            corpus_directory=str(FOMC_DIR.relative_to(HERE.parent.parent)),
            omp_num_threads=os.environ.get("OMP_NUM_THREADS", "unset"),
            elapsed_seconds=None),
        corpus=dict(all=corpus_summary(statements), training_window=corpus_summary(train),
                    training_window_end=TRAIN_END,
                    held_back=corpus_summary(later) if later else None,
                    per_year_1994_1999={str(y): sum(1 for s in statements if s.date[:4] == str(y))
                                        for y in range(1994, 2000)},
                    from_2020=corpus_summary([s for s in statements if s.date >= "2020-01-01"])),
        headline=headline,
        preprocessing=dict(
            note="Six pipelines over the same 149 training statements. Every row is a defensible "
                 "choice and no two rows give the same vocabulary.",
            stopword_count=len(STOPWORDS), stopwords=sorted(STOPWORDS),
            suffixes=list(_SUFFIXES), min_stem=MIN_STEM,
            table=dof.reset_index().to_dict(orient="records"),
            worked_example={w: dict(baseline=BASELINE.run(w),
                                    everything_on=PIPELINES[-1].run(w))
                            for w in ("The Committee decided to raise the target range to 5-1/4 percent.",
                                      "Inflation has moderated but remains elevated.")}),
        bpe=dict(
            learned_on="the 1994-2014 training window",
            **bpe_report(merges_train, probe),
            word_level_vocabulary=len(word_vocab),
            out_of_vocabulary_probe={w: dict(in_word_vocabulary=w in word_vocab,
                                             pieces=apply_bpe(w, merges_train))
                                     for w in ("reflationary", "disinflationary")}),
        tfidf=dict(
            fitted_on="the 1994-2014 training window", n_fit=model.n_fit, min_df=model.min_df,
            vocabulary_size=len(model.vocabulary), matrix_shape=list(matrix.shape),
            zero_share=float((model.counts(statements) == 0).mean()),
            token_coverage_training=model.coverage(train),
            token_coverage_held_back=model.coverage(later) if later else None,
            top_terms={p: top_terms(matrix, model, next(i for i, d in enumerate(dates)
                                                        if d.startswith(p)))
                       for p in ("2008-12", "2020-03-15", "2017-07", "2005-06")},
            slide_document_term_matrix=slide_document_term_matrix().to_dict(orient="split"),
            worked_example=dict(
                sentences=list(TOY_SENTENCES), labels=list(TOY_LABELS),
                vocabulary=list(toy["counts"].columns),
                counts=toy["counts"].to_dict(orient="split"),
                document_frequency_and_idf=toy["weights"].reset_index().to_dict(orient="records"),
                tfidf=toy["tfidf"].to_dict(orient="split"),
                top_terms_printable=toy["top_terms"].reset_index().to_dict(orient="records"),
                cosine_counts=toy["cosine_counts"].to_dict(orient="split"),
                cosine_tfidf=toy["cosine_tfidf"].to_dict(orient="split"),
                reading="the shared words -- the, committee, rates, and -- carry a small IDF and "
                        "almost no weight; what survives is each sentence's own vocabulary, which "
                        "is why the cosine between d1 and d2 falls from 0.571 on raw counts to "
                        "0.152 on TF-IDF")),
        embeddings=dict(
            fitted_on="the 1994-2014 training window", n_fit=emb.n_fit, window=emb.window,
            min_count=emb.min_count, dim=emb.dim, vocabulary_size=len(emb.vocabulary),
            ppmi_density=emb.ppmi_density,
            singular_values_head=_round(emb.singular_values[:20], 4),
            neighbors={w: neighbors(emb, w) for w in
                       ("inflation", "employment", "committee", "accommodation", "tightening")
                       if w in emb},
            analogy=_analogies(emb, (("increase", "increased", "strengthen"),
                                     ("increase", "increased", "moderate"),
                                     ("raise", "raised", "lower"))),
            out_of_vocabulary_note="A word that appears fewer than min_count times in the training "
                                   "window has no vector at all. That is a property of the method, "
                                   "not a defect of this run, and the analogy table says so."),
        signals=dict(
            tfidf=dict(kind="representation", shape=list(matrix.shape),
                       coverage=s1["coverage"],
                       delta_definition="delta_t = 1 - cos(tfidf_{t-1}, tfidf_t)",
                       largest_changes=s1["largest_changes"]),
            embedding_hawkdove=dict(
                kind="seed-word axis plus novelty index",
                hawk_seeds=list(HAWK_SEEDS), dove_seeds=list(DOVE_SEEDS),
                hawk_used=s2["hawk_used"], dove_used=s2["dove_used"],
                hawk_counts=s2["hawk_counts"], dove_counts=s2["dove_counts"],
                novelty_window=s2["novelty_window"], most_novel=s2["most_novel"]),
            decision_sentence=dict(
                kind="regex on the decision sentence",
                hawk_pattern=HAWK_PATTERN.pattern, dove_pattern=DOVE_PATTERN.pattern,
                scope_statement_nonzero_share=s3["nonzero_share"],
                scope_sentence_nonzero_share=s3_narrow["nonzero_share"],
                sentence_source=s3_narrow["sentence_source"],
                example_sentences={s.date: decision_sentence(s.text)
                                   for s in statements if s.date in
                                   ("1994-02-04", "2004-06-30", "2008-12-16", "2020-03-15", "2022-03-16")}),
            language_model_stance=dict(kind="interface only", **{k: v for k, v in s4.items()
                                                                 if k not in {"dates", "score"}})),
        validation=validation,
        validation_table=validation_table(list(validation.values())).to_dict(orient="records"),
        reproduction_of_lab7a=reproduction,
        series=dict(
            dates=dates,
            embedding_hawkdove=_round(s2["score"], 6),
            novelty=[None if not np.isfinite(v) else round(float(v), 6) for v in s2["novelty"]],
            decision_score=[int(v) for v in s3["score"]],
            decision_score_narrowed=[int(v) for v in s3_narrow["score"]],
            hawk_matches=[int(v) for v in s3["hawk_matches"]],
            dove_matches=[int(v) for v in s3["dove_matches"]],
            language_model_stance=([None if not np.isfinite(v) else float(v) for v in s4["score"]]
                                   if s4["available"] else None),
            delta=_round(s1["delta"], 6), delta_dates=s1["delta_dates"],
            cycle_label=[int(v) for v in cycle_labels(dates)],
            subsequent_policy_path=[None if not np.isfinite(v) else round(float(v), 6)
                                    for v in policy_path(dates)],
            embedding_hawkdove_lab7a_fitting=_round(lab7a_hd["score"], 6)),
        cycles=dict(hiking=[list(b) for b in HIKING], cutting=[list(b) for b in CUTTING],
                    source="Lab 7A, labs/Lec07_LLM_Lab/Lab7A_Text_as_Data.ipynb, cell 26"),
    )
    lda = fit_lda_gibbs(train)
    shares = lda.yearly_shares()
    record["lda"] = dict(
        fitted_on="the 1994-2014 training window", method="collapsed Gibbs sampling",
        pipeline=LDA_PIPELINE.steps(), K=lda.K, a=lda.a, b=lda.b, sweeps=lda.sweeps, seed=lda.seed,
        min_df=lda.min_df, max_df_share=lda.max_df_share, vocabulary_size=len(lda.vocabulary),
        tokens=lda.tokens, loglik_every_10_sweeps=_round(lda.loglik_trace, 2),
        top_words=[lda.top_words(k, 8) for k in range(lda.K)],
        yearly_shares=dict(years=list(shares.index), shares=_round(shares.values, 4)),
        theta=_round(lda.theta, 4), dates=lda.dates)
    record["lda_toy"] = dict(documents=list(TOY_LDA_DOCS), runs=toy_lda())
    record["lookahead"] = dict(
        what="document frequency and IDF of each term, fitted on the training window and on all "
             "statements; the gap is what a full-sample fit leaks into pre-2015 weights",
        rows=idf_lookahead(statements, LOOKAHEAD_TERMS))
    record["noise"] = noise_draws(emb)
    record["embedding_centred"] = embedding_centred_check(train, emb, model)
    record["_meta"]["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
    Path(path).write_text(json.dumps(record, indent=1, default=_json_default), encoding="utf-8")
    return record


def adjacent_similarity(statements: Sequence[Statement], model: Tfidf) -> dict:
    """Cosine between each statement and the one before it, under three weightings of the same
    fitted vocabulary: raw counts, presence (0/1), and TF-IDF with the window-fitted IDF.

    The same pair of documents gets three different answers; the gap between them is what the
    weighting decides. Nothing is fitted here: `model` was fitted on the training window and is
    applied to every statement, so later rows carry no look-ahead.
    """
    ordered = sorted(statements, key=lambda s: s.date)
    counts = model.counts(ordered)
    binary = (counts > 0).astype(float)
    tfidf = model.transform(ordered)
    dates = [s.date for s in ordered]
    rows = []
    for i in range(1, len(ordered)):
        rows.append(dict(previous=dates[i - 1], date=dates[i],
                         counts=round(cosine(counts[i - 1], counts[i]), 4),
                         binary=round(cosine(binary[i - 1], binary[i]), 4),
                         tfidf=round(cosine(tfidf[i - 1], tfidf[i]), 4)))
    def mean(key):
        return round(float(np.mean([r[key] for r in rows])), 4)
    def share_above(key, level):
        return round(float(np.mean([r[key] > level for r in rows])), 4)
    lowest = min(rows, key=lambda r: r["tfidf"])
    return dict(
        what="cosine between consecutive statements under three weightings of the window-fitted "
             "vocabulary; the window's IDF is applied to every statement",
        fitted_on=f"{model.n_fit} statements, vocabulary {len(model.vocabulary)}",
        pairs=len(rows), rows=rows,
        mean=dict(counts=mean("counts"), binary=mean("binary"), tfidf=mean("tfidf")),
        share_above_090=dict(counts=share_above("counts", 0.9), binary=share_above("binary", 0.9),
                             tfidf=share_above("tfidf", 0.9)),
        lowest_tfidf=lowest,
        december_2008=next(r for r in rows if r["date"] == "2008-12-16"))


def add_deck1_records(path: Path = RECORDED) -> None:
    """Add Deck 5.1's `adjacent` block (cosine between consecutive statements, three weightings) to
    the recorded file, leaving every other key as recorded."""
    import platform
    statements = load_corpus()
    train = training_window(statements)
    model = fit_tfidf(train)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    stamp = dict(run_date=time.strftime("%Y-%m-%d"), python=platform.python_version(), numpy=np.__version__,
                 note="added 2026-10-09 for Deck 5.1's cosine on real statements; "
                      "the rest of this file is unchanged")
    data["adjacent"] = dict(_recorded=stamp, **adjacent_similarity(statements, model))
    Path(path).write_text(json.dumps(data, indent=1, default=_json_default), encoding="utf-8")
    a = data["adjacent"]
    print(f"added adjacent to {Path(path).name}: {a['pairs']} pairs, mean counts {a['mean']['counts']}, "
          f"binary {a['mean']['binary']}, tfidf {a['mean']['tfidf']}")


def add_deck2_records(path: Path = RECORDED) -> None:
    """Add Deck 5.2's two blocks (`noise`, `embedding_centred`) to the recorded file, leaving every
    other key as recorded: a full re-run under numpy 2 reorders tied lists recorded under numpy 1.24."""
    import platform
    statements = load_corpus()
    train = training_window(statements)
    model, emb = fit_tfidf(train), fit_embeddings(train)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    stamp = dict(run_date=time.strftime("%Y-%m-%d"), python=platform.python_version(), numpy=np.__version__,
                 note="added 2026-10-03 for Deck 5.2's word2vec example and centring check; "
                      "the rest of this file is unchanged")
    data["noise"] = dict(_recorded=stamp, **noise_draws(emb))
    data["embedding_centred"] = dict(_recorded=stamp, **embedding_centred_check(train, emb, model))
    Path(path).write_text(json.dumps(data, indent=1, default=_json_default), encoding="utf-8")
    print(f"added noise, embedding_centred to {Path(path).name}")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"{type(obj)} is not JSON serialisable")


def self_test() -> None:
    """Run the whole module and print the numbers the lab turns on."""
    t0 = time.perf_counter()
    statements = load_corpus()
    train = training_window(statements)
    print(f"corpus: {corpus_summary(statements)}")
    print(f"training window: {corpus_summary(train)}")

    merges = learn_bpe(word_frequencies(train), 200)
    print("\nfirst 10 merges:", [f"{a}+{b}" for a, b in merges[:10]])
    for w in ("raising", "disinflationary", "reflationary"):
        print(f"   {w:>16s} -> {apply_bpe(w, merges)}")

    model = fit_tfidf(train)
    emb = fit_embeddings(train)
    print(f"\nTF-IDF vocabulary {len(model.vocabulary):,} fitted on {model.n_fit} statements; "
          f"embedding vocabulary {len(emb.vocabulary):,}")

    dates = [s.date for s in statements]
    hd = signal_embedding_hawkdove(statements, emb, model)
    reg = signal_decision_sentence(statements)
    for name, values in (("embedding hawk-dove", hd["score"]), ("decision-sentence regex", reg["score"])):
        result = validate_signal(dates, values, name, end=TRAIN_END)
        print(f"\n{name}, training window: hiking {result['hiking_mean']:+.3f}, "
              f"cutting {result['cutting_mean']:+.3f}, gap {result['gap']:+.3f}")
        for row in result["per_cycle"]:
            if row["covered"]:
                print(f"   {row['kind']:>7s}  {row['start']} to {row['end']}  n={row['n']:3d}  "
                      f"mean {row['mean']:+.3f}")
        print(f"   {'PASS' if result['passes'] else 'FAIL'}  {result['verdict']}")
    print(f"\n{time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["test"]:
        self_test()
    elif sys.argv[1:] == ["add", "deck2"]:
        add_deck2_records()
    elif sys.argv[1:] == ["add", "deck1"]:
        add_deck1_records()
    else:
        rec = build_records()
        h = rec["headline"]
        print(f"wrote {RECORDED.name} in {rec['_meta']['elapsed_seconds']} s")
        print("first 10 BPE merges:", rec["bpe"]["first_merges"][:10])
        print("embedding hawk-dove reproduces the failure:",
              h["embedding_hawkdove_reproduces_the_failure"])
        print("decision regex reproduces the success:", h["decision_regex_reproduces_the_success"])
        print("Lab 7A numbers reproduced:", rec["reproduction_of_lab7a"]["matches_lab7a"])
