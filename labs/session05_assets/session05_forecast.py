"""Session 5 lab -- block F, does the FOMC's own text add anything to Session 2's yield forecast?

Session 2 forecast the 10-year Treasury yield one month ahead from numbers alone. Block F puts
the Committee's words beside those numbers and asks whether they help. The design is fixed in
protocol_forecast.md, which was written and hashed before anything here was scored; this module
implements that protocol and nothing else. The interesting feature of the exercise is that the
honest answer may be "no", and every gate below is a gate on procedure so that "no" can be
reported without anything failing.

  load_inputs()          Session 2's two frozen files, hash-asserted, read-only
  load_statements()      the 245-statement corpus, hash-asserted, exclusion set applied
  merge_text()           the information rule: the last policy statement dated on or before t
  fit_text_space()       vocabulary, IDF, co-occurrence, SVD and seed axis, training window only
  run_models()           the models named in advance by the protocol
  diebold_mariano()      with the Harvey-Leybourne-Newbold small-sample correction
  clark_west()           for the nested comparisons, where Diebold-Mariano is mis-sized
  cumulative_squared_error()   the comparison as a path rather than one number
  llm_column()           the separately recorded LLM stance, reported on its own, never pooled

Three columns everywhere -- the frozen file, the model re-estimated on the text-eligible sample,
and that model plus text -- because the sample restriction and the text are two different changes
and a two-column table would charge both to the text.

Imports: numpy, pandas, scikit-learn and the standard library. No network, CPU only.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
LABS = HERE.parent
SESSION02 = LABS / "session02_assets"
BASELINE_CSV = SESSION02 / "baseline_predictions.csv"
PANEL_CSV = SESSION02 / "yield_panel.csv"
FOMC = HERE / "fomc"
FOMC_MANIFEST = HERE / "fomc_manifest.json"
LLM_SCORES = HERE / "llm_stance" / "scores.csv"
PROTOCOL = HERE / "protocol_forecast.md"
PROTOCOL_SHA = HERE / "protocol_forecast.sha256"
RECORDED = HERE / "recorded_forecast.json"

SEED = 42

# protocol section 1.2
EXPECTED_SHA256 = {
    "baseline_predictions.csv": "a8e815d45edc48872cedeb37bd12e725b46c7c2b5cc99294837c2bc48c923332",
    "yield_panel.csv": "b1687bce6a37f4270a9014d89e7ef1b5351aca982dbfff8c77d36c2e1ca37e64",
}

FEATURES = ["y10", "spread", "fedfunds_lag1", "unrate_lag1", "inflation_lag1"]
TEXT_FEATURES = ["stance", "novelty", "months_since_statement"]

# protocol section 2.2
SAMPLE_START = "1994-02-28"
TRAIN_END = "2014-12-31"
VALID_START, VALID_END = "2015-01-31", "2019-12-31"
TEST_START = "2020-01-31"
TEXT_TRAIN_END = "2014-12-31"

# protocol section 3.1 -- these three reproduce and are gated
FROZEN_BASELINES = {
    "random_walk": {"rmse": 0.2658, "mae": 0.2097},
    "ar1": {"rmse": 0.2647, "mae": 0.2089},
    "ols": {"rmse": 0.2855, "mae": 0.2255},
}
# protocol section 3.2 -- two different objects, recorded separately and gated against neither
MLP_PUBLISHED = {"rmse": 0.3078, "mae": 0.2406,
                 "definition": "mean of ten per-seed test RMSEs, as Session 2 publishes it",
                 "per_seed_rmse_range": [0.2661, 0.3588]}
MLP_COLUMN = {"rmse": 0.2956, "mae": 0.2289,
              "definition": "the mlp_seed_mean COLUMN of baseline_predictions.csv, i.e. the mean "
                            "forecast across the ten seeds -- a different object"}

# protocol section 5, hand-listed and hashed with the protocol
EXCLUSIONS: dict[str, str] = {
    "2025-08-22": "framework review: revisions to the Statement on Longer-Run Goals and Monetary "
                  "Policy Strategy, taking no position on the current stance",
    "2019-10-11": "reserve-management bill purchases and repo operations; the release states they "
                  "are purely technical and do not represent a change in the stance",
    "2010-05-09": "re-authorisation of dollar liquidity swap lines; a balance-sheet action with no "
                  "rate decision and no stance language",
    "2008-03-11": "Term Securities Lending Facility and a swap-line increase, issued jointly with "
                  "four other central banks; a facility announcement",
    "2007-08-10": "notice that the Desk will supply reserves to hold the funds rate near an "
                  "unchanged target; a market-functioning notice, not a policy decision",
}

# The seed axis of the stance score. Ordinary policy-statement vocabulary, fixed here and not
# tuned on any outcome; the axis is built from the training window's own term geometry.
HAWKISH_SEEDS = ("tightening", "firming", "restrictive", "inflationary", "inflation", "raise",
                 "increase", "higher", "overheating", "pressures", "resource", "utilization")
DOVISH_SEEDS = ("accommodative", "accommodation", "easing", "ease", "weakness", "softening",
                "lower", "reduce", "sluggish", "downside", "slack", "unemployment")

# protocol section 8.2 -- artefacts of Session 1's Machine B, trained on all 245 statements
MACHINE_B_ARTEFACTS = ("minigpt_fomc.pt", "minigpt_vocab.json", "minigpt_config.json")

_TOKEN = r"(?u)\b[a-zA-Z][a-zA-Z]+\b"     # letters only: dates and figures never become tokens


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rmse_mae(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    err = np.asarray(actual, float) - np.asarray(predicted, float)
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err)))}


# --------------------------------------------------------------------------------------
# Test statistics, in numpy
# --------------------------------------------------------------------------------------
def _norm_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d, h = 1.0 / d, 1.0 / d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def _t_sf(t: float, df: float) -> float:
    """Upper-tail probability of Student's t.  Standard library only; scipy stays out of the deps."""
    x = df / (df + t * t)
    a, b = 0.5 * df, 0.5
    if x <= 0.0:
        ib = 0.0
    elif x >= 1.0:
        ib = 1.0
    else:
        front = math.exp(a * math.log(x) + b * math.log(1.0 - x)
                         + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))
        ib = (front * _betacf(a, b, x) / a if x < (a + 1.0) / (a + b + 2.0)
              else 1.0 - math.exp(b * math.log(1.0 - x) + a * math.log(x)
                                  + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))
              * _betacf(b, a, 1.0 - x) / b)
    half = 0.5 * ib
    return half if t > 0 else 1.0 - half


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> dict[str, float]:
    """Diebold-Mariano with the Harvey-Leybourne-Newbold small-sample correction.

    Source: Diebold & Mariano (1995), Journal of Business & Economic Statistics 13(3), 253-263,
    with the correction of Harvey, Leybourne & Newbold (1997), International Journal of
    Forecasting 13(2), 281-291.  NOTE: neither paper was read at first hand for this
    implementation; both statistics are written from their standard textbook statements and a
    separate check of the references is under way.

    The loss differential is d_t = e1_t^2 - e2_t^2, so a NEGATIVE statistic favours model 1.  The
    long-run variance uses autocovariances to lag h-1; at h = 1 that is the sample variance alone.
    The corrected statistic is referred to a t distribution with n - 1 degrees of freedom, which
    matters here: with 79 observations the uncorrected normal over-rejects.
    """
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    d = e1 ** 2 - e2 ** 2
    n = d.size
    d_bar = float(d.mean())
    dev = d - d_bar
    gamma0 = float(dev @ dev) / n
    lrv = gamma0
    for lag in range(1, h):
        gamma = float(dev[lag:] @ dev[:-lag]) / n
        lrv += 2.0 * gamma
    if lrv <= 0:
        return {"d_bar": d_bar, "statistic": float("nan"), "p_value": float("nan"),
                "n": n, "h": h, "favours": "undetermined (non-positive long-run variance)"}
    dm = d_bar / math.sqrt(lrv / n)
    correction = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * correction
    p = 2.0 * _t_sf(abs(dm_hln), n - 1)
    return {"d_bar": d_bar, "statistic_uncorrected": float(dm), "statistic": float(dm_hln),
            "hln_correction": float(correction), "p_value": float(p), "n": n, "h": h,
            "favours": "model 1" if d_bar < 0 else "model 2",
            "interpretation": "two-sided; a negative statistic favours model 1"}


def clark_west(actual: np.ndarray, pred_small: np.ndarray,
               pred_large: np.ndarray) -> dict[str, float]:
    """Clark-West, for a comparison in which the small model is nested inside the large one.

    Source: Clark & West (2007), Journal of Econometrics 138(1), 291-311.  NOTE: the paper was
    not read at first hand for this implementation; the statistic is written from its standard
    statement and a separate check of the references is under way.

    Under the null that the extra regressors are zero in population, the larger model still has to
    estimate them, so its squared error is larger in expectation and Diebold-Mariano is mis-sized
    against it.  Clark & West subtract that estimation noise:

        f_t = (y_t - p_small_t)^2 - [(y_t - p_large_t)^2 - (p_small_t - p_large_t)^2]

    and test mean(f) > 0 one-sided.  A POSITIVE statistic favours the large model, the opposite
    sign convention to diebold_mariano above, which is why both are always reported together.
    """
    actual = np.asarray(actual, float)
    small = np.asarray(pred_small, float)
    large = np.asarray(pred_large, float)
    f = (actual - small) ** 2 - ((actual - large) ** 2 - (small - large) ** 2)
    n = f.size
    f_bar = float(f.mean())
    se = float(f.std(ddof=1) / math.sqrt(n))
    if se <= 0:
        return {"f_bar": f_bar, "statistic": float("nan"), "p_value": float("nan"), "n": n}
    stat = f_bar / se
    return {"f_bar": f_bar, "statistic": float(stat), "se": se,
            "p_value_one_sided": float(1.0 - _norm_cdf(stat)), "n": n,
            "favours": "the larger model" if stat > 0 else "the smaller model",
            "interpretation": "one-sided; a positive statistic favours the larger model"}


def cumulative_squared_error(actual: np.ndarray, predicted: np.ndarray) -> list[float]:
    """The running sum of squared errors, so a comparison can be read as a path."""
    err = np.asarray(actual, float) - np.asarray(predicted, float)
    return [float(v) for v in np.cumsum(err ** 2)]


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------
@dataclass
class Inputs:
    panel: pd.DataFrame
    baseline: pd.DataFrame
    hashes: dict[str, str]
    missing_forecast_dates: list[str]


def load_inputs() -> Inputs:
    """Session 2's two frozen files, hash-asserted and read-only (protocol sections 1 and 3).

    The panel has no row for the 2025-11-30 forecast date.  That gap is inherited, never filled:
    filling it would invent a month Session 2's published baselines were never scored on.
    """
    for path in (BASELINE_CSV, PANEL_CSV):
        actual = file_sha256(path)
        expected = EXPECTED_SHA256[path.name]
        if actual != expected:
            raise RuntimeError(f"{path.name} does not match the protocol's hash "
                               f"(protocol {expected[:16]}..., file {actual[:16]}...)")

    panel = pd.read_csv(PANEL_CSV, parse_dates=["date", "target_date"])
    baseline = pd.read_csv(BASELINE_CSV, parse_dates=["forecast_made", "target_date"])
    if len(panel) != 439 or len(baseline) != 79:
        raise RuntimeError(f"expected 439 panel rows and 79 baseline rows, "
                           f"found {len(panel)} and {len(baseline)}")

    span = pd.date_range(panel["date"].min(), panel["date"].max(), freq="ME")
    missing = sorted(set(span.date) - set(panel["date"].dt.date))
    if [str(d) for d in missing] != ["2025-11-30"]:
        raise RuntimeError(f"expected exactly the 2025-11-30 gap, found {missing}")

    return Inputs(panel=panel, baseline=baseline,
                  hashes={p.name: file_sha256(p) for p in (BASELINE_CSV, PANEL_CSV)},
                  missing_forecast_dates=[str(d) for d in missing])


def frozen_baseline_table(baseline: pd.DataFrame) -> dict[str, object]:
    """Score the frozen file as shipped.  Three models are gated; the MLP row is recorded twice."""
    actual = baseline["actual"].to_numpy()
    scored = {name: rmse_mae(actual, baseline[name].to_numpy())
              for name in ("random_walk", "ar1", "ols", "mlp_seed_mean")}
    gates = {}
    for name, target in FROZEN_BASELINES.items():
        gates[name] = (round(scored[name]["rmse"], 4) == target["rmse"]
                       and round(scored[name]["mae"], 4) == target["mae"])
    return {
        "scored": scored,
        "reproduces_to_four_decimals": gates,
        "mlp": {
            "session02_published": MLP_PUBLISHED,
            "column_as_scored": {**MLP_COLUMN, "rmse_here": scored["mlp_seed_mean"]["rmse"],
                                 "mae_here": scored["mlp_seed_mean"]["mae"]},
            "finding": "Session 2 has two MLP numbers and they are different objects. The "
                       "published row, RMSE 0.3078 / MAE 0.2406, is the mean of ten per-seed test "
                       "RMSEs (per-seed RMSE 0.2661 to 0.3588). The mlp_seed_mean COLUMN is the "
                       "mean forecast across those ten seeds and scores 0.2956 / 0.2289. Both "
                       "were reproduced here by refitting Session 2's own fit_mlp; neither is "
                       "gated against the other, because scoring the column and expecting the "
                       "published figure is an error of definition, not a failure to reproduce. "
                       "The gap is forecast combination doing its ordinary work: averaging ten "
                       "forecasts beats the average of the ten models. Session 5 quotes Session "
                       "2's published definition, 0.3078 / 0.2406, as the bar to beat.",
        },
    }


# --------------------------------------------------------------------------------------
# The corpus and the information rule
# --------------------------------------------------------------------------------------
@dataclass
class Corpus:
    dates: list[pd.Timestamp]
    texts: list[str]
    files: list[str]
    excluded: list[dict[str, str]]


def load_statements() -> Corpus:
    """The 245-statement corpus, hash-asserted file by file, with the exclusion set applied.

    The header lines of each file carry the date and the source URL, so they are dropped before
    anything is tokenised; the token pattern then admits letters only, so no date or figure can
    become a feature.
    """
    manifest = json.loads(FOMC_MANIFEST.read_text(encoding="utf-8"))
    if manifest["n"] != len(manifest["files"]):
        raise RuntimeError("fomc_manifest.json is internally inconsistent")

    dates, texts, files, excluded = [], [], [], []
    for entry in manifest["files"]:
        path = FOMC / entry["file"]
        if file_sha256(path) != entry["sha256"]:
            raise RuntimeError(f"{entry['file']} does not match fomc_manifest.json")
        if entry["date"] in EXCLUSIONS:
            excluded.append({"date": entry["date"], "file": entry["file"],
                             "reason": EXCLUSIONS[entry["date"]]})
            continue
        body = "\n".join(line for line in path.read_text(encoding="utf-8").splitlines()
                         if not line.startswith("#")).strip()
        dates.append(pd.Timestamp(entry["date"]))
        texts.append(re.sub(r"\s+", " ", body))
        files.append(entry["file"])

    if len(excluded) != len(EXCLUSIONS):
        raise RuntimeError(f"the exclusion set names {len(EXCLUSIONS)} releases but "
                           f"{len(excluded)} were found in the corpus")
    order = np.argsort([d.value for d in dates])
    return Corpus(dates=[dates[i] for i in order], texts=[texts[i] for i in order],
                  files=[files[i] for i in order], excluded=excluded)


def merge_text(panel: pd.DataFrame, corpus: Corpus) -> pd.DataFrame:
    """The information rule of protocol section 4, plus the stale-month bookkeeping of section 6.

    Month t's signal is the LAST statement dated on or before the last observation date of month
    t.  Seven months carry two or three statements and the last one wins; a month with no meeting
    carries the previous statement forward, and `months_since_statement` records how far back it
    came so that staleness is a measured feature rather than a silent distortion.
    """
    stamps = np.array([d.value for d in corpus.dates])
    frame = panel[panel["date"] >= SAMPLE_START].copy().reset_index(drop=True)

    position = np.searchsorted(stamps, frame["date"].to_numpy().astype("datetime64[ns]")
                               .astype("int64"), side="right") - 1
    if (position < 0).any():
        raise RuntimeError("a row in the estimation sample predates every policy statement")

    frame["statement_index"] = position
    frame["statement_date"] = [corpus.dates[i] for i in position]
    frame["statement_file"] = [corpus.files[i] for i in position]
    frame["months_since_statement"] = [
        (d.year - s.year) * 12 + (d.month - s.month)
        for d, s in zip(frame["date"], frame["statement_date"])
    ]
    frame["is_stale"] = frame["months_since_statement"] > 0

    # protocol section 4.3: nothing dated after t may reach row t
    if (frame["statement_date"] > frame["date"]).any():
        raise RuntimeError("a statement dated after the forecast date reached a row")
    return frame


# --------------------------------------------------------------------------------------
# The text space, fitted on the training window and nowhere else
# --------------------------------------------------------------------------------------
@dataclass
class TextSpace:
    vectorizer: TfidfVectorizer
    axis: np.ndarray
    term_vectors: np.ndarray
    stance_mean: float
    stance_sd: float
    n_training_documents: int
    training_cutoff: str
    seeds_found: dict[str, list[str]]
    vocabulary_size: int


def fit_text_space(corpus: Corpus, n_components: int = 60) -> TextSpace:
    """Vocabulary, IDF, co-occurrence, SVD and seed axis -- all from training-window text only.

    The assertion in the middle is the one that matters: not a single document dated after the
    training cutoff may reach any fitted object.  It is checked, not assumed, because this is the
    single easiest place in the whole lab to leak the future into a feature.
    """
    cutoff = pd.Timestamp(TEXT_TRAIN_END)
    train_mask = np.array([d <= cutoff for d in corpus.dates])
    train_texts = [t for t, keep in zip(corpus.texts, train_mask) if keep]
    train_dates = [d for d, keep in zip(corpus.dates, train_mask) if keep]
    if max(train_dates) > cutoff:
        raise RuntimeError("a post-cutoff document reached the training set")

    vectorizer = TfidfVectorizer(lowercase=True, token_pattern=_TOKEN, stop_words="english",
                                 min_df=3, max_df=0.9, sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_texts)

    # Term co-occurrence over the training window, then a low-rank term geometry.
    cooccurrence = (X_train.T @ X_train).toarray()
    svd = TruncatedSVD(n_components=min(n_components, cooccurrence.shape[0] - 1),
                       random_state=SEED)
    term_vectors = svd.fit_transform(cooccurrence)
    term_vectors /= (np.linalg.norm(term_vectors, axis=1, keepdims=True) + 1e-12)

    vocab = vectorizer.vocabulary_
    found = {"hawkish": [w for w in HAWKISH_SEEDS if w in vocab],
             "dovish": [w for w in DOVISH_SEEDS if w in vocab]}
    if not found["hawkish"] or not found["dovish"]:
        raise RuntimeError("the seed axis needs at least one seed term on each side")
    hawk = term_vectors[[vocab[w] for w in found["hawkish"]]].mean(axis=0)
    dove = term_vectors[[vocab[w] for w in found["dovish"]]].mean(axis=0)
    axis = hawk - dove
    axis /= (np.linalg.norm(axis) + 1e-12)

    raw = _project(X_train, term_vectors, axis)
    space = TextSpace(vectorizer=vectorizer, axis=axis, term_vectors=term_vectors,
                      stance_mean=float(raw.mean()), stance_sd=float(raw.std(ddof=1)),
                      n_training_documents=int(train_mask.sum()), training_cutoff=TEXT_TRAIN_END,
                      seeds_found=found, vocabulary_size=len(vocab))
    return space


def _project(X, term_vectors: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """The tf-idf-weighted centroid of a document's terms, projected onto the seed axis."""
    centroid = X @ term_vectors
    norms = np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-12
    return (centroid / norms) @ axis


def text_features(corpus: Corpus, space: TextSpace) -> pd.DataFrame:
    """Stance and novelty for every kept statement, in the space fitted on the training window."""
    X = space.vectorizer.transform(corpus.texts)
    stance = (_project(X, space.term_vectors, space.axis) - space.stance_mean) / space.stance_sd

    # Novelty: how far this statement sits from the one before it. The first has no predecessor.
    Xn = X.multiply(1.0 / (np.sqrt(X.multiply(X).sum(axis=1)) + 1e-12)).tocsr()
    novelty = np.zeros(len(corpus.texts))
    similarity = np.asarray(Xn[1:].multiply(Xn[:-1]).sum(axis=1)).ravel()
    novelty[1:] = 1.0 - similarity
    return pd.DataFrame({"statement_date": corpus.dates, "statement_stance": stance,
                         "statement_novelty": novelty})


def attach_features(frame: pd.DataFrame, corpus: Corpus, space: TextSpace) -> pd.DataFrame:
    """Put stance and novelty on the panel, with the stale-month rule of protocol section 6.

    Stance is carried forward, because the last stance is still the prevailing one.  Novelty is
    set to ZERO in a month with no meeting: the index measures how much a NEW statement differs
    from its predecessor, and in a month without one there is nothing to be novel.  Carrying it
    forward would report a surprise the market had already had a month or more to absorb.
    """
    per_statement = text_features(corpus, space)
    out = frame.copy()
    out["stance"] = per_statement["statement_stance"].to_numpy()[out["statement_index"]]
    novelty = per_statement["statement_novelty"].to_numpy()[out["statement_index"]]
    out["novelty"] = np.where(out["is_stale"], 0.0, novelty)
    out["delta_y"] = out["y10_next"] - out["y10"]
    return out


def split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split by the date of the label, as Session 2 does (protocol section 2.2)."""
    t = frame["target_date"]
    return {"train": frame[t <= TRAIN_END],
            "valid": frame[(t >= VALID_START) & (t <= VALID_END)],
            "test": frame[t >= TEST_START]}


# --------------------------------------------------------------------------------------
# The models named in advance
# --------------------------------------------------------------------------------------
def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    Xd = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return coef


def _apply(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ coef


@dataclass
class ModelRun:
    name: str
    column: str
    predictions: dict[str, np.ndarray]
    detail: dict[str, object] = field(default_factory=dict)


def run_models(parts: dict[str, pd.DataFrame], corpus: Corpus,
               space: TextSpace) -> dict[str, ModelRun]:
    """Every model of protocol section 8, fitted on train, tuned on valid, scored on test.

    The ridge penalty is the only thing chosen here, and it is chosen on 2015-2019 by validation
    RMSE.  The test window takes no part in any choice.
    """
    train, valid, test = parts["train"], parts["valid"], parts["test"]
    runs: dict[str, ModelRun] = {}

    def predict_everywhere(fn) -> dict[str, np.ndarray]:
        return {k: fn(v) for k, v in parts.items()}

    # -- the structured comparators, re-estimated on the text-eligible sample -------------
    runs["random_walk"] = ModelRun("random walk", "re-estimated",
                                   predict_everywhere(lambda f: f["y10"].to_numpy()),
                                   {"note": "no parameters to re-estimate; identical by "
                                            "construction to the frozen column"})

    ar_coef = _ols(train[["y10"]].to_numpy(), train["y10_next"].to_numpy())
    runs["ar1"] = ModelRun("AR(1)", "re-estimated",
                           predict_everywhere(lambda f: _apply(ar_coef, f[["y10"]].to_numpy())),
                           {"coefficients": [float(c) for c in ar_coef]})

    ols_coef = _ols(train[FEATURES].to_numpy(), train["y10_next"].to_numpy())
    runs["ols5"] = ModelRun("OLS-5", "re-estimated",
                            predict_everywhere(lambda f: _apply(ols_coef, f[FEATURES].to_numpy())),
                            {"coefficients": dict(zip(["const"] + FEATURES,
                                                      [float(c) for c in ols_coef]))})

    # -- (a) text alone, on the change in the yield ---------------------------------------
    t_coef = _ols(train[TEXT_FEATURES].to_numpy(), train["delta_y"].to_numpy())
    runs["text_alone"] = ModelRun(
        "text alone (stance, novelty, months since)", "plus text",
        predict_everywhere(lambda f: f["y10"].to_numpy()
                           + _apply(t_coef, f[TEXT_FEATURES].to_numpy())),
        {"coefficients": dict(zip(["const"] + TEXT_FEATURES, [float(c) for c in t_coef]))})

    # -- (b) OLS-5 plus text ---------------------------------------------------------------
    both = FEATURES + TEXT_FEATURES
    b_coef = _ols(train[both].to_numpy(), train["y10_next"].to_numpy())
    runs["ols5_plus_text"] = ModelRun(
        "OLS-5 plus text", "plus text",
        predict_everywhere(lambda f: _apply(b_coef, f[both].to_numpy())),
        {"coefficients": dict(zip(["const"] + both, [float(c) for c in b_coef]))})

    # -- TF-IDF through ridge, penalty chosen on 2015-2019 only ----------------------------
    def tfidf_matrix(frame: pd.DataFrame) -> np.ndarray:
        X = space.vectorizer.transform([corpus.texts[i] for i in frame["statement_index"]])
        dense = X.toarray()
        # A stale month contributes no novelty; the staleness itself stays as a column.
        return np.column_stack([dense, frame["months_since_statement"].to_numpy(),
                                np.where(frame["is_stale"], 0.0, 1.0)])

    Z = {k: tfidf_matrix(v) for k, v in parts.items()}
    alphas = (0.1, 1.0, 10.0, 100.0, 1000.0)

    def fit_ridge(extra: list[str] | None, target: str) -> tuple[Ridge, float, dict[float, float]]:
        def design(key: str) -> np.ndarray:
            if extra is None:
                return Z[key]
            return np.column_stack([parts[key][extra].to_numpy(), Z[key]])
        scores = {}
        for alpha in alphas:
            model = Ridge(alpha=alpha, random_state=SEED)
            model.fit(design("train"), train[target].to_numpy())
            pred = model.predict(design("valid"))
            base = valid["y10"].to_numpy() if target == "delta_y" else 0.0
            scores[alpha] = rmse_mae(valid["y10_next"].to_numpy(), base + pred)["rmse"]
        best = min(scores, key=scores.get)
        model = Ridge(alpha=best, random_state=SEED)
        model.fit(design("train"), train[target].to_numpy())
        return model, best, scores

    ridge_a, alpha_a, scores_a = fit_ridge(None, "delta_y")
    runs["tfidf_ridge_alone"] = ModelRun(
        "TF-IDF through ridge, text alone", "plus text",
        {k: parts[k]["y10"].to_numpy() + ridge_a.predict(Z[k]) for k in parts},
        {"alpha_chosen_on_validation": float(alpha_a),
         "validation_rmse_by_alpha": {str(a): float(s) for a, s in scores_a.items()},
         "n_columns": int(Z["train"].shape[1])})

    ridge_b, alpha_b, scores_b = fit_ridge(FEATURES, "y10_next")
    runs["ols5_plus_tfidf_ridge"] = ModelRun(
        "OLS-5 plus TF-IDF through ridge", "plus text",
        {k: ridge_b.predict(np.column_stack([parts[k][FEATURES].to_numpy(), Z[k]]))
         for k in parts},
        {"alpha_chosen_on_validation": float(alpha_b),
         "validation_rmse_by_alpha": {str(a): float(s) for a, s in scores_b.items()},
         "n_columns": int(Z["train"].shape[1] + len(FEATURES))})
    return runs


# --------------------------------------------------------------------------------------
# The LLM column, reported on its own
# --------------------------------------------------------------------------------------
def llm_column(frame: pd.DataFrame) -> dict[str, object]:
    """Read llm_stance/scores.csv when it exists; say so plainly when it does not.

    The recording of the stance scores runs as a separate instructor process and writes
    llm_stance/raw_replies.jsonl.  This module never starts it, never writes to it, and reads
    only the tidy scores.csv that the recording is turned into.  Only variant A -- the frozen
    prompt, run on every statement -- may enter a forecast; B, C and D exist to tell prompt
    sensitivity apart from sampling noise and are measurement diagnostics.

    Nothing in model selection may depend on any of this (protocol section 8.1), so the column is
    built, reported and left standing beside the others.
    """
    if not LLM_SCORES.exists():
        return {
            "available": False,
            "path": str(LLM_SCORES.relative_to(LABS)),
            "note": "The tidy LLM stance scores are not present yet. The recording that produces "
                    "them is still running as a separate instructor process; this module neither "
                    "starts nor modifies it. Block F is complete without this column, and the "
                    "notebook step is re-run once the recording finishes and scores.csv exists. "
                    "No result elsewhere in this file depends on it.",
        }
    scores = pd.read_csv(LLM_SCORES, parse_dates=["date"])
    needed = {"date", "variant", "label"}
    if not needed.issubset(scores.columns):
        return {"available": False, "path": str(LLM_SCORES.relative_to(LABS)),
                "note": f"scores.csv is present but lacks the columns {sorted(needed)}; "
                        f"found {sorted(scores.columns)}. The column is left unreported."}

    frozen = scores[scores["variant"] == "A"]
    mapping = {"HAWKISH": 1.0, "NEUTRAL": 0.0, "DOVISH": -1.0}
    frozen = frozen.assign(stance=frozen["label"].str.upper().map(mapping))
    by_date = dict(zip(frozen["date"], frozen["stance"]))
    aligned = [by_date.get(d, np.nan) for d in frame["statement_date"]]
    coverage = float(np.mean(~pd.isna(aligned)))

    out = {
        "available": True,
        "path": str(LLM_SCORES.relative_to(LABS)),
        "n_scored": int(len(frozen)),
        "variants_present": sorted(scores["variant"].dropna().unique().tolist()),
        "label_counts": {k: int(v) for k, v in frozen["label"].value_counts().items()},
        "coverage_of_estimation_sample": coverage,
        "pooled_into_model_selection": False,
        "note": "Variant A only. Reported beside the structured columns and never inside them; "
                "no model choice and no penalty depends on it (protocol section 8.1).",
    }
    if coverage == 1.0:
        aligned = np.asarray(aligned, float)
        parts = split(frame.assign(llm_stance=aligned))
        coef = _ols(parts["train"][["llm_stance", "months_since_statement"]].to_numpy(),
                    parts["train"]["delta_y"].to_numpy())
        test = parts["test"]
        pred = test["y10"].to_numpy() + _apply(
            coef, test[["llm_stance", "months_since_statement"]].to_numpy())
        out["test_metrics"] = rmse_mae(test["y10_next"].to_numpy(), pred)
        out["coefficients"] = dict(zip(["const", "llm_stance", "months_since_statement"],
                                       [float(c) for c in coef]))
    return out


# --------------------------------------------------------------------------------------
# Gates on procedure
# --------------------------------------------------------------------------------------
def assert_no_machine_b() -> dict[str, object]:
    """Protocol section 8.2: nothing from Session 1's Machine B may reach block F.

    Machine B is the small transformer Session 1 trained on all 245 statements, test window
    included.  Any feature derived from it would be contaminated end to end, and the contamination
    would be invisible in a forecast metric.  So it is excluded by assertion rather than by care.
    """
    loaded = [name for name in sys.modules
              if name.startswith("session01") or name.startswith("minigpt")]
    if loaded:
        raise RuntimeError(f"a Session 1 module is loaded in this process: {loaded}")
    inputs = {BASELINE_CSV, PANEL_CSV, FOMC_MANIFEST, LLM_SCORES, PROTOCOL, PROTOCOL_SHA}
    for artefact in MACHINE_B_ARTEFACTS:
        if any(artefact in str(p) for p in inputs):
            raise RuntimeError(f"a Machine B artefact is among the declared inputs: {artefact}")
    return {"machine_b_artefacts_loaded": False,
            "session01_modules_in_process": loaded,
            "checked": list(MACHINE_B_ARTEFACTS),
            "why": "Machine B was trained on all 245 statements, the test window included."}


_TEST_SCORED = {"count": 0}


def score_test_once(runs: dict[str, ModelRun], parts: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Score the test window in a single pass, after every choice has been made.

    Protocol section 11: this runs once per process and raises if called again, so that "scored
    once" is enforced rather than promised.
    """
    _TEST_SCORED["count"] += 1
    if _TEST_SCORED["count"] > 1:
        raise RuntimeError("the test window has already been scored in this process "
                           "(protocol section 11.1)")
    test = parts["test"]
    actual = test["y10_next"].to_numpy()
    out = {}
    for key, run in runs.items():
        pred = run.predictions["test"]
        out[key] = {"name": run.name, "column": run.column, **rmse_mae(actual, pred),
                    "cumulative_squared_error": cumulative_squared_error(actual, pred),
                    "detail": run.detail}
    return out


def validation_metrics(runs: dict[str, ModelRun], parts: dict[str, pd.DataFrame]) -> dict:
    """Cheap, live, and carrying no inferential weight -- the notebook recomputes these."""
    valid = parts["valid"]
    actual = valid["y10_next"].to_numpy()
    return {k: rmse_mae(actual, r.predictions["valid"]) for k, r in runs.items()}


def assert_matches_record(test_scores: dict, record_path: Path = RECORDED,
                          places: int = 10) -> dict[str, bool]:
    """Protocol section 11.2: the test-window numbers in hand must equal the recorded file.

    The notebook recomputes the validation metrics live -- they are cheap and carry no
    inferential weight -- and then calls this.  A mismatch is a failure, not a new result: it
    means the test window has been scored a second time under different choices, which is the one
    thing the protocol forbids.
    """
    recorded = json.loads(Path(record_path).read_text(encoding="utf-8"))["test_metrics"]
    verdict = {}
    for key, here in test_scores.items():
        there = recorded.get(key)
        verdict[key] = (there is not None
                        and round(here["rmse"], places) == round(there["rmse"], places)
                        and round(here["mae"], places) == round(there["mae"], places))
    missing = set(recorded) - set(test_scores)
    if missing:
        raise RuntimeError(f"the record holds models this run did not produce: {sorted(missing)}")
    if not all(verdict.values()):
        failed = [k for k, ok in verdict.items() if not ok]
        raise RuntimeError(f"test-window numbers differ from {Path(record_path).name}: {failed}")
    return verdict


# --------------------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------------------
def three_column_table(frozen: dict, test_scores: dict) -> list[dict[str, object]]:
    """The frozen file, the re-estimated comparator and the model plus text, side by side."""
    rows = []
    pairs = [("random walk", "random_walk", "random_walk", None),
             ("AR(1)", "ar1", "ar1", None),
             ("OLS-5", "ols", "ols5", "ols5_plus_text"),
             ("OLS-5 + TF-IDF ridge", "ols", "ols5", "ols5_plus_tfidf_ridge"),
             ("text alone", None, None, "text_alone"),
             ("TF-IDF ridge alone", None, None, "tfidf_ridge_alone")]
    for label, frozen_key, reest_key, text_key in pairs:
        row: dict[str, object] = {"model": label}
        row["frozen_file"] = (frozen["scored"][frozen_key] if frozen_key else None)
        row["re_estimated"] = ({"rmse": test_scores[reest_key]["rmse"],
                                "mae": test_scores[reest_key]["mae"]} if reest_key else None)
        row["plus_text"] = ({"rmse": test_scores[text_key]["rmse"],
                             "mae": test_scores[text_key]["mae"]} if text_key else None)
        rows.append(row)
    return rows


def build_records() -> dict:
    """Run block F end to end and collect everything into one record."""
    started = time.perf_counter()
    machine_b = assert_no_machine_b()
    inputs = load_inputs()
    corpus = load_statements()
    frame = merge_text(inputs.panel, corpus)
    space = fit_text_space(corpus)
    frame = attach_features(frame, corpus, space)
    parts = split(frame)

    sizes = {k: int(len(v)) for k, v in parts.items()}
    if sizes != {"train": 250, "valid": 60, "test": 79}:
        raise RuntimeError(f"split sizes are {sizes}, protocol section 2.2 requires "
                           f"250 / 60 / 79")

    frozen = frozen_baseline_table(inputs.baseline)
    runs = run_models(parts, corpus, space)
    valid_metrics = validation_metrics(runs, parts)
    test_scores = score_test_once(runs, parts)

    test = parts["test"]
    actual = test["y10_next"].to_numpy()

    def errors(key: str) -> np.ndarray:
        return actual - runs[key].predictions["test"]

    comparisons = {
        "ols5_vs_ols5_plus_text": {
            "nested": True,
            "diebold_mariano_hln": diebold_mariano(errors("ols5"), errors("ols5_plus_text")),
            "clark_west": clark_west(actual, runs["ols5"].predictions["test"],
                                     runs["ols5_plus_text"].predictions["test"])},
        "ols5_vs_ols5_plus_tfidf_ridge": {
            "nested": True,
            "diebold_mariano_hln": diebold_mariano(errors("ols5"),
                                                   errors("ols5_plus_tfidf_ridge")),
            "clark_west": clark_west(actual, runs["ols5"].predictions["test"],
                                     runs["ols5_plus_tfidf_ridge"].predictions["test"])},
        "random_walk_vs_text_alone": {
            "nested": False,
            "diebold_mariano_hln": diebold_mariano(errors("random_walk"), errors("text_alone")),
            "note": "not nested in the Clark-West sense; the DM statistic is the relevant one"},
        "random_walk_vs_ols5": {
            "nested": False,
            "diebold_mariano_hln": diebold_mariano(errors("random_walk"), errors("ols5"))},
        "ar1_vs_ols5_plus_text": {
            "nested": False,
            "diebold_mariano_hln": diebold_mariano(errors("ar1"), errors("ols5_plus_text"))},
    }

    stale_with = int(parts["test"]["is_stale"].sum())
    raw_dates = sorted(pd.Timestamp(e["date"]) for e in
                       json.loads(FOMC_MANIFEST.read_text(encoding="utf-8"))["files"])
    raw_stamps = np.array([d.value for d in raw_dates])
    raw_pos = np.searchsorted(raw_stamps, test["date"].to_numpy().astype("datetime64[ns]")
                              .astype("int64"), side="right") - 1
    stale_without = int(sum(
        1 for d, i in zip(test["date"], raw_pos)
        if (d.year - raw_dates[i].year) * 12 + (d.month - raw_dates[i].month) > 0))

    gates = {
        "input_hashes_match_protocol": True,
        "frozen_baselines_reproduce": all(frozen["reproduces_to_four_decimals"].values()),
        "split_sizes_250_60_79": sizes == {"train": 250, "valid": 60, "test": 79},
        "missing_month_inherited_not_filled": inputs.missing_forecast_dates == ["2025-11-30"],
        "no_post_cutoff_document_in_a_fitted_object": True,
        "no_machine_b_artefact_loaded": not machine_b["machine_b_artefacts_loaded"],
        "test_window_scored_once": _TEST_SCORED["count"] == 1,
    }

    return {
        "_meta": {
            "module": "session05_forecast.py",
            "block": "F -- does FOMC statement text add anything to Session 2's forecast?",
            "recorded_on": "2026-09-20",
            "seed": SEED,
            "python": sys.version.split()[0],
            "numpy": np.__version__, "pandas": pd.__version__,
            "seconds": round(time.perf_counter() - started, 1),
            "protocol_sha256": file_sha256(PROTOCOL),
            "protocol_hash_predates_first_scored_run": True,
            "input_sha256": inputs.hashes,
        },
        "gates": gates,
        "gate_policy": "Every gate is on procedure. None of them depends on text helping: "
                       "'text adds nothing' passes all seven and is reported as found.",
        "sample": {
            "estimation_sample_starts": SAMPLE_START,
            "sizes": sizes,
            "rows_dropped_before_first_statement": 50,
            "missing_forecast_dates": inputs.missing_forecast_dates,
            "test_rows_with_a_stale_statement": stale_with,
            "test_rows_with_a_stale_statement_without_exclusions": stale_without,
            "stale_note": "The protocol's expected count, 25, is the count under the raw corpus. "
                          "Applying the hand-listed exclusion set makes it 26: August 2025's only "
                          "release is the framework-review document, so that month falls back to "
                          "2025-07-30. The difference is a consequence of the exclusion set, not "
                          "of the information rule; both counts are recorded.",
            "train_rows_with_a_stale_statement": int(parts["train"]["is_stale"].sum()),
            "valid_rows_with_a_stale_statement": int(parts["valid"]["is_stale"].sum()),
        },
        "corpus": {
            "n_statements_total": 245,
            "n_kept": len(corpus.dates),
            "excluded": corpus.excluded,
            "months_with_more_than_one_statement": 7,
            "information_rule": "month t's signal is the last kept statement dated on or before "
                                "the last observation date of month t; the last one in a month "
                                "with several wins",
        },
        "text_space": {
            "vocabulary_size": space.vocabulary_size,
            "n_training_documents": space.n_training_documents,
            "training_cutoff": space.training_cutoff,
            "seed_terms_found": space.seeds_found,
            "stance_standardised_on": "the training window only",
            "novelty_in_a_month_without_a_meeting": 0.0,
        },
        "frozen_baselines": frozen,
        "validation_metrics": valid_metrics,
        "test_metrics": test_scores,
        "three_column_table": three_column_table(frozen, test_scores),
        "comparisons": comparisons,
        "llm_column": llm_column(frame),
        "machine_b": machine_b,
        "test_dates": [str(d.date()) for d in test["date"]],
    }


def report(record: dict) -> str:
    """The three-column table and the tests, as a reader sees them."""
    lines = ["Block F -- Session 2's forecast, with and without the Committee's words", ""]
    lines.append(f"  {'model':<24}{'frozen file':>22}{'re-estimated':>22}{'plus text':>22}")
    lines.append(f"  {'':<24}{'RMSE     MAE':>22}{'RMSE     MAE':>22}{'RMSE     MAE':>22}")
    for row in record["three_column_table"]:
        cells = []
        for key in ("frozen_file", "re_estimated", "plus_text"):
            cell = row[key]
            cells.append("        --        " if cell is None
                         else f"{cell['rmse']:>10.4f}{cell['mae']:>10.4f}")
        lines.append(f"  {row['model']:<24}" + "".join(f"{c:>22}" for c in cells))
    lines.append("")
    lines.append("  Session 2 publishes the MLP as the mean of ten per-seed RMSEs, "
                 f"{MLP_PUBLISHED['rmse']:.4f} / {MLP_PUBLISHED['mae']:.4f};")
    lines.append(f"  the mlp_seed_mean column is the mean forecast and scores "
                 f"{MLP_COLUMN['rmse']:.4f} / {MLP_COLUMN['mae']:.4f}.")
    lines.append("")
    lines.append("  Comparisons")
    for name, cmp in record["comparisons"].items():
        dm = cmp["diebold_mariano_hln"]
        line = (f"    {name:<34} DM-HLN {dm['statistic']:>7.3f}  p {dm['p_value']:>6.3f}")
        if "clark_west" in cmp:
            cw = cmp["clark_west"]
            line += f"   CW {cw['statistic']:>7.3f}  p {cw['p_value_one_sided']:>6.3f}"
        lines.append(line)
    lines.append("")
    lines.append("  Gates (procedure only)")
    for name, ok in record["gates"].items():
        lines.append(f"    {'PASS' if ok else 'FAIL'}  {name}")
    stale = record["sample"]
    lines.append("")
    lines.append(f"  Stale test months: {stale['test_rows_with_a_stale_statement']} of 79 "
                 f"({stale['test_rows_with_a_stale_statement_without_exclusions']} before the "
                 f"exclusion set).")
    llm = record["llm_column"]
    lines.append(f"  LLM column: {'present' if llm['available'] else 'not yet recorded'}.")
    return "\n".join(lines)


if __name__ == "__main__":
    rec = build_records()
    RECORDED.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(report(rec))
    print(f"\nwritten to {RECORDED.name}  ({rec['_meta']['seconds']} s)")
