"""Session 5 lab -- block E, measurement error in an LLM-generated variable.

Block E is the book's chapter 14 run inside the notebook.  The four scripts of that chapter are
vendored byte-identically under book_code/ch14/ and this module is a set of thin wrappers over
them: it loads each file, calls the chapter's OWN functions in the chapter's OWN order, and
returns the results as data instead of printed text.  Nothing here re-derives a statistic.  The
point of the block is that the lab reproduces the chapter rather than paraphrasing it, so when a
student changes a number in book_code/ the lab moves with it.

  bias_correction_table()   the four estimators of beta -- ideal, naive, validation-only,
                            bias-corrected -- with their intervals and the calibration equation
  validation_size_sweep()   interval width against the size of the hand-coded validation set
  per_class_kappa()         overall and per-class kappa, and what merging the failing class buys
  self_consistency()        the screen's threshold, precision, recall and F1, and the share of
                            stable errors it is blind to by construction
  order_effect_power()      items required for 80% power against a first-position effect of size
                            delta

Every one of these is gated against the digits printed in chapters/chapter14.tex (see CHAPTER
below, which carries the .tex line where each digit appears).  A gate that fails is recorded as a
disagreement with the chapter and left alone: this module never tunes itself towards the book.

Reproducibility rests on one fact about the chapter's scripts -- each draws from a single
np.random.default_rng(42) whose consumption order is fixed by its main().  The wrappers reproduce
that order call for call; a re-ordering silently changes every number.

Imports: numpy and the standard library.  No network, CPU only.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType

import numpy as np

HERE = Path(__file__).resolve().parent
BOOK_CODE = HERE / "book_code"
MANIFEST = BOOK_CODE / "MANIFEST.json"
RECORDED = HERE / "recorded_measure.json"

SEED = 42


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------
# What the chapter prints
# --------------------------------------------------------------------------------------
# Each entry is (value, decimals, chapter14.tex line).  `decimals` is the precision the chapter
# prints to and therefore the precision a gate may demand: the chapter rounds, so a match is
# equality after rounding to that many places, never to machine precision.
CHAPTER: dict[str, tuple[float, int, int]] = {
    # -- the measure itself (line 194)
    "corr_mstar_mhat":        (0.851, 3, 194),
    "mean_bias":              (0.0063, 4, 194),
    "error_on_size_slope":    (-0.0359, 4, 194),
    "error_on_size_se":       (0.0019, 4, 194),
    # -- the four estimators (line 195)
    "beta_ideal":             (-0.0806, 4, 195),
    "beta_naive":             (-0.0552, 4, 195),
    "naive_pct_of_truth":     (31.0, 1, 195),
    "beta_validation_only":   (-0.0866, 4, 195),
    "width_validation_only":  (0.0570, 4, 195),
    "beta_corrected":         (-0.0785, 4, 195),
    "corrected_pct_of_truth": (1.9, 1, 195),
    "width_corrected":        (0.0293, 4, 195),
    # -- what the correction buys (line 196)
    "corrected_narrower_pct": (49.0, 0, 196),
    "calibration_intercept":  (0.1423, 4, 196),
    "calibration_slope_mhat": (0.7031, 4, 196),
    "calibration_slope_c":    (0.0048, 4, 196),
    # -- the validation sweep (line 197)
    "sweep_50_to_200_pct":    (49.0, 0, 197),
    "sweep_200_to_800_pct":   (17.0, 0, 197),
    # -- per-class kappa (table at lines 313-334, read back at 336-338)
    "kappa_overall":          (0.844, 3, 330),
    "accuracy_overall":       (0.900, 3, 330),
    "kappa_rare_class":       (0.634, 3, 322),
    "recall_rare_class":      (0.528, 3, 322),
    "share_rare_class":       (0.120, 3, 322),
    "kappa_after_merge":      (0.881, 3, 338),
    # every row of the chapter's table 14.2, in the chapter's order
    "share_supply_chain":     (0.413, 3, 318),
    "recall_supply_chain":    (0.968, 3, 318),
    "kappa_supply_chain":     (0.891, 3, 318),
    "share_monetary":         (0.387, 3, 320),
    "recall_monetary":        (0.974, 3, 320),
    "kappa_monetary":         (0.876, 3, 320),
    "share_uncodable":        (0.080, 3, 324),
    "recall_uncodable":       (0.750, 3, 324),
    "kappa_uncodable":        (0.825, 3, 324),
    # -- order effects (line 557)
    "n_required_letter":      (1555, 0, 557),
    "n_required_symbol":      (34, 0, 557),
    "delta_letter":           (0.0355, 4, 557),
    "delta_symbol":           (0.2335, 4, 557),
    # -- the self-consistency screen (lines 684 and 686)
    "sc_threshold":           (0.38, 2, 684),
    "sc_precision":           (0.745, 3, 684),
    "sc_recall":              (0.755, 3, 684),
    "sc_mean_h_fabricated":   (0.514, 3, 684),
    "sc_mean_h_retrieved":    (0.169, 3, 684),
    "sc_stable_undetected_pct": (88.8, 1, 686),
    # -- the removal design, chapter footnote at line 555
    "design_bias_fixed":      (0.1326, 4, 555),
    "design_bias_randomised": (-0.0005, 4, 555),
    "design_bias_exhaustive": (-0.0001, 4, 555),
    "design_rmse_randomised": (0.0103, 4, 555),
    "design_rmse_exhaustive": (0.0105, 4, 555),
}

# Where the printed digits above come from. This is a provenance record, not a path the module
# opens: the statistics are computed by the vendored copies under book_code/, and the symbolic
# root keeps a private filesystem layout out of a file that ships with the lab.
CHAPTER_TEX = ("<BOOK_REPO:Book1-AI-for-Econonomist>/"
               "面向经济学家的人工智能/chapters/chapter14.tex")


@dataclass(frozen=True)
class Gate:
    """One printed digit of the chapter set beside what the vendored script produces here."""
    name: str
    chapter: float
    computed: float
    decimals: int
    tex_line: int

    @property
    def agrees(self) -> bool:
        return round(float(self.computed), self.decimals) == round(float(self.chapter),
                                                                   self.decimals)

    def __str__(self) -> str:
        mark = "agrees" if self.agrees else "DIFFERS"
        return (f"{mark:>7}  {self.name:<26} chapter {self.chapter:>10.4f}   "
                f"here {self.computed:>10.4f}   (chapter14.tex:{self.tex_line})")


def gate(name: str, computed: float) -> Gate:
    value, decimals, line = CHAPTER[name]
    return Gate(name=name, chapter=value, computed=float(computed), decimals=decimals,
                tex_line=line)


# --------------------------------------------------------------------------------------
# Loading the vendored chapter scripts
# --------------------------------------------------------------------------------------
_LOADED: dict[str, ModuleType] = {}


def _manifest_sha256(relative: str) -> str:
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    for entry in entries:
        if entry["path"] == relative:
            return entry["sha256"]
    raise KeyError(f"{relative} is not listed in {MANIFEST.name}")


def book_module(stem: str, chapter: str = "ch14") -> ModuleType:
    """Import one vendored chapter script by path, after checking it against MANIFEST.json.

    The scripts are not a package and their names would collide with anything else on the path,
    so they are loaded from their file and cached under a namespaced key.  The hash check is the
    whole reason the manifest exists: it makes an edited copy fail loudly instead of quietly
    changing the lab's numbers.
    """
    key = f"{chapter}/{stem}"
    if key in _LOADED:
        return _LOADED[key]
    path = BOOK_CODE / chapter / f"{stem}.py"
    if not path.exists():
        raise FileNotFoundError(f"vendored chapter script missing: {path}")
    actual = file_sha256(path)
    expected = _manifest_sha256(f"{chapter}/{stem}.py")
    if actual != expected:
        raise RuntimeError(f"{key}.py does not match MANIFEST.json "
                           f"(manifest {expected[:16]}..., file {actual[:16]}...)")
    spec = importlib.util.spec_from_file_location(f"book_{chapter}_{stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # book_code/ holds the vendored copies and their manifest, and nothing else: a __pycache__
    # written beside them would be an unvendored file in a folder whose whole point is that its
    # contents are byte-identical to the book's and accounted for in MANIFEST.json.
    was_writing = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = was_writing
    _LOADED[key] = module
    return module


# --------------------------------------------------------------------------------------
# E1  Bias correction for an LLM-generated regressor
# --------------------------------------------------------------------------------------
@dataclass
class BiasCorrection:
    """The chapter's four estimators of beta, their intervals and the calibration equation."""
    beta_true: float
    n: int
    n_validation: int
    corr_mstar_mhat: float
    mean_bias: float
    error_on_size_slope: float
    error_on_size_se: float
    estimators: dict[str, dict[str, float]]
    calibration: dict[str, float]
    width_validation_only: float
    width_corrected: float
    corrected_narrower_pct: float
    gates: list[Gate] = field(default_factory=list)


def bias_correction_table() -> BiasCorrection:
    """The four estimators of the chapter's worked example, from book_code/ch14/bias_correction.py.

    The call order reproduces that script's main() exactly -- simulate, then draw the validation
    index, then the three closed-form estimators, then regression calibration, then the bootstrap
    -- because all five draw from one generator and the order fixes the numbers.

    Naive is the estimator a researcher gets by treating the model's score as the variable; it is
    attenuated.  Validation-only is unbiased and imprecise.  The corrected estimator buys the
    precision back by calibrating the score on the hand-coded rows and using it full sample.
    """
    bc = book_module("bias_correction")
    rng = np.random.default_rng(bc.SEED)

    m_star, m_hat, c, y = bc.simulate(rng)
    valid_idx = rng.choice(len(y), size=bc.N_VALID, replace=False)

    corr = float(np.corrcoef(m_star, m_hat)[0, 1])
    err = m_hat - m_star
    slope_err_c, se_err_c = bc.ols(c.reshape(-1, 1), err)

    coef_ideal, se_ideal = bc.ols(np.column_stack([m_star, c]), y)
    coef_naive, se_naive = bc.ols(np.column_stack([m_hat, c]), y)
    coef_vo, se_vo = bc.ols(np.column_stack([m_star[valid_idx], c[valid_idx]]), y[valid_idx])
    coef_rc, cal_coef = bc.regression_calibration(m_star, m_hat, c, y, valid_idx)
    boot = bc.bootstrap_rc(rng, m_star, m_hat, c, y, valid_idx)
    se_rc = float(boot.std(ddof=1))
    lo_rc, hi_rc = (float(v) for v in np.percentile(boot, [2.5, 97.5]))

    def row(beta: float, se: float, lo: float, hi: float) -> dict[str, float]:
        return {"beta": float(beta), "se": float(se), "ci_low": float(lo), "ci_high": float(hi),
                "ci_width": float(hi - lo),
                "pct_of_truth": float(100.0 * (beta - bc.BETA_TRUE) / abs(bc.BETA_TRUE))}

    lo_i, hi_i = bc.ci(coef_ideal[1], se_ideal[1])
    lo_n, hi_n = bc.ci(coef_naive[1], se_naive[1])
    lo_v, hi_v = bc.ci(coef_vo[1], se_vo[1])
    estimators = {
        "ideal": row(coef_ideal[1], se_ideal[1], lo_i, hi_i),
        "naive": row(coef_naive[1], se_naive[1], lo_n, hi_n),
        "validation_only": row(coef_vo[1], se_vo[1], lo_v, hi_v),
        "corrected": row(coef_rc[1], se_rc, lo_rc, hi_rc),
    }
    # The chapter quotes the validation-only width as the normal-approximation width, not the
    # bootstrap one, so it is recomputed here the way the chapter's main() does.
    width_vo = float(2.0 * 1.96 * se_vo[1])
    width_rc = float(hi_rc - lo_rc)
    narrower = float(100.0 * (1.0 - width_rc / width_vo))

    out = BiasCorrection(
        beta_true=float(bc.BETA_TRUE), n=int(bc.N), n_validation=int(bc.N_VALID),
        corr_mstar_mhat=corr, mean_bias=float(np.mean(err)),
        error_on_size_slope=float(slope_err_c[1]), error_on_size_se=float(se_err_c[1]),
        estimators=estimators,
        calibration={"intercept": float(cal_coef[0]), "slope_m_hat": float(cal_coef[1]),
                     "slope_c": float(cal_coef[2])},
        width_validation_only=width_vo, width_corrected=width_rc,
        corrected_narrower_pct=narrower,
    )
    out.gates = [
        gate("corr_mstar_mhat", out.corr_mstar_mhat),
        gate("mean_bias", out.mean_bias),
        gate("error_on_size_slope", out.error_on_size_slope),
        gate("error_on_size_se", out.error_on_size_se),
        gate("beta_ideal", estimators["ideal"]["beta"]),
        gate("beta_naive", estimators["naive"]["beta"]),
        gate("naive_pct_of_truth", estimators["naive"]["pct_of_truth"]),
        gate("beta_validation_only", estimators["validation_only"]["beta"]),
        gate("width_validation_only", width_vo),
        gate("beta_corrected", estimators["corrected"]["beta"]),
        gate("corrected_pct_of_truth", estimators["corrected"]["pct_of_truth"]),
        gate("width_corrected", width_rc),
        gate("corrected_narrower_pct", narrower),
        gate("calibration_intercept", out.calibration["intercept"]),
        gate("calibration_slope_mhat", out.calibration["slope_m_hat"]),
        gate("calibration_slope_c", out.calibration["slope_c"]),
    ]
    return out


# --------------------------------------------------------------------------------------
# E2  How large the hand-coded validation set has to be
# --------------------------------------------------------------------------------------
@dataclass
class ValidationSweep:
    """Interval width of the corrected estimator against the size of the validation set."""
    sizes: list[int]
    rows: list[dict[str, float]]
    narrowing_50_to_200_pct: float
    narrowing_200_to_800_pct: float
    gates: list[Gate] = field(default_factory=list)


def validation_size_sweep(sizes: tuple[int, ...] = (50, 100, 200, 400, 800),
                          n_boot: int = 1000) -> ValidationSweep:
    """Block [4] of the chapter's bias_correction.py: what more hand labels actually buy.

    The sweep runs on a second generator, default_rng(SEED + 1), exactly as the chapter does, so
    it does not depend on how much of the first generator the estimator table consumed.  The data
    are the same draw, because they come from simulate() on a fresh default_rng(SEED).

    The curve is the budget rule of the chapter: width falls in the number of labels but at a
    decreasing rate, because past a few hundred documents the binding constraint stops being the
    calibration equation's own error and becomes the full-sample residual variance.
    """
    bc = book_module("bias_correction")
    rng = np.random.default_rng(bc.SEED)
    m_star, m_hat, c, y = bc.simulate(rng)

    rng2 = np.random.default_rng(bc.SEED + 1)
    rows, widths = [], {}
    for n_v in sizes:
        idx = rng2.choice(len(y), size=n_v, replace=False)
        coef_t, cal_t = bc.regression_calibration(m_star, m_hat, c, y, idx)
        boot = bc.bootstrap_rc(rng2, m_star, m_hat, c, y, idx, n_boot=n_boot)
        lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
        widths[n_v] = hi - lo
        rows.append({"n_validation": int(n_v), "beta_corrected": float(coef_t[1]),
                     "ci_low": lo, "ci_high": hi, "ci_width": float(hi - lo),
                     "calibration_slope_m_hat": float(cal_t[1])})

    def narrowing(small: int, large: int) -> float:
        return float(100.0 * (1.0 - widths[large] / widths[small]))

    out = ValidationSweep(sizes=[int(s) for s in sizes], rows=rows,
                          narrowing_50_to_200_pct=narrowing(50, 200),
                          narrowing_200_to_800_pct=narrowing(200, 800))
    out.gates = [gate("sweep_50_to_200_pct", out.narrowing_50_to_200_pct),
                 gate("sweep_200_to_800_pct", out.narrowing_200_to_800_pct)]
    return out


# --------------------------------------------------------------------------------------
# E3  Per-class agreement, and why the overall number hides the failure
# --------------------------------------------------------------------------------------
@dataclass
class PerClassKappa:
    """Overall and one-vs-rest kappa on the hand-coded audit subset, plus the merge remedy."""
    n_full: int
    n_audit: int
    labels: list[str]
    accuracy: float
    kappa_overall: float
    fleiss_band: str
    landis_koch_band: str
    classes: list[dict[str, object]]
    confusion: list[list[int]]
    merge: dict[str, object]
    gates: list[Gate] = field(default_factory=list)


def per_class_kappa() -> PerClassKappa:
    """The chapter's table 14.2, from book_code/ch14/human_llm_kappa.py.

    An overall kappa of 0.844 reads as excellent on the Fleiss bands, and the lesson of the table
    is that it is carried entirely by the two high-frequency narratives: the rare class, which is
    where the interesting hypothesis lives, is coded far worse and the overall figure cannot show
    it.  Merging the rare class upward raises the overall number without improving any coding --
    it removes the distinction the research question was about.
    """
    hk = book_module("human_llm_kappa")
    rng = np.random.default_rng(hk.SEED)
    human_full, llm_full = hk.simulate_coding(rng)

    n_audit = int(round(hk.AUDIT_FRAC * len(human_full)))
    audit = rng.choice(len(human_full), size=n_audit, replace=False)
    human, llm = human_full[audit], llm_full[audit]
    k = len(hk.LABELS)

    accuracy = float(np.mean(human == llm))
    kappa = float(hk.cohen_kappa(human, llm))
    cm = hk.confusion_matrix(human, llm, k)
    pck = hk.per_category_kappa(human, llm, k)

    classes = []
    for c in range(k):
        support = int(cm[c].sum())
        classes.append({
            "label": hk.LABELS[c],
            "share": float(np.mean(human == c)),
            "support": support,
            "recall": float(cm[c, c] / max(support, 1)),
            "kappa": float(pck[c]),
            "meets_threshold_0_75": bool(pck[c] >= 0.75),
        })

    merge_from, merge_into = 2, 0
    hm = np.where(human == merge_from, merge_into, human)
    lm = np.where(llm == merge_from, merge_into, llm)
    kappa_merged = float(hk.cohen_kappa(hm, lm))

    # The chapter's "rare class" is the low-frequency NARRATIVE that fails the 0.75 threshold --
    # the one the merge remedy then removes -- and not simply the smallest class: the residual
    # "uncodable" category is smaller still (0.080 against 0.120) and clears the threshold.
    rare = classes[merge_from]
    out = PerClassKappa(
        n_full=int(len(human_full)), n_audit=n_audit, labels=list(hk.LABELS),
        accuracy=accuracy, kappa_overall=kappa,
        fleiss_band=hk.fleiss_band(kappa), landis_koch_band=hk.landis_koch_band(kappa),
        classes=classes, confusion=[[int(v) for v in r] for r in cm],
        merge={"from": hk.LABELS[merge_from], "into": hk.LABELS[merge_into],
               "kappa_before": kappa, "kappa_after": kappa_merged,
               "what_it_costs": "the distinction itself, and with it any hypothesis about the "
                                "merged narrative; the coding did not improve"},
    )
    out.gates = [
        gate("kappa_overall", kappa),
        gate("accuracy_overall", accuracy),
        gate("share_rare_class", rare["share"]),
        gate("kappa_rare_class", rare["kappa"]),
        gate("recall_rare_class", rare["recall"]),
        gate("kappa_after_merge", kappa_merged),
    ]
    for row, stem in zip(classes, ("supply_chain", "monetary", None, "uncodable")):
        if stem is None:                  # already gated above as the rare class
            continue
        out.gates += [gate(f"share_{stem}", row["share"]), gate(f"recall_{stem}", row["recall"]),
                      gate(f"kappa_{stem}", row["kappa"])]
    return out


# --------------------------------------------------------------------------------------
# E4  The self-consistency screen and what it cannot see
# --------------------------------------------------------------------------------------
@dataclass
class SelfConsistency:
    """The screen's operating point, and the share of stable errors that pass it unflagged."""
    n_claims: int
    resamples: int
    by_kind: dict[str, dict[str, float]]
    threshold: float
    precision: float
    recall: float
    f1: float
    flagged_share: float
    fabricated_missed: int
    fabricated_total: int
    stable_error_total: int
    stable_error_undetected: int
    stable_error_undetected_pct: float
    gates: list[Gate] = field(default_factory=list)


def self_consistency() -> SelfConsistency:
    """The hallucination screen of book_code/ch14/self_consistency_check.py at its best threshold.

    The number that matters is the last one.  Resampling measures how much the model's answer
    moves, not whether it is true, so a false statement the model holds firmly resamples as
    consistently as a true one and the screen is blind to it by construction.  Raising the number
    of resamples does not help: it sharpens the score, it does not change what the score measures.
    """
    sc = book_module("self_consistency_check")
    rng = np.random.default_rng(sc.SEED)
    kinds, scores = sc.simulate(rng)

    by_kind = {}
    for index, name in enumerate(sc.KINDS):
        sel = kinds == index
        p10, p90 = (float(v) for v in np.percentile(scores[sel], [10, 90]))
        by_kind[name] = {"n": int(sel.sum()), "mean_h": float(scores[sel].mean()),
                         "sd_h": float(scores[sel].std(ddof=1)), "p10": p10, "p90": p90}

    threshold, met = sc.best_threshold(kinds, scores)
    fabricated = kinds == 1
    missed = int(np.sum(fabricated & (scores < threshold)))

    stable = kinds == 2                      # "confidently wrong": held firmly, and false
    undetected = int(np.sum(stable & (scores < threshold)))
    stable_total = int(stable.sum())

    out = SelfConsistency(
        n_claims=int(len(kinds)), resamples=int(sc.R_DEFAULT), by_kind=by_kind,
        threshold=float(threshold), precision=float(met["precision"]),
        recall=float(met["recall"]), f1=float(met["f1"]),
        flagged_share=float(met["flagged"]),
        fabricated_missed=missed, fabricated_total=int(fabricated.sum()),
        stable_error_total=stable_total, stable_error_undetected=undetected,
        stable_error_undetected_pct=float(100.0 * undetected / max(stable_total, 1)),
    )
    out.gates = [
        gate("sc_threshold", out.threshold),
        gate("sc_precision", out.precision),
        gate("sc_recall", out.recall),
        gate("sc_mean_h_fabricated", by_kind["fabricated"]["mean_h"]),
        gate("sc_mean_h_retrieved", by_kind["retrieved"]["mean_h"]),
        gate("sc_stable_undetected_pct", out.stable_error_undetected_pct),
    ]
    return out


# --------------------------------------------------------------------------------------
# E5  Detecting an option-order effect
# --------------------------------------------------------------------------------------
@dataclass
class OrderEffectPower:
    """Items needed for 80% power against a first-position effect, by the size of that effect."""
    deltas: list[dict[str, object]]
    designs: dict[str, dict[str, float]]
    letter_power_at_50_items: float
    gates: list[Gate] = field(default_factory=list)


def order_effect_power() -> OrderEffectPower:
    """Design calculations from book_code/ch14/prompt_permutation_test.py.

    Two questions, one calculation.  Which presentation removes a first-position effect: holding
    the order fixed leaves the whole effect standing, randomising the order across items removes
    it, and exhausting the orderings at a fixed budget buys variance rather than unbiasedness.
    And how many items are needed before the effect can be told from zero: the answer swings by a
    factor of forty across two label formats, which is why a robustness check run on a few dozen
    items is evidence of nothing either way.

    The generator is consumed in the chapter's order -- the three removal designs first, then the
    detection powers -- so the simulated figures match the chapter's footnote.
    """
    pp = book_module("prompt_permutation_test")
    rng = np.random.default_rng(pp.SEED)

    d_overall = round(pp.BT_POINTS[0][1] - 0.5, 4)
    d_letter = round(pp.BT_POINTS[1][1] - 0.5, 4)
    d_symbol = round(pp.BT_POINTS[2][1] - 0.5, 4)

    designs = {}
    for design, n_orders in (("fixed", 1), ("randomised", 1), ("exhaustive", pp.M_OPTIONS)):
        bias, rmse = pp.simulate_design(rng, d_overall, design, n_orders=n_orders)
        designs[design] = {"mean_bias": float(bias), "rmse": float(rmse)}

    grid = sorted({d_letter, 0.05, 0.10, d_overall, d_symbol})
    named = {d_letter: "letter labels", d_overall: "overall", d_symbol: "symbol labels"}
    deltas = []
    for d in grid:
        n_req = int(pp.n_required(d))
        deltas.append({"delta": float(d), "setting": named.get(d, ""),
                       "n_required_80pct_power": n_req,
                       "simulated_power_at_n": float(pp.empirical_power(rng, d, n_req))})
    power_50 = float(pp.empirical_power(rng, d_letter, 50))

    by_delta = {row["delta"]: row["n_required_80pct_power"] for row in deltas}
    out = OrderEffectPower(deltas=deltas, designs=designs, letter_power_at_50_items=power_50)
    out.gates = [
        gate("delta_letter", d_letter),
        gate("delta_symbol", d_symbol),
        gate("n_required_letter", by_delta[d_letter]),
        gate("n_required_symbol", by_delta[d_symbol]),
        gate("design_bias_fixed", designs["fixed"]["mean_bias"]),
        gate("design_bias_randomised", designs["randomised"]["mean_bias"]),
        gate("design_bias_exhaustive", designs["exhaustive"]["mean_bias"]),
        gate("design_rmse_randomised", designs["randomised"]["rmse"]),
        gate("design_rmse_exhaustive", designs["exhaustive"]["rmse"]),
    ]
    return out


# --------------------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------------------
def _serialise(obj) -> dict:
    data = asdict(obj)
    data["gates"] = [{"name": g.name, "chapter": g.chapter, "computed": g.computed,
                      "decimals": g.decimals, "tex_line": g.tex_line, "agrees": g.agrees}
                     for g in obj.gates]
    return data


def build_records() -> dict:
    """Run every block and collect it, with the gate verdicts, into one record."""
    started = time.perf_counter()
    blocks = {
        "bias_correction": bias_correction_table(),
        "validation_sweep": validation_size_sweep(),
        "per_class_kappa": per_class_kappa(),
        "self_consistency": self_consistency(),
        "order_effect_power": order_effect_power(),
    }
    gates = [g for block in blocks.values() for g in block.gates]
    disagreements = [g for g in gates if not g.agrees]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {
        "_meta": {
            "module": "session05_measure.py",
            "block": "E -- measurement error in an LLM-generated variable",
            "recorded_on": "2026-09-20",
            "seed": SEED,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "seconds": round(time.perf_counter() - started, 1),
            "book_code_head": manifest["book_repo"]["head"],
            "book_code_dirty": manifest["book_repo"]["dirty"],
            "book_code_sha256": {e["path"]: e["sha256"] for e in manifest["files"]},
            "chapter_source": CHAPTER_TEX,
            "how_the_numbers_are_produced": "The vendored chapter scripts are imported and their "
                                            "own functions called in their own order. No statistic "
                                            "in this module is a reimplementation.",
        },
        "gate_summary": {
            "checked": len(gates),
            "agree": len(gates) - len(disagreements),
            "differ": len(disagreements),
            "differing": [{"name": g.name, "chapter": g.chapter, "computed": g.computed,
                           "tex_line": g.tex_line} for g in disagreements],
            "policy": "A disagreement with the chapter is recorded here and left standing. The "
                      "scripts are vendored byte-identically and nothing in this module is "
                      "adjusted to close a gap.",
        },
        **{name: _serialise(block) for name, block in blocks.items()},
    }


def report(record: dict) -> str:
    """The gate table as a reader sees it, one line per printed digit."""
    lines = ["Block E -- the chapter's own code, against the chapter's printed digits", ""]
    for name in ("bias_correction", "validation_sweep", "per_class_kappa", "self_consistency",
                 "order_effect_power"):
        lines.append(f"  {name}")
        for g in record[name]["gates"]:
            mark = "agrees " if g["agrees"] else "DIFFERS"
            lines.append(f"    {mark}  {g['name']:<26} chapter {g['chapter']:>10.4f}   "
                         f"here {g['computed']:>10.4f}   (chapter14.tex:{g['tex_line']})")
        lines.append("")
    s = record["gate_summary"]
    lines.append(f"  {s['agree']} of {s['checked']} printed digits agree; {s['differ']} differ.")
    return "\n".join(lines)


if __name__ == "__main__":
    rec = build_records()
    RECORDED.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(report(rec))
    print(f"\nwritten to {RECORDED.name}  ({rec['_meta']['seconds']} s)")
