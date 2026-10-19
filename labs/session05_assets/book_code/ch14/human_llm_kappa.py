"""
human_llm_kappa.py — Human-Machine Agreement for Survey Coding
==============================================================

Companion to Chapter 15, §15.2.2 (Comparison with Hand Coding and
Validation) and Exercise "编程实践" 1.

Complements code/ch14/yin_vu_persico_kappa.py: that script measures agreement
between *two models*; this one measures agreement between *a model and a human
gold standard*, which is the quantity a referee asks for.  The two answer
different questions and neither substitutes for the other.

What this script does
---------------------
Simulates the open-ended survey-coding task of §15.2: n respondents explain
why they expect prices to rise, and each answer is coded into one of four
categories --

    0  supply-chain narrative      (high frequency)
    1  monetary narrative          (high frequency)
    2  opportunistic-firm narrative(low frequency)
    3  uncodable

An LLM codes the full sample; a random audit subset is coded by hand and
treated as the gold standard.  The script then computes:

    1) Overall accuracy and overall Cohen's kappa.
    2) The Fleiss (1981) and Landis-Koch (1977) interpretation bands side by
       side.  They disagree, and the disagreement matters: the kappa >= 0.75
       "excellent" cutpoint used in §15.2 is Fleiss's, NOT Landis-Koch's,
       whose bands are 0.61-0.80 substantial and 0.81-1.00 almost perfect.
    3) The confusion matrix.
    4) Per-category kappa (one-vs-rest).  This is the payload: overall kappa
       can clear 0.75 while a low-frequency category fails badly, because the
       high-frequency categories dominate the overall statistic.  Reporting
       only overall accuracy hides exactly the error that damages inference
       about minority narratives.
    5) The remedy of last resort -- merging the failing category into a
       coarser parent -- and what it costs.

The script is intentionally LLM-API-free: it does NOT call any external model.

Dependencies: numpy  (no GPU, no internet)
Tested with: Python 3.9+, numpy 1.26
"""

from __future__ import annotations

import numpy as np

SEED = 42
N = 3000            # full sample of open-ended answers
AUDIT_FRAC = 0.10   # share hand-coded as the gold standard

LABELS = ["supply-chain", "monetary", "opportunistic", "uncodable"]

# Population shares of the four narratives
CATEGORY_P = [0.42, 0.36, 0.12, 0.10]

# Probability the LLM reproduces the human label, by category.  The
# opportunistic narrative is expressed in colloquial, highly variable
# language and is frequently absorbed into its nearest high-frequency
# neighbour -- the "minority expression misclassification" of §15.2.
FIDELITY = [0.96, 0.94, 0.55, 0.72]


# ----------------------------------------------------------------------
# 1. Agreement statistics
# ----------------------------------------------------------------------
def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Two-rater Cohen's kappa for categorical labels."""
    assert a.shape == b.shape
    categories = np.unique(np.concatenate([a, b]))
    po = float(np.mean(a == b))
    pe = float(sum(np.mean(a == c) * np.mean(b == c) for c in categories))
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def fleiss_band(kappa: float) -> str:
    """Fleiss (1981), Statistical Methods for Rates and Proportions, 2nd ed."""
    if kappa < 0.40:
        return "poor (< 0.40)"
    if kappa <= 0.75:
        return "fair to good (0.40-0.75)"
    return "excellent (> 0.75)"


def landis_koch_band(kappa: float) -> str:
    """Landis & Koch (1977), Biometrics 33(1), 159-174.  No 0.75 cutpoint."""
    if kappa < 0.00:
        return "poor (< 0)"
    if kappa < 0.20:
        return "slight (0.00-0.20)"
    if kappa < 0.40:
        return "fair (0.21-0.40)"
    if kappa < 0.60:
        return "moderate (0.41-0.60)"
    if kappa < 0.80:
        return "substantial (0.61-0.80)"
    return "almost perfect (0.81-1.00)"


def confusion_matrix(human: np.ndarray, llm: np.ndarray, k: int) -> np.ndarray:
    m = np.zeros((k, k), dtype=int)
    for h, l in zip(human, llm):
        m[h, l] += 1
    return m


def per_category_kappa(
    human: np.ndarray, llm: np.ndarray, k: int
) -> list[float]:
    """One-vs-rest Cohen's kappa for each category."""
    out = []
    for c in range(k):
        out.append(cohen_kappa((human == c).astype(int),
                               (llm == c).astype(int)))
    return out


# ----------------------------------------------------------------------
# 2. Synthetic coding task
# ----------------------------------------------------------------------
def simulate_coding(
    rng: np.random.Generator, n: int = N
) -> tuple[np.ndarray, np.ndarray]:
    """Return (human_label, llm_label) for n answers."""
    human = rng.choice(len(LABELS), size=n, p=CATEGORY_P)
    llm = human.copy()
    for c, keep in enumerate(FIDELITY):
        in_c = np.where(human == c)[0]
        slips = in_c[rng.uniform(size=len(in_c)) > keep]
        # A slipped answer lands in a neighbouring category, weighted toward
        # the two high-frequency narratives.
        for i in slips:
            alt = [j for j in range(len(LABELS)) if j != c]
            w = np.array([CATEGORY_P[j] for j in alt], dtype=float)
            llm[i] = rng.choice(alt, p=w / w.sum())
    return human, llm


# ----------------------------------------------------------------------
# 3. Main
# ----------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(SEED)
    human_full, llm_full = simulate_coding(rng)

    n_audit = int(round(AUDIT_FRAC * len(human_full)))
    audit = rng.choice(len(human_full), size=n_audit, replace=False)
    human, llm = human_full[audit], llm_full[audit]
    k = len(LABELS)

    print("=" * 70)
    print(f"Human-machine agreement on survey coding (synthetic, seed={SEED})")
    print("=" * 70)
    print(f"Full sample n = {len(human_full)},  hand-coded audit "
          f"n = {n_audit}  ({AUDIT_FRAC:.0%})")
    print()

    # --- [1] Overall ---------------------------------------------------
    acc = float(np.mean(human == llm))
    kappa = cohen_kappa(human, llm)
    print("[1] Overall statistics on the audit subset")
    print(f"    accuracy       = {acc:.3f}")
    print(f"    Cohen's kappa  = {kappa:.3f}")
    print(f"      Fleiss (1981)      band: {fleiss_band(kappa)}")
    print(f"      Landis-Koch (1977) band: {landis_koch_band(kappa)}")
    print("    -> The 0.75 threshold used in this chapter is Fleiss's.")
    print("       Landis-Koch has no 0.75 cutpoint; citing it for 0.75 is a")
    print("       misattribution common in the applied literature.")
    print()

    # --- [2] Confusion matrix -----------------------------------------
    cm = confusion_matrix(human, llm, k)
    print("[2] Confusion matrix (rows = human gold standard, cols = LLM)")
    header = "".join(f"{s[:9]:>12}" for s in LABELS)
    corner = "human vs llm"
    print(f"    {corner:<16}{header}{'recall':>10}")
    for c in range(k):
        row = "".join(f"{cm[c, j]:>12}" for j in range(k))
        recall = cm[c, c] / max(cm[c].sum(), 1)
        print(f"    {LABELS[c]:<16}{row}{recall:>10.3f}")
    print()

    # --- [3] Per-category kappa ---------------------------------------
    pck = per_category_kappa(human, llm, k)
    print("[3] Per-category kappa (one-vs-rest) -- report this, not just "
          "accuracy")
    print(f"    {'category':<16}{'share':>8}{'kappa':>9}   verdict at 0.75")
    for c in range(k):
        share = float(np.mean(human == c))
        flag = "PASS" if pck[c] >= 0.75 else "FAIL -> action required"
        print(f"    {LABELS[c]:<16}{share:>8.3f}{pck[c]:>9.3f}   {flag}")
    print()
    failing = [c for c in range(k) if pck[c] < 0.75]
    if failing:
        names = ", ".join(LABELS[c] for c in failing)
        print(f"    Overall kappa is {kappa:.3f} "
              f"({fleiss_band(kappa)}), yet the categories")
        print(f"    [{names}] fall below 0.75.  The overall statistic is")
        print("    carried by the two high-frequency narratives.  An overall-")
        print("    accuracy-only report would have concealed this entirely.")
    print()

    # --- [4] The merge remedy, and its cost ---------------------------
    print("[4] Remedy: merge the failing low-frequency category upward")
    merge_from, merge_into = 2, 0   # opportunistic -> supply-chain parent
    hm = np.where(human == merge_from, merge_into, human)
    lm = np.where(llm == merge_from, merge_into, llm)
    kappa_merged = cohen_kappa(hm, lm)
    print(f"    merging '{LABELS[merge_from]}' into "
          f"'{LABELS[merge_into]}':")
    print(f"    kappa {kappa:.3f} -> {kappa_merged:.3f}"
          f"  ({fleiss_band(kappa_merged)})")
    print("    The statistic improves because the hard distinction was")
    print("    removed, not because the coding got better.  The cost is the")
    print("    research question: with the category gone, no hypothesis about")
    print("    opportunistic-pricing narratives can be tested.  Merge only")
    print("    when that narrative is not the object of study; otherwise send")
    print("    the category back to hand coding.")
    print()
    print("Note: illustrative calculation on simulated data with a fixed "
          "seed; not an empirical result.")


if __name__ == "__main__":
    main()
