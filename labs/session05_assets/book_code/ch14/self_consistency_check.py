"""
self_consistency_check.py — Self-Consistency Screening for Hallucinations
=========================================================================

Companion to Chapter 15, §15.5.1 (Identifying Hallucination Risk) and the
self-consistency equation of that section.

What this script does
---------------------
Implements the SelfCheckGPT-style screen: sample the same prompt R times and
score a claim by how much the samples disagree,

    h_hat(s) = 1 - (1/(R-1)) * sum_{j != 1} sim(s_1, s_j)

where sim is a similarity between two sampled answers (here a Jaccard
overlap on the set of asserted facts, standing in for an n-gram or embedding
similarity).  A claim whose resamples disagree is flagged for verification
against the primary source.

The simulation contains THREE kinds of claim, and the third is the point:

    "retrieved"          the model holds the fact; resamples mostly agree
    "fabricated"         the model invents; resamples mostly diverge
    "confidently wrong"  the model holds a WRONG fact firmly; resamples agree

Per-claim volatility is drawn from a distribution rather than fixed by type,
so the retrieved and fabricated score distributions genuinely OVERLAP.  The
screen therefore trades precision against recall like any real classifier --
a screen that separated the types perfectly would be an artefact of the
simulation, not a lesson about screening.

Types 1 and 2 are what the screen separates.  Type 3 is what it CANNOT see:
a systematic falsehood reproduced consistently scores exactly like a true
fact.  Self-consistency measures the variance of fabrication, not truth.
The script reports performance on types 1-2 and then, separately, how many
type-3 errors pass through -- which is why the chapter pairs this screen with
retrieval against an authoritative source rather than treating it as
sufficient.

A final block is the design calculation: what actually improves as R grows.
The separation between the type means barely moves; what tightens is the
sampling error of h_hat, and hence the screen's F1.

Every sentence of interpretation printed below interpolates a computed
value, so the narrative cannot drift away from the numbers.

The script is intentionally LLM-API-free: it does NOT call any external model.

Dependencies: numpy  (no GPU, no internet)
Tested with: Python 3.9+, numpy 1.26
"""

from __future__ import annotations

import numpy as np

SEED = 42
N_CLAIMS = 600
R_DEFAULT = 8

# Block [4] sweep: averaging over replications, because a single draw of
# n=400 leaves enough Monte Carlo noise to make the F1 trend unreadable.
N_SWEEP = 400
N_SWEEP_REPS = 8
R_GRID = (2, 3, 5, 8, 12, 20)

KINDS = ["retrieved", "fabricated", "confidently wrong"]
SHARES = [0.60, 0.25, 0.15]

# Per-claim "fact drift" is Beta-distributed: (mean, concentration).
# Lower concentration => wider spread => more overlap between types.
DRIFT_PRIOR = {
    "retrieved": (0.12, 9.0),
    "fabricated": (0.42, 7.0),
    "confidently wrong": (0.14, 9.0),
}

N_FACTS = 6          # facts asserted per answer
FACT_POOL = 40       # vocabulary of possible facts


# ----------------------------------------------------------------------
# 1. Similarity and the consistency score
# ----------------------------------------------------------------------
def jaccard(a: set[int], b: set[int]) -> float:
    """Jaccard overlap, standing in for a sentence-level similarity."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def h_hat(samples: list[set[int]]) -> float:
    """Inconsistency score: 1 minus mean similarity of sample 1 to the rest."""
    if len(samples) < 2:
        return 0.0
    sims = [jaccard(samples[0], s) for s in samples[1:]]
    return 1.0 - float(np.mean(sims))


# ----------------------------------------------------------------------
# 2. Synthetic resampling process
# ----------------------------------------------------------------------
def draw_samples(
    rng: np.random.Generator, drift: float, r: int
) -> list[set[int]]:
    """Draw r resampled answers, each a set of asserted facts."""
    anchor = set(rng.choice(FACT_POOL, size=N_FACTS, replace=False).tolist())
    out = [set(anchor)]
    for _ in range(r - 1):
        s = set()
        for f in anchor:
            if rng.uniform() < drift:
                s.add(int(rng.integers(0, FACT_POOL)))   # drifted fact
            else:
                s.add(int(f))
        out.append(s)
    return out


def simulate(
    rng: np.random.Generator, n: int = N_CLAIMS, r: int = R_DEFAULT
) -> tuple[np.ndarray, np.ndarray]:
    """Return (kind_index, h_score) for n claims."""
    kinds = rng.choice(len(KINDS), size=n, p=SHARES)
    scores = np.empty(n)
    for i, k in enumerate(kinds):
        mean, conc = DRIFT_PRIOR[KINDS[k]]
        drift = float(rng.beta(mean * conc, (1.0 - mean) * conc))
        scores[i] = h_hat(draw_samples(rng, drift, r))
    return kinds, scores


# ----------------------------------------------------------------------
# 3. Screen performance
# ----------------------------------------------------------------------
def screen_metrics(
    kinds: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    """Precision/recall for flagging fabricated claims (kind 1)."""
    flagged = scores >= threshold
    is_fab = kinds == 1
    tp = int(np.sum(flagged & is_fab))
    fp = int(np.sum(flagged & ~is_fab))
    fn = int(np.sum(~flagged & is_fab))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else (
        2 * precision * recall / (precision + recall)
    )
    return {"precision": precision, "recall": recall, "f1": f1,
            "flagged": float(np.mean(flagged))}


def best_threshold(
    kinds: np.ndarray, scores: np.ndarray
) -> tuple[float, dict[str, float]]:
    grid = np.linspace(0.05, 0.95, 91)
    best_t, best_m = float(grid[0]), screen_metrics(kinds, scores, grid[0])
    for t in grid[1:]:
        m = screen_metrics(kinds, scores, t)
        if m["f1"] > best_m["f1"]:
            best_t, best_m = float(t), m
    return best_t, best_m


# ----------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(SEED)
    kinds, scores = simulate(rng)

    print("=" * 70)
    print(f"Self-consistency screening for hallucinations (seed={SEED})")
    print("=" * 70)
    print(f"claims n = {len(kinds)},  resamples per claim R = {R_DEFAULT}")
    print()

    print("[1] Inconsistency score by claim type")
    print(f"    {'type':<20}{'n':>6}{'mean h':>10}{'sd':>8}"
          f"{'p10':>8}{'p90':>8}")
    stats = {}
    for k, nm in enumerate(KINDS):
        sel = kinds == k
        m, sd = scores[sel].mean(), scores[sel].std(ddof=1)
        p10, p90 = np.percentile(scores[sel], [10, 90])
        stats[nm] = (m, sd, p10, p90)
        print(f"    {nm:<20}{int(sel.sum()):>6}{m:>10.3f}{sd:>8.3f}"
              f"{p10:>8.3f}{p90:>8.3f}")
    overlap_lo = stats["fabricated"][2]
    overlap_hi = stats["retrieved"][3]
    print()
    print(f"    Fabricated claims score high (mean "
          f"{stats['fabricated'][0]:.3f}) and retrieved claims low "
          f"(mean {stats['retrieved'][0]:.3f}),")
    if overlap_lo < overlap_hi:
        print(f"    but the distributions overlap: the fabricated p10 "
              f"({overlap_lo:.3f}) sits below the")
        print(f"    retrieved p90 ({overlap_hi:.3f}), so no threshold "
              f"separates them cleanly.")
    else:
        print(f"    and in this draw they barely overlap (fabricated p10 "
              f"{overlap_lo:.3f} vs retrieved p90 {overlap_hi:.3f}).")
    print(f"    'Confidently wrong' claims score {stats[KINDS[2]][0]:.3f}, "
          f"essentially the same as")
    print(f"    retrieved ones -- the screen is blind to them by "
          f"construction.")
    print()

    thr, met = best_threshold(kinds, scores)
    print("[2] Screen performance for FABRICATED claims")
    print(f"    best threshold h* = {thr:.2f}")
    print(f"    precision = {met['precision']:.3f}   "
          f"recall = {met['recall']:.3f}   F1 = {met['f1']:.3f}")
    print(f"    share of all claims flagged for review = "
          f"{met['flagged']:.3f}")
    n_fab = int(np.sum(kinds == 1))
    missed = int(np.sum((kinds == 1) & (scores < thr)))
    print(f"    Even at the best threshold, {missed} of {n_fab} fabricated "
          f"claims go unflagged")
    print(f"    ({missed / max(n_fab, 1):.1%}).  The screen is a triage "
          f"device, not a proof.")
    print()

    print("[3] What the screen lets through")
    cw = kinds == 2
    passed = cw & (scores < thr)
    print(f"    'confidently wrong' claims        = {int(cw.sum())}")
    print(f"    of which the screen does NOT flag = {int(passed.sum())}"
          f"  ({passed.sum() / max(cw.sum(), 1):.1%})")
    print("    These are false statements the model repeats consistently.")
    print("    No number of resamples will surface them: consistency is not")
    print("    truth.  They are reachable only by checking the claim against")
    print("    an authoritative source -- which is why retrieval and")
    print("    source-level verification sit alongside this screen, not")
    print("    downstream of it.")
    print()

    print("[4] Design calculation: what actually improves as R grows")
    print(f"    (each row averages {N_SWEEP_REPS} independent replications "
          f"of n={N_SWEEP} claims)")
    print(f"    {'R':>4}{'mean h (retr.)':>17}{'mean h (fab.)':>16}"
          f"{'gap':>8}{'sd(h) retr.':>13}{'F1':>8}")
    rows = []
    for r in R_GRID:
        acc = []
        for rep in range(N_SWEEP_REPS):
            rng_r = np.random.default_rng(SEED + 1000 * rep + r)
            k_r, s_r = simulate(rng_r, n=N_SWEEP, r=r)
            _t, m = best_threshold(k_r, s_r)
            acc.append((s_r[k_r == 0].mean(), s_r[k_r == 1].mean(),
                        s_r[k_r == 0].std(ddof=1), m["f1"]))
        m_retr, m_fab, sd_retr, f1 = np.array(acc).mean(axis=0)
        rows.append((r, m_retr, m_fab, m_fab - m_retr, sd_retr, f1))
        print(f"    {r:>4}{m_retr:>17.3f}{m_fab:>16.3f}"
              f"{m_fab - m_retr:>8.3f}{sd_retr:>13.3f}{f1:>8.3f}")
    print()

    gaps = [row[3] for row in rows]
    sds = [row[4] for row in rows]
    f1s = [row[5] for row in rows]
    f1_max = max(f1s)
    # Smallest R reaching 95% of the best F1 on this grid
    r_star = next(R for (R, *_rest), f in zip(rows, f1s)
                  if f >= 0.95 * f1_max)

    print(f"    The gap between the type means is nearly flat in R "
          f"({gaps[0]:.3f} at R={rows[0][0]} vs {gaps[-1]:.3f} "
          f"at R={rows[-1][0]}):")
    print("    R does not make fabrication more detectable in expectation.")
    print("    What R buys is precision of the score itself -- sd(h) among "
          "retrieved claims")
    print(f"    falls from {sds[0]:.3f} to {sds[-1]:.3f}, and that is what "
          f"lifts F1 from {f1s[0]:.3f} to {f1s[-1]:.3f}.")
    print(f"    F1 is still improving at R={rows[-1][0]}, so there is no "
          f"sharp saturation point;")
    print(f"    the cheapest R reaching 95% of the best F1 on this grid is "
          f"R={r_star}.")
    print("    Since cost is linear in R while the gain is concave, that is "
          "the sensible")
    print("    operating point for a first-pass screen -- raise it only for "
          "claims whose")
    print("    verification cost is high enough to justify the extra calls.")
    print()
    print("Note: illustrative calculation on simulated data with a fixed "
          "seed; not an empirical result.")


if __name__ == "__main__":
    main()
