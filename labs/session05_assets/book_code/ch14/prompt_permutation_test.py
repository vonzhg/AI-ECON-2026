"""
prompt_permutation_test.py — Design Calculations for Option-Order Effects
=========================================================================

Companion to Chapter 15, §15.4.2 (Prompt Choice as a Confounder) and
Exercise "编程实践" 3.

This script deliberately does NOT inject a primacy effect and then
"discover" it -- that would be circular, and it would teach nothing.  It
answers the two design questions a researcher actually faces:

    (Q1) REMOVAL.  Which presentation design removes a primacy effect of
         size delta, and what does each design cost?
    (Q2) DETECTION.  How many items are needed before a primacy effect of
         size delta can be distinguished from zero at conventional levels?

Both are answered analytically and checked by simulation.  Every sentence of
interpretation printed below interpolates a computed value, so the narrative
cannot drift away from the numbers.

Calibration
-----------
The delta grid is taken from Brucks & Toubia (2025), PLOS ONE 20(4),
e0319159, who report the probability that GPT-4 selects the first-listed
option in a task where the truth is 0.50:

    overall          0.6321
    letter labels    0.5355
    symbol labels    0.7335
    aggregated over all 32 prompt architectures: 0.5001

Note the direction: SYMBOL labels induce the stronger bias, letters the
weaker.  (A widely repeated secondary summary has this backwards.)

The main result of Q1 is easy to state and easy to get wrong: what buys
unbiasedness is *randomising the option order across items*, not exhausting
the orderings within each item.  Once the order is randomised, the residual
error is pure sampling noise, and extra orderings buy variance reduction
only.  A design that holds the order fixed, by contrast, stays biased by
delta no matter how many items are added.

The script is intentionally LLM-API-free: it does NOT call any external model.

Dependencies: numpy  (matplotlib optional, for the figure; no GPU, no internet)
Tested with: Python 3.9+, numpy 1.26, matplotlib 3.8
"""

from __future__ import annotations

import math

import numpy as np

SEED = 42

# Brucks & Toubia (2025) calibration points: (label, first-option rate)
BT_POINTS = [
    ("overall", 0.6321),
    ("letter labels", 0.5355),
    ("symbol labels", 0.7335),
]
BT_AGGREGATED = 0.5001

Z_975 = 1.959964
Z_80 = 0.841621

TOTAL_CALLS = 1600      # budget held fixed across designs in Q1
M_OPTIONS = 4


# ----------------------------------------------------------------------
# 1. Q1 -- which design removes the effect
# ----------------------------------------------------------------------
def simulate_design(
    rng: np.random.Generator,
    delta: float,
    design: str,
    m_options: int = M_OPTIONS,
    total_calls: int = TOTAL_CALLS,
    n_orders: int = 1,
    n_rep: int = 300,
) -> tuple[float, float]:
    """Return (mean signed bias, RMSE) of the estimated share of option 0.

    The position effect raises the first-listed option's choice probability
    to 1/m + delta, spreading the remainder equally over the others.

    design = "fixed"       option 0 is always listed first
             "randomised"  the first-listed option is drawn at random per item
             "exhaustive"  each item is shown in n_orders balanced orderings,
                           cycling option 0 through the leading positions

    The total number of model calls is held at total_calls for every design,
    so the comparison isolates the presentation choice from sample size.
    """
    base = 1.0 / m_options
    p_first = base + delta
    p_other = (1.0 - p_first) / (m_options - 1)
    errs = np.empty(n_rep)

    for r in range(n_rep):
        if design == "fixed":
            first = np.zeros(total_calls, dtype=int)
        elif design == "randomised":
            first = rng.integers(0, m_options, total_calls)
        elif design == "exhaustive":
            # Balanced cycling: equal numbers of calls with each option first
            reps = int(math.ceil(total_calls / n_orders))
            cycle = np.tile(np.arange(n_orders) % m_options, reps)
            first = cycle[:total_calls]
        else:
            raise ValueError(f"unknown design: {design}")

        probs = np.full((total_calls, m_options), p_other)
        probs[np.arange(total_calls), first] = p_first
        u = rng.uniform(size=(total_calls, 1))
        picks = (probs.cumsum(axis=1) < u).sum(axis=1)
        errs[r] = np.mean(picks == 0) - base

    return float(np.mean(errs)), float(np.sqrt(np.mean(errs ** 2)))


# ----------------------------------------------------------------------
# 2. Q2 -- detection: required number of items
# ----------------------------------------------------------------------
def n_required(delta: float, power: float = 0.80) -> int:
    """Items needed to reject p = 0.5 against p = 0.5 + delta.

    Two-sided 5% test, normal approximation:
        n >= [ z_{0.975} * sqrt(0.25) + z_power * sqrt(p(1-p)) ]^2 / delta^2
    """
    if delta <= 0:
        return -1
    z_pow = Z_80 if abs(power - 0.80) < 1e-9 else _z_from_power(power)
    p1 = 0.5 + delta
    num = Z_975 * math.sqrt(0.25) + z_pow * math.sqrt(p1 * (1.0 - p1))
    return int(math.ceil(num ** 2 / delta ** 2))


def _z_from_power(power: float) -> float:
    """Inverse normal CDF via bisection (keeps scipy out of the deps)."""
    lo, hi = -6.0, 6.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < power:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def empirical_power(
    rng: np.random.Generator, delta: float, n_items: int, n_rep: int = 4000
) -> float:
    """Simulated power of the two-sided binomial test at the 5% level."""
    p1 = 0.5 + delta
    draws = rng.binomial(n_items, p1, n_rep)
    se = math.sqrt(0.25 / n_items)
    z = (draws / n_items - 0.5) / se
    return float(np.mean(np.abs(z) > Z_975))


# ----------------------------------------------------------------------
# 3. Main
# ----------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(SEED)

    print("=" * 70)
    print(f"Option-order effects: removal and detection design (seed={SEED})")
    print("=" * 70)
    print("Calibration from Brucks & Toubia (2025), PLOS ONE 20(4) e0319159")
    print()

    d_overall = round(BT_POINTS[0][1] - 0.5, 4)
    d_letter = round(BT_POINTS[1][1] - 0.5, 4)
    d_symbol = round(BT_POINTS[2][1] - 0.5, 4)

    print("[0] The delta implied by each reported rate")
    print(f"    {'setting':<18}{'first-option rate':>19}{'delta':>9}")
    for name, rate in BT_POINTS:
        print(f"    {name:<18}{rate:>19.4f}{rate - 0.5:>9.4f}")
    print(f"    Direction check: symbol labels ({BT_POINTS[2][1]:.4f}) induce "
          f"a STRONGER")
    print(f"    effect than letter labels ({BT_POINTS[1][1]:.4f}) -- a ratio "
          f"of {d_symbol / d_letter:.1f} to 1.")
    print(f"    Aggregating all 32 architectures returns {BT_AGGREGATED:.4f}.")
    print()

    # --- [1] Removal --------------------------------------------------
    print("[1] Q1 -- which presentation design removes the effect")
    print(f"    m = {M_OPTIONS} options, delta = {d_overall:.4f}, "
          f"total model calls held at {TOTAL_CALLS} for every design")
    print(f"    (the truth is a share of {1 / M_OPTIONS:.4f})")
    print()
    print(f"    {'design':<26}{'mean bias':>12}{'RMSE':>10}")
    results = {}
    for design, n_orders in (("fixed", 1), ("randomised", 1),
                             ("exhaustive", M_OPTIONS)):
        bias, rmse = simulate_design(rng, d_overall, design,
                                     n_orders=n_orders)
        results[design] = (bias, rmse)
        print(f"    {design:<26}{bias:>+12.4f}{rmse:>10.4f}")
    print()
    b_fixed = results["fixed"][0]
    b_rand = results["randomised"][0]
    r_rand = results["randomised"][1]
    r_exh = results["exhaustive"][1]
    print(f"    Holding the order fixed leaves a bias of {b_fixed:+.4f}, i.e. "
          f"essentially the whole")
    print(f"    delta of {d_overall:.4f}; it does not shrink with more items, "
          f"because it is not noise.")
    print(f"    Randomising the order across items cuts the bias to "
          f"{b_rand:+.4f} -- that step alone,")
    print(f"    not exhaustive permutation, is what buys unbiasedness.")
    print(f"    Exhausting the orderings changes RMSE only from {r_rand:.4f} "
          f"to {r_exh:.4f}: at a fixed")
    print(f"    budget it buys variance, not bias.  Its real advantage is "
          f"that the balance is")
    print(f"    guaranteed by construction rather than holding only in "
          f"expectation.")
    print()

    # --- [2] Detection ------------------------------------------------
    print("[2] Q2 -- detection: items needed to call delta non-zero")
    print("    (two-sided 5% test, 80% power)")
    print(f"    {'delta':>8}{'n required':>13}{'simulated power':>18}")
    n_by_delta: dict[float, int] = {}
    grid = sorted({d_letter, 0.05, 0.10, d_overall, d_symbol})
    for d in grid:
        n_req = n_required(d)
        n_by_delta[d] = n_req
        pw = empirical_power(rng, d, n_req)
        print(f"    {d:>8.4f}{n_req:>13}{pw:>18.3f}")
    print()
    pw50_letter = empirical_power(rng, d_letter, 50)
    print(f"    The letter-label effect (delta = {d_letter:.4f}) needs about "
          f"{n_by_delta[d_letter]} items;")
    print(f"    the symbol-label effect (delta = {d_symbol:.4f}) needs only "
          f"{n_by_delta[d_symbol]} -- a "
          f"{n_by_delta[d_letter] / n_by_delta[d_symbol]:.0f}-fold")
    print(f"    difference driven entirely by a formatting choice.")
    print(f"    A robustness check run on 50 items has power of only "
          f"{pw50_letter:.2f} against the")
    print(f"    smaller effect, so reporting 'no order effect' from such a "
          f"check is absence")
    print(f"    of evidence, not evidence of absence.")
    print()

    # --- [3] Feasibility of exhaustive permutation --------------------
    print("[3] When exhaustive permutation stops being feasible")
    print(f"    {'m options':>11}{'m! orderings':>15}")
    for m in (2, 3, 4, 5, 6, 8):
        print(f"    {m:>11}{math.factorial(m):>15}")
    print()
    print("    Exhausting m! orderings per item is affordable only for very")
    print("    small m.  Since randomisation across items already delivers")
    print("    unbiasedness (block [1]), the practical recipe for larger m is")
    print("    to randomise the order per item, or to use a Latin square that")
    print("    balances each option across each position -- not to enumerate")
    print("    all m! permutations.")
    print()
    print("Note: illustrative design calculation on simulated data with a")
    print("fixed seed; the delta values are taken from the cited study, but")
    print("no LLM was called here.")

    # --- Optional figure ---------------------------------------------
    try:
        import matplotlib.pyplot as plt

        deltas = np.linspace(0.02, 0.25, 60)
        ns = [n_required(float(d)) for d in deltas]
        fig, ax = plt.subplots(figsize=(6.4, 4))
        ax.plot(deltas, ns, color="tab:blue")
        ax.set_yscale("log")
        for name, rate in BT_POINTS:
            ax.axvline(rate - 0.5, linestyle="--", alpha=0.6, color="tab:red")
            ax.text(rate - 0.5, max(ns) * 0.5, name, rotation=90,
                    fontsize=8, ha="right", va="top")
        ax.set_xlabel(r"primacy effect $\delta$")
        ax.set_ylabel("items required (80% power, log scale)")
        ax.set_title("Detecting an option-order effect")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = "prompt_permutation_power.png"
        fig.savefig(out_path, dpi=150)
        print(f"\nFigure saved to {out_path}")
    except Exception as exc:  # pragma: no cover
        print(f"\n(matplotlib unavailable, skipping plot: {exc})")


if __name__ == "__main__":
    main()
