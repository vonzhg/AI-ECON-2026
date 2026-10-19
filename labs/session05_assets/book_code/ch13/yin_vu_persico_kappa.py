"""
yin_vu_persico_kappa.py — Two-LLM Inter-rater Reliability Demonstration
======================================================================

Companion to Chapter 14, §14.6.1 (Structural Limits of LLMs as Measurement
Instruments) and Exercise 6.

What this script does
---------------------
Reproduces the *qualitative* finding of Yin, Vu & Persico (NBER 35110, 2026)
on a simplified, fully synthetic dataset:

    * Generates 40 "task" items, each labelled by two synthetic "LLMs" (A and B)
      on a 3-level "AI exposure" scale (0, 1, 2).
    * The disagreement between A and B is controlled by a drift parameter rho:
        - rho = 0.0  -> A and B agree perfectly
        - rho = 1.0  -> A and B are statistically independent (with same marginals)
    * Computes:
        1) Mean and standard deviation of each rater's scores.
        2) Cohen's kappa between A and B.
        3) Landis-Koch interpretation band ("slight" / "fair" / "moderate" / ...).
        4) DiD-style coefficient beta_hat on post * exposure, estimated once with
           exposure_A and once with exposure_B (in a stylized panel).
        5) A plot of kappa(rho) for rho in [0, 1].

The script is intentionally LLM-API-free: it does NOT call any external
model.  The point is to make the *measurement-noise mechanics* tangible
without requiring credentials.

Dependencies: numpy, pandas, matplotlib  (no GPU, no internet)
Tested with: Python 3.9+, numpy 1.26, pandas 2.1, matplotlib 3.8
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1. Cohen's kappa
# ----------------------------------------------------------------------
def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Two-rater Cohen's kappa for categorical labels (any int range)."""
    assert a.shape == b.shape
    n = len(a)
    categories = np.unique(np.concatenate([a, b]))
    # Observed agreement
    po = np.mean(a == b)
    # Expected agreement by chance
    pe = 0.0
    for c in categories:
        pe += (np.mean(a == c) * np.mean(b == c))
    if pe >= 1.0:  # avoid divide-by-zero when ratings are constant
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def landis_koch_band(kappa: float) -> str:
    if kappa < 0.0:
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


# ----------------------------------------------------------------------
# 2. Synthetic two-LLM dataset
# ----------------------------------------------------------------------
def simulate_two_raters(
    n_tasks: int = 40,
    rho: float = 0.65,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two correlated rater outputs on a 0/1/2 scale.

    rho controls the disagreement rate.  Higher rho => more independent
    raters => lower observed kappa.  The marginals of A and B match.
    """
    rng = np.random.default_rng(seed)
    # The "true" latent exposure level for each task
    latent = rng.choice([0, 1, 2], size=n_tasks, p=[0.35, 0.45, 0.20])
    # Rater A flips its label with small probability proportional to rho
    flip_a = rng.uniform(size=n_tasks) < 0.10 * rho
    rater_a = np.where(flip_a, rng.choice([0, 1, 2], size=n_tasks), latent)
    # Rater B flips more often, mimicking a different vendor / version
    flip_b = rng.uniform(size=n_tasks) < 0.45 * rho
    rater_b = np.where(flip_b, rng.choice([0, 1, 2], size=n_tasks), latent)
    return rater_a.astype(int), rater_b.astype(int)


# ----------------------------------------------------------------------
# 3. Stylized DiD with two exposure measures
# ----------------------------------------------------------------------
def did_with_exposure(exposure: np.ndarray, seed: int = 7) -> float:
    """Run a stylized DiD: y = alpha + beta * post * exposure + u.

    Returns the OLS estimate of beta.  For a sanity-check rather than a
    real empirical setup.
    """
    rng = np.random.default_rng(seed)
    n_units = len(exposure)
    n_periods = 4
    post = np.concatenate([np.zeros(n_units), np.ones(n_units * (n_periods - 1))])
    expo = np.tile(exposure, n_periods).astype(float)
    # True coefficient = 0.7 in this synthetic world
    y = 1.0 + 0.7 * post * expo + 0.15 * rng.standard_normal(len(post))
    # OLS by hand
    x = post * expo
    x_demean = x - x.mean()
    y_demean = y - y.mean()
    if (x_demean ** 2).sum() < 1e-9:
        return float("nan")
    return float((x_demean * y_demean).sum() / (x_demean ** 2).sum())


# ----------------------------------------------------------------------
# 4. Main: report + (optional) plot
# ----------------------------------------------------------------------
def main(rho: float = 0.65) -> None:
    a, b = simulate_two_raters(n_tasks=40, rho=rho, seed=42)
    df = pd.DataFrame({"task_id": np.arange(len(a)),
                       "rater_A": a,
                       "rater_B": b})
    print("Sample of 40 synthetic (task, rater_A, rater_B) triplets:")
    print(df.head(10).to_string(index=False))
    print()
    print(f"Mean(A) = {a.mean():.3f}   SD(A) = {a.std(ddof=1):.3f}")
    print(f"Mean(B) = {b.mean():.3f}   SD(B) = {b.std(ddof=1):.3f}")
    print(f"Mean ratio  Mean(B) / Mean(A) = {b.mean() / max(a.mean(), 1e-9):.3f}")
    kappa = cohen_kappa(a, b)
    print(f"Cohen's kappa = {kappa:.3f}  ({landis_koch_band(kappa)})")
    print()
    # DiD comparison
    beta_a = did_with_exposure(a)
    beta_b = did_with_exposure(b)
    print(f"DiD beta_hat using rater A's exposure  = {beta_a:+.3f}")
    print(f"DiD beta_hat using rater B's exposure  = {beta_b:+.3f}")
    if beta_a * beta_b < 0:
        print("*** Estimated coefficients have OPPOSITE SIGNS across raters! ***")
    print()
    # kappa-vs-rho curve
    rho_grid = np.linspace(0.0, 1.0, 21)
    kappas = []
    for r in rho_grid:
        aa, bb = simulate_two_raters(n_tasks=2000, rho=r, seed=42)
        kappas.append(cohen_kappa(aa, bb))
    print("kappa(rho) on a denser grid (n_tasks=2000 to stabilize):")
    for r, k in zip(rho_grid, kappas):
        print(f"  rho={r:.2f}   kappa={k:.3f}   {landis_koch_band(k)}")

    # Optional plot — only if matplotlib is available
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433  (intentional late import)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rho_grid, kappas, marker="o")
        ax.axhline(0.36, color="red", linestyle="--",
                   label="YVP (2026) lowest pairwise kappa ≈ 0.36")
        ax.set_xlabel(r"Drift parameter $\rho$")
        ax.set_ylabel(r"Cohen's $\kappa$")
        ax.set_title(r"$\kappa$ as a function of inter-LLM drift")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = "kappa_vs_rho.png"
        fig.savefig(out_path, dpi=150)
        print(f"\nFigure saved to {out_path}")
    except Exception as exc:  # pragma: no cover
        print(f"\n(matplotlib unavailable, skipping plot: {exc})")


if __name__ == "__main__":
    main(rho=0.65)
