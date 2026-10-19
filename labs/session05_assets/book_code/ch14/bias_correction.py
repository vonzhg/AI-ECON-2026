"""
bias_correction.py — Correcting Bias from LLM-Generated Regressors
=================================================================

Companion to Chapter 15, §15.1.3 (Structured Extraction and Bias Correction
for Generated Regressors) and Exercise "编程实践" 2.

This script produces every number quoted in the worked example box
《A股年报 MD&A 的"融资压力"标注》.  The data are fully synthetic: the box is
an illustrative calculation, not an empirical result.

What this script does
---------------------
Units i = 1..n are Chinese A-share firm-years.  The latent construct m*_i is
the financing pressure expressed in the firm's MD&A section, in [0, 1].  The
regression of interest is

    y_i = alpha + beta * m*_i + gamma * c_i + u_i          (ideal)

where y_i is next year's investment rate and c_i is firm size.  m*_i is not
observed.  The researcher scores all n MD&A passages with an LLM to obtain
m_hat_i, and obtains m*_i only on a small randomly drawn validation set.

The key feature is that the LLM's measurement error is *not* classical:
larger firms write more formulaic, more guarded MD&A text, so the model
systematically understates their financing pressure and E[e_i | c_i] != 0.
This is the case Ludwig & Mullainathan (2025) warn about — the sign of the
naive bias need not be the classical attenuation direction.

Four estimators of beta are compared:

    1) Ideal (infeasible)      — regress on the true m*; benchmark only.
    2) Naive                   — substitute m_hat for m*.  Biased.
    3) Validation-set only     — unbiased but uses only n_v observations.
    4) Bias-corrected          — regression calibration: estimate
                                 E[m* | m_hat, c] on the validation set, then
                                 use the fitted values in the full sample.
                                 Standard errors by bootstrap, resampling the
                                 full sample and the validation set jointly.

The contrast that matters is (3) vs (4): both are approximately unbiased, but
(4) buys back full-sample precision with a few hundred hand-coded documents.
A final block traces interval width against validation-set size, which is the
design calculation behind a hand-coding budget.

The script is intentionally LLM-API-free: it does NOT call any external model.

Dependencies: numpy  (matplotlib optional, for the figure; no GPU, no internet)
Tested with: Python 3.9+, numpy 1.26, matplotlib 3.8
"""

from __future__ import annotations

import numpy as np

SEED = 42
N = 3000          # full-sample observations
N_VALID = 200     # hand-coded validation set
N_BOOT = 2000     # bootstrap replications

# True parameters of the synthetic world
ALPHA_TRUE = 0.020
BETA_TRUE = -0.080   # more financing pressure -> lower investment next year
GAMMA_TRUE = 0.010


# ----------------------------------------------------------------------
# 1. Synthetic data-generating process
# ----------------------------------------------------------------------
def simulate(
    rng: np.random.Generator,
    n: int = N,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (m_star, m_hat, c, y).

    The LLM measure m_hat departs from m_star in three ways, only the first
    of which is benign:
        (a) independent noise,
        (b) shrinkage toward the middle (the model avoids extreme calls),
        (c) a term proportional to firm size -- the non-classical part.
    """
    # Firm size (standardized log assets)
    c = rng.normal(0.0, 1.0, n)

    # Latent financing pressure: larger firms face looser constraints
    m_star = 0.45 - 0.10 * c + rng.normal(0.0, 0.15, n)
    m_star = np.clip(m_star, 0.0, 1.0)

    # Outcome: next year's investment rate
    u = rng.normal(0.0, 0.030, n)
    y = ALPHA_TRUE + BETA_TRUE * m_star + GAMMA_TRUE * c + u

    # LLM measure with non-classical error
    shrink = 0.85
    e_sys = -0.055 * c
    e_idio = rng.normal(0.0, 0.10, n)
    m_hat = 0.5 + shrink * (m_star - 0.5) + e_sys + e_idio
    m_hat = np.clip(m_hat, 0.0, 1.0)

    return m_star, m_hat, c, y


# ----------------------------------------------------------------------
# 2. OLS with homoskedastic standard errors
# ----------------------------------------------------------------------
def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS with an intercept prepended.  Returns (coefficients, std errors)."""
    Xd = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    dof = len(y) - Xd.shape[1]
    sigma2 = resid @ resid / dof
    XtX_inv = np.linalg.inv(Xd.T @ Xd)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    return beta, se


# ----------------------------------------------------------------------
# 3. Regression calibration and its bootstrap
# ----------------------------------------------------------------------
def regression_calibration(
    m_star: np.ndarray,
    m_hat: np.ndarray,
    c: np.ndarray,
    y: np.ndarray,
    valid_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate E[m* | m_hat, c] on the validation set, then use it full-sample.

    Assumptions required: (i) the validation set is drawn at random; (ii) the
    calibration equation extrapolates outside it; (iii) given (m_hat, c), the
    relation between m* and y does not depend on the residual of m_hat.

    Note that m_star is read *only* at valid_idx.  That is exactly the
    information constraint of the real problem: hand labels exist only on the
    validation set.
    """
    # Step 1: fit the calibration equation on the validation set
    Xv = np.column_stack([m_hat[valid_idx], c[valid_idx]])
    cal_coef, _ = ols(Xv, m_star[valid_idx])

    # Step 2: predict m* in the full sample, then run the target regression
    m_tilde = cal_coef[0] + cal_coef[1] * m_hat + cal_coef[2] * c
    coef, _ = ols(np.column_stack([m_tilde, c]), y)
    return coef, cal_coef


def bootstrap_rc(
    rng: np.random.Generator,
    m_star: np.ndarray,
    m_hat: np.ndarray,
    c: np.ndarray,
    y: np.ndarray,
    valid_idx: np.ndarray,
    n_boot: int = N_BOOT,
) -> np.ndarray:
    """Bootstrap the corrected estimator, resampling both samples jointly.

    Resampling both is necessary: the uncertainty comes from the full-sample
    regression *and* from the fact that the calibration equation is itself
    estimated.  Resampling only the full sample understates the standard error.
    """
    n = len(y)
    n_v = len(valid_idx)
    out = np.empty(n_boot)
    for b in range(n_boot):
        full_b = rng.integers(0, n, n)
        val_b = valid_idx[rng.integers(0, n_v, n_v)]

        Xv = np.column_stack([m_hat[val_b], c[val_b]])
        cal, _ = ols(Xv, m_star[val_b])
        m_tilde = cal[0] + cal[1] * m_hat[full_b] + cal[2] * c[full_b]
        coef, _ = ols(np.column_stack([m_tilde, c[full_b]]), y[full_b])
        out[b] = coef[1]
    return out


def ci(point: float, se: float, z: float = 1.96) -> tuple[float, float]:
    return point - z * se, point + z * se


def fmt(lo: float, hi: float) -> str:
    return f"[{lo:+.4f}, {hi:+.4f}]"


# ----------------------------------------------------------------------
# 4. Main: report + (optional) plot
# ----------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(SEED)

    m_star, m_hat, c, y = simulate(rng)
    valid_idx = rng.choice(len(y), size=N_VALID, replace=False)

    print("=" * 70)
    print(f"Bias correction for an LLM text measure (synthetic, seed={SEED})")
    print("=" * 70)
    print(f"Full sample n = {len(y)},  validation set n_v = {N_VALID}")
    print(f"True beta = {BETA_TRUE:+.4f}")
    print()

    # --- [1] Measure quality -----------------------------------------
    corr = np.corrcoef(m_star, m_hat)[0, 1]
    err = m_hat - m_star
    slope_err_c, se_err_c = ols(c.reshape(-1, 1), err)
    print("[1] Quality of the LLM measure")
    print(f"    corr(m*, m_hat)              = {corr:.3f}")
    print(f"    mean bias E[m_hat - m*]      = {np.mean(err):+.4f}")
    print(f"    slope of error on size c     = {slope_err_c[1]:+.4f}"
          f"  (se {se_err_c[1]:.4f})")
    print("    -> A significantly non-zero slope IS the non-classical part:")
    print("       the error is systematically related to a regressor.")
    print("       Note the mean bias is nearly zero, so a mean-bias check")
    print("       alone would have declared this measure clean.")
    print()

    # --- [2] Four estimators -----------------------------------------
    coef_ideal, se_ideal = ols(np.column_stack([m_star, c]), y)
    coef_naive, se_naive = ols(np.column_stack([m_hat, c]), y)
    coef_vo, se_vo = ols(
        np.column_stack([m_star[valid_idx], c[valid_idx]]), y[valid_idx]
    )
    coef_rc, cal_coef = regression_calibration(m_star, m_hat, c, y, valid_idx)
    boot = bootstrap_rc(rng, m_star, m_hat, c, y, valid_idx)
    se_rc = boot.std(ddof=1)
    lo_rc, hi_rc = np.percentile(boot, [2.5, 97.5])

    print("[2] Four estimators of beta")
    rows = [
        ("Ideal (true m*, infeasible)", coef_ideal[1], se_ideal[1],
         ci(coef_ideal[1], se_ideal[1])),
        ("Naive (m_hat as regressor)", coef_naive[1], se_naive[1],
         ci(coef_naive[1], se_naive[1])),
        ("Validation set only", coef_vo[1], se_vo[1],
         ci(coef_vo[1], se_vo[1])),
        ("Bias-corrected (calibration)", coef_rc[1], se_rc, (lo_rc, hi_rc)),
    ]
    print(f"    {'Estimator':<32}{'beta':>10}{'se':>9}   95% CI")
    for name, b, s, (lo, hi) in rows:
        print(f"    {name:<32}{b:>+10.4f}{s:>9.4f}   {fmt(lo, hi)}")
    print()

    # --- [3] Interpretation ------------------------------------------
    bias_naive = coef_naive[1] - BETA_TRUE
    bias_rc = coef_rc[1] - BETA_TRUE
    w_vo = 2 * 1.96 * se_vo[1]
    w_rc = hi_rc - lo_rc

    print("[3] Interpretation")
    print(f"    Naive bias     = {bias_naive:+.4f}"
          f"  ({100 * bias_naive / abs(BETA_TRUE):+.1f}% of the truth)")
    print(f"    Corrected bias = {bias_rc:+.4f}"
          f"  ({100 * bias_rc / abs(BETA_TRUE):+.1f}% of the truth)")
    print(f"    Naive attenuated toward zero: "
          f"{abs(coef_naive[1]) < abs(BETA_TRUE)}")
    print()
    print(f"    CI width: validation-only {w_vo:.4f}, corrected {w_rc:.4f}"
          f"  (ratio {w_vo / w_rc:.2f})")
    print(f"    -> {N_VALID} hand-coded documents buy back full-sample "
          f"precision:")
    print(f"       the corrected interval is {100 * (1 - w_rc / w_vo):.0f}% "
          f"narrower than validation-only.")
    print()
    print(f"    Calibration: E[m*|m_hat,c] = {cal_coef[0]:+.4f} "
          f"{cal_coef[1]:+.4f}*m_hat {cal_coef[2]:+.4f}*c")
    print(f"    The calibration slope on m_hat, {cal_coef[1]:.3f}, is well "
          f"below 1.  That shrinkage")
    print("       is the source of the naive attenuation; the correction "
          "scales the slope back up.")
    print(f"    The coefficient on c is only {cal_coef[2]:+.4f} in this draw, "
          f"because m_hat already")
    print("       carries a size-proportional error and the two are highly "
          "collinear.  This does")
    print("       NOT mean c may be dropped: without it the calibration is no "
          "longer a consistent")
    print("       estimate of E[m*|m_hat,c], and the error-on-size slope "
          "above goes unabsorbed.")
    print()

    # --- [4] Design calculation --------------------------------------
    print("[4] Design calculation: interval width vs validation-set size")
    print(f"    {'n_v':>6}{'beta_rc':>11}{'CI width':>12}")
    rng2 = np.random.default_rng(SEED + 1)
    widths: dict[int, float] = {}
    for n_v in (50, 100, 200, 400, 800):
        idx = rng2.choice(len(y), size=n_v, replace=False)
        coef_t, _ = regression_calibration(m_star, m_hat, c, y, idx)
        bt = bootstrap_rc(rng2, m_star, m_hat, c, y, idx, n_boot=1000)
        lo, hi = np.percentile(bt, [2.5, 97.5])
        widths[n_v] = hi - lo
        print(f"    {n_v:>6}{coef_t[1]:>+11.4f}{hi - lo:>12.4f}")
    print()
    print("    Width falls in n_v, but at a decreasing rate.  For small n_v")
    print("    the calibration equation's own estimation error dominates; past")
    print("    a few hundred documents the binding constraint becomes the")
    print("    full-sample residual variance, and extra labels buy little.")
    print(f"    Here 50 -> 200 narrows the interval by "
          f"{100 * (1 - widths[200] / widths[50]):.0f}%, while 200 -> 800 "
          f"narrows it by only "
          f"{100 * (1 - widths[800] / widths[200]):.0f}%.")
    print("    That curve is the budget rule: spend enough labels to kill the")
    print("    bias, then stop enlarging the validation set.")
    print()
    print("Note: illustrative calculation on simulated data with a fixed "
          "seed; not an empirical result.")

    # --- Optional figure ---------------------------------------------
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4))
        names = ["Ideal\n(infeasible)", "Naive", "Validation\nonly",
                 "Bias-\ncorrected"]
        for k, (_, b, _s, (lo, hi)) in enumerate(rows):
            ax.plot([k, k], [lo, hi], color="tab:blue", lw=2)
            ax.plot(k, b, "o", color="tab:blue")
        ax.axhline(BETA_TRUE, color="red", linestyle="--",
                   label=f"true beta = {BETA_TRUE:+.3f}")
        ax.set_xticks(range(4))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel(r"$\beta$ and 95% CI")
        ax.set_title("LLM-generated regressor: bias and precision")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        out_path = "bias_correction_ci.png"
        fig.savefig(out_path, dpi=150)
        print(f"\nFigure saved to {out_path}")
    except Exception as exc:  # pragma: no cover
        print(f"\n(matplotlib unavailable, skipping plot: {exc})")


if __name__ == "__main__":
    main()
