"""V1 plotting helpers."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from model import N


def plot_loss(losses, ax=None, label="V1"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(losses, lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("MSE Euler residual")
    ax.set_title(f"{label} training loss")
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_lifecycle(sim, axes=None):
    """Plot mean consumption, mean wealth, mean savings rate by cohort age."""
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    mean_c = sim["c"].mean(axis=0)
    mean_a = sim["a"].mean(axis=0)
    mean_s = sim["s"].mean(axis=0)
    ages = np.arange(N)
    axes[0].bar(ages, mean_c, color="C0")
    axes[0].set_xticks(ages)
    axes[0].set_xlabel("cohort age")
    axes[0].set_ylabel("mean consumption")
    axes[0].set_title("Lifecycle consumption")
    axes[1].bar(np.arange(1, N), mean_a, color="C2")
    axes[1].set_xticks(np.arange(1, N))
    axes[1].set_xlabel("cohort age")
    axes[1].set_ylabel("mean wealth")
    axes[1].set_title("Lifecycle wealth (entry)")
    axes[2].bar(np.arange(N - 1), mean_s, color="C3")
    axes[2].set_xticks(np.arange(N - 1))
    axes[2].set_xlabel("cohort age")
    axes[2].set_ylabel("mean savings rate")
    axes[2].set_title("Savings rate by age")
    return axes


def plot_aggregate(sim, T_show: int = 200, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    ax = axes[0]
    ax.plot(sim["K"][:T_show], color="C0")
    ax2 = ax.twinx()
    ax2.step(np.arange(T_show), sim["Z"][:T_show], where="post", color="C3", lw=0.8)
    ax.set_xlabel("period")
    ax.set_ylabel("aggregate K", color="C0")
    ax2.set_ylabel("TFP Z", color="C3")
    ax.set_title(f"Sample path (first {T_show} periods)")
    ax = axes[1]
    bins = np.linspace(sim["K"].min() * 0.95, sim["K"].max() * 1.05, 40)
    ax.hist(sim["K"][sim["zi"] == 0], bins=bins, alpha=0.55, color="C0", label="Z_lo")
    ax.hist(sim["K"][sim["zi"] == 1], bins=bins, alpha=0.55, color="C3", label="Z_hi")
    ax.set_xlabel("aggregate K")
    ax.set_ylabel("count")
    ax.set_title("Ergodic K conditional on Z")
    ax.legend()
    return axes
