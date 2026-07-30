"""V2 plotting helpers."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from model import N, Z_VALS


def plot_loss(losses, ax=None, label="V2"):
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
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    mean_c = sim["c"].mean(axis=0)
    mean_a = sim["a"].mean(axis=0)
    mean_s = sim["s"].mean(axis=0)
    axes[0].bar(np.arange(N), mean_c, color="C0")
    axes[0].set_xticks(np.arange(N)); axes[0].set_xlabel("cohort age"); axes[0].set_ylabel("mean consumption")
    axes[0].set_title("Lifecycle consumption")
    axes[1].bar(np.arange(1, N), mean_a, color="C2")
    axes[1].set_xticks(np.arange(1, N)); axes[1].set_xlabel("cohort age"); axes[1].set_ylabel("mean wealth")
    axes[1].set_title("Lifecycle wealth")
    axes[2].bar(np.arange(N - 1), mean_s, color="C3")
    axes[2].set_xticks(np.arange(N - 1)); axes[2].set_xlabel("cohort age"); axes[2].set_ylabel("mean savings rate")
    axes[2].set_title("Savings rate by age")
    return axes


def plot_aggregate_by_regime(sim, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.5))
    n_tfp = Z_VALS.shape[0]
    bins = np.linspace(sim["K"].min() * 0.95, sim["K"].max() * 1.05, 40)
    cmap = plt.get_cmap("viridis", n_tfp)
    for j in range(n_tfp):
        mask = sim["zi"] == j
        if mask.any():
            ax.hist(sim["K"][mask], bins=bins, alpha=0.6,
                    color=cmap(j), label=f"Z = {Z_VALS[j].item():.3f}")
    ax.set_xlabel("aggregate K")
    ax.set_ylabel("count")
    ax.set_title("Ergodic K by TFP regime")
    ax.legend(fontsize=8)
    return ax


def plot_tfp_chain(ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.2))
    z = Z_VALS.cpu().numpy()
    ax.bar(np.arange(z.size), z, color="C4")
    ax.set_xticks(np.arange(z.size))
    ax.set_xlabel("state index")
    ax.set_ylabel("Z")
    ax.set_title("Rouwenhorst TFP grid")
    return ax
