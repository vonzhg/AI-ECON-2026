"""V3 plotting helpers."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from model import N


def plot_loss(losses, ax=None, label="V3"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(losses, lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("total loss (capital + bond + FB)")
    ax.set_title(f"{label} training loss")
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_lifecycle(sim, axes=None):
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    mean_c = sim["c"].mean(axis=0)
    mean_k = sim["k"].mean(axis=0)
    mean_b = sim["b"].mean(axis=0)
    mean_sK = sim["s_K"].mean(axis=0)
    axes[0, 0].bar(np.arange(N), mean_c, color="C0")
    axes[0, 0].set_xticks(np.arange(N)); axes[0, 0].set_title("Lifecycle consumption")
    axes[0, 0].set_xlabel("cohort age")
    axes[0, 1].bar(np.arange(1, N), mean_k, color="C2")
    axes[0, 1].set_xticks(np.arange(1, N)); axes[0, 1].set_title("Lifecycle capital wealth")
    axes[0, 1].set_xlabel("cohort age")
    bcol = ["C3" if x >= 0 else "C1" for x in mean_b]
    axes[1, 0].bar(np.arange(1, N), mean_b, color=bcol)
    axes[1, 0].axhline(0, color="black", lw=0.5)
    axes[1, 0].set_xticks(np.arange(1, N)); axes[1, 0].set_title("Lifecycle bond holdings")
    axes[1, 0].set_xlabel("cohort age")
    axes[1, 1].bar(np.arange(N - 1), mean_sK, color="C4")
    axes[1, 1].set_xticks(np.arange(N - 1)); axes[1, 1].set_title("Capital savings rate")
    axes[1, 1].set_xlabel("cohort age")
    return axes


def plot_bond_price(sim, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(sim["p_b"][:200], color="C3")
    ax.set_xlabel("period")
    ax.set_ylabel("bond price $p_b$")
    ax.set_title("Endogenous bond price (first 200 periods)")
    ax.grid(True, alpha=0.3)
    return ax
