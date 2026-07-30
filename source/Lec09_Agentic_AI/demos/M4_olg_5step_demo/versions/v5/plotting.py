"""V5 plotting — adds homotopy-phase decomposition."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from model import N


def plot_homotopy_loss(history, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3.5))
    rec = np.array(history["all"])
    rk = rec[:, 0]; rb = rec[:, 1]; fb = rec[:, 2]; total = rec[:, 3]
    ax.plot(total, lw=0.6, color="black", label="total")
    ax.plot(rk,    lw=0.6, color="C0", alpha=0.8, label="R_K MSE")
    ax.plot(rb,    lw=0.6, color="C3", alpha=0.8, label="R_B MSE")
    ax.plot(fb,    lw=0.6, color="C2", alpha=0.8, label="FB MSE")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("residual MSE")
    cum = 0
    cmap = plt.get_cmap("Set2")
    for i, p in enumerate(history["phases"]):
        ax.axvspan(cum, cum + p["steps"], alpha=0.12, color=cmap(i))
        cum += p["steps"]
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("V5 homotopy — per-phase residual decomposition")
    ax.grid(True, which="both", alpha=0.2)
    return ax


def plot_phase_summary(history, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3.4))
    names = [p["name"] for p in history["phases"]]
    starts = [p["start_total"] for p in history["phases"]]
    ends = [p["end_total"] for p in history["phases"]]
    x = np.arange(len(names))
    ax.bar(x - 0.18, starts, 0.34, color="C1", label="phase start")
    ax.bar(x + 0.18, ends, 0.34, color="C2", label="phase end (50-step mean)")
    ax.set_yscale("log")
    ax.set_xticks(x, [n.replace("_", "\n") for n in names])
    ax.set_ylabel("total loss")
    ax.set_title("Per-phase residual snapshots")
    ax.legend()
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
    axes[0, 1].bar(np.arange(1, N), mean_k, color="C2")
    axes[0, 1].set_xticks(np.arange(1, N)); axes[0, 1].set_title("Lifecycle capital wealth")
    bcol = ["C3" if x >= 0 else "C1" for x in mean_b]
    axes[1, 0].bar(np.arange(1, N), mean_b, color=bcol)
    axes[1, 0].axhline(0, color="black", lw=0.5)
    axes[1, 0].set_xticks(np.arange(1, N)); axes[1, 0].set_title("Lifecycle bond holdings")
    axes[1, 1].bar(np.arange(N - 1), mean_sK, color="C4")
    axes[1, 1].set_xticks(np.arange(N - 1)); axes[1, 1].set_title("Capital savings rate")
    return axes
