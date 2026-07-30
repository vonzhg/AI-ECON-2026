"""V0 plotting — matplotlib figures the notebook displays."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import P, Z_VALS, device
from network import PolicyNet


def plot_loss(losses, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(losses, lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("MSE Euler residual")
    ax.set_title("V0 training loss")
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_path_and_hist(sim, T_show: int = 200, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    ax = axes[0]
    ax.plot(sim["K"][:T_show], color="C0", label="K_t")
    ax2 = ax.twinx()
    ax2.step(np.arange(T_show), sim["Z"][:T_show], where="post", color="C3", lw=0.8)
    ax2.set_yticks([P["Z_lo"], P["Z_hi"]])
    ax.set_xlabel("period (after burn-in)")
    ax.set_ylabel("aggregate K", color="C0")
    ax2.set_ylabel("TFP Z", color="C3")
    ax.set_title(f"Sample path (first {T_show} periods)")
    ax = axes[1]
    bins = np.linspace(sim["K"].min() * 0.95, sim["K"].max() * 1.05, 40)
    ax.hist(sim["K"][sim["zi"] == 0], bins=bins, alpha=0.55, color="C0", label="Z = Z_lo")
    ax.hist(sim["K"][sim["zi"] == 1], bins=bins, alpha=0.55, color="C3", label="Z = Z_hi")
    ax.set_xlabel("aggregate capital K")
    ax.set_ylabel("count")
    ax.set_title("Ergodic K conditional on Z")
    ax.legend()
    return axes


def plot_lifecycle(sim, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
    means_lo = [sim["cy"][sim["zi"] == 0].mean(),
                sim["cm"][sim["zi"] == 0].mean(),
                sim["co"][sim["zi"] == 0].mean()]
    means_hi = [sim["cy"][sim["zi"] == 1].mean(),
                sim["cm"][sim["zi"] == 1].mean(),
                sim["co"][sim["zi"] == 1].mean()]
    x = np.arange(3)
    ax.bar(x - 0.18, means_lo, 0.34, label="Z = Z_lo", color="C0")
    ax.bar(x + 0.18, means_hi, 0.34, label="Z = Z_hi", color="C3")
    ax.set_xticks(x, ["young", "middle", "old"])
    ax.set_ylabel("mean consumption")
    ax.set_title("Lifecycle consumption profile")
    ax.legend()
    return ax


@torch.no_grad()
def policy_slice(net: PolicyNet, K_grid: np.ndarray, z_idx: int, am_share: float):
    am = torch.tensor(am_share * K_grid, dtype=torch.float32, device=device)
    ao = torch.tensor((1 - am_share) * K_grid, dtype=torch.float32, device=device)
    Z = Z_VALS[torch.full_like(am, z_idx, dtype=torch.long)]
    sy, sm = net(Z, am, ao)
    return sy.cpu().numpy(), sm.cpu().numpy()


def plot_policy_slices(net, sim, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    K_grid = np.linspace(max(0.01, sim["K"].min() * 0.6), sim["K"].max() * 1.4, 80)
    am_share = sim["am"].mean() / (sim["am"].mean() + sim["ao"].mean())
    sy_lo, sm_lo = policy_slice(net, K_grid, 0, am_share)
    sy_hi, sm_hi = policy_slice(net, K_grid, 1, am_share)
    axes[0].plot(K_grid, sy_lo, color="C0", label="Z_lo")
    axes[0].plot(K_grid, sy_hi, color="C3", label="Z_hi")
    axes[0].set_xlabel("aggregate K"); axes[0].set_ylabel("$s_y$")
    axes[0].set_title("Young savings rate"); axes[0].legend()
    axes[1].plot(K_grid, sm_lo, color="C0", label="Z_lo")
    axes[1].plot(K_grid, sm_hi, color="C3", label="Z_hi")
    axes[1].set_xlabel("aggregate K"); axes[1].set_ylabel("$s_m$")
    axes[1].set_title("Middle savings rate"); axes[1].legend()
    return axes


def plot_residual_hist(R_y, R_m, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(R_y, bins=60, alpha=0.6, color="C0", label="young residual")
    ax.hist(R_m, bins=60, alpha=0.6, color="C3", label="middle residual")
    ax.set_xlabel("normalised Euler residual")
    ax.set_ylabel("count"); ax.legend()
    ax.set_title(
        f"Pointwise residuals  "
        f"(mean |R_y|={np.abs(R_y).mean():.2e},  mean |R_m|={np.abs(R_m).mean():.2e})"
    )
    return ax


def save_default_figures(net, sim, losses, R_y, R_m, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, ax = plt.subplots(figsize=(7, 3.2)); plot_loss(losses, ax)
    p = out_dir / "v0_loss.png"; fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5)); plot_path_and_hist(sim, axes=axes)
    p = out_dir / "v0_path_hist.png"; fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)
    fig, ax = plt.subplots(figsize=(6.5, 3.5)); plot_lifecycle(sim, ax)
    p = out_dir / "v0_lifecycle.png"; fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4)); plot_policy_slices(net, sim, axes)
    p = out_dir / "v0_policies.png"; fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)
    fig, ax = plt.subplots(figsize=(7, 3.2)); plot_residual_hist(R_y, R_m, ax)
    p = out_dir / "v0_residuals.png"; fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)
    return paths
