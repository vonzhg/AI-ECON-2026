"""V2 simulation."""
from __future__ import annotations

import math

import numpy as np
import torch

from model import N, P_MAT, Z_VALS, cohort_decisions, device, init_cloud, step_cloud
from network import PolicyNet
from train import euler_residuals


@torch.no_grad()
def run(net: PolicyNet, T: int = 5000, burn: int = 500) -> dict:
    z_idx, a = init_cloud(1)
    rec_zi: list[int] = []; rec_Z: list[float] = []; rec_K: list[float] = []
    rec_r: list[float] = []; rec_w: list[float] = []
    rec_a: list[np.ndarray] = []; rec_c: list[np.ndarray] = []; rec_s: list[np.ndarray] = []
    n_tfp = Z_VALS.shape[0]
    for t in range(T):
        Z = Z_VALS[z_idx]
        s = net(Z, a)
        out = cohort_decisions(Z, a, s)
        if t >= burn:
            rec_zi.append(int(z_idx.item())); rec_Z.append(float(Z.item()))
            rec_K.append(float(out["K"].item())); rec_r.append(float(out["r"].item()))
            rec_w.append(float(out["w"].item()))
            rec_a.append(a[0].cpu().numpy().copy())
            rec_c.append(out["c"][0].cpu().numpy().copy())
            rec_s.append(s[0].cpu().numpy().copy())
        a = out["a_next"]
        probs = P_MAT[z_idx]
        cdf = torch.cumsum(probs, dim=-1)
        u = torch.rand(z_idx.shape[0], 1, device=device)
        z_idx = (u > cdf).sum(dim=-1).long().clamp(max=n_tfp - 1)
    return dict(
        zi=np.array(rec_zi), Z=np.array(rec_Z), K=np.array(rec_K),
        r=np.array(rec_r), w=np.array(rec_w),
        a=np.stack(rec_a, axis=0), c=np.stack(rec_c, axis=0), s=np.stack(rec_s, axis=0),
    )


def print_summary(sim: dict) -> None:
    K = sim["K"]; r = sim["r"]; zi = sim["zi"]
    n_tfp = Z_VALS.shape[0]
    print(f"E[K]      = {K.mean():.4f}    std = {K.std():.4f}")
    for j in range(n_tfp):
        mask = zi == j
        if mask.any():
            print(f"E[K|state {j} (Z={Z_VALS[j].item():.3f})] = {K[mask].mean():.4f}   (n={int(mask.sum())})")
    print(f"E[r]      = {r.mean():.4f}")


def ergodic_residuals(net: PolicyNet, N_eval: int = 4096, burn: int = 200) -> np.ndarray:
    zi_e, a_e = init_cloud(N_eval)
    for _ in range(burn):
        zi_e, a_e = step_cloud(zi_e, a_e, net)
    with torch.no_grad():
        R = euler_residuals(zi_e, a_e, net)
    return R.cpu().numpy()


def validation_gate(sim: dict, losses: list[float]) -> dict:
    K = sim["K"]; zi = sim["zi"]; c = sim["c"]; s = sim["s"]
    final_rms = math.sqrt(min(losses[-50:]) + 1e-30)
    rms_ok = final_rms < 0.10
    # Procyclical: K rises with TFP state. Compare top vs bottom regimes.
    n_tfp = Z_VALS.shape[0]
    K_top = K[zi == n_tfp - 1].mean() if (zi == n_tfp - 1).any() else float("nan")
    K_bot = K[zi == 0].mean() if (zi == 0).any() else float("nan")
    procyclical = bool(K_top > K_bot)
    consumption_grows = bool(c.mean(axis=0)[N - 1] > c.mean(axis=0)[0])
    saving_peak_pre_retire = bool(int(np.argmax(s.mean(axis=0))) >= 2)
    # K dispersion across regimes should be larger than V1's 2-state version.
    by_state = np.array([K[zi == j].mean() if (zi == j).any() else np.nan for j in range(n_tfp)])
    by_state = by_state[~np.isnan(by_state)]
    spread = by_state.max() - by_state.min()
    return {
        "training_progressed": min(losses) < losses[0] / 5,
        "procyclical_top_state>bottom_state": procyclical,
        "rms_euler_residual_<_10pct": bool(rms_ok),
        "consumption_grows_with_age": consumption_grows,
        "savings_peak_in_pre_retirement": saving_peak_pre_retire,
        "K_spread_across_TFP_states": float(spread),
    }
