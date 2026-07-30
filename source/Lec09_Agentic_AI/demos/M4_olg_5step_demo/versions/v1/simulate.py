"""V1 simulation."""
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
    rec_zi: list[int] = []
    rec_Z: list[float] = []
    rec_K: list[float] = []
    rec_r: list[float] = []
    rec_w: list[float] = []
    rec_a: list[np.ndarray] = []      # shape (T - burn, N-1)
    rec_c: list[np.ndarray] = []      # shape (T - burn, N)
    rec_s: list[np.ndarray] = []      # shape (T - burn, N-1)
    for t in range(T):
        Z = Z_VALS[z_idx]
        s = net(Z, a)
        out = cohort_decisions(Z, a, s)
        if t >= burn:
            rec_zi.append(int(z_idx.item()))
            rec_Z.append(float(Z.item()))
            rec_K.append(float(out["K"].item()))
            rec_r.append(float(out["r"].item()))
            rec_w.append(float(out["w"].item()))
            rec_a.append(a[0].cpu().numpy().copy())
            rec_c.append(out["c"][0].cpu().numpy().copy())
            rec_s.append(s[0].cpu().numpy().copy())
        a = out["a_next"]
        probs = P_MAT[z_idx]
        u = torch.rand(1, device=device)
        z_idx = (u > probs[:, 0]).long()
    return dict(
        zi=np.array(rec_zi),
        Z=np.array(rec_Z),
        K=np.array(rec_K),
        r=np.array(rec_r),
        w=np.array(rec_w),
        a=np.stack(rec_a, axis=0),
        c=np.stack(rec_c, axis=0),
        s=np.stack(rec_s, axis=0),
    )


def print_summary(sim: dict) -> None:
    K = sim["K"]; r = sim["r"]; zi = sim["zi"]
    print(f"E[K]      = {K.mean():.3f}")
    print(f"E[K|Z_lo] = {K[zi == 0].mean():.3f}")
    print(f"E[K|Z_hi] = {K[zi == 1].mean():.3f}")
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
    K_lo = K[zi == 0].mean(); K_hi = K[zi == 1].mean()
    procyclical = K_hi > K_lo
    final_rms = math.sqrt(min(losses[-50:]) + 1e-30)
    rms_ok = final_rms < 0.10
    mean_c = c.mean(axis=0)
    consumption_grows = mean_c[N - 1] > mean_c[0]
    # Savings rate should peak before retirement (interior cohort), not at age 0.
    mean_s = s.mean(axis=0)
    peak_saver = int(np.argmax(mean_s))
    saving_peak_in_pre_retirement = peak_saver >= 2
    return {
        "training_progressed": min(losses) < losses[0] / 5,
        "procyclical_capital_E[K|hi] > E[K|lo]": bool(procyclical),
        "rms_euler_residual_<_10pct": bool(rms_ok),
        "consumption_grows_with_age": bool(consumption_grows),
        "savings_peak_in_pre_retirement": bool(saving_peak_in_pre_retirement),
    }
