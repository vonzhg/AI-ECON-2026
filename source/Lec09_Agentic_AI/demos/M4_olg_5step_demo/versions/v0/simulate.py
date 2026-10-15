"""V0 simulation — forward-simulate a single trained network."""
from __future__ import annotations

import math

import numpy as np
import torch

from model import P, P_MAT, Z_VALS, cohort_decisions, device, init_cloud, step_cloud
from network import PolicyNet
from train import euler_residuals


@torch.no_grad()
def run(net: PolicyNet, T: int = 5000, burn: int = 500) -> dict:
    z_idx, am, ao = init_cloud(1)
    rows = []
    for t in range(T):
        Z = Z_VALS[z_idx]
        sy, sm = net(Z, am, ao)
        out = cohort_decisions(Z, am, ao, sy, sm)
        if t >= burn:
            rows.append((
                int(z_idx.item()), float(Z.item()), float(out["K"].item()),
                float(out["cy"].item()), float(out["cm"].item()), float(out["co"].item()),
                float(out["r"].item()), float(out["w"].item()),
                float(am.item()), float(ao.item()),
                float(sy.item()), float(sm.item()),
            ))
        am, ao = out["am_next"], out["ao_next"]
        probs = P_MAT[z_idx]
        u = torch.rand(1, device=device)
        z_idx = (u > probs[:, 0]).long()
    cols = ["zi", "Z", "K", "cy", "cm", "co", "r", "w", "am", "ao", "sy", "sm"]
    return {c: np.array([r[i] for r in rows]) for i, c in enumerate(cols)}


def print_summary(sim: dict) -> None:
    K = sim["K"]; r = sim["r"]; zi = sim["zi"]
    print(f"E[K]      = {K.mean():.3f}")
    print(f"E[K|Z_lo] = {K[zi == 0].mean():.3f}")
    print(f"E[K|Z_hi] = {K[zi == 1].mean():.3f}")
    annual = (1 + r.mean()) ** (1 / 20) - 1
    print(f"E[r]      = {r.mean():.4f}   (annualised ≈ {annual:.4f})")


def ergodic_residuals(net: PolicyNet, N_eval: int = 4096, burn: int = 200):
    """Build a fresh ergodic cloud and evaluate Euler residuals on it."""
    zi_e, am_e, ao_e = init_cloud(N_eval)
    for _ in range(burn):
        zi_e, am_e, ao_e = step_cloud(zi_e, am_e, ao_e, net)
    with torch.no_grad():
        R_y, R_m = euler_residuals(zi_e, am_e, ao_e, net)
    return R_y.cpu().numpy(), R_m.cpu().numpy()


def validation_gate(sim: dict, losses: list[float]) -> dict:
    """Check the V0 validation criteria. Returns one bool per criterion."""
    K = sim["K"]; zi = sim["zi"]; cy = sim["cy"]; cm = sim["cm"]; co = sim["co"]
    K_lo = K[zi == 0].mean(); K_hi = K[zi == 1].mean()
    procyclical = K_hi > K_lo
    final_rms = math.sqrt(min(losses[-50:]) + 1e-30)
    rms_ok = final_rms < 0.08
    hump = (co.mean() > cm.mean()) and (cm.mean() > cy.mean())
    return {
        "training_progressed": min(losses) < losses[0] / 10,
        "procyclical_capital_E[K|hi] > E[K|lo]": bool(procyclical),
        "rms_euler_residual_<_8pct": bool(rms_ok),
        "lifecycle_hump_co>cm>cy": bool(hump),
    }
