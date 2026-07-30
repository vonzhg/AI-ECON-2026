"""V3 simulation — track both capital and bond paths."""
from __future__ import annotations

import math

import numpy as np
import torch

from model import N, P, P_MAT, Z_VALS, cohort_decisions, device, init_cloud, step_cloud
from network import PolicyNet
from train import euler_residuals


@torch.no_grad()
def run(net: PolicyNet, T: int = 5000, burn: int = 500, *, bonds_off: bool = False) -> dict:
    z_idx, k, b = init_cloud(1)
    rec_zi: list[int] = []; rec_Z: list[float] = []; rec_K: list[float] = []
    rec_r: list[float] = []; rec_w: list[float] = []; rec_pb: list[float] = []
    rec_k: list[np.ndarray] = []; rec_b: list[np.ndarray] = []
    rec_c: list[np.ndarray] = []; rec_sK: list[np.ndarray] = []
    n_tfp = Z_VALS.shape[0]
    for t in range(T):
        Z = Z_VALS[z_idx]
        s_K, b_next, p_b = net(Z, k, b)
        out = cohort_decisions(Z, k, b, s_K, b_next, p_b, bonds_off=bonds_off)
        if t >= burn:
            rec_zi.append(int(z_idx.item())); rec_Z.append(float(Z.item()))
            rec_K.append(float(out["K"].item())); rec_r.append(float(out["r"].item()))
            rec_w.append(float(out["w"].item())); rec_pb.append(float(p_b.item()))
            rec_k.append(k[0].cpu().numpy().copy())
            rec_b.append(b[0].cpu().numpy().copy())
            rec_c.append(out["c"][0].cpu().numpy().copy())
            rec_sK.append(s_K[0].cpu().numpy().copy())
        k = out["k_next"]
        b = out["b_next"]
        probs = P_MAT[z_idx]
        cdf = torch.cumsum(probs, dim=-1)
        u = torch.rand(z_idx.shape[0], 1, device=device)
        z_idx = (u > cdf).sum(dim=-1).long().clamp(max=n_tfp - 1)
    return dict(
        zi=np.array(rec_zi), Z=np.array(rec_Z), K=np.array(rec_K),
        r=np.array(rec_r), w=np.array(rec_w), p_b=np.array(rec_pb),
        k=np.stack(rec_k), b=np.stack(rec_b), c=np.stack(rec_c), s_K=np.stack(rec_sK),
    )


def print_summary(sim: dict) -> None:
    print(f"E[K]   = {sim['K'].mean():.4f}")
    print(f"E[r]   = {sim['r'].mean():.4f}")
    print(f"E[p_b] = {sim['p_b'].mean():.4f}    implied bond rate = {1.0/sim['p_b'].mean()-1:.4f}")
    mean_b = sim["b"].mean(axis=0)
    print(f"mean bond holdings by age 1..N-1: {np.round(mean_b, 4)}")
    print(f"max |sum b| (should be ≈ 0):       {np.abs(sim['b'].sum(axis=1)).max():.2e}")


def ergodic_residuals(net: PolicyNet, N_eval: int = 4096, burn: int = 200, *, bonds_off: bool = False):
    zi_e, k_e, b_e = init_cloud(N_eval)
    for _ in range(burn):
        zi_e, k_e, b_e = step_cloud(zi_e, k_e, b_e, net, bonds_off=bonds_off)
    with torch.no_grad():
        R_K, R_B, FB = euler_residuals(zi_e, k_e, b_e, net, bonds_off=bonds_off)
    return R_K.cpu().numpy(), R_B.cpu().numpy(), FB.cpu().numpy()


def validation_gate(sim: dict, losses: list[float]) -> dict:
    K = sim["K"]; zi = sim["zi"]; c = sim["c"]; b = sim["b"]; s_K = sim["s_K"]
    n_tfp = Z_VALS.shape[0]
    K_top = K[zi == n_tfp - 1].mean() if (zi == n_tfp - 1).any() else float("nan")
    K_bot = K[zi == 0].mean() if (zi == 0).any() else float("nan")
    final_rms = math.sqrt(min(losses[-50:]) + 1e-30)
    bond_clearing = bool(np.abs(b.sum(axis=1)).max() < 1e-4)
    # Old cohorts (pre-retirement) should hold non-trivial wealth in bonds (positive or negative).
    bond_dispersion = float(b.mean(axis=0).std())
    bond_lifecycle = bool(bond_dispersion > 1e-3)
    consumption_grows = bool(c.mean(axis=0)[N - 1] > c.mean(axis=0)[0])
    saving_peak_pre_retire = bool(int(np.argmax(s_K.mean(axis=0))) >= 2)
    p_b_in_range = bool(P["p_b_min"] < sim["p_b"].mean() < P["p_b_max"])
    return {
        "training_progressed": min(losses) < losses[0] / 5,
        "procyclical_top_state>bottom_state": bool(K_top > K_bot),
        "rms_total_loss_<_15pct": bool(final_rms < 0.15),
        "bond_market_clears": bond_clearing,
        "bond_lifecycle_dispersion": bond_lifecycle,
        "consumption_grows_with_age": consumption_grows,
        "savings_peak_in_pre_retirement": saving_peak_pre_retire,
        "bond_price_in_range": p_b_in_range,
    }
