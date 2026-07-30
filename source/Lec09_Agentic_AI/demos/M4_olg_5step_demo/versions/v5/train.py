"""V5 training — four-phase stabilising homotopy schedule.

Phase 1  (capital-only)      : bond_weight = 0, fb_weight = 0, bonds_off = True
Phase 2  (bond pretraining)  : bond_weight = 0.1, fb_weight = 0.0, bonds_off = False
Phase 3  (bond homotopy)     : bond_weight 0.1 → 1.0 (linear), fb_weight 0 → 0.5
Phase 4  (fine-tuning)       : bond_weight = 1.0, fb_weight = 0.5, lr = lr / 10

The model itself (cohort decisions, Euler residuals, FB) is identical to V4;
only the training schedule is new.  The schedule is logged step-by-step so the
notebook can plot a per-phase residual decomposition.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.optim as optim

from model import (
    EPS, N, P, P_MAT, Z_VALS, cohort_decisions, device, fb_residual,
    init_cloud, replace_random, step_cloud,
)
from network import PolicyNet


HP: dict = dict(
    hidden=192,
    N_cloud=512,
    batch_size=256,
    lr=8e-4,
    lr_decay=0.9998,
    pretrain_steps=800,
    pretrain_target_sK=0.40,
    pretrain_target_pb=0.80,
    log_every=500,
    seed=0,
    psi_K=None,
    # Phase budgets.
    phase1_steps=2000,        # capital-only
    phase2_steps=1500,        # bond pretraining
    phase3_steps=1500,        # bond homotopy
    phase4_steps=1500,        # fine-tuning
    fine_tune_lr_factor=0.10,
    # Final-phase loss weights.
    capital_weight=1.0,
    bond_weight=1.0,
    fb_weight=0.5,
)


def euler_residuals(z_idx, k, b, net, *, bonds_off=False, psi_K=None):
    psi = psi_K if psi_K is not None else P["psi_K"]
    Z = Z_VALS[z_idx]
    s_K, b_next, p_b = net(Z, k, b)
    today = cohort_decisions(Z, k, b, s_K, b_next, p_b, bonds_off=bonds_off, psi_K=psi)

    n_z = Z_VALS.shape[0]
    rhs_K = torch.zeros_like(s_K); rhs_B = torch.zeros_like(s_K)
    for jz in range(n_z):
        Zp = Z_VALS[jz].expand_as(Z)
        s_K_p, b_next_p, p_b_p = net(Zp, today["k_next"], today["b_next"])
        out_p = cohort_decisions(Zp, today["k_next"], today["b_next"],
                                 s_K_p, b_next_p, p_b_p, bonds_off=bonds_off, psi_K=psi)
        c_next = out_p["c"][..., 1:N]
        prob = P_MAT[z_idx, jz].unsqueeze(-1)
        mu_next = c_next.pow(-P["gamma"])
        rhs_K = rhs_K + prob * (1 + out_p["r"].unsqueeze(-1)) * mu_next
        rhs_B = rhs_B + prob * mu_next

    mu_today = today["c"][..., :N - 1].pow(-P["gamma"])
    marg_cost = 1 + psi * today["delta_k"]
    R_K = marg_cost - P["beta"] * rhs_K / mu_today
    if bonds_off:
        R_B = torch.zeros_like(R_K)
    else:
        R_B = 1 - P["beta"] * rhs_B / (p_b.unsqueeze(-1) * mu_today)

    slack = b_next - P["b_min"]
    multiplier = R_B.abs() + 1e-4
    FB = fb_residual(slack, multiplier)
    return R_K, R_B, FB


def pretrain(net: PolicyNet, n_steps: int, *, lr: float = 3e-3) -> float:
    opt = optim.Adam(net.parameters(), lr=lr)
    last = float("nan")
    target_sK = torch.tensor(HP["pretrain_target_sK"], device=device)
    target_pb = torch.tensor(HP["pretrain_target_pb"], device=device)
    for _ in range(n_steps):
        z_idx, k, b = init_cloud(HP["batch_size"])
        Z = Z_VALS[z_idx]
        s_K, b_next, p_b = net(Z, k, b)
        loss = ((s_K - target_sK) ** 2).mean() \
             + ((p_b - target_pb) ** 2).mean() \
             + 0.1 * (b_next ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.item())
    return last


def _train_block(
    net: PolicyNet, opt, sched,
    cloud, hp, n_steps, *, capital_weight, bond_weight, fb_weight,
    bonds_off=False, weight_schedule=None, log_label="", verbose=True,
):
    """Run n_steps of training with a given (constant or scheduled) weight set.

    weight_schedule: optional callable step → (capital_weight, bond_weight, fb_weight)
                     overriding the constant weights above. Used in homotopy phase.
    Returns updated cloud and a list of per-step residuals (R_K_mse, R_B_mse, FB_mse, total).
    """
    z_idx, k, b = cloud
    psi_K = hp.get("psi_K", None)
    rec: list[tuple[float, float, float, float]] = []
    for step in range(1, n_steps + 1):
        z_idx, k, b = step_cloud(z_idx, k, b, net, bonds_off=bonds_off, psi_K=psi_K)
        if step % 100 == 0:
            z_idx, k, b = replace_random(z_idx, k, b, frac=0.05)
        idx = torch.randperm(hp["N_cloud"], device=device)[: hp["batch_size"]]
        R_K, R_B, FB = euler_residuals(z_idx[idx], k[idx], b[idx], net,
                                       bonds_off=bonds_off, psi_K=psi_K)
        if weight_schedule is not None:
            wK, wB, wFB = weight_schedule(step)
        else:
            wK, wB, wFB = capital_weight, bond_weight, fb_weight
        rk_mse = (R_K ** 2).mean()
        rb_mse = (R_B ** 2).mean()
        fb_mse = (FB ** 2).mean()
        total = wK * rk_mse + wB * rb_mse + wFB * fb_mse
        opt.zero_grad(); total.backward(); opt.step(); sched.step()
        rec.append((float(rk_mse.item()), float(rb_mse.item()), float(fb_mse.item()), float(total.item())))
        if verbose and (step % hp["log_every"] == 0 or step == 1):
            lr_now = sched.get_last_lr()[0]
            print(f"  [{log_label}] step {step:5d}   total = {total.item():.3e}   "
                  f"R_K = {rk_mse.item():.3e}  R_B = {rb_mse.item():.3e}  FB = {fb_mse.item():.3e}  "
                  f"lr = {lr_now:.2e}")
    return (z_idx, k, b), rec


def homotopy_run(
    *, hp_overrides: dict | None = None, verbose: bool = True,
) -> tuple[PolicyNet, dict]:
    """Run the full four-phase homotopy. Returns (net, history) where
    history has per-phase metadata and per-step residuals."""
    hp = {**HP, **(hp_overrides or {})}
    torch.manual_seed(hp["seed"]); np.random.seed(hp["seed"])
    net = PolicyNet(hidden=hp["hidden"]).to(device)
    pretrain(net, hp["pretrain_steps"])

    opt = optim.Adam(net.parameters(), lr=hp["lr"])
    sched = optim.lr_scheduler.ExponentialLR(opt, gamma=hp["lr_decay"])
    cloud = init_cloud(hp["N_cloud"])

    history: dict = {"phases": [], "all": []}

    # ---- Phase 1: capital only ----
    if verbose:
        print("=== Phase 1: capital-only training ===")
    cloud, rec1 = _train_block(
        net, opt, sched, cloud, hp, hp["phase1_steps"],
        capital_weight=hp["capital_weight"], bond_weight=0.0, fb_weight=0.0,
        bonds_off=True, log_label="P1", verbose=verbose,
    )
    history["phases"].append(dict(name="capital_only", steps=hp["phase1_steps"],
                                  start_total=rec1[0][3], end_total=np.mean([r[3] for r in rec1[-50:]])))
    history["all"].extend(rec1)

    # ---- Phase 2: bond pretraining (small bond weight, no FB) ----
    if verbose:
        print("=== Phase 2: bond pretraining (low weight) ===")
    cloud, rec2 = _train_block(
        net, opt, sched, cloud, hp, hp["phase2_steps"],
        capital_weight=hp["capital_weight"], bond_weight=0.1, fb_weight=0.0,
        bonds_off=False, log_label="P2", verbose=verbose,
    )
    history["phases"].append(dict(name="bond_pretraining", steps=hp["phase2_steps"],
                                  start_total=rec2[0][3], end_total=np.mean([r[3] for r in rec2[-50:]])))
    history["all"].extend(rec2)

    # ---- Phase 3: bond-weight homotopy + FB ramp ----
    if verbose:
        print("=== Phase 3: bond homotopy ===")
    n3 = hp["phase3_steps"]
    bond_target = hp["bond_weight"]
    fb_target = hp["fb_weight"]
    def schedule(step: int):
        t = min(1.0, step / max(1, n3))
        wK = hp["capital_weight"]
        wB = 0.1 + (bond_target - 0.1) * t
        wFB = fb_target * t
        return wK, wB, wFB
    cloud, rec3 = _train_block(
        net, opt, sched, cloud, hp, n3,
        capital_weight=hp["capital_weight"], bond_weight=bond_target, fb_weight=fb_target,
        bonds_off=False, weight_schedule=schedule, log_label="P3", verbose=verbose,
    )
    history["phases"].append(dict(name="bond_homotopy", steps=n3,
                                  start_total=rec3[0][3], end_total=np.mean([r[3] for r in rec3[-50:]])))
    history["all"].extend(rec3)

    # ---- Phase 4: fine-tuning at lower LR ----
    if verbose:
        print("=== Phase 4: fine-tuning ===")
    for g in opt.param_groups:
        g["lr"] = g["lr"] * hp["fine_tune_lr_factor"]
    cloud, rec4 = _train_block(
        net, opt, sched, cloud, hp, hp["phase4_steps"],
        capital_weight=hp["capital_weight"], bond_weight=hp["bond_weight"], fb_weight=hp["fb_weight"],
        bonds_off=False, log_label="P4", verbose=verbose,
    )
    history["phases"].append(dict(name="fine_tuning", steps=hp["phase4_steps"],
                                  start_total=rec4[0][3], end_total=np.mean([r[3] for r in rec4[-50:]])))
    history["all"].extend(rec4)

    return net, history


def run(*, hp_overrides=None, verbose=True):
    """Same signature as V0..V4 but returns (net, history) instead of (net, losses).

    Pull out the total-loss column to mimic the older API:
        losses = [row[3] for row in history['all']]
    """
    return homotopy_run(hp_overrides=hp_overrides, verbose=verbose)
