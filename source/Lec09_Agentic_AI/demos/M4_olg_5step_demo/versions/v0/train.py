"""V0 training — pretraining + Euler-residual loss on a cloud."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.optim as optim

from model import (
    P, P_MAT, Z_VALS, cohort_decisions, device, init_cloud, prices,
    replace_random, step_cloud,
)
from network import PolicyNet


HP: dict = dict(
    hidden=64,
    N_cloud=512,
    batch_size=256,
    lr=1e-3,
    lr_decay=0.9998,
    n_steps=5000,
    pretrain_steps=400,
    pretrain_target=(0.30, 0.50),
    log_every=500,
    seed=0,
)


def euler_residuals(
    z_idx: torch.Tensor, am: torch.Tensor, ao: torch.Tensor, net: PolicyNet,
) -> tuple[torch.Tensor, torch.Tensor]:
    Z = Z_VALS[z_idx]
    sy, sm = net(Z, am, ao)
    today = cohort_decisions(Z, am, ao, sy, sm)

    rhs_y = torch.zeros_like(today["cy"])
    rhs_m = torch.zeros_like(today["cm"])
    n_z = Z_VALS.shape[0]
    for jz in range(n_z):
        Zp = Z_VALS[jz].expand_as(Z)
        Kp = today["am_next"] + today["ao_next"]
        rp, wp = prices(Zp, Kp)
        co_p = (1 + rp) * today["ao_next"]
        sy_p, sm_p = net(Zp, today["am_next"], today["ao_next"])
        inc_m_p = wp * P["eps_m"] + (1 + rp) * today["am_next"]
        cm_p = (1 - sm_p) * inc_m_p
        prob = P_MAT[z_idx, jz]
        rhs_y = rhs_y + prob * (1 + rp) * cm_p.pow(-P["gamma"])
        rhs_m = rhs_m + prob * (1 + rp) * co_p.pow(-P["gamma"])

    mu_cy = today["cy"].pow(-P["gamma"])
    mu_cm = today["cm"].pow(-P["gamma"])
    R_y = 1 - P["beta"] * rhs_y / mu_cy
    R_m = 1 - P["beta"] * rhs_m / mu_cm
    return R_y, R_m


def pretrain(net: PolicyNet, n_steps: int, target: tuple[float, float], lr: float = 3e-3) -> float:
    opt = optim.Adam(net.parameters(), lr=lr)
    tgt_y, tgt_m = target
    last = float("nan")
    for _ in range(n_steps):
        z_idx, am, ao = init_cloud(HP["batch_size"])
        Z = Z_VALS[z_idx]
        sy, sm = net(Z, am, ao)
        loss = ((sy - tgt_y) ** 2 + (sm - tgt_m) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.item())
    return last


def train(net: PolicyNet, hp: dict | None = None, *, verbose: bool = True) -> list[float]:
    hp = {**HP, **(hp or {})}
    z_idx, am, ao = init_cloud(hp["N_cloud"])
    opt = optim.Adam(net.parameters(), lr=hp["lr"])
    sched = optim.lr_scheduler.ExponentialLR(opt, gamma=hp.get("lr_decay", 1.0))
    losses: list[float] = []
    for step in range(1, hp["n_steps"] + 1):
        z_idx, am, ao = step_cloud(z_idx, am, ao, net)
        if step % 100 == 0:
            z_idx, am, ao = replace_random(z_idx, am, ao, frac=0.05)
        idx = torch.randperm(hp["N_cloud"], device=device)[: hp["batch_size"]]
        R_y, R_m = euler_residuals(z_idx[idx], am[idx], ao[idx], net)
        loss = (R_y ** 2 + R_m ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
        if verbose and (step % hp["log_every"] == 0 or step == 1):
            rms = math.sqrt(loss.item() + 1e-30)
            lr = sched.get_last_lr()[0]
            print(f"step {step:5d}   MSE = {loss.item():.4e}   "
                  f"log10|EE|_RMS = {math.log10(rms):.2f}   lr = {lr:.2e}")
    return losses


def run(*, hp_overrides: dict | None = None, verbose: bool = True) -> tuple[PolicyNet, list[float]]:
    """Default training entrypoint: pretrain + train."""
    hp = {**HP, **(hp_overrides or {})}
    torch.manual_seed(hp["seed"])
    np.random.seed(hp["seed"])
    net = PolicyNet(hidden=hp["hidden"]).to(device)
    pretrain(net, hp["pretrain_steps"], hp["pretrain_target"])
    losses = train(net, hp, verbose=verbose)
    return net, losses
