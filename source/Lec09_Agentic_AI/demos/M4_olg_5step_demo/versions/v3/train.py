"""V3 training — capital + bond Euler residuals + Fischer–Burmeister borrowing penalty."""
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
    n_steps=8000,
    pretrain_steps=800,
    pretrain_target_sK=0.40,
    pretrain_target_pb=0.80,
    capital_weight=1.0,
    bond_weight=1.0,
    fb_weight=0.5,            # weight on Fischer–Burmeister borrowing-limit residual
    log_every=500,
    seed=0,
    bonds_off=False,
)


def euler_residuals(z_idx: torch.Tensor, k: torch.Tensor, b: torch.Tensor, net: PolicyNet,
                    *, bonds_off: bool = False):
    """Return (R_K, R_B, FB) each of shape (M, N-1)."""
    Z = Z_VALS[z_idx]
    s_K, b_next, p_b = net(Z, k, b)
    today = cohort_decisions(Z, k, b, s_K, b_next, p_b, bonds_off=bonds_off)

    n_z = Z_VALS.shape[0]
    rhs_K = torch.zeros_like(s_K)
    rhs_B = torch.zeros_like(s_K)
    for jz in range(n_z):
        Zp = Z_VALS[jz].expand_as(Z)
        k_p = today["k_next"]
        b_p = today["b_next"]
        s_K_p, b_next_p, p_b_p = net(Zp, k_p, b_p)
        out_p = cohort_decisions(Zp, k_p, b_p, s_K_p, b_next_p, p_b_p, bonds_off=bonds_off)
        c_next = out_p["c"][..., 1:N]                                  # cohort age j+1 tomorrow
        rp = out_p["r"]
        prob = P_MAT[z_idx, jz].unsqueeze(-1)
        mu_next = c_next.pow(-P["gamma"])
        rhs_K = rhs_K + prob * (1 + rp.unsqueeze(-1)) * mu_next
        rhs_B = rhs_B + prob * mu_next

    mu_today = today["c"][..., :N - 1].pow(-P["gamma"])
    R_K = 1 - P["beta"] * rhs_K / mu_today
    if bonds_off:
        R_B = torch.zeros_like(R_K)
    else:
        # Bond Euler:  p_b · u'(c) = β · E[u'(c')]
        R_B = 1 - P["beta"] * rhs_B / (p_b.unsqueeze(-1) * mu_today)

    # Fischer–Burmeister on the borrowing limit  b_next ≥ b_min.
    # We pair the slack (b_next - b_min) with |R_B| as a proxy multiplier.
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
             + 0.1 * (b_next ** 2).mean()      # bonds start near zero
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.item())
    return last


def train(net: PolicyNet, hp: dict | None = None, *, verbose: bool = True) -> list[float]:
    hp = {**HP, **(hp or {})}
    z_idx, k, b = init_cloud(hp["N_cloud"])
    opt = optim.Adam(net.parameters(), lr=hp["lr"])
    sched = optim.lr_scheduler.ExponentialLR(opt, gamma=hp.get("lr_decay", 1.0))
    losses: list[float] = []
    bonds_off = hp.get("bonds_off", False)
    for step in range(1, hp["n_steps"] + 1):
        z_idx, k, b = step_cloud(z_idx, k, b, net, bonds_off=bonds_off)
        if step % 100 == 0:
            z_idx, k, b = replace_random(z_idx, k, b, frac=0.05)
        idx = torch.randperm(hp["N_cloud"], device=device)[: hp["batch_size"]]
        R_K, R_B, FB = euler_residuals(z_idx[idx], k[idx], b[idx], net, bonds_off=bonds_off)
        loss = (
            hp["capital_weight"] * (R_K ** 2).mean()
            + hp["bond_weight"] * (R_B ** 2).mean()
            + hp["fb_weight"] * (FB ** 2).mean()
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
        if verbose and (step % hp["log_every"] == 0 or step == 1):
            rms = math.sqrt(loss.item() + 1e-30)
            lr = sched.get_last_lr()[0]
            print(f"step {step:5d}   total = {loss.item():.4e}   "
                  f"R_K = {(R_K**2).mean().item():.3e}   R_B = {(R_B**2).mean().item():.3e}   "
                  f"FB = {(FB**2).mean().item():.3e}   lr = {lr:.2e}")
    return losses


def run(*, hp_overrides: dict | None = None, verbose: bool = True) -> tuple[PolicyNet, list[float]]:
    hp = {**HP, **(hp_overrides or {})}
    torch.manual_seed(hp["seed"])
    np.random.seed(hp["seed"])
    net = PolicyNet(hidden=hp["hidden"]).to(device)
    pretrain(net, hp["pretrain_steps"])
    losses = train(net, hp, verbose=verbose)
    return net, losses
