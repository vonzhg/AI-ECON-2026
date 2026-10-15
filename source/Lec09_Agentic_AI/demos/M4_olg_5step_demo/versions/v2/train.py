"""V2 training — euler-residual loop runs over n_tfp states (4 by default)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.optim as optim

from model import (
    N, P, P_MAT, Z_VALS, cohort_decisions, device, init_cloud, replace_random, step_cloud,
)
from network import PolicyNet


HP: dict = dict(
    hidden=128,
    N_cloud=512,
    batch_size=256,
    lr=1e-3,
    lr_decay=0.9998,
    n_steps=6000,
    pretrain_steps=600,
    pretrain_target=0.40,
    log_every=500,
    seed=0,
)


def euler_residuals(z_idx: torch.Tensor, a: torch.Tensor, net: PolicyNet) -> torch.Tensor:
    Z = Z_VALS[z_idx]
    s = net(Z, a)
    today = cohort_decisions(Z, a, s)

    n_z = Z_VALS.shape[0]
    rhs = torch.zeros_like(s)
    for jz in range(n_z):
        Zp = Z_VALS[jz].expand_as(Z)
        a_p = today["a_next"]
        s_p = net(Zp, a_p)
        out_p = cohort_decisions(Zp, a_p, s_p)
        c_next = out_p["c"][..., 1:N]
        prob = P_MAT[z_idx, jz].unsqueeze(-1)
        rhs = rhs + prob * (1 + out_p["r"].unsqueeze(-1)) * c_next.pow(-P["gamma"])

    mu_today = today["c"][..., :N - 1].pow(-P["gamma"])
    return 1 - P["beta"] * rhs / mu_today


def pretrain(net: PolicyNet, n_steps: int, target: float, lr: float = 3e-3) -> float:
    opt = optim.Adam(net.parameters(), lr=lr)
    last = float("nan")
    tgt = torch.tensor(target, device=device)
    for _ in range(n_steps):
        z_idx, a = init_cloud(HP["batch_size"])
        Z = Z_VALS[z_idx]
        s = net(Z, a)
        loss = ((s - tgt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.item())
    return last


def train(net: PolicyNet, hp: dict | None = None, *, verbose: bool = True) -> list[float]:
    hp = {**HP, **(hp or {})}
    z_idx, a = init_cloud(hp["N_cloud"])
    opt = optim.Adam(net.parameters(), lr=hp["lr"])
    sched = optim.lr_scheduler.ExponentialLR(opt, gamma=hp.get("lr_decay", 1.0))
    losses: list[float] = []
    for step in range(1, hp["n_steps"] + 1):
        z_idx, a = step_cloud(z_idx, a, net)
        if step % 100 == 0:
            z_idx, a = replace_random(z_idx, a, frac=0.05)
        idx = torch.randperm(hp["N_cloud"], device=device)[: hp["batch_size"]]
        R = euler_residuals(z_idx[idx], a[idx], net)
        loss = (R ** 2).mean()
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
    hp = {**HP, **(hp_overrides or {})}
    torch.manual_seed(hp["seed"])
    np.random.seed(hp["seed"])
    net = PolicyNet(hidden=hp["hidden"]).to(device)
    pretrain(net, hp["pretrain_steps"], hp["pretrain_target"])
    losses = train(net, hp, verbose=verbose)
    return net, losses
