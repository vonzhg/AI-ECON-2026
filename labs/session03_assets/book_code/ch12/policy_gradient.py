"""
策略梯度与演员—评论家（离散行动，神经网络版）：20×2 增长模型，与动态规划解对照

对应章节：第13章《强化学习核心算法》
  - §13.4 策略梯度（sec:ch13_pg）：带基线的 REINFORCE，train_reinforce
  - §13.5 演员—评论家（sec:ch13_ac）：一步演员—评论家，train_actor_critic
  - 选读：简化版 PPO-Clip，train_ppo。正文只在 §13.5.3（sec:ch13_ppo）用一段话概述 PPO，
    不引用这里的代码；默认不运行，加 --ppo 才运行。

共同设定
--------
- 环境：growth_env.py 的 BatchGrowthEnv，即第12章 20×2 练习环境的 E 个独立副本，向量化推进。
  起点在网格上均匀随机；只有 terminated 才去掉自举项（本模型没有终止状态），截断时照常自举。
- 策略 g_θ(k'|s)：打分网络 h_θ(s,a) 以 (k̃, z̃, k̃') 为输入、输出一个分数，对 20 个 k' 求值后
  在可行集上做 softmax（带掩码 softmax：不可行 k' 的分数换成 -1e9）。把 k' 放进输入而不是
  “每个 k' 一个输出头”，相邻 k' 共享参数，策略沿 k' 方向光滑；试验中每个 k' 一个输出头的写法
  学得明显更差，两种方法都停在资本只积累到 2 附近的储蓄不足的规则上。
  输出层初始化为零，所以初始策略在可行集上均匀。
- 评论家（也是 REINFORCE 的基线）V̂_ψ(s)：输入 (k̃, z̃)。两个网络都是两个 64 单元 tanh 隐层。
- 熵正则系数 0.01，防止策略过早集中、探索停止；Adam 学习率在训练中线性降到初值的 5%。

REINFORCE（带基线）
------------------
- 每次迭代并行采样 E=16 条 T=100 期轨迹；θ 在一次迭代内不变，所以先对全部 40 个状态算出
  g_θ(·|s)，轨迹内查表抽样，与逐步前向传播等价。
- 回报用截断回报 G_t = Σ_{j=0}^{T-1-t} γ^j r_{t+j+1}，不自举：REINFORCE 本来就不用价值估计承担尾部。
- 梯度按策略梯度定理带 γ^t：Σ_t γ^t ∇log g_θ(a_t|s_t)[G_t - V̂_ψ(s_t)]。use_gamma_t=True 为默认；
  设为 False（命令行 --reinforce-no-gamma-t）时得到的方向不再是贴现目标 J(θ) 的梯度。
  T=100 时，被截掉的尾部在 γ^t 加权下的权重为 γ^T≈0.006。
- 优势除以本批的标准差（只改变步长的尺度）。基线用均方误差拟合 G_t，并按 γ^t 加权：
  梯度里权重最大的是轨迹前段的完整回报，被截得很短的后段回报不应左右基线。
- 演员学习率 2e-2，基线 6e-2，3,000 次迭代，共 480 万个转移。学习率取得偏大是有意的：
  初始的均匀策略很快变成“普遍多储蓄”的分散策略，梯度信号相对噪声很弱；学习率减半时，
  6 个种子里约一半在 3,000 次迭代内离不开这个状态（贪心策略 ℓ_CE 停在 19% 左右）。

一步演员—评论家
---------------
- E=16 个副本同步推进，每一步用各自的 TD 误差 δ = r + γ(1-terminated)·V̂_ψ(s') - V̂_ψ(s)
  更新一次，对 E 条转移取平均；E=1 就是正文算法的逐步在线更新。评论家一步是半梯度 TD，
  演员一步用同一个 δ（对演员是常数）。
- 每段 20 期后截断并随机重启。截断照常自举，段长不引入偏差，只决定多久换一次起点、
  评论家能看到多大范围的状态。段长取 100 时评论家几乎只见到当前策略路径上的状态，
  种子 0、1、2 都没有学出来：两个停在普遍多储蓄的分散策略上（贪心策略 ℓ_CE 19%），
  一个停在把资本全部吃掉的策略上。
- use_gamma_t 默认 False，原因有二。其一，对每个状态一组参数（表格型 softmax）时，
  γ^t 只是按时刻缩放该状态这一步的步长，不改变各状态参数块内的更新方向与驻点；
  其二，与正文的一步演员—评论家算法一致。这里的网络在状态之间共享参数，省略 γ^t 会改变
  各状态梯度的相对权重，得到的方向并非严格的 J(θ) 梯度（Thomas 2014；Nota & Thomas 2020），
  这是持续在线学习的通行做法；本环境每 20 期重启一次，γ^t 中的 t 从重启时刻算起，并无经济含义。
- 演员学习率 3e-4，评论家 3e-3（评论家快于演员）；30,000 次更新，共 48 万个转移。

评估（训练结束后与 DP 基准对照）
------------------------------
- 贪心策略（每个状态取概率最大的 k'）：与 g* 相同的状态数，ℓ_CE 的全部状态平均与加权平均；
- 随机策略本身（按 g_θ 抽样执行）的 ℓ_CE，由线性方程组精确求出；
- 对照：均匀随机策略的 ℓ_CE。“加权”指按 g* 上 ε=0.1 的 ε-贪心策略的平稳分布加权。

依赖：numpy + pytorch（单线程）。
运行（在书稿目录下）：OMP_NUM_THREADS=1 python3 -B code/ch13/policy_gradient.py [--seed 0] [--reinforce-no-gamma-t] [--ppo]
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")   # 须在导入 torch 之前

import numpy as np
import torch
import torch.nn as nn

from growth_env import BatchGrowthEnv, GrowthBenchmark, format_rule, make_batch_env, make_env

torch.set_num_threads(1)
_MASKED_SCORE = -1e9


# --------------------------------------------------------------------------- #
# 网络与策略
# --------------------------------------------------------------------------- #
class ScoreNet(nn.Module):
    """打分网络 h_θ(s, a)：输入 (k̃, z̃, k̃')，输出一个分数；输出层初始化为零。"""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_state: torch.Tensor, k_next: torch.Tensor) -> torch.Tensor:
        """x_state: (B, 2)；k_next: (A,) 为全部 k' 的 k̃'。返回 (B, A) 的分数。"""
        B, A = x_state.shape[0], k_next.shape[0]
        inputs = torch.cat([x_state.unsqueeze(1).expand(B, A, 2), k_next.view(1, A, 1).expand(B, A, 1)], dim=-1)
        return self.net(inputs).squeeze(-1)


class CriticNet(nn.Module):
    """价值网络 V̂_ψ(s)：输入 (k̃, z̃)，输出标量。"""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GrowthPolicy:
    """演员（打分网络）、评论家与状态特征、可行集掩码放在一起；状态用展平索引 s = z*num_k + k。"""

    def __init__(self, env: BatchGrowthEnv, hidden_dim: int = 64) -> None:
        mdp = env.mdp
        k_tilde = (mdp.k_grid - mdp.k_grid[0]) / (mdp.k_grid[-1] - mdp.k_grid[0])
        z_tilde = np.arange(env.num_z) / max(env.num_z - 1, 1)
        s_all = np.arange(env.num_states)
        self.x_state = torch.tensor(np.stack([k_tilde[s_all % env.num_k], z_tilde[s_all // env.num_k]], axis=1),
                                    dtype=torch.float32)
        self.k_next = torch.tensor(k_tilde, dtype=torch.float32)
        mask = np.zeros((env.num_states, env.num_actions), dtype=bool)
        for s in s_all:
            mask[s] = mdp._feasible[s % env.num_k, s // env.num_k]
        self.mask = torch.from_numpy(mask)
        self.actor = ScoreNet(hidden_dim)
        self.critic = CriticNet(hidden_dim)

    def log_probs(self, s: torch.Tensor) -> torch.Tensor:
        """log g_θ(·|s)，带掩码 softmax；s 为 LongTensor (B,)，返回 (B, A)。"""
        scores = self.actor(self.x_state[s], self.k_next).masked_fill(~self.mask[s], _MASKED_SCORE)
        return torch.log_softmax(scores, dim=-1)

    def values(self, s: torch.Tensor) -> torch.Tensor:
        return self.critic(self.x_state[s])

    def entropy(self, logp: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return -(logp.exp() * logp.masked_fill(~self.mask[s], 0.0)).sum(dim=-1)

    def prob_table(self) -> np.ndarray:
        """全部状态上的 g_θ(·|s)（float64，不可行处为 0，每行和为 1）。"""
        with torch.no_grad():
            probs = self.log_probs(torch.arange(self.mask.shape[0])).exp().double().numpy()
        probs[~self.mask.numpy()] = 0.0
        return probs / probs.sum(axis=1, keepdims=True)


def sample_actions(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """逐行按概率抽一个行动（逆 CDF）；零概率（不可行）的行动不会被抽中。"""
    cdf = np.cumsum(probs, axis=1)
    cdf /= cdf[:, -1:]
    u = rng.random(probs.shape[0])
    return (cdf < u[:, None]).sum(axis=1)


def set_learning_rate(optimizers, base_lrs, fraction_done: float, final_fraction: float) -> None:
    scale = max(final_fraction, 1.0 - fraction_done)
    for opt, lr in zip(optimizers, base_lrs):
        for group in opt.param_groups:
            group["lr"] = lr * scale


def evaluate_policy(policy: GrowthPolicy, bench: GrowthBenchmark, weights: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    probs = policy.prob_table()
    g = probs.argmax(axis=1)
    metrics = bench.evaluate_greedy(g, weights)
    loss_stochastic = bench.ce_loss(bench.policy_value_stochastic(probs))
    metrics["ce_stochastic_all"] = float(loss_stochastic.mean())
    metrics["ce_stochastic_weighted"] = float(weights @ loss_stochastic)
    return metrics, g


def progress_line(label: str, metrics: Dict[str, float], seconds: float) -> str:
    return (f"  {label}（{seconds:3.0f} 秒）：贪心策略与 g* 相同 {metrics['agree']:2d}/40，"
            f"ℓ_CE 全部 {100 * metrics['ce_all']:.2f}%、加权 {100 * metrics['ce_weighted']:.2f}%；"
            f"随机策略本身 ℓ_CE 加权 {100 * metrics['ce_stochastic_weighted']:.2f}%")


# --------------------------------------------------------------------------- #
# REINFORCE（带基线）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReinforceConfig:
    num_iterations: int = 3000
    num_envs: int = 16
    horizon: int = 100
    lr_actor: float = 2e-2
    lr_critic: float = 6e-2
    entropy_coef: float = 0.01
    use_gamma_t: bool = True
    baseline_gamma_t: bool = True    # 基线的均方误差也按 γ^t 加权
    normalize_advantage: bool = True
    lr_final_fraction: float = 0.05
    hidden_dim: int = 64
    seed: int = 0


def train_reinforce(cfg: ReinforceConfig, bench: Optional[GrowthBenchmark] = None,
                    weights: Optional[np.ndarray] = None, report_every: int = 0) -> GrowthPolicy:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    env = make_batch_env(cfg.num_envs, seed=1000 + cfg.seed, episode_length=cfg.horizon)
    policy = GrowthPolicy(env, cfg.hidden_dim)
    opt_actor = torch.optim.Adam(policy.actor.parameters(), lr=cfg.lr_actor)
    opt_critic = torch.optim.Adam(policy.critic.parameters(), lr=cfg.lr_critic)
    gamma, E, T = env.gamma, cfg.num_envs, cfg.horizon
    time_weight = torch.tensor([gamma ** t if cfg.use_gamma_t else 1.0 for t in range(T)], dtype=torch.float32)
    time_weight = time_weight.repeat_interleave(E)          # 与 (T, E) 按行展平的顺序一致

    t0 = time.time()
    for it in range(cfg.num_iterations):
        set_learning_rate((opt_actor, opt_critic), (cfg.lr_actor, cfg.lr_critic),
                          it / cfg.num_iterations, cfg.lr_final_fraction)
        # 1. 用当前 θ 并行采样 E 条 T 期轨迹
        probs = policy.prob_table()
        states = np.empty((T, E), dtype=np.int64)
        actions = np.empty((T, E), dtype=np.int64)
        rewards = np.empty((T, E))
        s = env.reset()
        for t in range(T):
            a = sample_actions(probs[s], rng)
            s_next, r, terminated, truncated = env.step(a)
            states[t], actions[t], rewards[t] = s, a, r
            s = s_next
        # 所有副本都在第 T 期被截断（terminated 恒为 False）；REINFORCE 不为截掉的尾部补价值

        # 2. 截断回报 G_t
        returns = np.empty((T, E))
        acc = np.zeros(E)
        for t in reversed(range(T)):
            acc = rewards[t] + gamma * acc
            returns[t] = acc

        # 3. 损失。同一状态在批中重复出现，网络只对不重复的状态计算一次（梯度完全相同）
        S = torch.from_numpy(states.reshape(-1))
        A = torch.from_numpy(actions.reshape(-1))
        G = torch.tensor(returns.reshape(-1), dtype=torch.float32)
        uniq, inverse = torch.unique(S, return_inverse=True)
        logp = policy.log_probs(uniq)[inverse]
        v = policy.values(uniq)[inverse]
        advantage = (G - v).detach()                          # 基线对演员是常数
        if cfg.normalize_advantage:
            advantage = advantage / (advantage.std() + 1e-8)
        logp_a = logp.gather(1, A.view(-1, 1)).squeeze(1)
        entropy = policy.entropy(logp, S)
        loss_actor = -(time_weight * (logp_a * advantage + cfg.entropy_coef * entropy)).sum() / E
        if cfg.baseline_gamma_t:
            loss_critic = (time_weight * (G - v).pow(2)).sum() / time_weight.sum()
        else:
            loss_critic = (G - v).pow(2).mean()

        opt_critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()
        opt_actor.zero_grad()
        loss_actor.backward()
        opt_actor.step()

        if report_every and (it + 1) % report_every == 0 and bench is not None and weights is not None:
            m, _ = evaluate_policy(policy, bench, weights)
            print(progress_line(f"迭代 {it + 1:>5,d}", m, time.time() - t0), flush=True)
    return policy


# --------------------------------------------------------------------------- #
# 一步演员—评论家（同步并行的在线 TD(0)）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ActorCriticConfig:
    num_updates: int = 30_000
    num_envs: int = 16
    segment_length: int = 20
    lr_actor: float = 3e-4
    lr_critic: float = 3e-3
    entropy_coef: float = 0.01
    use_gamma_t: bool = False     # 见模块说明：持续在线学习默认不乘 γ^t
    lr_final_fraction: float = 0.05
    hidden_dim: int = 64
    seed: int = 0


def train_actor_critic(cfg: ActorCriticConfig, bench: Optional[GrowthBenchmark] = None,
                       weights: Optional[np.ndarray] = None, report_every: int = 0) -> GrowthPolicy:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    env = make_batch_env(cfg.num_envs, seed=2000 + cfg.seed, episode_length=cfg.segment_length)
    policy = GrowthPolicy(env, cfg.hidden_dim)
    opt_actor = torch.optim.Adam(policy.actor.parameters(), lr=cfg.lr_actor)
    opt_critic = torch.optim.Adam(policy.critic.parameters(), lr=cfg.lr_critic)
    gamma = env.gamma

    s = env.reset()
    discount_since_reset = np.ones(cfg.num_envs)              # 只在 use_gamma_t=True 时使用
    t0 = time.time()
    for it in range(cfg.num_updates):
        set_learning_rate((opt_actor, opt_critic), (cfg.lr_actor, cfg.lr_critic),
                          it / cfg.num_updates, cfg.lr_final_fraction)
        S = torch.from_numpy(s)
        logp = policy.log_probs(S)                             # 保留计算图（演员）
        probs = logp.detach().exp().double().numpy()
        a = sample_actions(probs / probs.sum(axis=1, keepdims=True), rng)
        s_next, r, terminated, truncated = env.step(a)

        v = policy.values(S)                                   # 保留计算图（评论家）
        with torch.no_grad():                                  # 目标里的 V̂(s') 当常数：半梯度
            v_next = policy.values(torch.from_numpy(s_next))
            target = (torch.tensor(r, dtype=torch.float32)
                      + gamma * torch.tensor(1.0 - terminated, dtype=torch.float32) * v_next)  # 截断照常自举
        delta = (target - v).detach()                          # 对演员是常数

        weight = torch.tensor(discount_since_reset if cfg.use_gamma_t else np.ones(cfg.num_envs), dtype=torch.float32)
        logp_a = logp.gather(1, torch.from_numpy(a).view(-1, 1)).squeeze(1)
        loss_actor = -(weight * (logp_a * delta + cfg.entropy_coef * policy.entropy(logp, S))).mean()
        loss_critic = (target - v).pow(2).mean()

        opt_actor.zero_grad()
        opt_critic.zero_grad()
        (loss_actor + loss_critic).backward()                  # 两个网络参数不重叠，合在一起反传
        opt_actor.step()
        opt_critic.step()

        done = terminated | truncated
        discount_since_reset = gamma * discount_since_reset
        if done.any():
            s = env.reset(done)                                # 只重启到期的副本
            discount_since_reset[done] = 1.0
        else:
            s = s_next

        if report_every and (it + 1) % report_every == 0 and bench is not None and weights is not None:
            m, _ = evaluate_policy(policy, bench, weights)
            print(progress_line(f"更新 {it + 1:>6,d}", m, time.time() - t0), flush=True)
    return policy


# --------------------------------------------------------------------------- #
# 选读：简化版 PPO-Clip（正文不引用；--ppo 时运行）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PPOConfig:
    num_iterations: int = 300
    num_envs: int = 16
    rollout_length: int = 64      # 每次迭代每个副本收集的步数
    segment_length: int = 20
    lam: float = 0.95             # GAE 参数
    clip_eps: float = 0.2
    lr: float = 1e-3
    num_epochs: int = 4
    minibatch_size: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    lr_final_fraction: float = 0.05
    hidden_dim: int = 64
    seed: int = 0


def train_ppo(cfg: PPOConfig, bench: Optional[GrowthBenchmark] = None,
              weights: Optional[np.ndarray] = None, report_every: int = 0) -> GrowthPolicy:
    """
    简化版 PPO-Clip：用 θ_old 收集一批转移，GAE 估计优势，再对同一批数据做若干轮小批量更新。
    剪切目标由 torch.min(unclipped, clipped) 实现，比率 ρ_t(θ) = g_θ(a_t|s_t)/g_θ_old(a_t|s_t)。
    截断与终止分开：δ_t 在截断处用 V̂(s_{t+1}) 自举；GAE 的递推在每段轨迹结束处断开。
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    env = make_batch_env(cfg.num_envs, seed=3000 + cfg.seed, episode_length=cfg.segment_length)
    policy = GrowthPolicy(env, cfg.hidden_dim)
    params = list(policy.actor.parameters()) + list(policy.critic.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    gamma, E, L = env.gamma, cfg.num_envs, cfg.rollout_length

    s = env.reset()
    t0 = time.time()
    for it in range(cfg.num_iterations):
        set_learning_rate((optimizer,), (cfg.lr,), it / cfg.num_iterations, cfg.lr_final_fraction)
        # 1. 用 θ_old 收集 L 步（θ 固定，先算概率表与价值表）
        probs = policy.prob_table()
        with torch.no_grad():
            value_table = policy.values(torch.arange(env.num_states)).numpy().astype(float)
        states = np.empty((L, E), dtype=np.int64)
        actions = np.empty((L, E), dtype=np.int64)
        rewards = np.empty((L, E))
        not_terminated = np.empty((L, E))
        not_done = np.empty((L, E))
        next_values = np.empty((L, E))
        for t in range(L):
            a = sample_actions(probs[s], rng)
            s_next, r, terminated, truncated = env.step(a)
            states[t], actions[t], rewards[t] = s, a, r
            not_terminated[t] = 1.0 - terminated
            done = terminated | truncated
            not_done[t] = 1.0 - done
            next_values[t] = value_table[s_next]              # 重置之前的下一期状态
            s = env.reset(done) if done.any() else s_next
        logp_old = np.log(probs[states, actions])

        # 2. GAE 优势与价值目标
        deltas = rewards + gamma * not_terminated * next_values - value_table[states]
        adv = np.zeros((L, E))
        running = np.zeros(E)
        for t in reversed(range(L)):
            running = deltas[t] + gamma * cfg.lam * not_done[t] * running
            adv[t] = running
        value_targets = adv + value_table[states]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        S = torch.from_numpy(states.reshape(-1))
        A = torch.from_numpy(actions.reshape(-1))
        LP_OLD = torch.tensor(logp_old.reshape(-1), dtype=torch.float32)
        ADV = torch.tensor(adv.reshape(-1), dtype=torch.float32)
        RET = torch.tensor(value_targets.reshape(-1), dtype=torch.float32)

        # 3. 多轮小批量优化同一批数据
        n = S.shape[0]
        for _ in range(cfg.num_epochs):
            perm = torch.from_numpy(rng.permutation(n))
            for start in range(0, n, cfg.minibatch_size):
                b = perm[start:start + cfg.minibatch_size]
                logp = policy.log_probs(S[b])
                logp_a = logp.gather(1, A[b].view(-1, 1)).squeeze(1)
                ratio = torch.exp(logp_a - LP_OLD[b])
                unclipped = ratio * ADV[b]
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * ADV[b]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = (policy.values(S[b]) - RET[b]).pow(2).mean()
                entropy = policy.entropy(logp, S[b]).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if report_every and (it + 1) % report_every == 0 and bench is not None and weights is not None:
            m, _ = evaluate_policy(policy, bench, weights)
            print(progress_line(f"迭代 {it + 1:>4d}", m, time.time() - t0), flush=True)
    return policy


# --------------------------------------------------------------------------- #
# 主程序
# --------------------------------------------------------------------------- #
def report_final(name: str, policy: GrowthPolicy, bench: GrowthBenchmark, weights: np.ndarray, seconds: float) -> None:
    m, g = evaluate_policy(policy, bench, weights)
    print(f"[{name}] 贪心策略与 g* 相同 {m['agree']}/40（储蓄偏少 {m['less']}、偏多 {m['more']}，最大偏离 {m['max_dev']} 格）；"
          f"ℓ_CE 全部 {100 * m['ce_all']:.3f}%，加权 {100 * m['ce_weighted']:.3f}%；"
          f"随机策略本身 ℓ_CE 全部 {100 * m['ce_stochastic_all']:.2f}%，加权 {100 * m['ce_stochastic_weighted']:.2f}%；"
          f"用时 {seconds:.0f} 秒")
    print("  贪心 k' 索引（第一行 z_L，第二行 z_H）：")
    for line in format_rule(g, bench.num_k, bench.num_z):
        print("    " + line)


def main() -> None:
    parser = argparse.ArgumentParser(description="20×2 增长模型上的 REINFORCE 与一步演员—评论家")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reinforce-no-gamma-t", action="store_true", help="REINFORCE 不乘 γ^t（对照用）")
    parser.add_argument("--ppo", action="store_true", help="另外运行选读的简化版 PPO")
    args = parser.parse_args()

    t_start = time.time()
    bench = GrowthBenchmark(make_env(seed=0))
    mu = bench.stationary_distribution(bench.epsilon_greedy_matrix(bench.g_star, 0.1))

    print("=" * 78)
    print("策略梯度与演员—评论家：20×2 增长模型（第12章练习环境），与值函数迭代的 g* 对照")
    print("=" * 78)
    loss_u = bench.ce_loss(bench.policy_value_stochastic(bench.uniform_random_matrix()))
    print(f"对照：均匀随机策略 ℓ_CE 全部 {100 * loss_u.mean():.2f}%，加权 {100 * (mu @ loss_u):.2f}%"
          f"（也是两种方法的初始策略）；种子 {args.seed}")
    print("g* 的 k' 索引（第一行 z_L，第二行 z_H）：")
    for line in format_rule(bench.g_star, bench.num_k, bench.num_z):
        print("    " + line)

    rcfg = ReinforceConfig(seed=args.seed, use_gamma_t=not args.reinforce_no_gamma_t)
    print(f"\nREINFORCE（带基线；{'乘' if rcfg.use_gamma_t else '不乘'} γ^t）：每次迭代 {rcfg.num_envs} 条 {rcfg.horizon} 期轨迹，"
          f"{rcfg.num_iterations} 次迭代；演员学习率 {rcfg.lr_actor}、基线 {rcfg.lr_critic}，线性衰减")
    t0 = time.time()
    policy_r = train_reinforce(rcfg, bench, mu, report_every=rcfg.num_iterations // 6)
    report_final("REINFORCE", policy_r, bench, mu, time.time() - t0)

    acfg = ActorCriticConfig(seed=args.seed)
    print(f"\n一步演员—评论家（{acfg.num_envs} 个副本同步，每段 {acfg.segment_length} 期、截断照常自举）：{acfg.num_updates:,d} 次更新；"
          f"演员学习率 {acfg.lr_actor}、评论家 {acfg.lr_critic}，线性衰减")
    t0 = time.time()
    policy_ac = train_actor_critic(acfg, bench, mu, report_every=acfg.num_updates // 6)
    report_final("演员—评论家", policy_ac, bench, mu, time.time() - t0)

    if args.ppo:
        pcfg = PPOConfig(seed=args.seed)
        print(f"\n选读：简化版 PPO-Clip（{pcfg.num_envs} 个副本 × {pcfg.rollout_length} 步/迭代，{pcfg.num_iterations} 次迭代，"
              f"GAE λ={pcfg.lam}，剪切 {pcfg.clip_eps}）")
        t0 = time.time()
        policy_ppo = train_ppo(pcfg, bench, mu, report_every=pcfg.num_iterations // 6)
        report_final("PPO（选读）", policy_ppo, bench, mu, time.time() - t0)

    print(f"\n总耗时 {time.time() - t_start:.0f} 秒")


if __name__ == "__main__":
    main()
