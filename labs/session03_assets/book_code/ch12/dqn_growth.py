"""
最小版深度 Q 网络（DQN）：20×2 增长模型，与动态规划解对照

对应章节：第13章《强化学习核心算法》§13.3.2 神经网络与 DQN（subsec:ch13_nn_dqn）

要点
----
- Q 网络 Q_ψ(s,·)：输入 (k̃, z̃)，两个 64 单元 ReLU 隐层，输出 20 个 k' 的 Q 值；
  不可行的 k' 在 argmax 与 max 中一律屏蔽（动作掩码）。
- 经验回放：容量 50,000 的环形缓冲区；每走一个环境步，从中均匀抽 64 条转移做一次梯度步。
- 目标网络：Q_{ψ⁻} 每 500 个梯度步与 Q_ψ 同步一次，
  TD 目标 y = r + γ(1 - terminated)·max_{a'∈A(s')} Q_{ψ⁻}(s', a')。
  每段轨迹 100 期后截断并重新随机起点；截断时照常自举，缓冲区只记录 terminated（本模型恒为 False）。
- 损失：Huber（SmoothL1），梯度范数裁剪到 10；Adam 学习率 1e-3，训练中线性降到初值的 5%，
  使最终的贪心策略不再随最后几批样本来回跳动。
- 探索：ε 在前一半训练步内从 1.0 线性降到 0.1，此后保持 0.1；衰减在训练预算之内完成。
- 默认训练 100,000 个环境步（1,000 段），种子 0。

评估（训练结束后与 DP 基准对照）
------------------------------
- 贪心策略与值函数迭代的 g* 相同的状态数、最大偏离格点数；
- 消费等价损失 ℓ_CE：全部 40 个状态的平均，以及按 g* 上 ε=0.1 的 ε-贪心策略的平稳分布加权的平均；
- 对照组：真正的均匀随机策略（每期在可行 k' 中均匀抽取），其 ℓ_CE 由线性方程组精确求出。

依赖：numpy + pytorch（单线程）。
运行（在书稿目录下）：OMP_NUM_THREADS=1 python3 -B code/ch13/dqn_growth.py [--seed 0] [--steps 100000]
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")   # 须在导入 torch 之前；小网络多线程只会更慢

import numpy as np
import torch
import torch.nn as nn

from growth_env import GrowthBenchmark, GrowthEnv, format_rule, make_env

torch.set_num_threads(1)


def state_features(state: Tuple[int, int], env: GrowthEnv) -> np.ndarray:
    """(k̃, z̃) ∈ [0,1]²。"""
    return np.array(env.features01(state), dtype=np.float32)


class QNetwork(nn.Module):
    """两层全连接 Q 网络：输出维度 = 行动数 = 资本网格点数。"""

    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """固定容量的环形经验回放缓冲区（numpy 数组存储，满了以后覆盖最旧的转移）。"""

    def __init__(self, capacity: int, obs_dim: int, num_actions: int) -> None:
        self.capacity = capacity
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s_next = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.mask_next = np.zeros((capacity, num_actions), dtype=bool)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(self, s: np.ndarray, a: int, r: float, s_next: np.ndarray, mask_next: np.ndarray, terminated: bool) -> None:
        i = self.ptr
        self.s[i], self.a[i], self.r[i] = s, a, r
        self.s_next[i], self.mask_next[i], self.terminated[i] = s_next, mask_next, float(terminated)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        j = rng.integers(self.size, size=batch_size)
        return (torch.from_numpy(self.s[j]), torch.from_numpy(self.a[j]), torch.from_numpy(self.r[j]),
                torch.from_numpy(self.s_next[j]), torch.from_numpy(self.mask_next[j]),
                torch.from_numpy(self.terminated[j]))


@dataclass(frozen=True)
class DQNConfig:
    total_steps: int = 100_000
    batch_size: int = 64
    buffer_capacity: int = 50_000
    warmup_steps: int = 1_000
    learning_rate: float = 1e-3
    lr_final_fraction: float = 0.05      # 学习率线性降到初值的这一比例
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_fraction: float = 0.5  # ε 在前 50% 的训练步内衰减完
    target_sync_every: int = 500         # 以梯度步计
    hidden_dim: int = 64
    seed: int = 0


def epsilon_at(cfg: DQNConfig, step: int) -> float:
    frac = min(1.0, step / (cfg.epsilon_decay_fraction * cfg.total_steps))
    return cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * frac


def greedy_policy(net: QNetwork, env: GrowthEnv) -> np.ndarray:
    """对全部状态取带掩码的 argmax。"""
    x = torch.from_numpy(np.stack([state_features(env.index_to_state(s), env) for s in range(env.num_states)]))
    mask = torch.from_numpy(np.stack([env.action_mask(env.index_to_state(s)) for s in range(env.num_states)]))
    with torch.no_grad():
        return net(x).masked_fill(~mask, float("-inf")).argmax(dim=1).numpy()


def train_dqn(env: GrowthEnv, cfg: DQNConfig, bench: Optional[GrowthBenchmark] = None,
              weights: Optional[np.ndarray] = None, report_every: int = 0) -> QNetwork:
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    gamma = env.gamma
    num_actions = env.num_actions

    online = QNetwork(2, num_actions, cfg.hidden_dim)
    target = QNetwork(2, num_actions, cfg.hidden_dim)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.SmoothL1Loss()
    buffer = ReplayBuffer(cfg.buffer_capacity, 2, num_actions)

    # 可行集与特征只依赖状态，事先缓存
    masks = [env.action_mask(env.index_to_state(s)) for s in range(env.num_states)]
    feasible = [np.flatnonzero(m) for m in masks]
    feats = [state_features(env.index_to_state(s), env) for s in range(env.num_states)]

    state = env.reset()
    updates = 0
    t0 = time.time()
    for step in range(1, cfg.total_steps + 1):
        eps = epsilon_at(cfg, step)
        lr = cfg.learning_rate * max(cfg.lr_final_fraction, 1.0 - (step - 1) / cfg.total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        s = env.state_index(*state)
        if rng.random() < eps:
            action = int(feasible[s][int(rng.random() * feasible[s].size)])
        else:
            with torch.no_grad():
                q_row = online(torch.from_numpy(feats[s]))
            action = int(q_row.masked_fill(~torch.from_numpy(masks[s]), float("-inf")).argmax())
        next_state, reward, terminated, truncated, _ = env.step(action)
        s2 = env.state_index(*next_state)
        buffer.add(feats[s], action, reward, feats[s2], masks[s2], terminated)
        state = env.reset() if (terminated or truncated) else next_state

        if buffer.size >= cfg.warmup_steps:
            s_b, a_b, r_b, s2_b, mask2_b, term_b = buffer.sample(cfg.batch_size, rng)
            with torch.no_grad():
                best_next = target(s2_b).masked_fill(~mask2_b, float("-inf")).max(dim=1).values
                # 每个状态至少有一个可行行动，best_next 有限；只有 terminated 才去掉自举项
                y = r_b + gamma * (1.0 - term_b) * best_next
            q_pred = online(s_b).gather(1, a_b.view(-1, 1)).squeeze(1)
            loss = loss_fn(q_pred, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
            optimizer.step()
            updates += 1
            if updates % cfg.target_sync_every == 0:
                target.load_state_dict(online.state_dict())

        if report_every and step % report_every == 0 and bench is not None and weights is not None:
            m = bench.evaluate_greedy(greedy_policy(online, env), weights)
            print(f"  {step:>7,d} 步（ε={eps:.2f}，学习率 {lr:.1e}，{time.time() - t0:3.0f} 秒）：与 g* 相同 {m['agree']:2d}/40，"
                  f"ℓ_CE 全部 {100 * m['ce_all']:.2f}%，加权 {100 * m['ce_weighted']:.2f}%", flush=True)
    return online


def main() -> None:
    parser = argparse.ArgumentParser(description="20×2 增长模型上的最小 DQN")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100_000)
    args = parser.parse_args()

    t_start = time.time()
    env = make_env(seed=args.seed)
    bench = GrowthBenchmark(env)
    mu = bench.stationary_distribution(bench.epsilon_greedy_matrix(bench.g_star, 0.1))
    cfg = DQNConfig(total_steps=args.steps, seed=args.seed)

    print("=" * 78)
    print("DQN：20×2 增长模型（第12章练习环境），与值函数迭代的 g* 对照")
    print("=" * 78)
    print(f"训练 {cfg.total_steps:,d} 步（每段 100 期，截断时照常自举）；回放容量 {cfg.buffer_capacity:,d}，批量 {cfg.batch_size}；"
          f"目标网络每 {cfg.target_sync_every} 个梯度步同步")
    print(f"ε 在前 {int(cfg.epsilon_decay_fraction * cfg.total_steps):,d} 步内由 {cfg.epsilon_start} 线性降到 {cfg.epsilon_end}；"
          f"Adam 学习率 {cfg.learning_rate} 线性降到 {cfg.learning_rate * cfg.lr_final_fraction:.0e}；种子 {cfg.seed}")
    loss_u = bench.ce_loss(bench.policy_value_stochastic(bench.uniform_random_matrix()))
    print(f"对照：均匀随机策略（每期在可行 k' 中均匀抽取）ℓ_CE 全部平均 {100 * loss_u.mean():.2f}%，"
          f"加权 {100 * (mu @ loss_u):.2f}%")
    print("说明：“加权”指按 g* 上 ε=0.1 的 ε-贪心策略的平稳分布加权。\n训练过程：")

    net = train_dqn(env, cfg, bench=bench, weights=mu, report_every=max(cfg.total_steps // 5, 1))
    g = greedy_policy(net, env)
    m = bench.evaluate_greedy(g, mu)
    print(f"\n[DQN 贪心策略] 与 g* 相同 {m['agree']}/40（储蓄偏少 {m['less']}、偏多 {m['more']}，最大偏离 {m['max_dev']} 格）；"
          f"ℓ_CE 全部平均 {100 * m['ce_all']:.3f}%，加权 {100 * m['ce_weighted']:.3f}%")
    print("k' 索引：DQN（上两行）与 g*（下两行），每组第一行 z_L、第二行 z_H")
    for line in format_rule(g, env.num_k, env.num_z) + format_rule(bench.g_star, env.num_k, env.num_z):
        print("  " + line)
    print(f"\n总耗时 {time.time() - t_start:.0f} 秒")


if __name__ == "__main__":
    main()
