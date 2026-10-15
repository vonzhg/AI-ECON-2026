"""
表格型时序差分控制：SARSA 与 Q学习，并与动态规划解逐格对照

对应章节：第13章《强化学习核心算法》§13.2 SARSA 与 Q学习（sec:ch13_eval_control）；
TD 目标、步长与截断的处理见 §13.1（sec:ch13_td）。

设定
----
- 环境：growth_env.py（第12章 20×2 练习环境：40 个状态、542 个可行的状态—行动对）。
  每段轨迹 T=100 期，起点在网格上均匀随机；第 T 期截断时照常自举，然后重新开始。
- 行为策略：固定 ε=0.1 的 ε-贪心。随机行动只在可行 k' 中抽取；贪心行动是带掩码的 argmax。
- 步长：每个状态—行动对单独计数，η = C(s,a)^(-0.6)。它满足 Robbins–Monro 条件；
  第一次更新时 η=1，Q 表的初值 0 随即被第一个目标覆盖。
- SARSA（on-policy）：目标 r + γ Q(s', a')，a' 按同一个 ε-贪心策略抽取；
  Q学习（off-policy）：目标 r + γ max_{a'∈A(s')} Q(s', a')。
- 两种算法使用同样的种子与步数，环境的随机数与算法的随机数分开。

评估（训练结束后与 DP 基准对照，学习算法本身不调用）
----------------------------------------------------
- 贪心策略 g：Q 在可行集上的 argmax。报告与值函数迭代的 g* 相同的状态数，
  其余状态分成储蓄偏少（k' 低于 g*）与偏多。
- 消费等价损失 ℓ_CE(s) = exp{(1-γ)[V*(s) - V^g(s)]} - 1：全部 40 个状态的简单平均；
  以及按“g* 上 ε=0.1 的 ε-贪心策略”的平稳分布加权的平均。
- 参照：固定 ε 时 SARSA 评估的是 ε-贪心策略自己的价值，它的理论极限是最优 ε-软策略；
  脚本先打印该策略的贪心部分与 g* 的差别，再报告学到的策略与它相同的状态数。

运行（在书稿目录下）：python3 -B code/ch13/tabular_td_control.py [--steps 1000000] [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from growth_env import GrowthBenchmark, GrowthEnv, format_rule, make_env


def masked_argmax(q_row: np.ndarray, feasible: np.ndarray, rng: np.random.Generator) -> int:
    """在可行行动 feasible（索引数组）中取 argmax；并列时随机选一个。"""
    values = q_row[feasible]
    best = feasible[values == values.max()]
    if best.size == 1:
        return int(best[0])
    return int(best[int(rng.random() * best.size)])


def epsilon_greedy(q_row: np.ndarray, feasible: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """以概率 ε 在可行集中均匀抽取，否则取带掩码的 argmax。"""
    if rng.random() < epsilon:
        return int(feasible[int(rng.random() * feasible.size)])
    return masked_argmax(q_row, feasible, rng)


@dataclass(frozen=True)
class TDControlConfig:
    num_steps: int = 1_000_000
    epsilon: float = 0.1
    step_exponent: float = 0.6   # η = C(s,a)^(-step_exponent)
    seed: int = 0


def train_td_control(
    env: GrowthEnv,
    method: str,
    cfg: TDControlConfig,
    checkpoints: Sequence[int] = (),
    on_checkpoint: Optional[Callable[[int, np.ndarray], None]] = None,
) -> np.ndarray:
    """
    method = "sarsa" 或 "q"。返回 Q 表 (num_states, num_actions)；不可行格子保持 0，从不参与比较。
    """
    if method not in ("sarsa", "q"):
        raise ValueError("method must be 'sarsa' or 'q'")
    gamma = env.gamma
    rng = np.random.default_rng(cfg.seed)
    q = np.zeros((env.num_states, env.num_actions))
    count = np.zeros((env.num_states, env.num_actions), dtype=np.int64)
    # 可行集只依赖状态，事先按状态缓存（等价于每步调用 env.action_mask）
    feasible: List[np.ndarray] = [
        np.flatnonzero(env.action_mask(env.index_to_state(s))) for s in range(env.num_states)
    ]
    marks = set(int(c) for c in checkpoints)

    step = 0
    while step < cfg.num_steps:
        s = env.state_index(*env.reset())
        a = epsilon_greedy(q[s], feasible[s], cfg.epsilon, rng)
        while True:
            next_state, r, terminated, truncated, _ = env.step(a)
            s2 = env.state_index(*next_state)
            if method == "sarsa":
                a2 = epsilon_greedy(q[s2], feasible[s2], cfg.epsilon, rng)   # 实际将要执行的行动
                bootstrap = q[s2, a2]
            else:
                bootstrap = q[s2, feasible[s2]].max()                         # 贪心行动的价值
            # 只有真正的终止才去掉自举项；截断（truncated）时照常自举
            target = r + (0.0 if terminated else gamma * bootstrap)
            count[s, a] += 1
            q[s, a] += (target - q[s, a]) * float(count[s, a]) ** (-cfg.step_exponent)
            step += 1
            if on_checkpoint is not None and step in marks:
                on_checkpoint(step, q)
            if terminated or truncated or step >= cfg.num_steps:
                break
            s = s2
            a = a2 if method == "sarsa" else epsilon_greedy(q[s], feasible[s], cfg.epsilon, rng)
    return q


def epsilon_soft_optimum(bench: GrowthBenchmark, epsilon: float, max_iter: int = 200) -> np.ndarray:
    """
    ε-软策略迭代：在“以 1-ε 按贪心行动、以 ε 均匀试探”的策略族里求最优，返回其贪心部分。
    这是固定 ε 的 SARSA 的理论极限。
    """
    g = bench.g_star.copy()
    for _ in range(max_iter):
        V = bench.policy_value_stochastic(bench.epsilon_greedy_matrix(g, epsilon))
        g_new = bench.greedy_from_q(bench.reward + bench.gamma * bench.P @ V)
        if np.array_equal(g_new, g):
            break
        g = g_new
    return g


def main() -> None:
    parser = argparse.ArgumentParser(description="20×2 增长模型上的 SARSA 与 Q学习")
    parser.add_argument("--steps", type=int, default=1_000_000, help="每次训练的总步数")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="随机种子")
    args = parser.parse_args()

    t_start = time.time()
    eps = 0.1
    bench = GrowthBenchmark(make_env(seed=0))
    mu = bench.stationary_distribution(bench.epsilon_greedy_matrix(bench.g_star, eps))
    checkpoints = sorted({c for c in (100_000, 300_000, 1_000_000, 3_000_000) if c < args.steps} | {args.steps})

    print("=" * 78)
    print("表格型 SARSA 与 Q学习：20×2 增长模型（第12章练习环境），与值函数迭代的 g* 对照")
    print("=" * 78)
    print(f"状态 {bench.num_states} 个，可行状态—行动对 {int(bench.feasible.sum())} 个；γ={bench.gamma}；"
          f"每段 T=100 期，截断时照常自举")
    print(f"行为策略 ε={eps}；步长 η=C(s,a)^-0.6；种子 {args.seeds}；每次 {args.steps} 步")
    print("g* 的 k' 索引（第一行 z_L，第二行 z_H）：")
    for line in format_rule(bench.g_star, bench.num_k, bench.num_z):
        print("  " + line)

    g_soft = epsilon_soft_optimum(bench, eps)
    loss_soft = bench.evaluate_greedy(g_soft, mu)
    print(f"参照：最优 ε-软策略（ε={eps}，SARSA 的理论极限）的贪心部分与 g* 不同的状态 "
          f"{bench.num_states - loss_soft['agree']} 个（储蓄偏少 {loss_soft['less']}、偏多 {loss_soft['more']}）；"
          f"ℓ_CE 全部平均 {100 * loss_soft['ce_all']:.3f}%，加权 {100 * loss_soft['ce_weighted']:.3f}%")
    loss_eps = bench.ce_loss(bench.policy_value_stochastic(bench.epsilon_greedy_matrix(bench.g_star, eps)))
    print(f"参照：g* 上 ε={eps} 的 ε-贪心策略本身（含随机行动）ℓ_CE 全部平均 {100 * loss_eps.mean():.2f}%，"
          f"加权 {100 * (mu @ loss_eps):.2f}%")
    print("说明：“加权”指按 g* 上 ε-贪心策略的平稳分布加权；“少/多”指 k' 低于/高于 g*。")

    names = {"sarsa": "SARSA", "q": "Q学习"}
    for method in ("sarsa", "q"):
        print("\n" + "-" * 78)
        print(f"{names[method]}")
        print("-" * 78)
        finals = []
        for seed in args.seeds:
            t0 = time.time()
            env = make_env(seed=1000 + seed)
            cfg = TDControlConfig(num_steps=args.steps, epsilon=eps, seed=seed)

            def report(step: int, q: np.ndarray, seed: int = seed) -> None:
                g = bench.greedy_from_q(q)
                m = bench.evaluate_greedy(g, mu)
                same_soft = int(np.sum(g == g_soft))
                print(f"  种子{seed} {step:>9,d} 步：与 g* 相同 {m['agree']:2d}/40（少 {m['less']:2d}、多 {m['more']:2d}，"
                      f"最大偏离 {m['max_dev']} 格）；与 ε-软最优相同 {same_soft:2d}/40；"
                      f"ℓ_CE 全部 {100 * m['ce_all']:.3f}%，加权 {100 * m['ce_weighted']:.3f}%")
                if step == args.steps:
                    finals.append(m)

            train_td_control(env, method, cfg, checkpoints=checkpoints, on_checkpoint=report)
            print(f"  （种子{seed} 用时 {time.time() - t0:.0f} 秒）")
        if finals:
            ce_all = np.array([m["ce_all"] for m in finals]) * 100
            ce_w = np.array([m["ce_weighted"] for m in finals]) * 100
            agree = np.array([m["agree"] for m in finals])
            print(f"  {names[method]} {args.steps:,d} 步小结（{len(finals)} 个种子）：与 g* 相同 {agree.min()}–{agree.max()}/40；"
                  f"ℓ_CE 全部 {ce_all.min():.3f}%–{ce_all.max():.3f}%，加权 {ce_w.min():.3f}%–{ce_w.max():.3f}%")

    print(f"\n总耗时 {time.time() - t_start:.0f} 秒")


if __name__ == "__main__":
    main()
