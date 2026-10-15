"""
首次访问 Monte Carlo 策略评估（教学骨架）

对应章节：第12章《强化学习基础》§“样本备份与最基本的 Monte Carlo 方法”与§“首次访问与每次访问”

设定
----
- 环境：来自 growth_mdp.DiscreteGrowthMDP。
- 给定一个策略 g（默认：在每个状态选可行集合中最靠近"中位"的 k'），
  生成 N 条 episode，每条长度 H，用首次访问规则估计 V^g(s)。
- 与 DP 解 V^g 对比，绘制误差随 N 的收敛曲线。

教学要点
--------
- 这是编程练习 (b) 的骨架答案。
- 关键在于"首次访问"：同一条轨迹中若状态 s 多次出现，只用第一次访问
  时的回报 G_t 来更新 V(s)，保证样本独立性更强（详见§“首次访问与每次访问”）。
- 与 every-visit 的对比留作扩展练习。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from growth_mdp import DiscreteGrowthMDP, make_default_env, value_iteration


def median_feasible_policy(env: DiscreteGrowthMDP) -> Callable[[Tuple[int, int]], int]:
    """构造一个简单的确定性策略：在可行集合中选中位 k'。"""
    def policy(state: Tuple[int, int]) -> int:
        mask = env.action_mask(state)
        feasible = np.flatnonzero(mask)
        return int(feasible[len(feasible) // 2])
    return policy


def evaluate_policy_dp(env: DiscreteGrowthMDP, policy_table: np.ndarray, tol: float = 1e-6, max_iter: int = 2000) -> np.ndarray:
    """
    DP 精确策略评估（用于对照 MC 估计）。

    参数
    ----
    policy_table : ndarray (num_k, num_z)
        每个状态指定的 k_next 索引。

    返回
    ----
    V^g 的形状 (num_k, num_z)。
    """
    beta = env.params.beta
    V = np.zeros((env.num_k, env.num_z))
    for _ in range(max_iter):
        EV = V @ env.p_z.T  # (num_k, num_z) -> E[V(k', z') | z] indexed by (k', z)
        V_new = np.empty_like(V)
        for k_idx in range(env.num_k):
            for z_idx in range(env.num_z):
                a = int(policy_table[k_idx, z_idx])
                r = env._reward[k_idx, z_idx, a]
                V_new[k_idx, z_idx] = r + beta * EV[a, z_idx]
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    return V


def policy_to_table(env: DiscreteGrowthMDP, policy: Callable[[Tuple[int, int]], int]) -> np.ndarray:
    """把一个确定性策略函数转成 (num_k, num_z) 的查表，用于 DP 评估。"""
    table = np.zeros((env.num_k, env.num_z), dtype=int)
    for k_idx in range(env.num_k):
        for z_idx in range(env.num_z):
            table[k_idx, z_idx] = policy((k_idx, z_idx))
    return table


@dataclass(frozen=True)
class MCConfig:
    num_episodes: int = 1000
    episode_length: int = 100
    seed: int = 42


def first_visit_mc_prediction(
    env: DiscreteGrowthMDP,
    policy: Callable[[Tuple[int, int]], int],
    cfg: MCConfig,
    snapshot_every: int = 100,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    首次访问 MC 策略评估。

    返回
    ----
    V_hat : ndarray (num_k, num_z)
    snapshots : dict[int -> V_hat] —— 用于绘制收敛曲线（在指定 episode 数时的中间估计）
    """
    beta = env.params.beta
    returns_sum = np.zeros((env.num_k, env.num_z))
    returns_count = np.zeros((env.num_k, env.num_z), dtype=int)
    snapshots: Dict[int, np.ndarray] = {}

    for ep in range(1, cfg.num_episodes + 1):
        state = env.reset()
        trajectory: List[Tuple[Tuple[int, int], float]] = []
        done = False
        while not done:
            a = policy(state)
            next_state, r, done, _ = env.step(a)
            trajectory.append((state, r))
            state = next_state

        # 从后往前累计回报 G_t = r_{t+1} + gamma * G_{t+1}
        G = 0.0
        visited: set = set()
        # 首次访问：从前往后扫描记录"首次时刻"，但回报需要从后往前算
        first_index: Dict[Tuple[int, int], int] = {}
        for t, (s, _) in enumerate(trajectory):
            if s not in first_index:
                first_index[s] = t

        # 计算每个时刻的回报 G_t
        T = len(trajectory)
        G_values = np.zeros(T)
        running = 0.0
        for t in range(T - 1, -1, -1):
            running = trajectory[t][1] + beta * running
            G_values[t] = running

        # 用首次访问的 G_t 更新均值
        for s, t in first_index.items():
            k_idx, z_idx = s
            returns_sum[k_idx, z_idx] += G_values[t]
            returns_count[k_idx, z_idx] += 1

        if ep % snapshot_every == 0:
            with np.errstate(invalid="ignore"):
                snap = np.where(returns_count > 0, returns_sum / np.maximum(returns_count, 1), 0.0)
            snapshots[ep] = snap.copy()

    with np.errstate(invalid="ignore"):
        V_hat = np.where(returns_count > 0, returns_sum / np.maximum(returns_count, 1), 0.0)
    return V_hat, snapshots


def plot_convergence(snapshots: Dict[int, np.ndarray], V_true: np.ndarray, out_path: str = "mc_convergence.png") -> None:
    """绘制 ||V_hat - V_true||_inf 随 episode 数的下降。"""
    import matplotlib.pyplot as plt
    episodes = sorted(snapshots.keys())
    errs = [float(np.max(np.abs(snapshots[ep] - V_true))) for ep in episodes]
    plt.figure(figsize=(6, 4))
    plt.plot(episodes, errs, marker="o")
    plt.xlabel("episode 数 N")
    plt.ylabel(r"$\|\hat V^g - V^g\|_\infty$")
    plt.title("首次访问 MC 估计的收敛曲线")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)


if __name__ == "__main__":
    env = make_default_env(seed=42)
    policy = median_feasible_policy(env)
    cfg = MCConfig(num_episodes=1000, episode_length=100, seed=42)
    V_hat, snapshots = first_visit_mc_prediction(env, policy, cfg, snapshot_every=100)

    # 计算 DP 精确解作为参照
    policy_tab = policy_to_table(env, policy)
    V_true = evaluate_policy_dp(env, policy_tab)
    err_inf = float(np.max(np.abs(V_hat - V_true)))
    print(f"first-visit MC after {cfg.num_episodes} episodes:")
    print(f"  V_hat(k_min, z_L) = {V_hat[0, 0]:.3f},  V_true = {V_true[0, 0]:.3f}")
    print(f"  V_hat(k_max, z_H) = {V_hat[-1, -1]:.3f}, V_true = {V_true[-1, -1]:.3f}")
    print(f"  ||V_hat - V_true||_inf = {err_inf:.3f}")
