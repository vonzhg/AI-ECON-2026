"""
第13章共用环境：包装第12章的离散增长模型，并附动态规划基准

对应章节：第13章《强化学习核心算法》§13.1–§13.5 的全部增长模型示例
（tabular_td_control.py、linear_td_approx.py、dqn_growth.py、policy_gradient.py 共用）。
GrowthEnv 是逐步交互的接口；BatchGrowthEnv 把 E 个独立副本放在一起向量化推进，只为加快并行采样。

模型：与第12章编程练习 (a) 相同，不在本文件重新定义
----------------------------------------------------
- 从 code/ch12/growth_mdp.py 导入 DiscreteGrowthMDP 与 make_default_env，网格、参数、
  奖励与可行集都取自那里：资本网格 20 点（0.1 到 5.0 等距），z ∈ {0.9, 1.1}，
  切换概率 0.2，f(k)=k^0.36，δ=0.10，γ=0.95（第12章代码记作 beta），u(c)=ln c。
- 状态 s=(k_idx, z_idx)，行动 a = 下一期资本 k' 的网格索引；只允许 c>0 的行动（动作掩码）。
- 展平记号 s = z_idx * num_k + k_idx，与 code/ch12/running_example.py 一致。

截断不等于终止
--------------
增长模型是无穷期问题，没有终止状态。step 返回两个标志：
- terminated：到达真正的终止状态（本模型恒为 False）；
- truncated ：模拟满 episode_length 期后人为截断。
TD 目标只在 terminated 时去掉自举项：target = r + γ(1 - terminated)·V(s')。
截断时照常自举，然后 reset 开始新的一段轨迹；若把截断当作终止，等于假定第 T 期之后
价值为零，会低估储蓄的价值。

与第12章 step 的一处实现差别
----------------------------
第12章的 step 用 rng.choice(p=...) 抽 z'，每步约 20 微秒；本文件用同一个转移矩阵 P_z
做逆 CDF 抽样，每步约 4 微秒，分布相同、随机数序列不同。奖励、可行集、转移概率都直接
读自被包装的 DiscreteGrowthMDP 对象。

动态规划基准（学习算法不调用，只用于事后评估）
----------------------------------------------
GrowthBenchmark 给出：V*、Q*、g*（值函数迭代，容差 1e-12）；确定性或随机策略的精确价值
（解线性方程组）；消费等价损失 ℓ_CE(s) = exp{(1-γ)[V*(s) - V^g(s)]} - 1（对数效用）；
ε-贪心与均匀随机策略的行动概率矩阵；策略诱导的马尔可夫链的平稳分布。

运行（在书稿目录下）：python3 -B code/ch13/growth_env.py（烟雾测试并打印基准摘要）
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# 不在 code/ch12/ 下留下 __pycache__（第12章目录只读）
sys.dont_write_bytecode = True
_CH12_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "ch11"))
if _CH12_DIR not in sys.path:
    sys.path.append(_CH12_DIR)

from growth_mdp import DiscreteGrowthMDP, make_default_env  # noqa: E402  第12章的模型


class GrowthEnv:
    """第13章算法共用的交互接口：reset / step / action_mask，内部是第12章的 MDP。"""

    def __init__(self, mdp: DiscreteGrowthMDP, episode_length: int = 100, seed: int = 0) -> None:
        self.mdp = mdp
        self.k_grid = mdp.k_grid
        self.z_grid = mdp.z_grid
        self.p_z = mdp.p_z
        self.params = mdp.params
        self.gamma = float(mdp.params.beta)
        self.episode_length = int(episode_length)
        self.rng = np.random.default_rng(seed)
        # 学习算法看不到 P_z；它只用于在 step 里抽下一期冲击
        self._p_z_cdf = np.cumsum(self.p_z, axis=1)
        self._t = 0
        self._k_idx = 0
        self._z_idx = 0

    # ---- 维度与索引 -------------------------------------------------------
    @property
    def num_k(self) -> int:
        return self.mdp.num_k

    @property
    def num_z(self) -> int:
        return self.mdp.num_z

    @property
    def num_states(self) -> int:
        return self.num_k * self.num_z

    @property
    def num_actions(self) -> int:
        return self.mdp.num_actions

    def state_index(self, k_idx: int, z_idx: int) -> int:
        return int(z_idx) * self.num_k + int(k_idx)

    def index_to_state(self, s: int) -> Tuple[int, int]:
        z_idx, k_idx = divmod(int(s), self.num_k)
        return k_idx, z_idx

    def features01(self, state: Tuple[int, int]) -> Tuple[float, float]:
        """(k̃, z̃)：把 k 线性标准化到 [0,1]，z 的索引标准化到 [0,1]。"""
        k_idx, z_idx = state
        k_lo, k_hi = float(self.k_grid[0]), float(self.k_grid[-1])
        k_tilde = (float(self.k_grid[k_idx]) - k_lo) / (k_hi - k_lo)
        z_tilde = float(z_idx) / max(self.num_z - 1, 1)
        return k_tilde, z_tilde

    # ---- 交互 ---------------------------------------------------------------
    def action_mask(self, state: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """可行的 k'（c>0）为 True；直接调用第12章的 action_mask。"""
        return self.mdp.action_mask((self._k_idx, self._z_idx) if state is None else state)

    def reset(self, k_idx: Optional[int] = None, z_idx: Optional[int] = None) -> Tuple[int, int]:
        """开始新的一段轨迹；不指定起点时在 (k, z) 网格上均匀随机抽取。"""
        self._t = 0
        self._k_idx = int(k_idx) if k_idx is not None else int(self.rng.integers(self.num_k))
        self._z_idx = int(z_idx) if z_idx is not None else int(self.rng.integers(self.num_z))
        return self._k_idx, self._z_idx

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, bool, Dict[str, float]]:
        """
        执行一期：选 k'=action，得到 u(c)，抽下一期 z'。

        返回 (next_state, reward, terminated, truncated, info)。
        terminated 恒为 False（无穷期问题没有终止状态）；truncated 表示本段轨迹已满 episode_length 期。
        """
        a = int(action)
        k_idx, z_idx = self._k_idx, self._z_idx
        if not self.mdp._feasible[k_idx, z_idx, a]:
            raise ValueError("Infeasible action: violates c > 0.")
        reward = float(self.mdp._reward[k_idx, z_idx, a])
        consumption = float(self.mdp._consumption[k_idx, z_idx, a])
        z_next = int(np.searchsorted(self._p_z_cdf[z_idx], self.rng.random(), side="right"))
        z_next = min(z_next, self.num_z - 1)
        self._t += 1
        self._k_idx, self._z_idx = a, z_next
        terminated = False
        truncated = self._t >= self.episode_length
        return (a, z_next), reward, terminated, truncated, {"t": self._t, "consumption": consumption}


def make_env(seed: int = 0, episode_length: int = 100) -> GrowthEnv:
    """第12章练习环境（make_default_env 的 20×2 网格）加上本章的接口。"""
    return GrowthEnv(make_default_env(seed=seed), episode_length=episode_length, seed=seed)


class BatchGrowthEnv:
    """
    E 个相互独立的环境副本，一次推进一步（numpy 向量化），供策略梯度方法并行采样。

    语义与 E 个 GrowthEnv 完全相同：各副本有自己的时钟，满 episode_length 期时 truncated=True，
    terminated 恒为 False；reset(which) 只重置指定的副本。状态以展平索引 s = z*num_k + k 表示。
    """

    def __init__(self, mdp: DiscreteGrowthMDP, num_envs: int, episode_length: int = 100, seed: int = 0) -> None:
        self.mdp = mdp
        self.num_envs = int(num_envs)
        self.num_k, self.num_z = mdp.num_k, mdp.num_z
        self.num_states, self.num_actions = self.num_k * self.num_z, mdp.num_actions
        self.gamma = float(mdp.params.beta)
        self.episode_length = int(episode_length)
        self.rng = np.random.default_rng(seed)
        self._p_z_cdf = np.cumsum(mdp.p_z, axis=1)
        self._t = np.zeros(self.num_envs, dtype=np.int64)
        self._k = np.zeros(self.num_envs, dtype=np.int64)
        self._z = np.zeros(self.num_envs, dtype=np.int64)

    def states(self) -> np.ndarray:
        return self._z * self.num_k + self._k

    def reset(self, which: Optional[np.ndarray] = None) -> np.ndarray:
        """重置 which（布尔数组）为 True 的副本，起点在网格上均匀随机；which=None 时全部重置。"""
        idx = np.arange(self.num_envs) if which is None else np.flatnonzero(which)
        self._t[idx] = 0
        self._k[idx] = self.rng.integers(self.num_k, size=idx.size)
        self._z[idx] = self.rng.integers(self.num_z, size=idx.size)
        return self.states()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """返回 (next_states, rewards, terminated, truncated)，均为长度 E 的数组。"""
        a = np.asarray(actions, dtype=np.int64)
        if not np.all(self.mdp._feasible[self._k, self._z, a]):
            raise ValueError("Infeasible action: violates c > 0.")
        rewards = self.mdp._reward[self._k, self._z, a].astype(float)
        u = self.rng.random(self.num_envs)
        z_next = (u[:, None] >= self._p_z_cdf[self._z]).sum(axis=1)
        self._k = a
        self._z = np.minimum(z_next, self.num_z - 1)
        self._t += 1
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = self._t >= self.episode_length
        return self.states(), rewards, terminated, truncated


def make_batch_env(num_envs: int, seed: int = 0, episode_length: int = 100) -> BatchGrowthEnv:
    return BatchGrowthEnv(make_default_env(seed=seed), num_envs=num_envs, episode_length=episode_length, seed=seed)


class GrowthBenchmark:
    """动态规划基准：只用于评估学习结果，学习算法本身不调用。"""

    def __init__(self, env: GrowthEnv, tol: float = 1e-12) -> None:
        self.gamma = env.gamma
        self.num_k, self.num_z = env.num_k, env.num_z
        self.num_states, self.num_actions = env.num_states, env.num_actions
        S, A, NK = self.num_states, self.num_actions, self.num_k
        mdp = env.mdp

        self.feasible = np.zeros((S, A), dtype=bool)
        self.reward = np.zeros((S, A))          # 不可行处填 0，只在 feasible 为 True 处使用
        self.P = np.zeros((S, A, S))
        for k in range(NK):
            for z in range(self.num_z):
                s = z * NK + k
                self.feasible[s] = mdp._feasible[k, z]
                self.reward[s] = np.where(mdp._feasible[k, z], mdp._reward[k, z], 0.0)
                for a in range(A):
                    for z2 in range(self.num_z):
                        self.P[s, a, z2 * NK + a] = mdp.p_z[z, z2]
        self.feasible_list: List[np.ndarray] = [np.flatnonzero(self.feasible[s]) for s in range(S)]

        V = np.zeros(S)
        for _ in range(100000):
            V_new = np.where(self.feasible, self.reward + self.gamma * self.P @ V, -np.inf).max(axis=1)
            if np.max(np.abs(V_new - V)) < tol:
                V = V_new
                break
            V = V_new
        self.Q_star = np.where(self.feasible, self.reward + self.gamma * self.P @ V, -np.inf)
        self.g_star = self.Q_star.argmax(axis=1)
        self.V_star = self.policy_value(self.g_star)

    # ---- 策略的精确价值 -----------------------------------------------------
    def policy_value(self, g: np.ndarray) -> np.ndarray:
        """确定性策略 g（每个状态一个 k' 索引）的精确价值：(I - γ P_g) V = r_g。"""
        idx = np.arange(self.num_states)
        g = np.asarray(g, dtype=int)
        if not np.all(self.feasible[idx, g]):
            raise ValueError("policy selects an infeasible action")
        return np.linalg.solve(np.eye(self.num_states) - self.gamma * self.P[idx, g], self.reward[idx, g])

    def transition_matrix(self, pi: np.ndarray) -> np.ndarray:
        """随机策略 pi（S×A 行动概率）诱导的状态转移矩阵 P_pi（S×S）。"""
        return np.einsum("sa,sat->st", pi, self.P)

    def expected_reward(self, pi: np.ndarray) -> np.ndarray:
        return (pi * self.reward).sum(axis=1)

    def policy_value_stochastic(self, pi: np.ndarray) -> np.ndarray:
        """随机策略的精确价值：(I - γ P_pi) V = r_pi。"""
        if np.any(pi[~self.feasible] > 0):
            raise ValueError("policy puts mass on infeasible actions")
        return np.linalg.solve(np.eye(self.num_states) - self.gamma * self.transition_matrix(pi),
                               self.expected_reward(pi))

    # ---- 策略族 ---------------------------------------------------------------
    def deterministic_matrix(self, g: np.ndarray) -> np.ndarray:
        pi = np.zeros((self.num_states, self.num_actions))
        pi[np.arange(self.num_states), np.asarray(g, dtype=int)] = 1.0
        return pi

    def epsilon_greedy_matrix(self, g: np.ndarray, eps: float) -> np.ndarray:
        """以 1-ε 按 g 行动、以 ε 在可行 k' 中均匀抽取。"""
        pi = np.zeros((self.num_states, self.num_actions))
        for s in range(self.num_states):
            feas = self.feasible_list[s]
            pi[s, feas] = eps / feas.size
            pi[s, int(g[s])] += 1.0 - eps
        return pi

    def uniform_random_matrix(self) -> np.ndarray:
        pi = self.feasible.astype(float)
        return pi / pi.sum(axis=1, keepdims=True)

    def greedy_from_q(self, q: np.ndarray) -> np.ndarray:
        """在可行集上取 argmax（不可行行动不参与比较）。"""
        return np.where(self.feasible, q, -np.inf).argmax(axis=1)

    # ---- 平稳分布与福利 -------------------------------------------------------
    def stationary_distribution(self, pi: np.ndarray) -> np.ndarray:
        """解 μ(I - P_pi) = 0、Σμ = 1（本章用到的策略都使状态链不可约）。"""
        S = self.num_states
        P_pi = self.transition_matrix(pi)
        A_mat = np.vstack([(np.eye(S) - P_pi).T, np.ones((1, S))])
        b = np.zeros(S + 1)
        b[-1] = 1.0
        mu = np.linalg.lstsq(A_mat, b, rcond=None)[0]
        mu = np.clip(mu, 0.0, None)
        return mu / mu.sum()

    def ce_loss(self, V: np.ndarray) -> np.ndarray:
        """消费等价损失 ℓ_CE(s) = exp{(1-γ)[V*(s) - V(s)]} - 1（对数效用下永久消费需提高的比例）。"""
        return np.exp((1.0 - self.gamma) * (self.V_star - V)) - 1.0

    def evaluate_greedy(self, g: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
        """确定性策略 g 与 g* 的对照：一致状态数、ℓ_CE 全部平均与加权平均、最大偏离格点数。"""
        g = np.asarray(g, dtype=int)
        loss = self.ce_loss(self.policy_value(g))
        diff = g - self.g_star
        return {
            "agree": int(np.sum(diff == 0)),
            "less": int(np.sum(diff < 0)),
            "more": int(np.sum(diff > 0)),
            "max_dev": int(np.max(np.abs(diff))),
            "ce_all": float(loss.mean()),
            "ce_weighted": float(weights @ loss),
        }


def format_rule(g: np.ndarray, num_k: int, num_z: int) -> List[str]:
    """把 k' 索引按 z 分行打印。"""
    rows = np.asarray(g, dtype=int).reshape(num_z, num_k)
    return [" ".join(f"{x:2d}" for x in row) for row in rows]


if __name__ == "__main__":
    env = make_env(seed=0)
    bench = GrowthBenchmark(env)
    # 烟雾测试：沿 g* 走满一段轨迹，确认截断标志在第 T 期出现、terminated 始终为 False
    state = env.reset(k_idx=10, z_idx=0)
    total, n_steps = 0.0, 0
    while True:
        a = int(bench.g_star[env.state_index(*state)])
        state, r, terminated, truncated, _ = env.step(a)
        total += r
        n_steps += 1
        if terminated or truncated:
            break
    print(f"一段轨迹 {n_steps} 期后 truncated=True、terminated={terminated}；未贴现奖励和 {total:.3f}")
    print(f"状态数 {env.num_states}，可行 (s,a) 对 {int(bench.feasible.sum())}；γ={env.gamma}")
    print(f"V* 范围 [{bench.V_star.min():.3f}, {bench.V_star.max():.3f}]")
    print("g* 的 k' 索引（第一行 z_L，第二行 z_H）：")
    for line in format_rule(bench.g_star, env.num_k, env.num_z):
        print("  " + line)
    pi_eps = bench.epsilon_greedy_matrix(bench.g_star, 0.1)
    mu = bench.stationary_distribution(pi_eps)
    for name, pi in [("ε=0.1 的 ε-贪心(g*)", pi_eps), ("均匀随机", bench.uniform_random_matrix())]:
        loss = bench.ce_loss(bench.policy_value_stochastic(pi))
        print(f"{name}：ℓ_CE 全部状态平均 {100 * loss.mean():.2f}%，按 ε-贪心(g*) 平稳分布加权 {100 * (mu @ loss):.2f}%")
