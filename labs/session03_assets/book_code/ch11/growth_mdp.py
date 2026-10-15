"""
离散增长模型环境（教学骨架）

对应章节：第12章《强化学习基础》§“马尔可夫决策过程（MDP）”

设定
----
- 状态：s = (k_idx, z_idx)，资本网格索引 + 冲击网格索引。
- 行动：a = k_next_idx，选择下一期资本 k' 的网格索引。
- 奖励：u(c) = ln(c)，其中 c = z f(k) + (1-δ)k - k'。
- 转移：k' 由行动确定；z' 按 P_Z(·|z) 抽样（学习算法不直接调用）。

教学要点
--------
- 这是编程练习 (a) 的"骨架答案"：学生需要实现的核心接口 reset/step
  已经给出，重点在于理解可行行动集（动作掩码）与不可行行动的处理。
- 模拟器内部知道 P_Z，但学习算法只通过 step 返回值看到 z 的实现序列，
  并不需要查询转移矩阵——这正是从 DP 到 RL 的关键转变。
- 所有随机性集中在 np.random.default_rng 中，便于复现（seed=42）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class GrowthParams:
    """偏好与技术参数。"""
    beta: float = 0.95
    delta: float = 0.10
    alpha: float = 0.36
    min_consumption: float = 1e-12


class DiscreteGrowthMDP:
    """
    离散增长模型 MDP（用于第12章 Monte Carlo 演示）。

    参数
    ----
    k_grid : ndarray (K,)
        严格递增的资本网格（K 个网格点）。
    z_grid : ndarray (Z,)
        TFP 网格（Z 个状态）。
    p_z : ndarray (Z, Z)
        冲击转移矩阵；模拟器内部使用，学习算法不可见。
    episode_length : int
        每条 episode 的长度（截断无穷期）。
    seed : int
        随机种子。
    params : GrowthParams
        模型参数。
    """

    def __init__(
        self,
        k_grid: np.ndarray,
        z_grid: np.ndarray,
        p_z: np.ndarray,
        episode_length: int = 100,
        seed: int = 42,
        params: GrowthParams = GrowthParams(),
    ) -> None:
        self.k_grid = np.asarray(k_grid, dtype=float)
        self.z_grid = np.asarray(z_grid, dtype=float)
        self.p_z = np.asarray(p_z, dtype=float)
        self.episode_length = int(episode_length)
        self.params = params

        # 输入合法性检查
        assert self.k_grid.ndim == 1 and self.k_grid.size >= 2
        assert np.all(self.k_grid[1:] > self.k_grid[:-1]), "k_grid 必须严格递增"
        assert self.z_grid.ndim == 1 and self.z_grid.size >= 1
        assert np.all(self.z_grid > 0), "z_grid 必须为正"
        assert self.p_z.shape == (self.z_grid.size, self.z_grid.size)
        assert np.allclose(self.p_z.sum(axis=1), 1.0), "p_z 每行和应为 1"

        self.rng = np.random.default_rng(seed)

        # 预计算消费、可行性、奖励 (K, Z, K) -> (k, z, k')
        k_now = self.k_grid[:, None, None]
        z_now = self.z_grid[None, :, None]
        k_next = self.k_grid[None, None, :]
        consumption = z_now * np.power(k_now, params.alpha) + (1 - params.delta) * k_now - k_next
        feasible = consumption > params.min_consumption
        reward = np.full_like(consumption, fill_value=-np.inf)
        reward[feasible] = np.log(consumption[feasible])

        self._consumption = consumption
        self._feasible = feasible
        self._reward = reward
        self._t = 0
        self._k_idx = 0
        self._z_idx = 0

    @property
    def num_k(self) -> int:
        return int(self.k_grid.size)

    @property
    def num_z(self) -> int:
        return int(self.z_grid.size)

    @property
    def num_actions(self) -> int:
        return self.num_k

    def action_mask(self, state: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """返回当前 (k, z) 下哪些 k' 满足 c >= 0。"""
        k_idx, z_idx = (self._k_idx, self._z_idx) if state is None else state
        return self._feasible[int(k_idx), int(z_idx)].copy()

    def reset(self, k_idx: Optional[int] = None, z_idx: Optional[int] = None) -> Tuple[int, int]:
        """开始新 episode。若不指定起点，则随机选取（探索性起点的核心机制）。"""
        self._t = 0
        self._k_idx = int(k_idx) if k_idx is not None else int(self.rng.integers(self.num_k))
        self._z_idx = int(z_idx) if z_idx is not None else int(self.rng.integers(self.num_z))
        return self._k_idx, self._z_idx

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, dict]:
        """
        执行一步交互。

        返回
        ----
        next_state : (k_idx, z_idx)
        reward     : float —— u(c)
        done       : bool  —— 是否达到 episode_length
        info       : dict  —— 含 't' 与 'consumption'
        """
        action = int(action)
        if not bool(self._feasible[self._k_idx, self._z_idx, action]):
            raise ValueError("Infeasible action: violates c >= 0.")

        c = float(self._consumption[self._k_idx, self._z_idx, action])
        r = float(self._reward[self._k_idx, self._z_idx, action])

        # 学习算法看不到 P_Z，只看到 step 返回的 z_next
        z_next = int(self.rng.choice(self.num_z, p=self.p_z[self._z_idx]))
        k_next = action

        self._t += 1
        done = self._t >= self.episode_length
        self._k_idx, self._z_idx = k_next, z_next
        return (k_next, z_next), r, done, {"t": self._t, "consumption": c}


def make_default_env(seed: int = 42) -> DiscreteGrowthMDP:
    """
    构造编程练习的标准教学环境。
    参数与练习题题面一致：K=20、Z=2、对称切换概率 0.2。
    """
    k_grid = np.linspace(0.1, 5.0, 20)
    z_grid = np.array([0.9, 1.1], dtype=float)
    p_z = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=float)
    return DiscreteGrowthMDP(
        k_grid=k_grid,
        z_grid=z_grid,
        p_z=p_z,
        episode_length=100,
        seed=seed,
    )


def value_iteration(env: DiscreteGrowthMDP, tol: float = 1e-6, max_iter: int = 1000) -> np.ndarray:
    """
    DP 参考解（值函数迭代），用于与 MC 估计比较。
    返回 V* 的形状 (num_k, num_z)。
    """
    beta = env.params.beta
    V = np.zeros((env.num_k, env.num_z))
    for _ in range(max_iter):
        # EV_next[k', z] = E[V(k', z') | z] = sum_{z'} p_z[z, z'] * V[k', z']
        EV_next = V @ env.p_z.T  # 形状 (num_k, num_z)
        V_new = np.empty_like(V)
        for k_idx in range(env.num_k):
            for z_idx in range(env.num_z):
                feasible = env._feasible[k_idx, z_idx]
                r = env._reward[k_idx, z_idx]            # 按 k' 索引
                ev = EV_next[:, z_idx]                   # 按 k' 索引
                vals = r + beta * ev
                V_new[k_idx, z_idx] = vals[feasible].max()
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    return V


if __name__ == "__main__":
    env = make_default_env(seed=42)
    # 烟雾测试：跑两条轨迹，确认 reset/step 接口工作
    for ep in range(2):
        s = env.reset()
        total_r = 0.0
        for _ in range(5):
            mask = env.action_mask(s)
            a = int(np.flatnonzero(mask)[0])  # 选最小可行 k'
            s, r, done, _ = env.step(a)
            total_r += r
            if done:
                break
        print(f"episode {ep}: total reward over 5 steps = {total_r:.3f}")
    # DP 参考解
    V_star = value_iteration(env)
    print(f"V*(k_min, z_L) = {V_star[0, 0]:.3f},  V*(k_max, z_H) = {V_star[-1, -1]:.3f}")
