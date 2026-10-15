"""
线性函数逼近下的策略评估：最佳线性拟合、TD 不动点、LSTD 与半梯度 TD(0)

对应章节：第13章《强化学习核心算法》§13.3.1 线性逼近与半梯度 TD（subsec:ch13_linear_fa）

设定
----
- 环境：growth_env.py（第12章 20×2 练习环境）。
- 被评估的策略 g_ε：在值函数迭代的最优策略 g* 上构造的 ε-贪心策略（ε=0.1），
  每期以 1-ε 的概率按 g* 选 k'，以 ε 的概率在可行 k' 中均匀抽取。
  它的精确价值 V^{g_ε} 由线性方程组求出，是本脚本全部误差的基准。
- 基函数 φ(k,z) = (1, k̃, k̃², 1{z=z_H}, k̃·1{z=z_H})，k̃ 是把 k 线性标准化到 [0,1] 后的值；
  线性逼近 V̂_ψ(s) = φ(s)ᵀψ。
- 误差一律按 g_ε 的平稳分布 μ 加权：‖V̂_ψ - V^{g_ε}‖_μ = sqrt(Σ_s μ(s)[φ(s)ᵀψ - V^{g_ε}(s)]²)。

打印
----
1. 精确部分
   - 按 μ 加权的最佳线性拟合 ψ_LS 及其误差；
   - TD 不动点 ψ_TD：解 Φᵀ D (Φ - γ P_g Φ) ψ = Φᵀ D r_g（D = diag(μ)，P_g、r_g 为 g_ε 的转移矩阵与期望奖励）；
   - Tsitsiklis–Van Roy（1997）的误差界：‖Φψ_TD - V‖_μ ≤ ‖Φψ_LS - V‖_μ / (1-γ)。
2. 抽样部分：同一条按 g_ε 行动、长 100 万期、中途不重置的样本路径
   - LSTD：用样本矩代替上式的期望，Â = (1/n)Σ_t φ(s_t)[φ(s_t) - γφ(s_{t+1})]ᵀ，
     b̂ = (1/n)Σ_t φ(s_t) r_{t+1}，ψ = Â⁻¹b̂。它是以 φ(s_t) 为工具变量、
     φ(s_t) - γφ(s_{t+1}) 为内生回归元的工具变量估计；
   - 半梯度 TD(0)：ψ ← ψ + η_t δ_t φ(s_t)，δ_t = r_{t+1} + γφ(s_{t+1})ᵀψ - φ(s_t)ᵀψ，
     η_t = 0.2(1 + t/10⁴)^(-0.6)（满足 Robbins–Monro 条件）；
   - 在第 1 万、10 万、100 万期打印系数、误差、与 ψ_TD 的距离。
3. 对照：把精确的 V^{g_ε} 代入，一步 TD 误差的均方根也不小。它来自奖励与冲击的随机性，
   所以 TD 误差的大小不能用来衡量逼近误差；衡量逼近误差要与精确价值比较。

依赖：只用 numpy。运行（在书稿目录下）：python3 -B code/ch13/linear_td_approx.py
"""

from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np

from growth_env import GrowthBenchmark, GrowthEnv, make_env

EPSILON = 0.1
NUM_STEPS = 1_000_000
CHECKPOINTS = (10_000, 100_000, 1_000_000)
START_STATE = (14, 0)          # k=3.71（g* 遍历集内），z=z_L


def features(state: Tuple[int, int], env: GrowthEnv) -> np.ndarray:
    """φ(k,z) = (1, k̃, k̃², 1{z=z_H}, k̃·1{z=z_H})。"""
    k_tilde, _ = env.features01(state)
    z_high = 1.0 if state[1] == env.num_z - 1 else 0.0
    return np.array([1.0, k_tilde, k_tilde ** 2, z_high, k_tilde * z_high])


def feature_matrix(env: GrowthEnv) -> np.ndarray:
    """Φ：第 s 行是展平状态 s 的特征。"""
    return np.stack([features(env.index_to_state(s), env) for s in range(env.num_states)])


def weighted_rmse(values: np.ndarray, target: np.ndarray, mu: np.ndarray) -> float:
    return float(np.sqrt(mu @ (values - target) ** 2))


def simulate_path(env: GrowthEnv, g: np.ndarray, epsilon: float, num_steps: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """按 g_ε 行动的一条连续样本路径；env.episode_length = num_steps，中途不截断、不重置。"""
    rng = np.random.default_rng(seed)
    feasible: List[np.ndarray] = [
        np.flatnonzero(env.action_mask(env.index_to_state(s))) for s in range(env.num_states)
    ]
    states = np.empty(num_steps + 1, dtype=np.int64)
    rewards = np.empty(num_steps)
    s = env.state_index(*env.reset(*START_STATE))
    states[0] = s
    for t in range(num_steps):
        if rng.random() < epsilon:
            a = int(feasible[s][int(rng.random() * feasible[s].size)])
        else:
            a = int(g[s])
        next_state, r, terminated, truncated, _ = env.step(a)
        s = env.state_index(*next_state)
        states[t + 1] = s
        rewards[t] = r
        if terminated or truncated:   # 本模型 terminated 恒为 False；truncated 只在路径末端出现
            assert t == num_steps - 1
    return states, rewards


def lstd(Phi: np.ndarray, states: np.ndarray, rewards: np.ndarray, gamma: float, n: int) -> np.ndarray:
    """前 n 期样本上的 LSTD（工具变量 φ(s_t)，内生回归元 φ(s_t) - γφ(s_{t+1})）。"""
    phi0 = Phi[states[:n]]
    phi1 = Phi[states[1:n + 1]]
    A_hat = phi0.T @ (phi0 - gamma * phi1) / n
    b_hat = phi0.T @ rewards[:n] / n
    return np.linalg.solve(A_hat, b_hat)


def semi_gradient_td(Phi: np.ndarray, states: np.ndarray, rewards: np.ndarray, gamma: float,
                     checkpoints: Tuple[int, ...]) -> List[Tuple[int, np.ndarray]]:
    """同一条路径上的半梯度 TD(0)；终止才去掉自举项，这条路径没有终止。"""
    psi = np.zeros(Phi.shape[1])
    out = []
    marks = set(checkpoints)
    for t in range(len(rewards)):
        phi_s = Phi[states[t]]
        delta = rewards[t] + gamma * (Phi[states[t + 1]] @ psi) - phi_s @ psi
        psi = psi + 0.2 * (1.0 + t / 1e4) ** (-0.6) * delta * phi_s
        if t + 1 in marks:
            out.append((t + 1, psi.copy()))
    return out


def fmt_vec(v: np.ndarray) -> str:
    return "(" + ", ".join(f"{x:7.3f}" for x in v) + ")"


def main() -> None:
    t_start = time.time()
    env = make_env(seed=2024, episode_length=NUM_STEPS)
    bench = GrowthBenchmark(env)
    gamma = env.gamma

    pi = bench.epsilon_greedy_matrix(bench.g_star, EPSILON)
    P_g = bench.transition_matrix(pi)
    r_g = bench.expected_reward(pi)
    V = bench.policy_value_stochastic(pi)
    mu = bench.stationary_distribution(pi)
    D = np.diag(mu)
    Phi = feature_matrix(env)

    print("=" * 78)
    print(f"线性 TD：评估 g* 上 ε={EPSILON} 的 ε-贪心策略 g_ε（20×2 增长模型）")
    print("=" * 78)
    print("基函数 φ = (1, k̃, k̃², 1{z=z_H}, k̃·1{z=z_H})；误差 = 按 g_ε 平稳分布 μ 加权的均方根")
    v_mean = mu @ V
    print(f"V^{{g_ε}} 范围 [{V.min():.3f}, {V.max():.3f}]；按 μ 加权均值 {v_mean:.3f}，"
          f"加权标准差 {weighted_rmse(V, np.full_like(V, v_mean), mu):.3f}（只用常数项时的误差）")

    psi_ls = np.linalg.solve(Phi.T @ D @ Phi, Phi.T @ D @ V)
    err_ls = weighted_rmse(Phi @ psi_ls, V, mu)
    psi_td = np.linalg.solve(Phi.T @ D @ (Phi - gamma * P_g @ Phi), Phi.T @ D @ r_g)
    err_td = weighted_rmse(Phi @ psi_td, V, mu)

    print("\n[精确部分]                     系数 ψ = (常数, k̃, k̃², z_H, k̃·z_H)                 误差")
    print(f"  最佳线性拟合 ψ_LS   {fmt_vec(psi_ls)}   {err_ls:.4f}")
    print(f"  TD 不动点 ψ_TD      {fmt_vec(psi_td)}   {err_td:.4f}")
    print(f"  TD 不动点误差 / 最佳拟合误差 = {err_td / err_ls:.2f}；定理上界 1/(1-γ) = {1 / (1 - gamma):.0f}，"
          f"即误差不超过 {err_ls / (1 - gamma):.3f}")
    print(f"  TD 不动点在各状态上的最大绝对误差 {np.max(np.abs(Phi @ psi_td - V)):.3f}")

    t0 = time.time()
    states, rewards = simulate_path(env, bench.g_star, EPSILON, NUM_STEPS, seed=7)
    t_sim = time.time() - t0
    print(f"\n[抽样部分] 一条按 g_ε 行动的样本路径，起点 (k={env.k_grid[START_STATE[0]]:.2f}, z_L)，"
          f"共 {NUM_STEPS:,d} 期（模拟用时 {t_sim:.0f} 秒）")
    print("  方法 / 样本期数                 系数 ψ                                             误差   ‖ψ-ψ_TD‖")
    for n in CHECKPOINTS:
        psi = lstd(Phi, states, rewards, gamma, n)
        print(f"  LSTD      n={n:>9,d} {fmt_vec(psi)}   {weighted_rmse(Phi @ psi, V, mu):.4f}   "
              f"{np.linalg.norm(psi - psi_td):.3f}")
    t0 = time.time()
    for n, psi in semi_gradient_td(Phi, states, rewards, gamma, CHECKPOINTS):
        print(f"  半梯度TD  n={n:>9,d} {fmt_vec(psi)}   {weighted_rmse(Phi @ psi, V, mu):.4f}   "
              f"{np.linalg.norm(psi - psi_td):.3f}")
    print(f"  （半梯度 TD 用时 {time.time() - t0:.0f} 秒；步长 η_t = 0.2(1+t/10⁴)^-0.6，初值 ψ=0）")

    delta_exact = rewards + gamma * V[states[1:]] - V[states[:-1]]
    delta_td = rewards + gamma * (Phi @ psi_td)[states[1:]] - (Phi @ psi_td)[states[:-1]]
    print("\n[对照] 沿同一条路径的一步 TD 误差均方根："
          f"代入精确 V^{{g_ε}} 为 {np.sqrt(np.mean(delta_exact ** 2)):.3f}，"
          f"代入 TD 不动点为 {np.sqrt(np.mean(delta_td ** 2)):.3f}")
    print("  两者几乎相同：TD 误差主要来自奖励与冲击的随机性，不能用来衡量逼近误差。")
    print(f"\n总耗时 {time.time() - t_start:.0f} 秒")


if __name__ == "__main__":
    main()
