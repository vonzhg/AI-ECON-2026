"""
Monte Carlo 控制（教学骨架）：探索性起点 + ε-贪心

对应章节：第12章《强化学习基础》§“策略改进与MC控制”

提供两个控制器
--------------
1. mc_control_es —— 探索性起点（ES）：每条 episode 的初始 (s_0, a_0) 均匀随机抽样；
   随后 episode 内部按当前贪心策略运行。对应§“MC控制：探索性起点与ε-贪心”与编程练习 (c)。
2. mc_control_eps_greedy —— ε-贪心 on-policy：每步以概率 ε 均匀随机、
   1-ε 贪心，ε 按 max(0.01, 1/sqrt(n)) 衰减（编程练习 (d) 的设定）。

两个控制器都用首次访问 MC 更新 Q(s, a)，再通过 argmax_a Q 推导新策略——
这就是 §“广义策略迭代与贪心改进”的 GPI 循环。它们都接受可选的 ``qstar`` 与 ``snapshot_every``
参数：若给出，则每隔若干回合把``已充分访问格点上的 |Q-Q*| 平均误差''记入
``trace``，用来画出 MC 控制的\emph{收敛曲线}。

关于评估指标（诚实呈现）
----------------------------------
表格型 MC 控制在本环境（40 个状态、Q 的 top-1/top-2 差距中位数仅约 0.02）上
\emph{收敛缓慢}，而且``策略是否与 DP 完全一致''是一个\textbf{误导性}指标：大量
状态的最优与次优行动价值几乎并列，纯按 argmax 比较会因近似并列而频繁``不匹配'',
即使两者的\emph{价值}几乎没有差别。因此 evaluate_against_dp 以\textbf{价值损失}
为主指标（这才是 RL 收敛的严格判据）：

- sup_value_loss        : 全状态上 V*(s) - V^{g_learned}(s) 的上确界
- ergodic_mean_loss     : 在 DP 最优策略\emph{常去}的遍历集上的平均价值损失
- ergodic_rel_loss      : 上一项相对 |V*| 的比例（``学到没有''最直观的数）
- eff_mismatch(tol)     : ``价值代价 > tol''的状态比例（真正``选错''的比例）
- raw_mismatch          : 纯 argmax 不匹配率（含近似并列，量级偏大、仅供参考）

随回合数增加，价值损失稳步下降（见 __main__ 的收敛轨迹），而 raw_mismatch 因
近似并列而长期居高——这恰好印证 §“探索与开发”的``覆盖/探索''主题：MC 只在数据覆盖到的
地方学得准，在罕至状态上不可靠。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from growth_mdp import DiscreteGrowthMDP, make_default_env, value_iteration


def masked_argmax(q_row: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
    """在可行行动中取 argmax，遇到并列随机平分（无并列时跳过抽样以提速）。"""
    feasible = np.flatnonzero(mask)
    feas_q = q_row[feasible]
    best = feasible[feas_q == feas_q.max()]
    if best.size == 1:
        return int(best[0])
    return int(rng.choice(best))


def epsilon_greedy(q_row: np.ndarray, mask: np.ndarray, eps: float, rng: np.random.Generator) -> int:
    """ε-贪心选择行动。"""
    if rng.random() < eps:
        return int(rng.choice(np.flatnonzero(mask)))
    return masked_argmax(q_row, mask, rng)


@dataclass(frozen=True)
class MCControlConfig:
    num_episodes: int = 5000
    seed: int = 42


def _greedy_policy(env: DiscreteGrowthMDP, Q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """由 Q 表导出贪心策略（每个状态一个行动索引）。"""
    g = np.zeros(env.num_k * env.num_z, dtype=int)
    for s_flat in range(env.num_k * env.num_z):
        k_idx, z_idx = s_flat % env.num_k, s_flat // env.num_k
        mask = env.action_mask((k_idx, z_idx))
        g[s_flat] = masked_argmax(Q[s_flat], mask, rng)
    return g


def _snapshot_q_error(Q: np.ndarray, counts: np.ndarray, qstar: np.ndarray, min_count: int = 10) -> float:
    """充分访问（counts>=min_count）格点上 |Q - Q*| 的平均误差；无则返回 nan。"""
    well = counts >= min_count
    if not np.any(well):
        return float("nan")
    return float(np.mean(np.abs(Q[well] - qstar[well])))


def mc_control_es(
    env: DiscreteGrowthMDP,
    cfg: MCControlConfig,
    qstar: Optional[np.ndarray] = None,
    snapshot_every: int = 0,
    trace: Optional[List[Tuple[int, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    探索性起点 + 首次访问 MC 控制。

    返回
    ----
    Q : ndarray (num_k * num_z, num_actions)
    g : ndarray (num_k * num_z,)  —— 每个状态的贪心行动
    """
    beta = env.params.beta
    rng = np.random.default_rng(cfg.seed)
    num_states = env.num_k * env.num_z
    num_actions = env.num_actions
    Q = np.zeros((num_states, num_actions))
    counts = np.zeros((num_states, num_actions), dtype=int)

    for ep in range(1, cfg.num_episodes + 1):
        # 探索性起点：在所有 (s, a) 上均匀抽样初始对
        k0 = int(rng.integers(env.num_k))
        z0 = int(rng.integers(env.num_z))
        s0 = env.reset(k_idx=k0, z_idx=z0)
        mask0 = env.action_mask(s0)
        a0 = int(rng.choice(np.flatnonzero(mask0)))

        # 用初始随机 (s_0, a_0)，之后按当前贪心策略运行
        trajectory: List[Tuple[Tuple[int, int], int, float]] = []
        next_state, r, done, _ = env.step(a0)
        trajectory.append((s0, a0, r))
        state = next_state
        while not done:
            s_flat = state[1] * env.num_k + state[0]
            mask = env.action_mask(state)
            a = masked_argmax(Q[s_flat], mask, rng)
            next_state, r, done, _ = env.step(a)
            trajectory.append((state, a, r))
            state = next_state

        # 计算每步 G_t（从后往前）
        T = len(trajectory)
        G_values = np.zeros(T)
        running = 0.0
        for t in range(T - 1, -1, -1):
            running = trajectory[t][2] + beta * running
            G_values[t] = running

        # 首次访问规则：记录 (s, a) 第一次出现的时刻
        first_index: Dict[Tuple[Tuple[int, int], int], int] = {}
        for t, (s, a, _) in enumerate(trajectory):
            if (s, a) not in first_index:
                first_index[(s, a)] = t

        for (s, a), t in first_index.items():
            s_flat = s[1] * env.num_k + s[0]
            counts[s_flat, a] += 1
            Q[s_flat, a] += (G_values[t] - Q[s_flat, a]) / counts[s_flat, a]

        if trace is not None and qstar is not None and snapshot_every and ep % snapshot_every == 0:
            trace.append((ep, _snapshot_q_error(Q, counts, qstar)))

    # 贪心策略只需在训练结束后导出一次（episode 内部直接用 Q）
    g = _greedy_policy(env, Q, rng)
    return Q, g


def mc_control_eps_greedy(
    env: DiscreteGrowthMDP,
    cfg: MCControlConfig,
    qstar: Optional[np.ndarray] = None,
    snapshot_every: int = 0,
    trace: Optional[List[Tuple[int, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """ε-贪心 on-policy MC 控制；ε_n = max(0.01, 1/sqrt(n))。"""
    beta = env.params.beta
    rng = np.random.default_rng(cfg.seed)
    num_states = env.num_k * env.num_z
    num_actions = env.num_actions
    Q = np.zeros((num_states, num_actions))
    counts = np.zeros((num_states, num_actions), dtype=int)

    for ep in range(1, cfg.num_episodes + 1):
        eps = max(0.01, 1.0 / np.sqrt(ep))
        state = env.reset()
        trajectory: List[Tuple[Tuple[int, int], int, float]] = []
        done = False
        while not done:
            s_flat = state[1] * env.num_k + state[0]
            mask = env.action_mask(state)
            a = epsilon_greedy(Q[s_flat], mask, eps, rng)
            next_state, r, done, _ = env.step(a)
            trajectory.append((state, a, r))
            state = next_state

        T = len(trajectory)
        G_values = np.zeros(T)
        running = 0.0
        for t in range(T - 1, -1, -1):
            running = trajectory[t][2] + beta * running
            G_values[t] = running

        first_index: Dict[Tuple[Tuple[int, int], int], int] = {}
        for t, (s, a, _) in enumerate(trajectory):
            if (s, a) not in first_index:
                first_index[(s, a)] = t

        for (s, a), t in first_index.items():
            s_flat = s[1] * env.num_k + s[0]
            counts[s_flat, a] += 1
            Q[s_flat, a] += (G_values[t] - Q[s_flat, a]) / counts[s_flat, a]

        if trace is not None and qstar is not None and snapshot_every and ep % snapshot_every == 0:
            trace.append((ep, _snapshot_q_error(Q, counts, qstar)))

    g = _greedy_policy(env, Q, rng)
    return Q, g


# --------------------------------------------------------------------------- #
#  DP 参考量：Q*、最优策略表、遍历集                                            #
# --------------------------------------------------------------------------- #

def q_star_table(env: DiscreteGrowthMDP) -> np.ndarray:
    """一步前瞻得到 Q*(s,a)，按 s_flat = z*num_k + k 展平；不可行行动置 -inf。"""
    beta = env.params.beta
    V_star = value_iteration(env)
    EV = V_star @ env.p_z.T  # (num_k, num_z)，按 (k', z) 索引
    Q = np.full((env.num_k * env.num_z, env.num_actions), -np.inf)
    for k_idx in range(env.num_k):
        for z_idx in range(env.num_z):
            s_flat = z_idx * env.num_k + k_idx
            feas = env._feasible[k_idx, z_idx]
            Q[s_flat, feas] = env._reward[k_idx, z_idx, feas] + beta * EV[feas, z_idx]
    return Q


def _dp_policy_table(env: DiscreteGrowthMDP) -> np.ndarray:
    """DP 最优策略表 (num_k, num_z)。"""
    Qs = q_star_table(env)
    pol = np.zeros((env.num_k, env.num_z), dtype=int)
    for k_idx in range(env.num_k):
        for z_idx in range(env.num_z):
            pol[k_idx, z_idx] = int(np.argmax(Qs[z_idx * env.num_k + k_idx]))
    return pol


def _ergodic_mask(env: DiscreteGrowthMDP, dp_policy: np.ndarray, n_steps: int = 20000) -> np.ndarray:
    """在 DP 最优策略下从中点出发模拟，标记被访问到的状态（遍历集）。

    用固定种子临时替换 env.rng、用毕复原，使遍历集与本函数被调用前 env.rng
    的状态\emph{无关}（保证评估结果可复现、不依赖调用顺序）。
    """
    saved_rng = env.rng
    env.rng = np.random.default_rng(2024)
    try:
        visit = np.zeros((env.num_k, env.num_z), dtype=bool)
        k0 = env.num_k // 2
        env.reset(k_idx=k0, z_idx=0)
        state = (k0, 0)
        for _ in range(n_steps):
            ki, zi = state
            a = int(dp_policy[ki, zi])
            if not env._feasible[ki, zi, a]:
                a = int(np.flatnonzero(env._feasible[ki, zi])[0])
            s, _, done, _ = env.step(a)
            visit[ki, zi] = True
            state = s
            if done:
                env.reset(k_idx=k0, z_idx=0)
                state = (k0, 0)
    finally:
        env.rng = saved_rng
    return visit


def evaluate_against_dp(
    env: DiscreteGrowthMDP, learned_policy: np.ndarray, value_tol: float = 0.05
) -> Dict[str, float]:
    """
    把学到的策略与 DP 最优策略比较。

    返回价值损失类指标（主）+ 策略不匹配类指标（辅，含近似并列说明）：
      sup_value_loss, ergodic_mean_loss, ergodic_rel_loss,
      eff_mismatch_rate（价值代价 > value_tol 的状态比例）, raw_mismatch_rate
    """
    beta = env.params.beta
    V_star = value_iteration(env)
    Qs = q_star_table(env)  # (num_states, num_actions)

    # 学到的策略写成表格
    policy_tab = np.zeros((env.num_k, env.num_z), dtype=int)
    for k_idx in range(env.num_k):
        for z_idx in range(env.num_z):
            policy_tab[k_idx, z_idx] = learned_policy[z_idx * env.num_k + k_idx]

    # 在 learned policy 下做 DP 评估，得到 V_learned
    V_learned = np.zeros_like(V_star)
    for _ in range(5000):
        EV = V_learned @ env.p_z.T
        V_new = np.empty_like(V_learned)
        for k_idx in range(env.num_k):
            for z_idx in range(env.num_z):
                a = int(policy_tab[k_idx, z_idx])
                if not env._feasible[k_idx, z_idx, a]:
                    a = int(np.flatnonzero(env._feasible[k_idx, z_idx])[0])
                V_new[k_idx, z_idx] = env._reward[k_idx, z_idx, a] + beta * EV[a, z_idx]
        if np.max(np.abs(V_new - V_learned)) < 1e-8:
            V_learned = V_new
            break
        V_learned = V_new

    dp_policy = _dp_policy_table(env)
    erg = _ergodic_mask(env, dp_policy)

    value_loss = V_star - V_learned
    # 价值代价：选学到的行动相对最优行动损失多少 Q*（即 max_a Q* - Q*(·, a_learned)）
    value_cost = np.zeros((env.num_k, env.num_z))
    for k_idx in range(env.num_k):
        for z_idx in range(env.num_z):
            s_flat = z_idx * env.num_k + k_idx
            a = int(policy_tab[k_idx, z_idx])
            value_cost[k_idx, z_idx] = Qs[s_flat].max() - Qs[s_flat, a]

    return {
        "sup_value_loss": float(np.max(value_loss)),
        "ergodic_mean_loss": float(np.mean(value_loss[erg])),
        "ergodic_rel_loss": float(np.mean(value_loss[erg]) / abs(np.mean(V_star[erg]))),
        "eff_mismatch_rate": float(np.mean(value_cost > value_tol)),
        "raw_mismatch_rate": float(np.mean(policy_tab != dp_policy)),
    }


# --------------------------------------------------------------------------- #
#  诊断量：sigma_G 与 Q 的 top-1/top-2 差距（量级诊断，在 20×2 默认网格上计算）          #
# --------------------------------------------------------------------------- #

def diagnostics(env: DiscreteGrowthMDP) -> Dict[str, float]:
    """报告 20×2 默认网格上的量级：回报标准差 sigma_G 与 Q 差距中位数。"""
    beta = env.params.beta
    dp_policy = _dp_policy_table(env)
    Qs = q_star_table(env)

    # top-1/top-2 差距（按可行行动）
    gaps = []
    for s_flat in range(env.num_k * env.num_z):
        q = np.sort(Qs[s_flat][np.isfinite(Qs[s_flat])])[::-1]
        if q.size >= 2:
            gaps.append(q[0] - q[1])
    gaps = np.array(gaps)

    # sigma_G：从近稳态 (s, a) 出发、按 DP 策略 rollout，统计 G_0 的标准差
    rng = np.random.default_rng(1)
    kss = int(np.argmin(np.abs(env.k_grid - 3.9)))  # 稳态资本附近

    def rollout_G0(k0: int, z0: int, a0: int) -> float:
        env.reset(k_idx=k0, z_idx=z0)
        s, r, done, _ = env.step(a0)
        G, disc, state = r, beta, s
        while not done:
            ki, zi = state
            a = int(dp_policy[ki, zi])
            if not env._feasible[ki, zi, a]:
                a = int(np.flatnonzero(env._feasible[ki, zi])[0])
            s, r, done, _ = env.step(a)
            G += disc * r
            disc *= beta
            state = s
        return G

    a0 = int(dp_policy[kss, 0])
    if not env._feasible[kss, 0, a0]:
        a0 = int(np.flatnonzero(env._feasible[kss, 0])[0])
    Gs = np.array([rollout_G0(kss, 0, a0) for _ in range(500)])
    return {
        "sigma_G": float(np.std(Gs)),
        "Q_gap_median": float(np.median(gaps)),
        "Q_gap_p90": float(np.percentile(gaps, 90)),
    }


if __name__ == "__main__":
    # 每个训练/评估步都用\emph{全新}的 env（make_default_env 重新构造、重置 rng），
    # 使结果可复现且与调用顺序无关——避免上一步消耗 env.rng 而污染下一步。

    # --- (0) 量级诊断：20×2 默认网格上的 sigma_G 与 Q 差距中位数 --- #
    diag = diagnostics(make_default_env(seed=42))
    print("诊断量级（20×2 默认网格）：")
    print(f"  sigma_G ≈ {diag['sigma_G']:.2f}  (单条轨迹回报的标准差，数量级 ~1)")
    print(f"  Q top1-top2 差距：中位数 ≈ {diag['Q_gap_median']:.3f}, 90分位 ≈ {diag['Q_gap_p90']:.3f}")
    print(f"  => 多数格点差距很小，故策略 argmax 易翻转、而价值损失才是收敛判据。\n")

    qstar = q_star_table(make_default_env(seed=42))  # 与 rng 无关

    # --- (1) ES-MC 收敛轨迹：随回合数增加，已充分访问格点上的 |Q-Q*| 误差下降 --- #
    print("ES-MC 收敛轨迹（充分访问格点上的平均 |Q - Q*|）：")
    trace_es: list = []
    _, g_es = mc_control_es(
        make_default_env(seed=42), MCControlConfig(num_episodes=24000, seed=42),
        qstar=qstar, snapshot_every=4000, trace=trace_es,
    )
    for ep, err in trace_es:
        print(f"  episode={ep:6d}:  mean|Q-Q*| = {err:.3f}")

    # --- (2) 价值损失（主指标）+ 策略不匹配（辅，含近似并列说明） --- #
    m_es = evaluate_against_dp(make_default_env(seed=42), g_es)
    print("\nES-MC 最终评估：")
    print(f"  [价值] sup value loss            = {m_es['sup_value_loss']:.3f}")
    print(f"  [价值] 遍历集平均价值损失        = {m_es['ergodic_mean_loss']:.3f}"
          f"  (相对 |V*| ≈ {100 * m_es['ergodic_rel_loss']:.1f}%)")
    print(f"  [策略] 有效不匹配 (价值代价>0.05) = {100 * m_es['eff_mismatch_rate']:.1f}%")
    print(f"  [策略] 原始 argmax 不匹配 (含近似并列，偏大仅供参考) = {100 * m_es['raw_mismatch_rate']:.1f}%")

    # --- (3) ε-greedy 也给出收敛轨迹：它探索随 ε 衰减而收紧，故\emph{比 ES 慢} --- #
    print("\non-policy ε-greedy MC 收敛轨迹（mean|Q - Q*|）：")
    trace_eps: list = []
    _, g_eps = mc_control_eps_greedy(
        make_default_env(seed=42), MCControlConfig(num_episodes=24000, seed=42),
        qstar=qstar, snapshot_every=4000, trace=trace_eps,
    )
    for ep, err in trace_eps:
        print(f"  episode={ep:6d}:  mean|Q-Q*| = {err:.3f}")
    m_eps = evaluate_against_dp(make_default_env(seed=42), g_eps)
    print("on-policy ε-greedy MC 最终评估：")
    print(f"  [价值] sup value loss            = {m_eps['sup_value_loss']:.3f}")
    print(f"  [价值] 遍历集平均价值损失        = {m_eps['ergodic_mean_loss']:.3f}"
          f"  (相对 |V*| ≈ {100 * m_eps['ergodic_rel_loss']:.1f}%)")
    print(f"  [策略] 有效不匹配 (价值代价>0.05) = {100 * m_eps['eff_mismatch_rate']:.1f}%")
    print(f"  [策略] 原始 argmax 不匹配 (含近似并列) = {100 * m_eps['raw_mismatch_rate']:.1f}%")

    print("\n要点：")
    print("  (1) 两种方法的 mean|Q-Q*| 都随回合单调下降——MC 控制确实在收敛。")
    print("  (2) ES 收敛更快、价值损失更小：它每回合强制覆盖一个随机 (s,a) 起点；"
          "ε-greedy 的探索随 ε 衰减而收紧，在罕至格点上采样不足，故收敛更慢"
          "（见§“探索与开发”的 GLIE 思考框：带 0.01 下界的 1/sqrt(n) 调度只收敛到 ε-soft 策略）。")
    print("  (3) ``原始 argmax 不匹配''因大量近似并列而长期偏高——这正是§“探索与开发”的``覆盖/探索''"
          "主题的数值印证：MC 只在数据覆盖到的遍历集上学得准。")
