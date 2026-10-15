"""
第12章贯穿案例：7×2 离散增长模型上的四组数值实验（E0–E3）

对应章节：第12章《强化学习基础》
- E0：校准、Q* 表、最优与次优行动的差距、单条轨迹回报的标准差（§“样本备份与最基本的 Monte Carlo 方法”）
- E1：单个格点上的贪心改进——估计精度与样本覆盖（§“广义策略迭代与贪心改进”示例框）
- E2：全模型上的纯贪心、探索性起点与 ε-贪心 MC 控制（§“MC 控制”示例框）
- E3：on-policy 与 off-policy：重要性采样的方差（§“On-policy 与 off-policy 方法”示例框）
- E4：同一格点上 ε-贪心与 Boltzmann 探索的选择概率（§“探索与开发”）

正文中这些示例框里的每一个数字、以及图中的坐标，都由本脚本打印。修改本脚本后必须同步修改正文。

设计要点
--------
- 模型：资本网格 7 点（1.5 到 6.0 等距），TFP 取 0.9 与 1.1，转移矩阵对角 0.8；
  β=0.95，δ=0.10，α=0.36，u(c)=ln c。状态 s=(k,z) 按 s = z*7 + k 展平。
- 截断：每条轨迹模拟 T_TRUNC=300 期；首次访问只使用 t < T_FIRST=150 的回报，
  被省略的尾部不超过 β^150·max|V*|。
- 福利度量：对数效用下的消费等价损失 ℓ(s) = exp{(1-β)[V*(s) - V^g(s)]} - 1。
- 默认不写任何文件；全部结果打印到标准输出。固定随机种子，重复运行输出一致。

运行：python3 -B code/ch12/running_example.py
"""

from __future__ import annotations

import math
import random
import sys
import time
from typing import Dict, List

import numpy as np

from growth_mdp import DiscreteGrowthMDP

# ----------------------------------------------------------------------------
# 校准
# ----------------------------------------------------------------------------
K_GRID = np.linspace(1.5, 6.0, 7)
Z_GRID = np.array([0.9, 1.1])
P_Z = np.array([[0.8, 0.2], [0.2, 0.8]])
T_TRUNC = 300
T_FIRST = 150
BAR_K, BAR_Z = 3, 0  # 贯穿案例的特定格点 \bar s = (k=3.75, z=0.9)

ENV = DiscreteGrowthMDP(K_GRID, Z_GRID, P_Z, episode_length=T_TRUNC, seed=0)
BETA = ENV.params.beta
NK, NZ = ENV.num_k, ENV.num_z
NS, NA = NK * NZ, NK


def sidx(k: int, z: int) -> int:
    return z * NK + k


# 展平后的奖励、可行集与转移核
FEAS = np.zeros((NS, NA), dtype=bool)
R = np.full((NS, NA), -np.inf)
for k in range(NK):
    for z in range(NZ):
        FEAS[sidx(k, z)] = ENV._feasible[k, z]
        R[sidx(k, z)] = ENV._reward[k, z]
RZ = np.where(FEAS, R, 0.0)
P = np.zeros((NS, NA, NS))
for k in range(NK):
    for z in range(NZ):
        for a in range(NA):
            for z2 in range(NZ):
                P[sidx(k, z), a, sidx(a, z2)] = P_Z[z, z2]
FEAS_LIST = [list(np.flatnonzero(FEAS[s])) for s in range(NS)]
S_BAR = sidx(BAR_K, BAR_Z)


def fmt(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}"


# ----------------------------------------------------------------------------
# DP 基准：值函数迭代、策略价值、消费等价损失、遍历集
# ----------------------------------------------------------------------------
def value_iteration(tol: float = 1e-6):
    V = np.zeros(NS)
    n = 0
    while True:
        Q = np.where(FEAS, RZ + BETA * P @ V, -np.inf)
        V_new = Q.max(axis=1)
        n += 1
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, n
        V = V_new


def exact_solution():
    V = np.zeros(NS)
    for _ in range(20000):
        V_new = np.where(FEAS, RZ + BETA * P @ V, -np.inf).max(axis=1)
        if np.max(np.abs(V_new - V)) < 1e-13:
            V = V_new
            break
        V = V_new
    Q = np.where(FEAS, RZ + BETA * P @ V, -np.inf)
    return V, Q, Q.argmax(axis=1)


V_STAR, Q_STAR, G_STAR = exact_solution()


def policy_value(g: np.ndarray) -> np.ndarray:
    idx = np.arange(NS)
    return np.linalg.solve(np.eye(NS) - BETA * P[idx, g], RZ[idx, g])


def ce_loss(g: np.ndarray) -> np.ndarray:
    return np.exp((1 - BETA) * (V_STAR - policy_value(g))) - 1.0


def ergodic_set(g: np.ndarray) -> np.ndarray:
    Pg = P[np.arange(NS), g]
    mu = np.zeros(NS)
    mu[S_BAR] = 1.0
    for _ in range(5000):
        mu = mu @ Pg
    return mu > 1e-10


ERGODIC = ergodic_set(G_STAR)


# ----------------------------------------------------------------------------
# 向量化模拟：从 (s0, a0) 出发，此后按确定性策略 g 行动，返回 T 期贴现回报
# ----------------------------------------------------------------------------
def simulate_returns(s0: int, a0: int, g: np.ndarray, n: int, T: int, rng: np.random.Generator) -> np.ndarray:
    s = np.full(n, s0)
    a = np.full(n, a0)
    G = np.zeros(n)
    disc = 1.0
    for t in range(T):
        if t > 0:
            a = g[s]
        G += disc * R[s, a]
        z = s // NK
        z_next = np.where(rng.random(n) < P_Z[z, 0], 0, 1)
        s = z_next * NK + a
        disc *= BETA
    return G


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ----------------------------------------------------------------------------
# E0
# ----------------------------------------------------------------------------
def run_e0() -> Dict[str, float]:
    print("=" * 78)
    print("E0  校准与 Q* 表")
    print("=" * 78)
    alpha, delta = ENV.params.alpha, ENV.params.delta
    k_ss = (alpha * BETA / (1 - BETA * (1 - delta))) ** (1 / (1 - alpha))
    _, n_vfi = value_iteration(1e-6)
    print(f"资本网格: {np.round(K_GRID, 3).tolist()}；z 网格: {Z_GRID.tolist()}；P_z 对角 0.8")
    print(f"状态数 {NS}，可行格子数 {int(FEAS.sum())}；z=1 时的确定性稳态资本 {k_ss:.2f}")
    print(f"值函数迭代（V_0=0，容差 1e-6）轮数: {n_vfi}；理论近似 ln(1e-6)/ln(β) = {math.log(1e-6)/math.log(BETA):.0f}")
    print(f"V* 范围: [{V_STAR.min():.3f}, {V_STAR.max():.3f}]；max|V*| = {np.abs(V_STAR).max():.3f}")
    print(f"截断尾部上界 β^(T_TRUNC-T_FIRST)·max|V*| = {BETA ** (T_TRUNC - T_FIRST) * np.abs(V_STAR).max():.4f}")
    print("g* 的 k' 索引（行 z=0.9 / z=1.1）:")
    print(G_STAR.reshape(NZ, NK))
    print(f"g* 的遍历集（从 s̄ 出发）: {[(int(s % NK), int(s // NK)) for s in np.flatnonzero(ERGODIC)]}")

    gaps = []
    for s in range(NS):
        q = np.sort(Q_STAR[s][FEAS[s]])[::-1]
        if q.size > 1:
            gaps.append(q[0] - q[1])
    gaps = np.array(gaps)
    p10, p50, p90 = np.percentile(gaps, [10, 50, 90])
    print(f"最优与次优 k' 的 Q* 差距：p10={p10:.3f} 中位数={p50:.3f} p90={p90:.3f}")

    rng = np.random.default_rng(2026)
    G = simulate_returns(S_BAR, int(G_STAR[S_BAR]), G_STAR, 20000, T_TRUNC, rng)
    sigma_G = float(G.std())
    print(f"σ_G（s̄ 处按 g* 行动，T={T_TRUNC}，20000 条）= {sigma_G:.3f}；样本均值 {G.mean():.3f} vs V*(s̄) {V_STAR[S_BAR]:.3f}")
    for N in (100, 10000):
        print(f"  N={N}: 标准误 σ_G/√N = {sigma_G / math.sqrt(N):.4f}")
    n_needed = 2 * (1.645 * sigma_G / p50) ** 2
    print(f"两两比较以 95% 把握分开中位数差距所需每格轨迹数 2(1.645σ/Δ)^2 ≈ {n_needed:.0f}")

    print("\nQ* 表（z=0.9 半张；行=当期资本 k，列=下一期资本 k'；— 为不可行；* 为行最大值）")
    header = "k \\ k' & " + " & ".join(f"{x:.2f}" for x in K_GRID) + r" \\"
    print(header)
    for k in range(NK):
        s = sidx(k, 0)
        cells = []
        for a in range(NA):
            if not FEAS[s, a]:
                cells.append("—")
            else:
                v = f"{Q_STAR[s, a] + 0.0:.3f}".replace("-0.000", "0.000")
                cells.append(r"\textbf{" + v + "}" if a == G_STAR[s] else v)
        print(f"{K_GRID[k]:.2f} & " + " & ".join(cells) + r" \\")
    return {"sigma_G": sigma_G, "gap_median": float(p50)}


# ----------------------------------------------------------------------------
# E1：单个格点 s̄ 上的贪心改进
# ----------------------------------------------------------------------------
def run_e1(reps: int = 2000, rounds: int = 300, pool_size: int = 40000) -> None:
    print("\n" + "=" * 78)
    print("E1  单个格点 s̄=(k=3.75, z=0.9)：估计精度与样本覆盖")
    print("=" * 78)
    arms = FEAS_LIST[S_BAR]
    q_true = np.array([Q_STAR[S_BAR, a] for a in arms])
    A = len(arms)
    best = int(np.argmax(q_true))
    order = np.argsort(q_true)[::-1]
    gap = q_true[order[0]] - q_true[order[1]]
    print("可行 k':", [round(float(K_GRID[a]), 2) for a in arms])
    print("Q*(s̄,·):", [fmt(x) for x in q_true])
    print(f"最优 k'={K_GRID[arms[best]]:.2f}；次优 k'={K_GRID[arms[order[1]]]:.2f}；差距 Δ={gap:.3f}；"
          f"折合永久消费 {100 * (math.exp((1 - BETA) * gap) - 1):.2f}%")

    rng = np.random.default_rng(12)
    pool = np.stack([simulate_returns(S_BAR, a, G_STAR, pool_size, T_TRUNC, rng) for a in arms])
    print("各行动回报标准差:", [fmt(x, 2) for x in pool.std(axis=1)])

    rules = ["greedy", "full", "sched", "rand"]
    names = {"greedy": "纯贪心", "full": "全面采样", "sched": "定期复查", "rand": "随机复查"}
    L, EPS = 10, 0.1
    marks = [0, 20, 100, 300]
    curves = {}
    for N0, lab in [(50, "精确"), (1, "粗糙")]:
        for rule in rules:
            r = np.random.default_rng(1000 + N0 * 10 + rules.index(rule))
            n = np.full((reps, A), float(N0))
            q = pool[np.arange(A)[None, :, None], r.integers(pool_size, size=(reps, A, N0))].mean(axis=2)
            hit = np.zeros(rounds + 1)
            dep = 0.0
            exe = 0.0
            nsamp = 0
            rows = np.arange(reps)
            for t in range(rounds + 1):
                g = np.argmax(q, axis=1)
                hit[t] = np.mean(g == best)
                if t == rounds:
                    break
                dep += np.mean(q_true[best] - q_true[g])
                if rule == "greedy" or (rule == "sched" and t % L != L - 1):
                    sample_mask = np.zeros((reps, A), dtype=bool)
                    sample_mask[rows, g] = True
                    exe += np.mean(q_true[best] - q_true[g])
                elif rule == "rand":
                    explore = r.random(reps) < EPS
                    i = np.where(explore, r.integers(A, size=reps), g)
                    sample_mask = np.zeros((reps, A), dtype=bool)
                    sample_mask[rows, i] = True
                    exe += np.mean(q_true[best] - q_true[i])
                else:  # full，或定期复查的复查轮
                    sample_mask = np.ones((reps, A), dtype=bool)
                x = pool[np.arange(A)[None, :], r.integers(pool_size, size=(reps, A))]
                n = n + sample_mask
                q = np.where(sample_mask, q + (x - q) / n, q)
                nsamp += int(sample_mask.sum())
            curves[(lab, rule)] = hit
            exe_txt = (f"；学习期实际执行行动的平均损失 {exe / rounds:.3f}"
                       f"（折合永久消费 {100 * (math.exp((1 - BETA) * exe / rounds) - 1):.2f}%）") if rule in ("greedy", "rand") else ""
            print(f"{lab}+{names[rule]}: 选对概率 " + "、".join(f"t={m}:{hit[m]:.2f}" for m in marks)
                  + f"；每轮样本数 {nsamp / (reps * rounds):.2f}；部署决策平均损失 {dep / rounds:.3f}"
                  + f"（折合永久消费 {100 * (math.exp((1 - BETA) * dep / rounds) - 1):.2f}%）" + exe_txt)

    # 两两比较的正态近似（只含最优与次优两个行动）
    i1, i2 = order[0], order[1]
    sig_pair = math.sqrt((pool[i1].var() + pool[i2].var()) / 2)
    print(f"\n两行动核对（k'={K_GRID[arms[i1]]:.2f} 与 {K_GRID[arms[i2]]:.2f}，σ={sig_pair:.3f}）：")
    r = np.random.default_rng(77)
    for N0 in (1, 50):
        approx = normal_cdf(gap / (sig_pair * math.sqrt(2 / N0)))
        m1 = pool[i1][r.integers(pool_size, size=(200000, N0))].mean(axis=1)
        m2 = pool[i2][r.integers(pool_size, size=(200000, N0))].mean(axis=1)
        print(f"  N0={N0}: 正态近似选对概率 {approx:.3f}；模拟 {np.mean(m1 > m2):.3f}")

    print("\nTikZ 坐标（每 10 轮）:")
    for key, hit in curves.items():
        pts = " ".join(f"({t},{hit[t]:.3f})" for t in range(0, rounds + 1, 10))
        print(f"% {key[0]}+{names[key[1]]}\n\\addplotcoords{{{pts}}}")


# ----------------------------------------------------------------------------
# E2：全模型 MC 控制（首次访问）
# ----------------------------------------------------------------------------
R_LIST = [[float(R[s, a]) for a in range(NA)] for s in range(NS)]
PZ0 = [float(P_Z[z, 0]) for z in range(NZ)]
BETA_POW = [BETA ** t for t in range(T_TRUNC)]
G0_POLICY = np.array([s % NK if FEAS[s, s % NK] else FEAS_LIST[s][-1] for s in range(NS)])  # 保持资本不变


def episode(s: int, a0: int, Q: List[List[float]], eps: float, rnd: random.Random, cont=None):
    """生成一条 T_TRUNC 期轨迹；cont 给定时按固定策略 cont 行动，否则按 Q 的 ε-贪心行动。"""
    S = [0] * T_TRUNC
    A = [0] * T_TRUNC
    Rw = [0.0] * T_TRUNC
    a = a0
    for t in range(T_TRUNC):
        if t > 0:
            feas = FEAS_LIST[s]
            if cont is not None:
                a = cont[s]
            elif eps > 0 and rnd.random() < eps:
                a = feas[int(rnd.random() * len(feas))]
            else:
                row = Q[s]
                a = max(feas, key=row.__getitem__)
        S[t], A[t], Rw[t] = s, a, R_LIST[s][a]
        z = s // NK
        s = (0 if rnd.random() < PZ0[z] else 1) * NK + a
    G = [0.0] * T_TRUNC
    acc = 0.0
    for t in range(T_TRUNC - 1, -1, -1):
        acc = Rw[t] + BETA * acc
        G[t] = acc
    return S, A, G


def run_control(method: str, seed: int, episodes: int, checkpoints: List[int]) -> List[dict]:
    rnd = random.Random(seed)
    Q = [[-math.inf] * NA for _ in range(NS)]
    C = [[0] * NA for _ in range(NS)]
    g0 = [int(x) for x in G0_POLICY]
    for s in range(NS):  # 初始评估：每格一条轨迹，此后按“保持资本不变”行动
        for a in FEAS_LIST[s]:
            _, _, G = episode(s, a, Q, 0.0, rnd, cont=g0)
            Q[s][a] = G[0]
            C[s][a] = 1
    out = []

    def snapshot(e: int):
        g = np.array([max(FEAS_LIST[s], key=Q[s].__getitem__) for s in range(NS)])
        loss = ce_loss(g)
        cnt = np.array(C)
        out.append({
            "episode": e,
            "ce_all": float(loss.mean()),
            "ce_erg": float(loss[ERGODIC].mean()),
            "cells10": int((cnt[FEAS] >= 10).sum()),
            "wrong": int((g != G_STAR).sum()),
            "bar_ok": bool(g[S_BAR] == G_STAR[S_BAR]),
        })

    if 0 in checkpoints:
        snapshot(0)
    for e in range(1, episodes + 1):
        if method == "es":
            s0 = int(rnd.random() * NS)
            feas = FEAS_LIST[s0]
            a0 = feas[int(rnd.random() * len(feas))]
            eps = 0.0
        else:
            s0 = S_BAR
            eps = 0.0 if method == "greedy" else (0.1 if method == "eps" else 1 / math.sqrt(e))
            feas = FEAS_LIST[s0]
            if eps > 0 and rnd.random() < eps:
                a0 = feas[int(rnd.random() * len(feas))]
            else:
                a0 = max(feas, key=Q[s0].__getitem__)
        S, A, G = episode(s0, a0, Q, eps, rnd)
        seen = set()
        for t in range(T_FIRST):
            key = S[t] * NA + A[t]
            if key in seen:
                continue
            seen.add(key)
            s, a = S[t], A[t]
            C[s][a] += 1
            Q[s][a] += (G[t] - Q[s][a]) / C[s][a]
        if e in checkpoints:
            snapshot(e)
    return out


def run_e2(seeds: int = 20, episodes: int = 6000) -> None:
    print("\n" + "=" * 78)
    print(f"E2  全模型首次访问 MC 控制（{seeds} 个种子，{episodes} 回合；纯贪心与 ε-贪心固定从 s̄ 出发）")
    print("=" * 78)
    ce_g0 = ce_loss(G0_POLICY)
    print(f"初始策略“保持资本不变”的消费等价损失：全部状态 {100 * ce_g0.mean():.2f}%，遍历集 {100 * ce_g0[ERGODIC].mean():.2f}%")
    checkpoints = [0, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 6000]
    checkpoints = [c for c in checkpoints if c <= episodes]
    names = {"greedy": "纯贪心（固定起点）", "es": "探索性起点（随机重置）", "eps": "ε=0.1（固定起点）", "glie": "ε=1/√n（固定起点）"}
    for method in ["greedy", "es", "eps", "glie"]:
        t0 = time.time()
        runs = [run_control(method, 500 + sd, episodes, checkpoints) for sd in range(seeds)]
        print(f"\n{names[method]}（{time.time() - t0:.0f} 秒）")
        for j, e in enumerate(checkpoints):
            ce_all = np.array([r[j]["ce_all"] for r in runs]) * 100
            ce_erg = np.array([r[j]["ce_erg"] for r in runs]) * 100
            cells = np.array([r[j]["cells10"] for r in runs])
            wrong = np.array([r[j]["wrong"] for r in runs])
            bar = np.mean([r[j]["bar_ok"] for r in runs])
            print(f"  e={e:5d}: CE全部 中位数 {np.median(ce_all):.2f}% [{ce_all.min():.2f},{ce_all.max():.2f}]"
                  f"；CE遍历集 中位数 {np.median(ce_erg):.2f}% [{ce_erg.min():.2f},{ce_erg.max():.2f}]"
                  f"；≥10样本格子 中位数 {np.median(cells):.0f} [{cells.min()},{cells.max()}]"
                  f"；错选状态 中位数 {np.median(wrong):.0f} [{wrong.min()},{wrong.max()}]；s̄ 选对比例 {bar:.2f}")
        print("  TikZ 坐标 CE全部中位数:", " ".join(
            f"({e},{np.median([r[j]['ce_all'] for r in runs]) * 100:.3f})" for j, e in enumerate(checkpoints) if e > 0))
        print("  TikZ 坐标 ≥10样本格子中位数:", " ".join(
            f"({e},{np.median([r[j]['cells10'] for r in runs]):.1f})" for j, e in enumerate(checkpoints) if e > 0))


# ----------------------------------------------------------------------------
# E3：on-policy 与 off-policy
# ----------------------------------------------------------------------------
def behavior_matrix(eps: float) -> np.ndarray:
    B = np.zeros((NS, NA))
    for s in range(NS):
        B[s, FEAS_LIST[s]] = eps / len(FEAS_LIST[s])
        B[s, G_STAR[s]] += 1 - eps
    return B


def t_period_value(pi: np.ndarray, T: int) -> np.ndarray:
    v = np.zeros(NS)
    for _ in range(T):
        v = (pi * RZ).sum(axis=1) + BETA * np.einsum("sa,sat,t->s", pi, P, v)
    return v


def eps_soft_optimum(eps: float):
    """ε-软策略迭代：在“以 1-ε 选贪心、以 ε 均匀试探”的策略族中求最优（Sutton–Barto §5.4）。"""
    g = G_STAR.copy()
    for _ in range(100):
        pi = np.zeros((NS, NA))
        for s in range(NS):
            pi[s, FEAS_LIST[s]] = eps / len(FEAS_LIST[s])
            pi[s, g[s]] += 1 - eps
        V = np.linalg.solve(np.eye(NS) - BETA * np.einsum("sa,sat->st", pi, P), (pi * RZ).sum(axis=1))
        Q = np.where(FEAS, RZ + BETA * P @ V, -np.inf)
        g_new = Q.argmax(axis=1)
        if np.array_equal(g_new, g):
            return g, V, Q
        g = g_new
    return g, V, Q


def run_e3(n_traj: int = 500, reps: int = 200) -> None:
    print("\n" + "=" * 78)
    print(f"E3  on-policy 与 off-policy（从 s̄ 出发；每次 {n_traj} 条轨迹，重复 {reps} 次）")
    print("=" * 78)
    for eps in (0.1, 0.3):
        g_e, V_e, Q_e = eps_soft_optimum(eps)
        q_row = np.sort(Q_e[S_BAR][FEAS[S_BAR]])[::-1]
        dv = V_STAR[S_BAR] - V_e[S_BAR]
        print(f"最优 ε-软策略（ε={eps}）：贪心部分与 g* 不同的状态数 {int((g_e != G_STAR).sum())}；"
              f"s̄ 上贪心部分选 k'={K_GRID[g_e[S_BAR]]:.2f}，与次优差距 {q_row[0] - q_row[1]:.3f}；"
              f"V_ε(s̄)={V_e[S_BAR]:.3f} vs V*(s̄)={V_STAR[S_BAR]:.3f}，折合永久消费 {100 * (math.exp((1 - BETA) * dv) - 1):.2f}%")
    target = np.zeros((NS, NA))
    target[np.arange(NS), G_STAR] = 1.0
    rng = np.random.default_rng(33)
    feas_sorted = np.argsort(~FEAS, axis=1, kind="stable")  # 每行可行行动排在前面
    nfeas_all = FEAS.sum(axis=1)
    for eps in (0.1, 0.3):
        B = behavior_matrix(eps)
        for T in (5, 10, 20, 50):
            v_t = t_period_value(target, T)[S_BAR]
            v_b = t_period_value(B, T)[S_BAR]
            m = n_traj * reps
            s = np.full(m, S_BAR)
            G = np.zeros(m)
            w = np.ones(m)
            disc = 1.0
            for t in range(T):
                explore = rng.random(m) < eps
                # 在可行集中均匀抽取：第 j 个可行行动
                j = (rng.random(m) * nfeas_all[s]).astype(int)
                a_rand = feas_sorted[s, j]
                a = np.where(explore, a_rand, G_STAR[s])
                b_prob = B[s, a]
                g_prob = target[s, a]
                w *= g_prob / b_prob
                G += disc * R[s, a]
                z = s // NK
                s = np.where(rng.random(m) < P_Z[z, 0], 0, 1) * NK + a
                disc *= BETA
            G = G.reshape(reps, n_traj)
            w = w.reshape(reps, n_traj)
            ois = (w * G).mean(axis=1)
            wsum = w.sum(axis=1)
            ok = wsum > 0
            wis = (w * G).sum(axis=1)[ok] / wsum[ok]
            on_policy = G.mean(axis=1)
            wis_txt = f"{wis.mean():.3f}（{wis.std():.3f}）" if ok.sum() > 1 else "无可用轨迹"
            print(f"ε={eps}, T={T}: 目标价值 {v_t:.3f}；行为策略价值 {v_b:.3f}（on-policy 样本均值 {on_policy.mean():.3f}）"
                  f"；普通IS {ois.mean():.3f}（{ois.std():.3f}）；加权IS {wis_txt}"
                  f"［可用重复 {ok.sum()}/{reps}］；权重非零比例 {100 * np.mean(w > 0):.1f}%"
                  f"；价值差折合 T 期消费 {100 * (math.exp((v_t - v_b) * (1 - BETA) / (1 - BETA ** T)) - 1):.2f}%")


# ----------------------------------------------------------------------------
# E4：ε-贪心与 Boltzmann 在 s̄ 上的选择概率
# ----------------------------------------------------------------------------
def run_e4() -> None:
    print("\n" + "=" * 78)
    print("E4  s̄ 上 ε-贪心与 Boltzmann 的选择概率（以 Q* 行为估计值）")
    print("=" * 78)
    arms = FEAS_LIST[S_BAR]
    q = np.array([Q_STAR[S_BAR, a] for a in arms])
    best = q.max()
    print("k':", [f"{K_GRID[a]:.2f}" for a in arms], " Q*:", [fmt(x) for x in q])
    p_eps = np.full(len(q), 0.1 / len(q))
    p_eps[np.argmax(q)] += 0.9
    print("ε=0.1:", [fmt(x) for x in p_eps], f" 每次决策期望价值损失 {np.dot(p_eps, best - q):.4f}")
    for tau in (0.1, 0.3, 1.0):
        p = np.exp((q - best) / tau)
        p /= p.sum()
        print(f"τ={tau}:", [fmt(x) for x in p], f" 每次决策期望价值损失 {np.dot(p, best - q):.4f}")


def main() -> None:
    quick = "--quick" in sys.argv
    t0 = time.time()
    run_e0()
    run_e1(reps=400 if quick else 2000)
    run_e2(seeds=3 if quick else 20, episodes=1000 if quick else 6000)
    run_e3(reps=50 if quick else 200)
    run_e4()
    print(f"\n总耗时 {time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
