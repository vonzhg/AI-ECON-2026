"""
第13章贯穿案例：沿用第12章 7×2 离散增长模型的五组数值实验（E5–E9）

对应章节：第13章《强化学习核心算法》
- E5：同一条样本路径上的 TD 与 MC（§13.1.4 示例框；表 tab:ch13_td_mc，图 fig:ch13_td_mc_error）
- E6：增长模型上的 SARSA 与 Q 学习（§13.2.3 示例框；表 tab:ch13_sarsa_q）
- E7：线性 TD、最佳拟合与 LSTD（§13.3.1 示例框；表 tab:ch13_linear_td）
- E8：基线降低策略梯度的方差（§13.4.2 示例框；表 tab:ch13_baseline）
- E9：同一数据协议下的演员—评论家、REINFORCE、SARSA 与 Q 学习（§13.5.2 示例框；图 fig:ch13_ac_growth，表 tab:ch13_all_methods）

正文中这些示例框里的每一个数字、以及图中的坐标，都由本脚本打印。修改本脚本后必须同步修改正文。

设计要点
--------
- 模型、精确解（V*、Q*、g*）与消费等价损失全部从 code/ch12/running_example.py 导入，
  保证与第12章 E0–E4 是同一个模型、同一套精确解。
- 被评估策略 g_ε：以 g* 为基础、ε=0.1 的 ε-贪心策略（第12章 E3 的行为策略），价值可精确求出。
- 误差与“加权”损失都按 g_ε 下马尔可夫链的平稳分布 d̄ 加权；“全部”指 14 个状态的简单平均。
- 无穷期问题没有终止状态：TD 类更新始终保留自举项。E5–E8 沿连续样本路径进行；E9 中“重置”指每 300 期
  把模拟器重置到均匀抽取的状态，重置只改变下一段的起点，不把任何价值置零。
- 初值 ln c_ss/(1-γ) 是“稳态消费永久持续”的价值（约 4.27）。
- 各实验对 20 条独立链（种子）取中位数；只依赖 numpy；固定随机种子，重复运行输出一致；默认不写文件。

运行：python3 -B code/ch13/running_example.py          （完整版，正文数字以此为准）
      python3 -B code/ch13/running_example.py --quick  （冒烟测试）
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import time

import numpy as np

# ----------------------------------------------------------------------------
# 从第12章导入同一个模型与精确解
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
CH12 = os.path.normpath(os.path.join(HERE, "..", "ch11"))
sys.path.insert(0, CH12)
_spec = importlib.util.spec_from_file_location("ch12_running_example", os.path.join(CH12, "running_example.py"))
ch12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ch12)

NS, NA, NK, NZ = ch12.NS, ch12.NA, ch12.NK, ch12.NZ
K_GRID, Z_GRID, P_Z = ch12.K_GRID, ch12.Z_GRID, ch12.P_Z
GAMMA = ch12.BETA
FEAS, R, RZ, P = ch12.FEAS, ch12.R, ch12.RZ, ch12.P
FEAS_LIST = ch12.FEAS_LIST
S_BAR = ch12.S_BAR
V_STAR, Q_STAR, G_STAR = ch12.V_STAR, ch12.Q_STAR, ch12.G_STAR
NFEAS = FEAS.sum(axis=1)
FEAS_SORTED = np.argsort(~FEAS, axis=1, kind="stable")  # 每行可行行动排在前面（k' 升序）
KS = np.array([K_GRID[s % NK] for s in range(NS)])
ZS = np.array([Z_GRID[s // NK] for s in range(NS)])

QUICK = "--quick" in sys.argv
SEEDS = 4 if QUICK else 20

_prm = ch12.ENV.params
K_SS = (_prm.alpha / (1 / GAMMA - 1 + _prm.delta)) ** (1 / (1 - _prm.alpha))
C_SS = K_SS ** _prm.alpha - _prm.delta * K_SS
V0 = math.log(C_SS) / (1 - GAMMA)


def sname(s: int) -> str:
    return f"(k={KS[s]:.2f}, z={ZS[s]:.1f})"


# ----------------------------------------------------------------------------
# 策略、价值、平稳分布与福利度量
# ----------------------------------------------------------------------------
def eps_greedy(g: np.ndarray, eps: float) -> np.ndarray:
    pi = np.zeros((NS, NA))
    for s in range(NS):
        pi[s, FEAS_LIST[s]] = eps / NFEAS[s]
        pi[s, g[s]] += 1 - eps
    return pi


def chain(pi: np.ndarray) -> np.ndarray:
    return np.einsum("sa,sat->st", pi, P)


def value_of(pi: np.ndarray) -> np.ndarray:
    return np.linalg.solve(np.eye(NS) - GAMMA * chain(pi), (pi * RZ).sum(axis=1))


def det_value(g: np.ndarray) -> np.ndarray:
    ar = np.arange(NS)
    return np.linalg.solve(np.eye(NS) - GAMMA * P[ar, g], RZ[ar, g])


def stationary(pi: np.ndarray) -> np.ndarray:
    Pp = chain(pi)
    d = np.zeros(NS)
    d[S_BAR] = 1.0
    for _ in range(20000):
        d = d @ Pp
    return d / d.sum()


def ce_from_value(V: np.ndarray) -> np.ndarray:
    """对数效用下的消费等价损失 ℓ_CE(s) = exp{(1-γ)[V*(s) - V(s)]} - 1。"""
    return np.exp((1 - GAMMA) * (V_STAR - V)) - 1.0


PI_EPS = eps_greedy(G_STAR, 0.1)
V_EPS = value_of(PI_EPS)
D_EPS = stationary(PI_EPS)
PI_UNIF = eps_greedy(G_STAR, 1.0)  # ε=1 即在可行集上均匀


def losses_det(Grows: np.ndarray):
    """一组确定性策略（rows, NS）的消费等价损失：返回（全部平均，d̄ 加权，错选状态数）。"""
    L = np.array([ce_from_value(det_value(g)) for g in Grows])
    return L.mean(axis=1), L @ D_EPS, (Grows != G_STAR).sum(axis=1)


def losses_stoch(PIrows: np.ndarray):
    L = np.array([ce_from_value(value_of(pi)) for pi in PIrows])
    return L.mean(axis=1), L @ D_EPS


def med(x) -> float:
    return float(np.median(np.asarray(x)))


def rng_of(tag: int) -> np.random.Generator:
    return np.random.default_rng(1300 + tag)


def simulate_path(pi: np.ndarray, n_chains: int, n_steps: int, rng: np.random.Generator, s0: int = S_BAR):
    """按固定随机策略 pi 模拟 n_chains 条连续样本路径；返回状态 (n, T+1) 与行动 (n, T)。"""
    cum = np.cumsum(pi, axis=1)
    S = np.empty((n_chains, n_steps + 1), dtype=np.int64)
    A = np.empty((n_chains, n_steps), dtype=np.int64)
    s = np.full(n_chains, s0, dtype=np.int64)
    S[:, 0] = s
    for t in range(n_steps):
        c = cum[s]
        u = rng.random(n_chains) * c[:, -1]
        a = (u[:, None] >= c).sum(axis=1)
        z2 = np.where(rng.random(n_chains) < P_Z[s // NK, 0], 0, 1)
        s = z2 * NK + a
        A[:, t] = a
        S[:, t + 1] = s
    return S, A


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ----------------------------------------------------------------------------
# E0'：本章用到的基准量
# ----------------------------------------------------------------------------
def run_basics() -> None:
    banner("基准量（与第12章同一个模型）")
    print(f"γ={GAMMA}，稳态资本 k_ss={K_SS:.3f}，稳态消费 c_ss={C_SS:.4f}，初值 ln c_ss/(1-γ)={V0:.2f}")
    print(f"g_ε（ε=0.1，以 g* 为基础）：V_ε(s̄)={V_EPS[S_BAR]:.3f}，V*(s̄)={V_STAR[S_BAR]:.3f}；"
          f"按 d̄ 加权的 V_ε 均值 {D_EPS @ V_EPS:.2f}；V_ε 的范围 [{V_EPS.min():.2f}, {V_EPS.max():.2f}]")
    lg = ce_from_value(V_EPS)
    print(f"g_ε 相对 g* 的消费等价损失：全部平均 {100 * lg.mean():.2f}%，d̄ 加权 {100 * (lg @ D_EPS):.2f}%")
    order = np.argsort(-D_EPS)
    print("平稳分布 d̄（从大到小）：", "  ".join(f"{sname(s)}:{D_EPS[s]:.3f}" for s in order))
    print(f"d̄ 小于 0.01 的状态数：{int((D_EPS < 0.01).sum())}；这些状态的 d̄ 之和 {D_EPS[D_EPS < 0.01].sum():.4f}")
    lu = ce_from_value(value_of(PI_UNIF))
    print(f"均匀随机策略的消费等价损失：全部平均 {100 * lu.mean():.2f}%，d̄ 加权 {100 * (lu @ D_EPS):.2f}%")


# ----------------------------------------------------------------------------
# E5：同一条样本路径上的 TD 与 MC
# ----------------------------------------------------------------------------
CK5 = [300, 500, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 50000, 100000]


def weighted_rmse(V: np.ndarray) -> np.ndarray:
    return np.sqrt((((V - V_EPS) ** 2) * D_EPS).sum(axis=-1))


def run_e5(S: np.ndarray, A: np.ndarray) -> dict:
    n = S.shape[0]
    N = CK5[-1] if not QUICK else 20000
    cks = [c for c in CK5 if c <= N]
    idx = np.arange(n)
    Rw = R[S[:, :N], A[:, :N]]
    inits = [0.0, V0, float(D_EPS @ V_EPS)]
    init_names = ["0", f"ln c_ss/(1-γ)={V0:.2f}", f"真值的加权均值 {D_EPS @ V_EPS:.2f}（仅作对照）"]
    ni = len(inits)
    banner(f"E5  同一条样本路径上的 TD 与 MC（评估 g_ε；{n} 条链，每条 {N} 步；d̄ 加权均方根误差的中位数）")
    t0 = time.time()

    # --- TD(0)：三种步长；TD(0.9)：步长 C(s)^-0.6、累积资格迹 ---
    kind = np.tile(np.arange(3), ni)  # 0：常数 0.1；1：C^-0.6；2：1/C
    V = np.repeat(np.array(inits), 3)[:, None, None] * np.ones((1, n, NS))
    VL = np.array(inits)[:, None, None] * np.ones((1, n, NS))
    e = np.zeros((n, NS))
    C = np.zeros((n, NS))
    rec_td = np.zeros((len(cks), ni * 3, n))
    rec_l = np.zeros((len(cks), ni, n))
    j = 0
    const = np.full(n, 0.1)
    for t in range(N):
        s = S[:, t]
        s2 = S[:, t + 1]
        r = Rw[:, t]
        C[idx, s] += 1
        c = C[idx, s]
        eta = np.stack([const, c ** -0.6, 1.0 / c])[kind]
        d = r + GAMMA * V[:, idx, s2] - V[:, idx, s]
        V[:, idx, s] += eta * d
        dl = r + GAMMA * VL[:, idx, s2] - VL[:, idx, s]
        e *= GAMMA * 0.9
        e[idx, s] += 1.0
        VL += (np.maximum(C, 1.0) ** -0.6 * e)[None] * dl[:, :, None]
        if j < len(cks) and t + 1 == cks[j]:
            rec_td[j] = weighted_rmse(V)
            rec_l[j] = weighted_rmse(VL)
            j += 1

    # --- MC：每 300 期一段，首次访问、只用段内前 150 期出发的回报，步长 1/C ---
    seg, t1 = 300, 150
    VM = np.array(inits)[:, None, None] * np.ones((1, n, NS))
    CM = np.zeros((n, NS))
    hist_end, hist_err = [0], [weighted_rmse(VM)]
    for start in range(0, N - seg + 1, seg):
        r = Rw[:, start:start + seg]
        G = np.zeros((n, seg))
        acc = np.zeros(n)
        for jj in range(seg - 1, -1, -1):
            acc = r[:, jj] + GAMMA * acc
            G[:, jj] = acc
        first = np.full((n, NS), np.nan)
        for jj in range(t1 - 1, -1, -1):
            first[idx, S[:, start + jj]] = G[:, jj]
        vis = ~np.isnan(first)
        CM += vis
        VM += np.where(vis[None], (np.nan_to_num(first)[None] - VM) / np.maximum(CM, 1.0)[None], 0.0)
        hist_end.append(start + seg)
        hist_err.append(weighted_rmse(VM))
    hist_end = np.array(hist_end)
    rec_mc = np.array([hist_err[int(np.searchsorted(hist_end, ck, side="right")) - 1] for ck in cks])  # (cks, ni, n)
    print(f"（耗时 {time.time() - t0:.0f} 秒）")

    labels = ["MC，步长 1/C", "TD(0)，η=0.1", "TD(0)，η=C^-0.6", "TD(0)，η=1/C", "TD(0.9)，η=C^-0.6"]
    table_ck = [c for c in (1000, 10000, 100000) if c <= N]
    out = {}
    for i in range(ni):
        print(f"\n初值 {init_names[i]}：")
        curves = [rec_mc[:, i], rec_td[:, 3 * i], rec_td[:, 3 * i + 1], rec_td[:, 3 * i + 2], rec_l[:, i]]
        for lab, cur in zip(labels, curves):
            cells = []
            for ck in table_ck:
                jj = cks.index(ck)
                v = cur[jj]
                cells.append(f"{ck:>6d} 步 {med(v):.3f} [{np.percentile(v, 10):.3f},{np.percentile(v, 90):.3f}]")
            print(f"  {lab:<18s} " + "；".join(cells))
            out[(i, lab)] = [med(cur[cks.index(ck)]) for ck in table_ck]
        if i == 0:
            print("  TikZ 坐标（初值 0；横轴 log10 步数，纵轴误差中位数）：")
            for lab, cur in zip(labels, curves):
                print(f"    {lab}: " + " ".join(f"({math.log10(ck):.3f},{med(cur[jj]):.3f})" for jj, ck in enumerate(cks)))
    return out


# ----------------------------------------------------------------------------
# E6：SARSA 与 Q 学习
# ----------------------------------------------------------------------------
CK6 = [1000, 3000, 10000, 30000, 100000, 300000]


def eps_greedy_act(Q: np.ndarray, s: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    m = s.shape[0]
    greedy_a = Q[np.arange(m), s].argmax(axis=1)
    j = (rng.random(m) * NFEAS[s]).astype(np.int64)
    rand_a = FEAS_SORTED[s, j]
    return np.where(rng.random(m) < eps, rand_a, greedy_a)


def run_e6() -> dict:
    banner("E6  增长模型上的 SARSA 与 Q 学习")
    print("精确部分：SARSA 在 ε 固定时逼近的是最优 ε-软策略的价值（第12章定理），其贪心部分与 g* 的差别")
    g_soft = {}
    for eps in (0.1, 0.2, 0.3):
        g_e, V_e, Q_e = ch12.eps_soft_optimum(eps)
        g_soft[eps] = (g_e, V_e, Q_e)
        diff = np.flatnonzero(g_e != G_STAR)
        txt = "；".join(f"{sname(s)} g* 选 {K_GRID[G_STAR[s]]:.2f}、ε-软最优选 {K_GRID[g_e[s]]:.2f}" for s in diff)
        lg = ce_from_value(det_value(g_e))
        print(f"  ε={eps}：不同的状态数 {len(diff)}（{txt}）；其贪心部分作为确定性策略的损失：全部 {100 * lg.mean():.2f}%，加权 {100 * (lg @ D_EPS):.2f}%")
        # 对照：对“以 g* 为基础的 ε-贪心”评估一次再取贪心
        Qg = np.where(FEAS, RZ + GAMMA * P @ value_of(eps_greedy(G_STAR, eps)), -np.inf)
        print(f"         对照（Q^{{g*的ε-贪心}} 上取贪心）不同的状态数 {int((Qg.argmax(axis=1) != G_STAR).sum())}")
    g_e, V_e, Q_e = g_soft[0.1]
    diff = np.flatnonzero(g_e != G_STAR)
    focus = int(min(diff, key=lambda s: (-ZS[s], KS[s])))  # 优先高生产率、低资本的那一格
    feas = FEAS_LIST[focus]
    print(f"\n表 tab:ch13_sarsa_q：格点 {sname(focus)} 上各可行 k' 的 Q* 与 Q_ε（ε=0.1 最优 ε-软策略的行动价值）")
    print("  k':    " + "  ".join(f"{K_GRID[a]:>6.2f}" for a in feas))
    print("  Q*:    " + "  ".join(f"{Q_STAR[focus, a]:>6.3f}" for a in feas) + f"   argmax k'={K_GRID[G_STAR[focus]]:.2f}")
    print("  Q_ε:   " + "  ".join(f"{Q_e[focus, a]:>6.3f}" for a in feas) + f"   argmax k'={K_GRID[g_e[focus]]:.2f}")
    a_lo, a_hi = sorted({int(g_e[focus]), int(G_STAR[focus])})
    z = focus // NK
    ev_star = [P_Z[z] @ V_STAR[[0 * NK + a, 1 * NK + a]] for a in (a_lo, a_hi)]
    ev_eps = [P_Z[z] @ V_e[[0 * NK + a, 1 * NK + a]] for a in (a_lo, a_hi)]
    du = R[focus, a_hi] - R[focus, a_lo]
    print(f"  从 k'={K_GRID[a_lo]:.2f} 提高到 {K_GRID[a_hi]:.2f}：即期效用变化 {du:.3f}；"
          f"贴现的延续价值增量：按 g* 行事 γΔE[V*]={GAMMA * (ev_star[1] - ev_star[0]):.3f}，"
          f"按 g_ε 行事 γΔE[V_ε]={GAMMA * (ev_eps[1] - ev_eps[0]):.3f}")
    for kz in (0, 1):
        vs = V_STAR[kz * NK:(kz + 1) * NK]
        ve = V_e[kz * NK:(kz + 1) * NK]
        print(f"  z={Z_GRID[kz]:.1f}：相邻资本格点之间 V* 的增量 " + " ".join(f"{x:.3f}" for x in np.diff(vs))
              + "；V_ε 的增量 " + " ".join(f"{x:.3f}" for x in np.diff(ve)))

    # --- 抽样部分 ---
    N = CK6[-1] if not QUICK else 30000
    cks = [c for c in CK6 if c <= N]
    rows = 2 * SEEDS
    is_sarsa = np.arange(rows) < SEEDS
    idx = np.arange(rows)
    rng = rng_of(6)
    Q = np.repeat(np.where(FEAS, V0, -np.inf)[None], rows, axis=0)
    C = np.zeros((rows, NS, NA))
    s = np.full(rows, S_BAR, dtype=np.int64)
    a = eps_greedy_act(Q, s, 0.1, rng)
    snaps = []
    j = 0
    t0 = time.time()
    for t in range(N):
        r = R[s, a]
        s2 = np.where(rng.random(rows) < P_Z[s // NK, 0], 0, 1) * NK + a
        a2 = eps_greedy_act(Q, s2, 0.1, rng)
        nxt = np.where(is_sarsa, Q[idx, s2, a2], Q[idx, s2].max(axis=1))
        C[idx, s, a] += 1
        Q[idx, s, a] += C[idx, s, a] ** -0.6 * (r + GAMMA * nxt - Q[idx, s, a])
        s, a = s2, a2
        if j < len(cks) and t + 1 == cks[j]:
            Gr = Q.argmax(axis=2)
            ce_all, ce_w, wrong = losses_det(Gr)
            cells = ((C >= 10) & FEAS[None]).sum(axis=(1, 2))
            snaps.append(dict(ce_all=ce_all, ce_w=ce_w, wrong=wrong, cells=cells, focus=Gr[:, focus].copy()))
            j += 1
    print(f"\n抽样部分：从 s̄ 出发的连续过程，ε=0.1，步长 C(s,a)^-0.6，初值 {V0:.2f}，{N} 步，{SEEDS} 个种子（耗时 {time.time() - t0:.0f} 秒）")
    out = {}
    for name, mask in (("SARSA", is_sarsa), ("Q学习", ~is_sarsa)):
        print(f"  {name}:")
        for ck, sn in zip(cks, snaps):
            ca, cw, wr, ce = 100 * sn["ce_all"][mask], 100 * sn["ce_w"][mask], sn["wrong"][mask], sn["cells"][mask]
            fo = sn["focus"][mask]
            print(f"    {ck:>6d} 步：损失 全部 中位数 {med(ca):.2f}% [{ca.min():.2f},{ca.max():.2f}]，加权 中位数 {med(cw):.2f}% [{cw.min():.2f},{cw.max():.2f}]"
                  f"（加权为 0 的种子 {int((cw < 1e-9).sum())}/{mask.sum()}）；错选状态 中位数 {med(wr):.0f} [{wr.min()},{wr.max()}]；"
                  f"≥10 样本格子 中位数 {med(ce):.0f} [{ce.min()},{ce.max()}]；{sname(focus)} 选 g* 的比例 {np.mean(fo == G_STAR[focus]):.2f}、选 ε-软最优的比例 {np.mean(fo == g_e[focus]):.2f}")
            out[(name, ck)] = dict(ce_all=med(ca), ce_w=med(cw), wrong=med(wr), cells=med(ce))
        print(f"    TikZ 坐标（加权损失中位数，横轴 log10 步数）: " + " ".join(
            f"({math.log10(ck):.3f},{med(100 * sn['ce_w'][mask]):.3f})" for ck, sn in zip(cks, snaps)))
    return out


# ----------------------------------------------------------------------------
# E7：线性 TD、最佳拟合与 LSTD
# ----------------------------------------------------------------------------
def features(name: str) -> np.ndarray:
    kt = (KS - K_GRID[0]) / (K_GRID[-1] - K_GRID[0])
    h = (ZS == Z_GRID[1]).astype(float)
    one = np.ones(NS)
    if name == "5项":
        cols = [one, kt, kt ** 2, h, kt * h]
    elif name == "4项":
        lk = np.log(KS)
        cols = [one, lk, h, lk * h]
    else:
        cols = [one, kt, h]
    return np.stack(cols, axis=1)


def norm_d(x: np.ndarray) -> np.ndarray:
    return np.sqrt((x ** 2 * D_EPS).sum(axis=-1))


def run_e7(S: np.ndarray, A: np.ndarray) -> dict:
    banner("E7  线性 TD、最佳拟合与 LSTD（评估 g_ε；误差按 d̄ 加权）")
    Pg = chain(PI_EPS)
    rg = (PI_EPS * RZ).sum(axis=1)
    Dm = np.diag(D_EPS)
    out = {}
    for name, desc in (("5项", "1, k̃, k̃², 1{z=1.1}, k̃·1{z=1.1}"), ("4项", "1, ln k, 1{z=1.1}, ln k·1{z=1.1}"), ("3项", "1, k̃, 1{z=1.1}")):
        Phi = features(name)
        psi_bf = np.linalg.solve(Phi.T @ Dm @ Phi, Phi.T @ Dm @ V_EPS)
        e_bf = float(norm_d(Phi @ psi_bf - V_EPS))
        Amat = Phi.T @ Dm @ (Phi - GAMMA * Pg @ Phi)
        psi_td = np.linalg.solve(Amat, Phi.T @ Dm @ rg)
        err_td = Phi @ psi_td - V_EPS
        e_td = float(norm_d(err_td))
        out[name] = dict(psi_td=psi_td, e_bf=e_bf, e_td=e_td)
        print(f"{name}基（{desc}）：最佳拟合误差 {e_bf:.3f}；TD 不动点误差 {e_td:.3f}（最大绝对误差 {np.abs(err_td).max():.3f}）；"
              f"Tsitsiklis–Van Roy 界 {e_bf / (1 - GAMMA):.3f}；勾股定理给出的界 {e_bf / math.sqrt(1 - GAMMA ** 2):.3f}")
    # 抽样：LSTD 与半梯度 TD（5 项基）
    Phi = features("5项")
    psi_td = out["5项"]["psi_td"]
    n = S.shape[0]
    N = S.shape[1] - 1
    rows = np.arange(n)
    Nss = np.zeros((n, NS, NS))
    np.add.at(Nss, (np.repeat(rows, N), S[:, :-1].ravel(), S[:, 1:].ravel()), 1.0)
    Nsa = np.zeros((n, NS, NA))
    np.add.at(Nsa, (np.repeat(rows, N), S[:, :-1].ravel(), A.ravel()), 1.0)
    psi_l = np.zeros((n, Phi.shape[1]))
    for i in range(n):
        occ = Nss[i].sum(axis=1)
        Ahat = Phi.T @ (occ[:, None] * Phi) - GAMMA * Phi.T @ Nss[i] @ Phi
        bhat = Phi.T @ (Nsa[i] * RZ).sum(axis=1)
        psi_l[i] = np.linalg.solve(Ahat, bhat)
    e_l = norm_d(psi_l @ Phi.T - V_EPS)
    out["lstd_med"] = med(e_l)
    print(f"\n5 项基，{N} 步样本（{n} 条链）：")
    print("  TD 不动点系数:      (" + ", ".join(f"{x:.3f}" for x in psi_td) + ")")
    print("  LSTD 系数（链 0）:  (" + ", ".join(f"{x:.3f}" for x in psi_l[0]) + ")")
    print(f"  LSTD 误差 中位数 {med(e_l):.3f} [{e_l.min():.3f},{e_l.max():.3f}]；系数与不动点的最大偏差 中位数 {med(np.abs(psi_l - psi_td).max(axis=1)):.3f}")
    # 半梯度 TD：η_t = 0.1(1+t/1000)^-0.6
    psi = np.zeros((n, Phi.shape[1]))
    cks = [c for c in (1000, 10000, 100000, 200000) if c <= N]
    Rw = RZ[S[:, :-1], A]
    j = 0
    for t in range(N):
        f1 = Phi[S[:, t]]
        f2 = Phi[S[:, t + 1]]
        dlt = Rw[:, t] + GAMMA * (f2 * psi).sum(axis=1) - (f1 * psi).sum(axis=1)
        psi += 0.1 * (1 + t / 1000) ** -0.6 * dlt[:, None] * f1
        if j < len(cks) and t + 1 == cks[j]:
            e_sg = norm_d(psi @ Phi.T - V_EPS)
            dist = norm_d((psi - psi_td) @ Phi.T)
            print(f"  半梯度 TD（η_t=0.1(1+t/1000)^-0.6）{cks[j]:>6d} 步：误差 中位数 {med(e_sg):.3f}；与不动点的加权距离 中位数 {med(dist):.3f}；"
                  "链 0 系数 (" + ", ".join(f"{x:.2f}" for x in psi[0]) + ")")
            j += 1
    return out


# ----------------------------------------------------------------------------
# E8：基线与策略梯度的方差
# ----------------------------------------------------------------------------
def run_e8(n_traj: int = 400, T: int = 300, T1: int = 150) -> dict:
    banner(f"E8  基线降低策略梯度的方差（表格 softmax，θ=0 即可行集上均匀；从 s̄ 出发 {n_traj} 条 {T} 期轨迹，t<{T1}，含 γ^t）")
    rng = rng_of(8)
    V_U = value_of(PI_UNIF)
    S, A = simulate_path(PI_UNIF, 2 * n_traj, T, rng)
    Rw = R[S[:, :-1], A]
    G = np.zeros((2 * n_traj, T))
    acc = np.zeros(2 * n_traj)
    for t in range(T - 1, -1, -1):
        acc = Rw[:, t] + GAMMA * acc
        G[:, t] = acc
    # 学到的基线：用另一半轨迹做首次访问 MC 平均
    tot = np.zeros(NS)
    cnt = np.zeros(NS)
    for i in range(n_traj, 2 * n_traj):
        seen = set()
        for t in range(T1):
            s = S[i, t]
            if s not in seen:
                seen.add(s)
                tot[s] += G[i, t]
                cnt[s] += 1
    V_hat = np.where(cnt > 0, tot / np.maximum(cnt, 1), (tot.sum() / max(cnt.sum(), 1)))
    print(f"学到的基线与真实价值的 d̄ 加权均方根差 {norm_d(V_hat - V_U):.3f}（有样本的状态 {int((cnt > 0).sum())}/{NS}）")
    baselines = [("不加基线", np.zeros(NS)), ("真实价值 V^g 为基线", V_U), ("学到的 V̂ 为基线", V_hat)]
    rows = np.arange(n_traj)
    grads = np.zeros((len(baselines), n_traj, NS * NA))
    for t in range(T1):
        s = S[:n_traj, t]
        a = A[:n_traj, t]
        score = -PI_UNIF[s]
        score[rows, a] += 1.0
        flat = s[:, None] * NA + np.arange(NA)[None]
        for b, (_, bv) in enumerate(baselines):
            adv = G[:n_traj, t] - bv[s]
            grads[b][rows[:, None], flat] += (GAMMA ** t) * adv[:, None] * score
    mask = FEAS.ravel()
    # 精确梯度（定理的状态—行动形式）
    Pu = chain(PI_UNIF)
    d_disc = (1 - GAMMA) * np.linalg.solve((np.eye(NS) - GAMMA * Pu).T, np.eye(NS)[S_BAR])
    Q_U = np.where(FEAS, RZ + GAMMA * P @ V_U, 0.0)
    g_exact = (d_disc[:, None] * PI_UNIF * (Q_U - V_U[:, None]) / (1 - GAMMA)).ravel()[mask]
    print(f"精确梯度（定理的状态—行动形式）范数 {np.linalg.norm(g_exact):.3f}")
    out = {}
    v_none = None
    for b, (name, _) in enumerate(baselines):
        gb = grads[b][:, mask]
        var_tot = float(gb.var(axis=0, ddof=1).sum())
        mean_v = gb.mean(axis=0)
        if v_none is None:
            v_none = var_tot
        print(f"  {name:<14s}：梯度估计各分量方差之和 {var_tot:.1f}（为不加基线的 {var_tot / v_none:.3f} 倍）；"
              f"均值范数 {np.linalg.norm(mean_v):.3f}；均值与精确梯度之差的范数 {np.linalg.norm(mean_v - g_exact):.3f}"
              f"（其标准误约 {math.sqrt(var_tot / n_traj):.3f}）")
        out[name] = var_tot
    print(f"  达到同样精度所需轨迹数之比（不加基线 : 真实价值基线）≈ {out['不加基线'] / out['真实价值 V^g 为基线']:.1f} : 1")
    return out


# ----------------------------------------------------------------------------
# E9：表格型演员—评论家、REINFORCE 与 TD 控制，在同一数据协议下比较
# ----------------------------------------------------------------------------
CK9 = [10000, 30000, 100000, 300000, 1000000]
K3 = [2, 2 + NK]  # k=3.00 的两个状态：g* 下经济长期停留的资本水平


def softmax_rows(theta_s: np.ndarray, s: np.ndarray) -> np.ndarray:
    logit = np.where(FEAS[s], theta_s, -np.inf)
    logit = logit - logit.max(axis=1, keepdims=True)
    p = np.exp(logit)
    return p / p.sum(axis=1, keepdims=True)


def policy_tables(theta: np.ndarray) -> np.ndarray:
    logit = np.where(FEAS[None], theta, -np.inf)
    logit = logit - logit.max(axis=2, keepdims=True)
    p = np.exp(logit)
    return p / p.sum(axis=2, keepdims=True)


def eval_actor(theta: np.ndarray):
    Gr = np.where(FEAS[None], theta, -np.inf).argmax(axis=2)
    ga, gw, wrong = losses_det(Gr)
    pis = policy_tables(theta)
    sa, sw = losses_stoch(pis)
    maxp = (pis.max(axis=2) * D_EPS).sum(axis=1)
    return dict(g_all=ga, g_w=gw, wrong=wrong, s_all=sa, s_w=sw, maxp=maxp)


AC_SETTINGS = [
    # 名称，演员步长基数，是否按 C(s)^-0.8 递减，是否每 300 期重置到随机状态
    ("不重置，常数 0.5", 0.5, False, False),
    ("重置，常数 0.1", 0.1, False, True),
    ("重置，常数 0.5", 0.5, False, True),
    ("重置，常数 1.0", 1.0, False, True),
    ("重置，递减 C(s)^-0.8", 1.0, True, True),
]
RESET_EVERY = 300


def report(name: str, cks, snaps, masks_shares=None) -> list:
    print(f"  {name}:")
    rows_out = []
    for j, (ck, sn) in enumerate(zip(cks, snaps)):
        gw = np.maximum(sn['g_w'], 0.0)
        line = (f"    {ck:>7d} 步：贪心策略损失 加权 {100 * med(gw):.2f}% [{100 * np.percentile(gw, 10):.2f},{100 * np.percentile(gw, 90):.2f}]"
                f"（加权损失低于 1% 的种子 {int((gw < 0.01).sum())}/{len(gw)}）、全部 {100 * med(sn['g_all']):.2f}%、错选 {med(sn['wrong']):.1f}")
        if "s_w" in sn:
            line += f"；随机策略本身损失 加权 {100 * med(sn['s_w']):.2f}%、全部 {100 * med(sn['s_all']):.2f}%；加权最大行动概率 {med(sn['maxp']):.2f}"
        if "share3" in sn:
            line += f"；该区间在 k=3.00 停留的时间比例 中位数 {med(sn['share3']):.3f}、最大 {np.max(sn['share3']):.3f}"
        print(line)
        rows_out.append(dict((k, med(v)) for k, v in sn.items()))
    print("    TikZ 坐标（贪心策略加权损失 %）: " + " ".join(
        f"({math.log10(ck):.3f},{100 * med(sn['g_w']):.3f})" for ck, sn in zip(cks, snaps)))
    return rows_out


def run_e9() -> dict:
    N = CK9[-1] if not QUICK else 30000
    cks = [c for c in CK9 if c <= N]
    out = {}
    banner(f"E9  控制方法的比较（{SEEDS} 个种子，最多 {N} 步；“重置”指每 {RESET_EVERY} 期把模拟器重置到均匀抽取的状态，"
           f"不重置时从 s̄ 出发一直走下去）")

    # --- 表格型演员—评论家：演员为每个状态一组 softmax 参数，评论家为表格 V̂（步长 C(s)^-0.6，初值 V0） ---
    nset = len(AC_SETTINGS)
    rows = nset * SEEDS
    idx = np.arange(rows)
    base = np.repeat(np.array([st[1] for st in AC_SETTINGS]), SEEDS)
    decr = np.repeat(np.array([st[2] for st in AC_SETTINGS]), SEEDS)
    reset = np.repeat(np.array([st[3] for st in AC_SETTINGS]), SEEDS)
    theta = np.zeros((rows, NS, NA))
    Vc = np.full((rows, NS), V0)
    Cs = np.zeros((rows, NS))
    s = np.full(rows, S_BAR, dtype=np.int64)
    rng = rng_of(9)
    snaps, prev = [], np.zeros((rows, NS))
    j = 0
    t0 = time.time()
    for t in range(1, N + 1):
        p = softmax_rows(theta[idx, s], s)
        cum = np.cumsum(p, axis=1)
        u = rng.random(rows) * cum[:, -1]
        a = (u[:, None] >= cum).sum(axis=1)
        r = R[s, a]
        s2 = np.where(rng.random(rows) < P_Z[s // NK, 0], 0, 1) * NK + a
        Cs[idx, s] += 1
        c = Cs[idx, s]
        dlt = r + GAMMA * Vc[idx, s2] - Vc[idx, s]
        Vc[idx, s] += c ** -0.6 * dlt  # 评论家：TD(0)；重置前的最后一步照常自举
        score = -p
        score[idx, a] += 1.0
        eta = np.where(decr, base * c ** -0.8, base)
        theta[idx, s] += eta[:, None] * dlt[:, None] * score  # 演员：连续任务中不乘 γ^t
        if t % RESET_EVERY == 0:
            s2 = np.where(reset, rng.integers(0, NS, rows), s2)
        s = s2
        if j < len(cks) and t == cks[j]:
            ev = eval_actor(theta)
            dC = Cs - prev
            ev["share3"] = dC[:, K3].sum(axis=1) / dC.sum(axis=1)
            prev = Cs.copy()
            snaps.append(ev)
            j += 1
    print(f"表格型演员—评论家（耗时 {time.time() - t0:.0f} 秒）")
    Gfin = np.where(FEAS[None], theta, -np.inf).argmax(axis=2)
    _, gw_fin, _ = losses_det(Gfin)
    K3_STATES = [2, 2 + NK]
    for k, (name, _, _, _) in enumerate(AC_SETTINGS):
        m = slice(k * SEEDS, (k + 1) * SEEDS)
        out["AC " + name] = report("演员—评论家，" + name, cks, [dict((key, v[m]) for key, v in sn.items()) for sn in snaps])
        stuck = gw_fin[m] >= 0.01
        if stuck.any():
            dk = Gfin[m][stuck] - G_STAR[None]
            down3 = np.mean([np.all(K_GRID[row[K3_STATES]] < 3.0) for row in Gfin[m][stuck]])
            print(f"    最终加权损失不低于 1% 的 {int(stuck.sum())} 个种子：与 g* 不同的状态中，少储蓄的个数 中位数 {med((dk < 0).sum(axis=1)):.0f}、"
                  f"多储蓄的个数 中位数 {med((dk > 0).sum(axis=1)):.0f}；在 k=3.00 的两个状态上都选择降低资本的种子比例 {down3:.2f}")

    # --- REINFORCE（带基线）：每条轨迹 300 期、从均匀抽取的状态出发；t<150 的项进入梯度，含 γ^t ---
    T, T1 = 300, 150
    rf_steps = [0.1, 0.5]
    rows = len(rf_steps) * SEEDS
    idx = np.arange(rows)
    step = np.repeat(np.array(rf_steps), SEEDS)
    theta = np.zeros((rows, NS, NA))
    Vb = np.full((rows, NS), V0)
    Cb = np.zeros((rows, NS))
    ck_ep = [max(1, round(ck / T)) for ck in cks]
    disc = GAMMA ** np.arange(T1)
    Sbuf = np.empty((rows, T), dtype=np.int64)
    Abuf = np.empty((rows, T), dtype=np.int64)
    Pbuf = np.empty((rows, T, NA))
    snaps = []
    j = 0
    t0 = time.time()
    for ep in range(1, ck_ep[-1] + 1):
        s = rng.integers(0, NS, rows)
        for t in range(T):
            p = softmax_rows(theta[idx, s], s)
            cum = np.cumsum(p, axis=1)
            u = rng.random(rows) * cum[:, -1]
            a = (u[:, None] >= cum).sum(axis=1)
            Sbuf[:, t], Abuf[:, t], Pbuf[:, t] = s, a, p
            s = np.where(rng.random(rows) < P_Z[s // NK, 0], 0, 1) * NK + a
        Rw = R[Sbuf, Abuf]
        G = np.zeros((rows, T))
        acc = np.zeros(rows)
        for t in range(T - 1, -1, -1):
            acc = Rw[:, t] + GAMMA * acc
            G[:, t] = acc
        grad = np.zeros((rows, NS, NA))
        for t in range(T1):
            st, at = Sbuf[:, t], Abuf[:, t]
            adv = G[:, t] - Vb[idx, st]
            sc = -Pbuf[:, t]
            sc[idx, at] += 1.0
            grad[idx, st] += (disc[t] * adv)[:, None] * sc
            Cb[idx, st] += 1
            Vb[idx, st] += Cb[idx, st] ** -0.6 * (G[:, t] - Vb[idx, st])  # 基线：整段回报的增量平均
        theta += step[:, None, None] * grad
        if j < len(ck_ep) and ep == ck_ep[j]:
            snaps.append(eval_actor(theta))
            j += 1
    print(f"REINFORCE（耗时 {time.time() - t0:.0f} 秒；检查点处的步数为 300 的整数倍，与表头相差不足 300 步）")
    for k, stp in enumerate(rf_steps):
        m = slice(k * SEEDS, (k + 1) * SEEDS)
        out[f"REINFORCE {stp}"] = report(f"REINFORCE，重置（每条轨迹从随机状态出发），步长 {stp}", cks,
                                          [dict((key, v[m]) for key, v in sn.items()) for sn in snaps])

    # --- SARSA 与 Q 学习：同样每 300 期重置；ε=0.1，步长 C(s,a)^-0.6，初值 V0 ---
    rows = 2 * SEEDS
    idx = np.arange(rows)
    is_sarsa = np.arange(rows) < SEEDS
    Q = np.repeat(np.where(FEAS, V0, -np.inf)[None], rows, axis=0)
    C = np.zeros((rows, NS, NA))
    s = np.full(rows, S_BAR, dtype=np.int64)
    a = eps_greedy_act(Q, s, 0.1, rng)
    snaps = []
    j = 0
    t0 = time.time()
    for t in range(1, N + 1):
        r = R[s, a]
        s2 = np.where(rng.random(rows) < P_Z[s // NK, 0], 0, 1) * NK + a
        a2 = eps_greedy_act(Q, s2, 0.1, rng)
        nxt = np.where(is_sarsa, Q[idx, s2, a2], Q[idx, s2].max(axis=1))
        C[idx, s, a] += 1
        Q[idx, s, a] += C[idx, s, a] ** -0.6 * (r + GAMMA * nxt - Q[idx, s, a])
        if t % RESET_EVERY == 0:
            s2 = rng.integers(0, NS, rows)
            a2 = eps_greedy_act(Q, s2, 0.1, rng)
        s, a = s2, a2
        if j < len(cks) and t == cks[j]:
            ca, cw, wr = losses_det(Q.argmax(axis=2))
            snaps.append(dict(g_all=ca, g_w=cw, wrong=wr))
            j += 1
    print(f"SARSA 与 Q 学习（耗时 {time.time() - t0:.0f} 秒）")
    for name, mask in (("SARSA", is_sarsa), ("Q学习", ~is_sarsa)):
        out[name] = report(f"{name}，重置，ε=0.1", cks, [dict((key, v[mask]) for key, v in sn.items()) for sn in snaps])
    return out


def main() -> None:
    t_all = time.time()
    run_basics()
    rng = rng_of(5)
    n_path = CK5[-1] if not QUICK else 20000
    n_path7 = 200000 if not QUICK else 20000
    S, A = simulate_path(PI_EPS, SEEDS, max(n_path, n_path7), rng)
    o5 = run_e5(S, A)
    run_e6()
    o7 = run_e7(S[:, :n_path7 + 1], A[:, :n_path7])
    run_e8()
    o9 = run_e9()
    banner("汇总：表 tab:ch13_all_methods 使用的数字")
    last = -1
    print("评估（初值 0，最后一个检查点的加权均方根误差中位数）：MC {:.3f}；TD(0) C^-0.6 {:.3f}；TD(0.9) {:.3f}；线性 TD 不动点（5项）{:.3f}；LSTD {:.3f}".format(
        o5[(0, "MC，步长 1/C")][last], o5[(0, "TD(0)，η=C^-0.6")][last], o5[(0, "TD(0.9)，η=C^-0.6")][last], o7["5项"]["e_td"], o7["lstd_med"]))
    for key in ("SARSA", "Q学习", "REINFORCE 0.5", "AC 重置，常数 0.5"):
        r = o9[key][last]
        print(f"控制（重置，最后一个检查点）{key}：加权 {100 * max(r['g_w'], 0):.2f}%，14 个状态平均 {100 * r['g_all']:.2f}%")
    print("MC 控制（探索性起点）的数字见 code/ch12/running_example.py 的 E2。")
    print(f"\n总耗时 {time.time() - t_all:.0f} 秒")


if __name__ == "__main__":
    main()
