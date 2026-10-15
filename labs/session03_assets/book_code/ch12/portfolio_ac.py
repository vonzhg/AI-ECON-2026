"""
连续行动的策略梯度：风险资产份额 ω 的 Beta 策略与 sigmoid-高斯策略

对应章节：第13章《强化学习核心算法》§13.8.3 练习题（比较两种连续策略参数化）；
方法见 §13.4 策略梯度（sec:ch13_pg）与 §13.5 演员—评论家（sec:ch13_ac）。
承接第12章练习 4(d)“投资组合选择的连续行动 MDP”，份额记号沿用那里的 ω。

设定
----
- 每期把份额 ω_t ∈ [0,1] 投入风险资产、1-ω_t 投入无风险资产；无风险毛回报 R_f=1.02，
  风险毛回报 R_e 对数正态、跨期独立同分布（E[R_e]=e^{0.05}，对数回报波动 0.15）。
- 奖励 u(组合毛回报)，u 为 CRRA（相对风险厌恶 2）；贴现因子 0.97；每段 50 期。
  状态 s_t = (log W_t, 上期 R_e - 1)。
- 重要性质：回报独立同分布、奖励只依赖当期组合回报，于是最优份额是“短视”的，
  ω* = argmax_ω E[u(ω R_e + (1-ω) R_f)]，与状态和期数无关——问题对 ω 实际上是静态的。
  脚本用 40 点 Gauss–Hermite 积分求出 ω*≈0.672（Merton 近似 (μ-r_f)/(γσ²) 给出 0.671），
  作为两种策略的基准；状态变量只是让网络“有东西可看”，理想的策略应当忽略它们。

两种连续策略参数化
------------------
- Beta 策略：网络输出 (a, b) > 1，ω ~ Beta(a, b)，天然落在 (0,1)；贪心行动取均值 a/(a+b)。
- Sigmoid-高斯策略：网络输出 (m, log σ)，x ~ N(m, σ²)，ω = sigmoid(x)；
  TransformedDistribution 自动加上 sigmoid 变换的雅可比项；贪心行动取 sigmoid(m)（中位数）。

梯度估计（得分函数 + 评论家基线）
--------------------------------
- 每段轨迹结束后，演员（策略网络）的损失为 -Σ_t log g_θ(ω_t|s_t)·[G_t - V̂_ψ(s_t)]，评论家 V̂_ψ 用均方误差拟合 G_t。
- ω_t 必须用 sample() 抽取（不带梯度）。若用 rsample()，抽样本身对 θ 可导，
  log_prob(ω_t) 的梯度会混入一条路径梯度，得到的就不再是得分函数估计量。
  实现上先在无梯度模式下抽完整段 ω_t，段末再把整段一次代入 log g_θ(ω_t|s_t)。
- 这里的 G_t 不乘 γ^t：本问题每期的最优 ω 相同，按时刻怎样加权都不改变最优点。

打印：固定 ω 的精确期望贴现回报（积分计算，无抽样噪声）；两种策略在每个种子下学到的贪心 ω
（评估状态上的均值与 10%/90% 分位），以及相对 ω* 的每期确定性等价回报损失（基点）。
ω 在 ω* 附近的福利代价是二阶小量（ω 从 0.67 变到 0.5 或 0.9，每期只差几个基点），
得分函数梯度的信号因此很弱，学到的 ω 随种子变化很大；这正是比较两种参数化时要注意的地方。

依赖：numpy + pytorch（单线程）。运行（在书稿目录下）：OMP_NUM_THREADS=1 python3 -B code/ch13/portfolio_ac.py
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")   # 须在导入 torch 之前

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta, Distribution, Normal, TransformedDistribution
from torch.distributions.transforms import SigmoidTransform

torch.set_num_threads(1)
Distribution.set_default_validate_args(False)    # 关闭逐次参数检查，每步快约四成


# --------------------------------------------------------------------------- #
# 环境
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PortfolioParams:
    beta: float = 0.97          # 贴现因子
    r_f: float = 1.02           # 无风险毛回报
    mu_e: float = 0.05          # 风险资产对数期望毛回报 ln E[R_e]
    sigma_e: float = 0.15       # 风险资产对数回报波动
    crra: float = 2.0           # 相对风险厌恶系数
    episode_length: int = 50


def crra_utility(gross: np.ndarray, crra: float) -> np.ndarray:
    if crra == 1.0:
        return np.log(gross)
    return (np.power(gross, 1.0 - crra) - 1.0) / (1.0 - crra)


class PortfolioEnv:
    """份额 ω ∈ [0,1] 的连续行动环境。"""

    def __init__(self, params: PortfolioParams = PortfolioParams(), seed: int = 0) -> None:
        self.p = params
        self.rng = np.random.default_rng(seed)
        self._t = 0
        self._w = 1.0
        self._last_re = float(np.exp(params.mu_e))

    @property
    def state_dim(self) -> int:
        return 2

    def _obs(self) -> np.ndarray:
        return np.array([np.log(self._w), self._last_re - 1.0], dtype=np.float32)

    def reset(self) -> np.ndarray:
        self._t = 0
        self._w = 1.0
        self._last_re = float(np.exp(self.p.mu_e))
        return self._obs()

    def step(self, omega: float) -> Tuple[np.ndarray, float, bool, dict]:
        omega = float(np.clip(omega, 0.0, 1.0))
        r_e = float(np.exp(self.p.mu_e - 0.5 * self.p.sigma_e ** 2 + self.p.sigma_e * self.rng.standard_normal()))
        gross = omega * r_e + (1.0 - omega) * self.p.r_f
        reward = float(crra_utility(np.array(gross), self.p.crra))
        self._w *= gross
        self._last_re = r_e
        self._t += 1
        done = self._t >= self.p.episode_length   # 有限期问题：第 50 期是真正的终点
        return self._obs(), reward, done, {"gross": gross, "r_e": r_e}


# --------------------------------------------------------------------------- #
# 基准：短视最优份额 ω*（Gauss–Hermite 积分）
# --------------------------------------------------------------------------- #
class MyopicBenchmark:
    def __init__(self, p: PortfolioParams, nodes: int = 40) -> None:
        x, w = np.polynomial.hermite_e.hermegauss(nodes)   # 标准正态的 Gauss–Hermite 节点
        self.p = p
        self.weights = w / w.sum()
        self.r_e = np.exp(p.mu_e - 0.5 * p.sigma_e ** 2 + p.sigma_e * x)
        grid = np.linspace(0.0, 1.0, 10001)
        vals = self.expected_utility(grid)
        self.omega_star = float(grid[int(np.argmax(vals))])
        self.eu_star = float(vals.max())

    def expected_utility(self, omega: np.ndarray) -> np.ndarray:
        """E[u(ω R_e + (1-ω) R_f)]，ω 可以是数组。"""
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        gross = omega[:, None] * self.r_e[None, :] + (1.0 - omega[:, None]) * self.p.r_f
        return crra_utility(gross, self.p.crra) @ self.weights

    def certainty_equivalent(self, eu: np.ndarray) -> np.ndarray:
        """期望效用对应的确定性等价毛回报。"""
        c = self.p.crra
        return np.exp(eu) if c == 1.0 else np.power(1.0 + (1.0 - c) * np.asarray(eu), 1.0 / (1.0 - c))

    def discounted_value(self, omega: float) -> float:
        """固定份额 ω 时 50 期期望贴现回报的精确值。"""
        p = self.p
        return float(self.expected_utility(omega)[0] * (1.0 - p.beta ** p.episode_length) / (1.0 - p.beta))


# --------------------------------------------------------------------------- #
# 连续策略：两种参数化
# --------------------------------------------------------------------------- #
class BetaPolicy(nn.Module):
    """网络输出 (a, b) > 1，ω ~ Beta(a, b) ∈ (0, 1)。"""

    def __init__(self, state_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(),
                                  nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.head = nn.Linear(hidden_dim, 2)

    def dist(self, s: torch.Tensor) -> Beta:
        ab = torch.nn.functional.softplus(self.head(self.body(s))) + 1.0   # a, b ≥ 1：单峰，避免端点处病态
        return Beta(ab[..., 0], ab[..., 1])

    def greedy_action(self, s: torch.Tensor) -> torch.Tensor:
        return self.dist(s).mean                                          # a/(a+b)


class SigmoidGaussianPolicy(nn.Module):
    """网络输出 (m, log σ)，x ~ N(m, σ²)，ω = sigmoid(x) ∈ (0, 1)。"""

    def __init__(self, state_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(),
                                  nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.head = nn.Linear(hidden_dim, 2)

    def dist(self, s: torch.Tensor) -> TransformedDistribution:
        raw = self.head(self.body(s))
        base = Normal(raw[..., 0], torch.exp(torch.clamp(raw[..., 1], -5.0, 2.0)))
        return TransformedDistribution(base, [SigmoidTransform()])        # 自动加上雅可比项

    def greedy_action(self, s: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self.body(s))[..., 0])             # sigmoid(m)，即分布的中位数


class ContinuousCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                 nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# --------------------------------------------------------------------------- #
# 训练：得分函数估计 + 评论家基线（每段轨迹更新一次）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ContinuousPGConfig:
    num_iterations: int = 800
    gamma: Optional[float] = None   # 默认取环境的贴现因子
    lr_actor: float = 3e-3          # 演员学习率（Adam）
    lr_critic: float = 1e-2         # 评论家学习率（Adam）
    hidden_dim: int = 64
    seed: int = 0


def discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    g, out = 0.0, np.zeros_like(rewards, dtype=float)
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


def train_continuous_pg(env: PortfolioEnv, policy_cls, cfg: ContinuousPGConfig) -> Tuple[nn.Module, ContinuousCritic]:
    """
    带评论家基线的得分函数策略梯度；ω 用 sample() 抽取，梯度只经过 log_prob。
    policy_cls 为 BetaPolicy 或 SigmoidGaussianPolicy；网络在设定随机种子之后才构造，保证可复现。
    """
    gamma = float(env.p.beta if cfg.gamma is None else cfg.gamma)
    torch.manual_seed(cfg.seed)
    env.rng = np.random.default_rng(cfg.seed + 1)

    policy = policy_cls(env.state_dim, cfg.hidden_dim)
    critic = ContinuousCritic(env.state_dim, cfg.hidden_dim)
    opt_a = torch.optim.Adam(policy.parameters(), lr=cfg.lr_actor)
    opt_v = torch.optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    for _ in range(cfg.num_iterations):
        states: List[np.ndarray] = []
        omegas: List[float] = []
        rewards: List[float] = []

        state = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                # 不带梯度的样本；夹到 (1e-4, 1-1e-4) 使 log_prob 有限
                omega = float(policy.dist(torch.from_numpy(state)).sample().clamp(1e-4, 1.0 - 1e-4))
            next_state, reward, done, _ = env.step(omega)
            states.append(state)
            omegas.append(omega)
            rewards.append(reward)
            state = next_state

        S = torch.from_numpy(np.stack(states))
        G = torch.tensor(discounted_returns(np.asarray(rewards), gamma), dtype=torch.float32)
        v = critic(S)
        loss_v = (G - v).pow(2).mean()
        advantage = (G - v).detach()                     # 基线对演员是常数
        # θ 在一段轨迹内不变，段末把整段 (s_t, ω_t) 一起代入 log g_θ(ω_t|s_t)，与逐步计算相同；
        # ω_t 是数据，梯度只经过 log g_θ：得分函数估计
        logp = policy.dist(S).log_prob(torch.tensor(omegas, dtype=torch.float32))
        loss_a = -(logp * advantage).mean()

        opt_v.zero_grad()
        loss_v.backward()
        opt_v.step()
        opt_a.zero_grad()
        loss_a.backward()
        opt_a.step()

    return policy, critic


def evaluate(env: PortfolioEnv, policy: nn.Module, bench: MyopicBenchmark,
             num_episodes: int = 200, seed: int = 7) -> Tuple[float, float, float, float]:
    """
    按贪心 ω 模拟状态路径；每个访问到的状态上的期望效用用积分精确计算（去掉回报抽样噪声）。
    返回 (期望贴现回报, 贪心 ω 的均值, 10% 分位, 90% 分位)。
    """
    gamma = float(env.p.beta)
    env.rng = np.random.default_rng(seed)
    values, omegas = [], []
    for _ in range(num_episodes):
        state = env.reset()
        done, total, discount = False, 0.0, 1.0
        while not done:
            with torch.no_grad():
                omega = float(policy.greedy_action(torch.from_numpy(state)))
            omegas.append(omega)
            total += discount * float(bench.expected_utility(omega)[0])
            discount *= gamma
            state, _, done, _ = env.step(omega)
        values.append(total)
    omegas_arr = np.asarray(omegas)
    return (float(np.mean(values)), float(omegas_arr.mean()),
            float(np.percentile(omegas_arr, 10)), float(np.percentile(omegas_arr, 90)))


def main() -> None:
    parser = argparse.ArgumentParser(description="连续份额 ω：Beta 策略与 sigmoid-高斯策略")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="随机种子（每个种子各训练一次）")
    args = parser.parse_args()

    t_start = time.time()
    params = PortfolioParams()
    bench = MyopicBenchmark(params)
    ce_star = float(bench.certainty_equivalent(bench.eu_star))
    merton = (params.mu_e - np.log(params.r_f)) / (params.crra * params.sigma_e ** 2)
    print("=" * 78)
    print("连续份额 ω 的策略梯度：Beta 策略与 sigmoid-高斯策略")
    print("=" * 78)
    print(f"短视最优份额 ω* = {bench.omega_star:.3f}（Merton 近似 {merton:.3f}）；"
          f"每期确定性等价毛回报 {ce_star:.5f}")
    for omega in (0.0, 0.5, bench.omega_star, 0.9, 1.0):
        loss_bp = 1e4 * (ce_star - float(bench.certainty_equivalent(bench.expected_utility(omega)[0])))
        print(f"[固定 ω={omega:.3f}] 50 期期望贴现回报（精确）{bench.discounted_value(omega):.4f}；"
              f"每期确定性等价回报比 ω* 低 {loss_bp:.1f} 个基点")
    cfg0 = ContinuousPGConfig()
    print(f"训练：每次更新用 1 段 {params.episode_length} 期轨迹，{cfg0.num_iterations} 次更新；"
          f"演员学习率 {cfg0.lr_actor}、评论家 {cfg0.lr_critic}；种子 {args.seeds}")

    for name, policy_cls in (("Beta", BetaPolicy), ("Sigmoid-高斯", SigmoidGaussianPolicy)):
        for seed in args.seeds:
            t0 = time.time()
            env = PortfolioEnv(params, seed=0)
            policy, _ = train_continuous_pg(env, policy_cls, ContinuousPGConfig(seed=seed))
            value, mean_omega, p10, p90 = evaluate(env, policy, bench)
            loss_bp = 1e4 * (ce_star - float(bench.certainty_equivalent(bench.expected_utility(mean_omega)[0])))
            print(f"[{name} 策略，种子{seed}] 贪心 ω 均值 {mean_omega:.3f}（10%–90% 分位 {p10:.3f}–{p90:.3f}）；"
                  f"期望贴现回报 {value:.4f}（ω* 为 {bench.discounted_value(bench.omega_star):.4f}）；"
                  f"按均值 ω 计每期确定性等价回报比 ω* 低 {loss_bp:.1f} 个基点；训练 {time.time() - t0:.0f} 秒")
    print(f"\n总耗时 {time.time() - t_start:.0f} 秒")


if __name__ == "__main__":
    main()
