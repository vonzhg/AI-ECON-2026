"""Session 3: the rows of Deck 3.5 that the book's code does not ship.

Course-owned (the book's code under book_code/ stays unchanged).  Every function runs on the book's
20 x 2 exercise grid through its own interfaces: `GrowthBenchmark` for the model-in-hand methods,
`reset` / `step` / `action_mask` for the model-free ones (so a `ModelBlinded` wrapper works).

    policy_iteration      Howard's algorithm: evaluate the rule exactly, improve by a max     (with the model)
    td_prediction         tabular TD(0), n-step TD, Monte Carlo (n = None) and TD(lambda)     (without it)
    train_td_control_x    expected SARSA and double Q-learning, in the book's SARSA/Q set-up  (without it)

Step sizes follow the book's TD control script: a per-cell count C and eta = C^(-0.6).
"""

from __future__ import annotations

import numpy as np

import session03_env as s3  # noqa: F401  puts the book's code on the path
from tabular_td_control import TDControlConfig, epsilon_greedy, masked_argmax  # noqa: E402  book ch. 12


# ------------------------------------------------------------------ with the model
def policy_iteration(bench, g0, weights, max_iter: int = 100) -> list[dict]:
    """Howard's policy iteration from the rule g0.  Each round: V^g by one linear solve (evaluation
    until V = T^g V), then g <- argmax_a r + gamma P V^g over feasible k' (improvement).  Returns
    one record per round: the rule's grade before improving it, and how many states changed."""
    g = np.asarray(g0, dtype=int).copy()
    out = []
    for it in range(max_iter):
        m = bench.evaluate_greedy(g, weights)
        V = bench.policy_value(g)
        g_new = bench.greedy_from_q(bench.reward + bench.gamma * bench.P @ V)
        changed = int(np.sum(g_new != g))
        out.append({"round": it, "changed": changed, **m})
        if changed == 0:
            break
        g = g_new
    return out


def median_rule(bench) -> np.ndarray:
    """The median feasible k' in every state: block B1's rule, on the 20 x 2 grid."""
    return np.array([f[len(f) // 2] for f in bench.feasible_list])


# ------------------------------------------------------------------ without the model: evaluation
def td_prediction(env, pi: np.ndarray, *, n: int | None = 1, lam: float | None = None,
                  num_steps: int = 200_000, seed: int = 0, step_exponent: float = 0.6) -> np.ndarray:
    """Value the stochastic rule pi (S x A action probabilities) from `step` alone.

    lam is None:   n-step TD.  The target for s_t is r_{t+1} + ... + gamma^(n-1) r_{t+n} + gamma^n V(s_{t+n}),
                   cut at the end of the episode, where the target bootstraps on the last state (the
                   episode is truncated, not terminated).  n = 1 is TD(0); n = None is Monte Carlo:
                   the whole path's discounted return, no bootstrap.
    lam = l:       TD(lambda) with accumulating traces, online: e <- gamma l e + 1{s_t = s}, then every
                   state moves by eta_s delta_t e(s).  lam = 0 is TD(0); lam -> 1 approaches Monte Carlo.

    Each state's step is eta = C(s)^(-step_exponent), C its visit count.  pi is used only to draw actions."""
    rng = np.random.default_rng(seed)
    gamma = env.gamma
    S = env.num_states
    V = np.zeros(S)
    C = np.zeros(S)
    cdf = np.cumsum(pi, axis=1)
    act = lambda s: int(min(np.searchsorted(cdf[s], rng.random(), side="right"), pi.shape[1] - 1))
    step = 0
    while step < num_steps:
        s = env.state_index(*env.reset())
        states, rewards = [s], []
        e = np.zeros(S) if lam is not None else None
        while True:
            nxt, r, terminated, truncated, _ = env.step(act(s))
            s2 = env.state_index(*nxt)
            step += 1
            if lam is not None:
                delta = r + (0.0 if terminated else gamma * V[s2]) - V[s]
                e *= gamma * lam
                e[s] += 1.0
                C[s] += 1
                live = e > 1e-8
                V[live] += np.where(C[live] > 0, C[live], 1.0) ** (-step_exponent) * delta * e[live]
            else:
                states.append(s2)
                rewards.append(r)
            if terminated or truncated or step >= num_steps:
                break
            s = s2
        if lam is None:
            _nstep_episode(V, C, states, rewards, n, gamma, step_exponent, bootstrap_end=not terminated)
    return V


def _nstep_episode(V, C, states, rewards, n, gamma, step_exponent, bootstrap_end):
    """Apply the n-step updates of one finished episode, in time order (n = None: Monte Carlo)."""
    T = len(rewards)
    disc = gamma ** np.arange(T)
    for t in range(T):
        h = T - t if n is None else min(n, T - t)
        G = float(disc[:h] @ np.asarray(rewards[t:t + h]))
        if n is not None and (t + h < T or bootstrap_end):
            G += gamma ** h * V[states[t + h]]
        s = states[t]
        C[s] += 1
        V[s] += (G - V[s]) * C[s] ** (-step_exponent)


def weighted_rmse(V_hat, V, w) -> float:
    return float(np.sqrt(w @ (V_hat - V) ** 2))


# ------------------------------------------------------------------ without the model: control
def train_td_control_x(env, method: str, cfg: TDControlConfig) -> np.ndarray:
    """method = "expected_sarsa" or "double_q", in the set-up of the book's train_td_control.

    expected SARSA: target r + gamma sum_a' pi_eps(a'|s') Q(s', a'), the epsilon-greedy mean over feasible k'
                    in place of SARSA's one drawn k''.
    double Q:       two tables; behave epsilon-greedy on Q1 + Q2; at each step one table, chosen by a coin,
                    picks the argmax at s' and the other one values it.  Returns (Q1 + Q2) / 2.
    """
    if method not in ("expected_sarsa", "double_q"):
        raise ValueError("method must be 'expected_sarsa' or 'double_q'")
    gamma, eps = env.gamma, cfg.epsilon
    rng = np.random.default_rng(cfg.seed)
    S, A = env.num_states, env.num_actions
    feasible = [np.flatnonzero(env.action_mask(env.index_to_state(s))) for s in range(S)]
    q = [np.zeros((S, A)), np.zeros((S, A))]
    cnt = [np.zeros((S, A), dtype=np.int64), np.zeros((S, A), dtype=np.int64)]
    behave = (lambda s: q[0][s]) if method == "expected_sarsa" else (lambda s: q[0][s] + q[1][s])

    step = 0
    while step < cfg.num_steps:
        s = env.state_index(*env.reset())
        while True:
            a = epsilon_greedy(behave(s), feasible[s], eps, rng)
            nxt, r, terminated, truncated, _ = env.step(a)
            s2 = env.state_index(*nxt)
            f2 = feasible[s2]
            if method == "expected_sarsa":
                row = q[0][s2, f2]
                best = row == row.max()
                probs = np.full(f2.size, eps / f2.size) + (1.0 - eps) * best / best.sum()
                i, boot = 0, float(probs @ row)
            else:
                i = int(rng.random() < 0.5)
                a_star = masked_argmax(q[i][s2], f2, rng)
                boot = q[1 - i][s2, a_star]
            target = r + (0.0 if terminated else gamma * boot)
            cnt[i][s, a] += 1
            q[i][s, a] += (target - q[i][s, a]) * float(cnt[i][s, a]) ** (-cfg.step_exponent)
            step += 1
            if terminated or truncated or step >= cfg.num_steps:
                break
            s = s2
    return q[0] if method == "expected_sarsa" else (q[0] + q[1]) / 2
