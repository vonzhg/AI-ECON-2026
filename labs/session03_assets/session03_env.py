"""Session 3 access to the book's discrete growth model.

The model, its simulator and its dynamic-programming benchmark are the book's own code, vendored
unchanged (two import-path fixes; see book_code/MANIFEST.json) under book_code/ch11 and ch12:

    Miao & Feng, AI for Economists, ch. 11 (code/ch11) and ch. 12 (code/ch12).

This file only puts those folders on the import path, builds the two grids the session uses, and adds
a wrapper that proves a learner never reads the transition matrix.

    7 x 2   the running example of the text: every number quoted on the slides
    20 x 2  the exercise environment: every algorithm in the lab

Calibration, both grids: alpha = 0.36, beta = 0.95, delta = 0.10, u = ln c,
z in {0.9, 1.1} with switching probability 0.2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BOOK = HERE / "book_code"
for sub in ("ch11", "ch12"):
    p = str(BOOK / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
sys.dont_write_bytecode = True

from growth_mdp import DiscreteGrowthMDP, make_default_env  # noqa: E402  book ch. 11
from growth_env import GrowthBenchmark, GrowthEnv, make_env  # noqa: E402  book ch. 12

K7 = np.array([1.5, 2.25, 3.0, 3.75, 4.5, 5.25, 6.0])
Z2 = np.array([0.9, 1.1])
P_Z = np.array([[0.8, 0.2], [0.2, 0.8]])


def make_env_7x2(seed: int = 0, episode_length: int = 100) -> GrowthEnv:
    """The text's running example: 14 states, 68 feasible state-action pairs."""
    mdp = DiscreteGrowthMDP(K7, Z2, P_Z, episode_length=episode_length, seed=seed)
    return GrowthEnv(mdp, episode_length=episode_length, seed=seed)


def make_env_20x2(seed: int = 0, episode_length: int = 100) -> GrowthEnv:
    """The exercise environment: 40 states, 542 feasible state-action pairs."""
    return make_env(seed=seed, episode_length=episode_length)


def vfi_trace(env: GrowthEnv, tol: float = 1e-6) -> dict[str, np.ndarray]:
    """Value function iteration from V = 0, recording the sup-norm change and the number of states
    whose greedy action still differs from the final one."""
    b = GrowthBenchmark(env)
    V = np.zeros(b.num_states)
    change, greedy = [], []
    while True:
        Q = np.where(b.feasible, b.reward + b.gamma * b.P @ V, -np.inf)
        V_new = Q.max(axis=1)
        change.append(float(np.max(np.abs(V_new - V))))
        greedy.append(int(np.sum(Q.argmax(axis=1) != b.g_star)))
        V = V_new
        if change[-1] < tol:
            break
    return {"change": np.array(change), "wrong_states": np.array(greedy)}


def branches(env: GrowthEnv, state, action: int):
    """The full backup as code: every branch of one transition, with its weight.  Returns a list of
    (probability, next_state, reward) over tomorrow's productivity, read from the same objects
    `step` uses -- so the rewards match `step` to the digit and the weights are the rows of P_z.
    A learner may call `step` (one drawn branch, Decks 3.2-3.4) or `branches` (every branch
    weighted, Deck 3.4's model-in-hand actor-critic); a blinded learner may only call `step`."""
    if isinstance(env, ModelBlinded):
        env.touched.append("branches")
        raise AttributeError("the learner may not call 'branches': the model is hidden")
    k_idx, z_idx = state
    a = int(action)
    if not env.mdp._feasible[k_idx, z_idx, a]:
        raise ValueError("Infeasible action: violates c > 0.")
    reward = float(env.mdp._reward[k_idx, z_idx, a])
    return [(float(env.p_z[z_idx, z_next]), (a, z_next), reward) for z_next in range(env.num_z)]


class ModelBlinded:
    """Hand a learner this instead of the environment: reset, step and the action mask work;
    anything that would reveal the transition matrix raises, and so does `branches`."""

    _HIDDEN = {"p_z", "_p_z_cdf", "mdp", "P", "rng", "branches"}

    def __init__(self, env: GrowthEnv) -> None:
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "touched", [])

    def __getattr__(self, name: str):
        if name in ModelBlinded._HIDDEN:
            self.touched.append(name)
            raise AttributeError(f"the learner may not read '{name}': the model is hidden")
        return getattr(object.__getattribute__(self, "_env"), name)


def recorded_digest(recorded: dict, script: str) -> list[str]:
    """English one-liners read out of a recorded full-budget run (the scripts print in Chinese).
    Every number is parsed from the recorded text; nothing is typed in by hand."""
    import re
    text = recorded["scripts"][script]["stdout"]
    secs = recorded["scripts"][script]["seconds"]
    out: list[str] = []
    if script == "ch11/mc_control.py":
        errs = re.findall(r"episode=\s*24000:\s+mean\|Q-Q\*\| = ([\d.]+)", text)
        loss = re.findall(r"遍历集平均价值损失\s+= ([\d.]+)\s+\(相对 \|V\*\| ≈ ([\d.]+)%\)", text)
        for name, e, (v, rel) in zip(("exploring starts", "epsilon-greedy"), errs, loss):
            out.append(f"{name:17s} 24,000 episodes: mean |Q - Q*| = {e}; value loss on the ergodic set {v} ({rel}% of |V*|)")
    elif script == "ch12/tabular_td_control.py":
        for zh, en in (("SARSA", "SARSA"), ("Q学习", "Q-learning")):
            m = re.search(zh + r" 1,000,000 步小结（3 个种子）：与 g\* 相同 ([\d–]+)/40；ℓ_CE 全部 ([\d.%–]+)，加权 ([\d.%–]+)", text)
            out.append(f"{en:10s} 1,000,000 steps, 3 seeds: agrees with g* in {m.group(1)}/40 states; "
                       f"loss {m.group(2)} (all), {m.group(3)} (weighted)")
    elif script == "ch12/dqn_growth.py":
        m = re.search(r"\[DQN 贪心策略\] 与 g\* 相同 (\d+)/40.*?ℓ_CE 全部平均 ([\d.]+%)，加权 ([\d.]+%)", text)
        out.append(f"DQN, 100,000 steps, seed 0: agrees with g* in {m.group(1)}/40 states; loss {m.group(2)} (all), {m.group(3)} (weighted)")
    elif script == "ch12/policy_gradient.py":
        for zh, en in (("REINFORCE", "REINFORCE, 3,000 iterations"), ("演员—评论家", "actor-critic, 30,000 updates")):
            m = re.search(r"\[" + zh + r"\] 贪心策略与 g\* 相同 (\d+)/40.*?ℓ_CE 全部 ([\d.]+%)，加权 ([\d.]+%)", text)
            out.append(f"{en:28s} seed 0: agrees with g* in {m.group(1)}/40 states; loss {m.group(2)} (all), {m.group(3)} (weighted)")
    elif script == "ch12/portfolio_ac.py":
        for zh, en in (("Beta", "Beta"), ("Sigmoid-高斯", "sigmoid-Gaussian")):
            for seed, mean, bp in re.findall(r"\[" + zh + r" 策略，种子(\d)\] 贪心 ω 均值 ([\d.]+).*?低 ([\d.]+) 个基点", text):
                out.append(f"{en:17s} seed {seed}: mean greedy share {mean}; certainty-equivalent return {bp} basis points below omega*")
    else:
        raise KeyError(script)
    return [f"recorded, full budget ({script}, {secs} s):"] + ["  " + line for line in out]
