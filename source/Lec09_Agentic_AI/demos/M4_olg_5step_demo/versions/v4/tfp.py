"""V3 — aggregate TFP discretized via Tauchen's method.

Pure-stdlib (well, plus numpy) discretization of an AR(1) process

    log z_{t+1} = rho * log z_t + eps_t,   eps_t ~ N(0, sigma^2)

into ``n_tfp`` states. Returns the TFP grid (in levels, ``z = exp(grid)``)
and the transition matrix. ``n_tfp = 1`` returns a degenerate process
where TFP is constant at 1.0 — this is the V3 → V2 reduction.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


def tauchen(
    n: int,
    rho: float,
    sigma: float,
    m: float = 3.0,
) -> Tuple[List[float], List[List[float]]]:
    """Discretize AR(1) ``log z_{t+1} = rho * log z_t + eps``, ``eps ~ N(0, sigma^2)``.

    Returns ``(grid, transition)`` where:
      * ``grid[i]`` is the level of TFP at state i (i.e., ``exp(log_grid[i])``).
      * ``transition[i][j]`` is the probability of moving from state i to state j.

    For ``n == 1`` returns the degenerate single-state process at ``z = 1.0``,
    used as the V3 → V2 reduction.
    """
    if n < 1:
        raise ValueError("tauchen: n must be >= 1")
    if n == 1:
        return [1.0], [[1.0]]
    if sigma <= 0.0:
        # σ = 0 is degenerate — collapse to a single point at log z = 0.
        return [1.0] * n, [[1.0 if j == 0 else 0.0 for j in range(n)] for _ in range(n)]

    sigma_z = sigma / math.sqrt(max(1.0 - rho * rho, 1.0e-12))
    log_max = m * sigma_z
    log_min = -log_max
    step = (log_max - log_min) / (n - 1)
    log_grid = [log_min + i * step for i in range(n)]

    def cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    transition: List[List[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
    for i, x in enumerate(log_grid):
        for j, y in enumerate(log_grid):
            mu = rho * x
            if j == 0:
                transition[i][j] = cdf((y + step / 2.0 - mu) / sigma)
            elif j == n - 1:
                transition[i][j] = 1.0 - cdf((y - step / 2.0 - mu) / sigma)
            else:
                transition[i][j] = (
                    cdf((y + step / 2.0 - mu) / sigma)
                    - cdf((y - step / 2.0 - mu) / sigma)
                )
        s = sum(transition[i])
        if s > 0.0:
            transition[i] = [p / s for p in transition[i]]

    grid = [math.exp(x) for x in log_grid]
    return grid, transition


def tfp_stationary_distribution(transition: List[List[float]]) -> List[float]:
    """Power-iterate the TFP transition matrix to get the stationary distribution."""
    n = len(transition)
    dist = [1.0 / n for _ in range(n)]
    P = np.array(transition)
    for _ in range(10_000):
        new = list(np.array(dist) @ P)
        err = max(abs(new[i] - dist[i]) for i in range(n))
        dist = [float(x) for x in new]
        if err < 1.0e-14:
            break
    s = sum(dist)
    return [x / s for x in dist]
