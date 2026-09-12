"""Aiyagari (1994) household problem -- partial equilibrium value function iteration.

The household solves

    V(a, z) = max_{a'} u(c) + beta * E[ V(a', z') | z ]
    s.t.     c = (1 + r) * a + w * z - a'
             a' >= a_min

with CRRA utility u(c) = c^(1 - sigma) / (1 - sigma), and a two-state Markov
chain for the idiosyncratic labour endowment z.

Calibration is annual: beta = 0.96, sigma = 2, r = 4%, w = 1.

Course material for the Lecture 9 agent lab. Not a reference implementation --
do not reuse as a template; see ../README.md.
"""

import numpy as np

BETA = 0.96
SIGMA = 2.0
R = 0.04
W = 1.0

A_MIN, A_MAX, N_A = 0.0, 20.0, 150
Z_VALS = np.array([0.7, 1.3])
P = np.array([[0.9, 0.1],
              [0.1, 0.9]])

MAX_ITER = 400
TOL = 1e-6


def utility(c):
    return c ** (1.0 - SIGMA) / (1.0 - SIGMA)


def solve(verbose=True):
    a_grid = np.linspace(A_MIN, A_MAX, N_A)
    n_z = len(Z_VALS)
    V = np.zeros((N_A, n_z))
    policy = np.zeros((N_A, n_z), dtype=int)

    for it in range(MAX_ITER):
        V_new = np.empty_like(V)
        EV = P @ V.T

        for iz in range(n_z):
            cash = (1.0 + R) * a_grid + W * Z_VALS[iz]
            for ia in range(N_A):
                c = cash[ia] - a_grid
                u = utility(c)
                total = u + BETA * EV[iz]
                best = np.argmax(total)
                V_new[ia, iz] = total[best]
                policy[ia, iz] = best

        V = V_new

        if verbose and it % 100 == 0:
            print(f"iter {it:4d}  V[0,0] = {V[0, 0]: .6f}")

    if verbose:
        print(f"done after {MAX_ITER} iterations")
    return a_grid, V, policy


def aggregate_assets(a_grid, policy):
    """Mean assets under the stationary distribution."""
    dist = np.ones((len(a_grid), len(Z_VALS)))
    dist /= dist.sum()
    for _ in range(500):
        new = np.zeros_like(dist)
        for iz in range(len(Z_VALS)):
            for ia in range(len(a_grid)):
                ia_next = policy[ia, iz]
                for jz in range(len(Z_VALS)):
                    new[ia_next, jz] += dist[ia, iz] * P[iz, jz]
        dist = new
    return float((dist.sum(axis=1) * a_grid).sum())


if __name__ == "__main__":
    grid, V, pol = solve()
    print(f"aggregate assets = {aggregate_assets(grid, pol):.4f}")
