# Ramsey_RA_simulation_module_v2.py
"""
Simulation Module for Deep Ramsey Algorithm.
VERSION 2: Streamlined and aligned with v2 modules.

Handles:
- t=0 Ramsey problem (direct optimization)
- t>=1 policy rollout
- Visualization of simulation paths
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os


# =============================================================================
# T=0 RAMSEY PROBLEM
# =============================================================================

def solve_t0_problem(b0: float, g0_idx: int, value_net: nn.Module,
                     config: dict, device: torch.device) -> tuple:
    """
    Solve the t=0 Ramsey problem via direct optimization.
    
    At t=0, the government chooses (l0, μ'(g_L), μ'(g_H)) to maximize:
        U(c0, l0) + β * E[V(B', μ', g')]
    
    Returns:
        (mu0, mu_next_g0, mu_next_g1)
    """
    econ = config['economic_parameters']
    beta = econ['beta']
    gamma_l = econ['gamma_l']
    mu_min = econ['mu_min']
    mu_max = econ['mu_max']
    
    zagg_vec = torch.tensor(econ['zagg_vec'], device=device, dtype=torch.float32).squeeze()
    pi_zagg = torch.tensor(econ['pi_zagg'], device=device, dtype=torch.float32)
    
    g0_val = zagg_vec[g0_idx].item()
    
    # Bounds
    l0_min = g0_val + 0.01  # c0 > 0 requires l0 > g0
    l0_max = 0.999
    
    def objective(x):
        """Negative of value (for minimization)."""
        l0, mu_next_g0, mu_next_g1 = x
        
        # Check bounds
        if (l0 <= l0_min or l0 >= l0_max or
            mu_next_g0 < mu_min or mu_next_g0 > mu_max or
            mu_next_g1 < mu_min or mu_next_g1 > mu_max):
            return 1e10
        
        c0 = l0 - g0_val
        if c0 <= 0:
            return 1e10
        
        mu0 = 1.0 / c0
        if mu0 < mu_min or mu0 > mu_max:
            return 1e10
        
        # Current utility
        u0 = np.log(c0) + gamma_l * np.log(1.0 - l0)
        
        # Bond price and next debt
        E_mu_next = pi_zagg[g0_idx, 0].item() * mu_next_g0 + pi_zagg[g0_idx, 1].item() * mu_next_g1
        q0 = beta * E_mu_next / mu0
        
        if q0 <= 0:
            return 1e10
        
        tau0 = 1.0 - gamma_l * c0 / (1.0 - l0 + 1e-8)
        tau0 = max(0.01, min(0.99, tau0))
        
        x0 = c0 + g0_val
        B_next = (b0 + g0_val - tau0 * x0) / q0
        
        # Continuation values
        with torch.no_grad():
            input_g0 = torch.tensor([[B_next, mu_next_g0, zagg_vec[0].item()]], 
                                   device=device, dtype=torch.float32)
            input_g1 = torch.tensor([[B_next, mu_next_g1, zagg_vec[1].item()]], 
                                   device=device, dtype=torch.float32)
            
            V_g0 = value_net(input_g0).item()
            V_g1 = value_net(input_g1).item()
        
        E_V_next = pi_zagg[g0_idx, 0].item() * V_g0 + pi_zagg[g0_idx, 1].item() * V_g1
        
        total_value = u0 + beta * E_V_next
        return -total_value
    
    # Optimize
    l0_init = (l0_min + l0_max) / 2
    mu_init = (mu_min + mu_max) / 2
    x0 = [l0_init, mu_init, mu_init]
    bounds = [(l0_min, l0_max), (mu_min, mu_max), (mu_min, mu_max)]
    
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    
    if result.success:
        l0_opt, mu_next_g0_opt, mu_next_g1_opt = result.x
        c0_opt = l0_opt - g0_val
        mu0_opt = 1.0 / c0_opt
        return mu0_opt, mu_next_g0_opt, mu_next_g1_opt
    else:
        print(f"  [T=0] Optimization failed for b0={b0:.3f}, using initial guess")
        c0_init = l0_init - g0_val
        mu0_init = 1.0 / c0_init
        return mu0_init, mu_init, mu_init


# =============================================================================
# SIMULATION
# =============================================================================

def run_simulation(b_init: float, g_init_idx: int,
                   policy_net: nn.Module, value_net: nn.Module,
                   config: dict, device: torch.device, T: int = 100) -> pd.DataFrame:
    """
    Run a single simulation path.
    
    t=0: Solve Ramsey problem directly
    t>=1: Follow trained policy network
    
    Returns:
        DataFrame with columns: t, b, mu, g_idx, g_val, c, l, tau, q, b_next
    """
    econ = config['economic_parameters']
    bounds = config['feasibility_bounds']
    
    beta = econ['beta']
    gamma_l = econ['gamma_l']
    mu_min = econ['mu_min']
    mu_max = econ['mu_max']
    zagg_vec = torch.tensor(econ['zagg_vec'], device=device, dtype=torch.float32).squeeze()
    pi_zagg = torch.tensor(econ['pi_zagg'], device=device, dtype=torch.float32)
    
    b_min = bounds['b_min_initial']
    b_max = bounds['b_max_initial']
    
    results = []
    
    # === Solve t=0 ===
    mu0, mu_next_g0_opt, mu_next_g1_opt = solve_t0_problem(
        b_init, g_init_idx, value_net, config, device)
    
    B_t = b_init
    mu_t = mu0
    g_t_idx = g_init_idx
    
    policy_net.eval()
    
    # === Simulate ===
    for t in range(T):
        g_t_val = zagg_vec[g_t_idx].item()
        
        c_t = 1.0 / mu_t
        x_t = c_t + g_t_val
        l_t = 1.0 - x_t
        
        # Get policy
        if t == 0:
            mu_next_g0 = mu_next_g0_opt
            mu_next_g1 = mu_next_g1_opt
        else:
            with torch.no_grad():
                state = torch.tensor([[B_t, mu_t, g_t_val]], device=device, dtype=torch.float32)
                logits = policy_net(state)
                mu_next_g0 = (torch.sigmoid(logits[0, 0]) * (mu_max - mu_min) + mu_min).item()
                mu_next_g1 = (torch.sigmoid(logits[0, 1]) * (mu_max - mu_min) + mu_min).item()
        
        # Bond price
        E_mu_next = pi_zagg[g_t_idx, 0].item() * mu_next_g0 + pi_zagg[g_t_idx, 1].item() * mu_next_g1
        q_t = beta * E_mu_next / mu_t
        
        # Tax
        tau_t = 1.0 - gamma_l * c_t / (l_t + 1e-8)
        tau_t = max(0.01, min(0.99, tau_t))
        
        # Next debt
        B_next = (B_t + g_t_val - tau_t * x_t) / q_t
        B_next_clamped = np.clip(B_next, b_min, b_max)
        
        results.append({
            't': t,
            'b': B_t,
            'mu': mu_t,
            'g_idx': g_t_idx,
            'g_val': g_t_val,
            'c': c_t,
            'l': l_t,
            'tau': tau_t,
            'q': q_t,
            'b_next': B_next_clamped
        })
        
        # Transition
        g_next_idx = torch.multinomial(pi_zagg[g_t_idx], 1).item()
        mu_next = mu_next_g0 if g_next_idx == 0 else mu_next_g1
        
        B_t = B_next_clamped
        mu_t = mu_next
        g_t_idx = g_next_idx
    
    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_simulation_timeseries(sim_dfs: list, save_path: str = 'figures/simulation_timeseries.png'):
    """Plot time series of key variables across simulations."""
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Simulation Results', fontsize=14)
    
    variables = [
        ('b', 'Debt (B)'),
        ('mu', 'Multiplier (μ)'),
        ('g_val', 'Govt Spending (g)'),
        ('c', 'Consumption (c)'),
        ('l', 'Leisure (l)'),
        ('tau', 'Tax Rate (τ)')
    ]
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(sim_dfs)))
    
    for (var, label), ax in zip(variables, axes.flatten()):
        for i, df in enumerate(sim_dfs):
            b_init = df['b'].iloc[0]
            ax.plot(df['t'], df[var], color=colors[i], 
                   label=f'B₀={b_init:.2f}', alpha=0.8)
        
        ax.set_xlabel('Time (t)')
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Simulation] Time series saved to: {save_path}")


def plot_simulation_statespace(sim_dfs: list, config: dict,
                               save_path: str = 'figures/simulation_statespace.png'):
    """Plot (B, μ) state-space trajectories."""
    
    bounds = config['feasibility_bounds']
    econ = config['economic_parameters']
    
    b_min = bounds['b_min_initial']
    b_max = bounds['b_max_initial']
    mu_min = econ['mu_min']
    mu_max = econ['mu_max']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(sim_dfs)))
    
    for i, df in enumerate(sim_dfs):
        b_init = df['b'].iloc[0]
        
        # Plot trajectory
        ax.scatter(df['b'], df['mu'], c=[colors[i]], s=10, alpha=0.5)
        
        # Mark start and end
        ax.scatter(df['b'].iloc[0], df['mu'].iloc[0], 
                  marker='o', s=100, c=[colors[i]], edgecolors='black', 
                  label=f'Start B₀={b_init:.2f}', zorder=5)
        ax.scatter(df['b'].iloc[-1], df['mu'].iloc[-1],
                  marker='X', s=120, c=[colors[i]], edgecolors='black', zorder=5)
    
    ax.set_xlabel('Debt (B)')
    ax.set_ylabel('Multiplier (μ)')
    ax.set_xlim(b_min - 0.1, b_max + 0.1)
    ax.set_ylim(mu_min - 0.1, mu_max + 0.1)
    ax.set_title('State Space Trajectories')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Simulation] State space saved to: {save_path}")


def run_and_plot_simulations(policy_net: nn.Module, value_net: nn.Module,
                             config: dict, device: torch.device):
    """Run multiple simulations and generate plots."""
    
    sim_cfg = config['simulation']
    bounds = config['feasibility_bounds']
    
    T = sim_cfg['T_periods']
    n_paths = sim_cfg['n_paths']
    
    b_min = bounds['b_min_initial']
    b_max = bounds['b_max_initial']
    
    # Starting points
    b_starts = np.linspace(b_min + 0.1, (b_min + b_max) / 2, n_paths)
    g_start_idx = 0
    
    print(f"\n[Simulation] Running {n_paths} paths for T={T} periods...")
    
    sim_dfs = []
    policy_net.eval()
    value_net.eval()
    
    with torch.no_grad():
        for b0 in b_starts:
            df = run_simulation(b0, g_start_idx, policy_net, value_net, config, device, T)
            sim_dfs.append(df)
            print(f"  Completed: B₀={b0:.3f}")
    
    plot_simulation_timeseries(sim_dfs, 'figures/simulation_timeseries.png')
    plot_simulation_statespace(sim_dfs, config, 'figures/simulation_statespace.png')
    
    print("[Simulation] Complete.")
    return sim_dfs
