"""
Simulation Module for Ramsey Optimal Taxation.

This module simulates economic trajectories using trained policy and value
networks. It handles the special t=0 Ramsey problem and generates diagnostic
plots for analyzing model behavior.

Key Features:
    - t=0 optimization: Solves the initial Ramsey problem without commitment
    - Forward simulation: Follows the policy network for t ≥ 1
    - State-space visualization: Plots trajectories in (B, μ) space
    - Time series plots: Shows evolution of key variables

Mathematical Background:
    At t=0: The government chooses (c₀, l₀, μ'(.)) to maximize V subject to
            the implementability constraint (without prior commitment).
    
    For t≥1: The government follows the pre-committed policy μ'(B, μ, g)
             learned by the policy network.

Authors: Zhigang Feng
Version: 2.0 (Streamlined)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import json
import os
from typing import List, Tuple, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

class SimConfig:
    """Configuration for simulations."""
    
    def __init__(self, config: dict):
        """
        Load simulation configuration.
        
        Args:
            config: Configuration dictionary
        """
        # Economic parameters
        econ = config.get('economic_parameters', config)
        self.beta = econ.get('beta', config.get('beta', 0.9))
        self.gamma_l = econ.get('gamma_l', config.get('gamma_l', 0.3))
        self.zagg_vec = econ.get('zagg_vec', config.get('zagg_vec'))
        self.pi_zagg = econ.get('pi_zagg', config.get('pi_zagg'))
        
        # State bounds
        bounds = config.get('state_bounds', config)
        self.mu_min = bounds.get('mu_min', config.get('mu_min', 1.27))
        self.mu_max = bounds.get('mu_max', config.get('mu_max', 2.51))
        
        # Penalty/constraint bounds (may be dynamic)
        penalty = config.get('penalty_params', {})
        self.b_min = penalty.get('b_min', -0.5)
        self.b_max = penalty.get('b_max', 3.5)
        self.tau_min = penalty.get('tau_min', 0.0)
        self.tau_max = penalty.get('tau_max', 1.0)
        
        # Simulation settings
        sim = config.get('simulation', {})
        self.T_sim = sim.get('T_sim', 100)
        self.num_starting_points = sim.get('num_starting_points', 3)
        
        # Model I/O
        io_cfg = config.get('model_io', config)
        self.model_number = io_cfg.get('model_number_output', config.get('model_number_output', 102))


# =============================================================================
# T=0 OPTIMIZER
# =============================================================================

class RamseyT0Optimizer:
    """
    Solves the t=0 Ramsey problem.
    
    At t=0, the government has no prior commitment and chooses:
        (l₀, μ'(g=0), μ'(g=1))
    to maximize:
        U(c₀, l₀) + β·E[V(B', μ', g') | g₀]
    
    subject to:
        c₀ = l₀ - g₀  (resource constraint at t=0)
        μ₀ = 1/c₀     (FOC consistency)
        B' = (B₀ + g₀ - τ₀·x₀) / q₀  (budget constraint)
    
    This differs from t≥1 where the government is committed to the policy
    function μ'(B, μ, g).
    """
    
    def __init__(self, value_net: nn.Module, config: SimConfig, device: torch.device):
        """
        Initialize the t=0 optimizer.
        
        Args:
            value_net: Trained value network V(B, μ, g)
            config: Simulation configuration
            device: PyTorch device
        """
        self.value_net = value_net
        self.config = config
        self.device = device
        
        # Convert to tensors
        self.g_vals = torch.tensor(config.zagg_vec, device=device, dtype=torch.float32).squeeze()
        self.pi_zagg = torch.tensor(config.pi_zagg, device=device, dtype=torch.float32)
    
    def solve(self, B0: float, g0_idx: int) -> Tuple[float, float, float]:
        """
        Solve the t=0 Ramsey problem.
        
        Args:
            B0: Initial debt level
            g0_idx: Initial shock state index (0 or 1)
        
        Returns:
            Tuple of (μ₀, μ'(g=0), μ'(g=1))
        """
        cfg = self.config
        g0_val = self.g_vals[g0_idx].item()
        
        # Define search bounds
        l0_min = g0_val + 0.01  # l > g to ensure c > 0
        l0_max = 0.999
        
        def objective(x):
            """Objective: -V (we minimize, so this maximizes V)."""
            l0, mu_next_g0, mu_next_g1 = x
            
            # Check bounds
            if (l0 <= l0_min or l0 >= l0_max or
                mu_next_g0 < cfg.mu_min or mu_next_g0 > cfg.mu_max or
                mu_next_g1 < cfg.mu_min or mu_next_g1 > cfg.mu_max):
                return 1e10
            
            # Compute allocations
            c0 = l0 - g0_val
            if c0 <= 0:
                return 1e10
            
            mu0 = 1.0 / c0
            if mu0 < cfg.mu_min or mu0 > cfg.mu_max:
                return 1e10
            
            # Convert to tensors
            c0_t = torch.tensor(c0, device=self.device, dtype=torch.float32)
            l0_t = torch.tensor(l0, device=self.device, dtype=torch.float32)
            mu0_t = torch.tensor(mu0, device=self.device, dtype=torch.float32)
            mu_g0_t = torch.tensor(mu_next_g0, device=self.device, dtype=torch.float32)
            mu_g1_t = torch.tensor(mu_next_g1, device=self.device, dtype=torch.float32)
            
            # Current period utility
            U0 = torch.log(c0_t) + cfg.gamma_l * torch.log(1.0 - l0_t)
            
            # Compute bond price and next-period debt
            E_mu_next = self.pi_zagg[g0_idx, 0] * mu_g0_t + self.pi_zagg[g0_idx, 1] * mu_g1_t
            q0 = cfg.beta * E_mu_next / mu0_t
            
            if q0.item() <= 0:
                return 1e10
            
            x0 = c0 + g0_val
            tau0 = 1.0 - cfg.gamma_l * c0 / (1.0 - l0 + 1e-8)
            tau0 = max(tau0, 0.01)
            B_next = (B0 + g0_val - tau0 * x0) / q0.item()
            
            B_next_t = torch.tensor(B_next, device=self.device, dtype=torch.float32)
            
            # Expected continuation value
            input_g0 = torch.stack([B_next_t, mu_g0_t, self.g_vals[0]])
            input_g1 = torch.stack([B_next_t, mu_g1_t, self.g_vals[1]])
            
            V_g0 = self.value_net(input_g0.unsqueeze(0))
            V_g1 = self.value_net(input_g1.unsqueeze(0))
            
            E_V_next = self.pi_zagg[g0_idx, 0] * V_g0 + self.pi_zagg[g0_idx, 1] * V_g1
            
            # Total value
            total_V = U0 + cfg.beta * E_V_next.squeeze()
            
            return -total_V.item()
        
        # Initial guess
        l0_init = (l0_min + l0_max) / 2.0
        mu_init = (cfg.mu_min + cfg.mu_max) / 2.0
        x0 = [l0_init, mu_init, mu_init]
        
        # Bounds
        bounds = [
            (l0_min, l0_max),
            (cfg.mu_min, cfg.mu_max),
            (cfg.mu_min, cfg.mu_max)
        ]
        
        # Optimize
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            l0_opt, mu_g0_opt, mu_g1_opt = result.x
            c0_opt = l0_opt - g0_val
            mu0_opt = 1.0 / c0_opt
            return mu0_opt, mu_g0_opt, mu_g1_opt
        else:
            print(f"Warning: t=0 optimization failed for B0={B0:.2f}")
            # Return initial guess
            c0_init = l0_init - g0_val
            mu0_init = 1.0 / c0_init
            return mu0_init, mu_init, mu_init


# =============================================================================
# SIMULATOR
# =============================================================================

class RamseySimulator:
    """
    Simulates economic trajectories using trained networks.
    
    The simulation proceeds as:
    1. t=0: Solve Ramsey problem to get (μ₀, μ'(g=0), μ'(g=1))
    2. t≥1: Follow policy network μ'(B, μ, g)
    
    At each period, computes:
        - Allocations: c, l, x = c + g
        - Tax rate: τ = 1 - γ_l·c/l
        - Bond price: q = β·E[μ']/μ
        - Next debt: B' = (B + g - τx)/q
    """
    
    def __init__(self, policy_net: nn.Module, value_net: nn.Module,
                 config: SimConfig, device: torch.device):
        """
        Initialize simulator.
        
        Args:
            policy_net: Trained policy network
            value_net: Trained value network
            config: Simulation configuration
            device: PyTorch device
        """
        self.policy_net = policy_net
        self.value_net = value_net
        self.config = config
        self.device = device
        
        # Pre-compute tensors
        self.g_vals = torch.tensor(config.zagg_vec, device=device, dtype=torch.float32).squeeze()
        self.pi_zagg = torch.tensor(config.pi_zagg, device=device, dtype=torch.float32)
        
        # t=0 optimizer
        self.t0_optimizer = RamseyT0Optimizer(value_net, config, device)
    
    def simulate(self, B_init: float, g_init_idx: int, T: int = None) -> pd.DataFrame:
        """
        Run a single simulation trajectory.
        
        Args:
            B_init: Initial debt level
            g_init_idx: Initial shock state (0 or 1)
            T: Number of periods (default: config.T_sim)
        
        Returns:
            DataFrame with columns: t, b, mu, g_idx, g_val, c, l, tau, q, b_next
        """
        if T is None:
            T = self.config.T_sim
        
        cfg = self.config
        results = []
        
        # Solve t=0 problem
        print(f"  Simulating B_init={B_init:.2f}... solving t=0 problem")
        mu0, mu_next_g0, mu_next_g1 = self.t0_optimizer.solve(B_init, g_init_idx)
        
        # Initialize state
        B = B_init
        mu = mu0
        g_idx = g_init_idx
        
        for t in range(T):
            g_val = self.g_vals[g_idx].item()
            
            # Compute allocations
            c = 1.0 / mu
            x = c + g_val
            l = 1.0 - x
            
            # Get policy (t=0 uses optimizer result, t≥1 uses network)
            if t == 0:
                mu_g0 = mu_next_g0
                mu_g1 = mu_next_g1
            else:
                input_t = torch.tensor([[B, mu, g_val]], device=self.device, dtype=torch.float32)
                with torch.no_grad():
                    logits = self.policy_net(input_t)
                mu_g0 = (torch.sigmoid(logits[0, 0]) * (cfg.mu_max - cfg.mu_min) + cfg.mu_min).item()
                mu_g1 = (torch.sigmoid(logits[0, 1]) * (cfg.mu_max - cfg.mu_min) + cfg.mu_min).item()
            
            # Compute prices
            E_mu_next = self.pi_zagg[g_idx, 0].item() * mu_g0 + self.pi_zagg[g_idx, 1].item() * mu_g1
            q = cfg.beta * E_mu_next / mu
            
            # Tax rate
            tau = 1.0 - cfg.gamma_l * c / (l + 1e-8)
            tau = max(tau, 0.01)
            
            # Next-period debt
            B_next = (B + g_val - tau * x) / q
            B_next = np.clip(B_next, cfg.b_min, cfg.b_max)
            
            # Store results
            results.append({
                't': t,
                'b': B,
                'mu': mu,
                'g_idx': g_idx,
                'g_val': g_val,
                'c': c,
                'l': l,
                'tau': tau,
                'q': q,
                'b_next': B_next
            })
            
            # Transition to next period
            g_next_idx = torch.multinomial(self.pi_zagg[g_idx], 1).item()
            mu_next = mu_g0 if g_next_idx == 0 else mu_g1
            
            B = B_next
            mu = mu_next
            g_idx = g_next_idx
        
        return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_time_series(sim_dfs: List[pd.DataFrame], 
                     save_path: str = 'figures/simulation_time_series.png',
                     title: str = 'Simulation Results'):
    """
    Plot time series of simulation variables.
    
    Creates a 3×2 grid showing: B, μ, g, c, l, τ
    
    Args:
        sim_dfs: List of simulation DataFrames
        save_path: Output file path
        title: Plot title
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(title, fontsize=14)
    
    variables = [
        ('b', 'Debt (B)'),
        ('mu', 'Multiplier (μ)'),
        ('g_val', 'Govt. Spending (g)'),
        ('c', 'Consumption (c)'),
        ('l', 'Labor (l)'),
        ('tau', 'Tax Rate (τ)')
    ]
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(sim_dfs)))
    
    for (var, label), ax in zip(variables, axes.flatten()):
        for i, df in enumerate(sim_dfs):
            b_init = df['b'].iloc[0]
            ax.plot(df['t'], df[var], label=f'B₀={b_init:.2f}', 
                   color=colors[i], linewidth=1.5)
        
        ax.set_xlabel('Time (t)')
        ax.set_ylabel(label)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Time series plot saved to: {save_path}")


def plot_state_space_paths(sim_dfs: List[pd.DataFrame], config: SimConfig,
                           save_path: str = 'figures/simulation_state_space.png',
                           title: str = 'State-Space Trajectories (B vs μ)'):
    """
    Plot simulation trajectories in (B, μ) state space.
    
    Shows scatter points along each trajectory with start/end markers.
    
    Args:
        sim_dfs: List of simulation DataFrames
        config: Simulation configuration
        save_path: Output file path
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(title, fontsize=14)
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(sim_dfs)))
    
    for i, df in enumerate(sim_dfs):
        b_init = df['b'].iloc[0]
        g_init = df['g_idx'].iloc[0]
        
        # Scatter trajectory
        ax.scatter(df['b'], df['mu'], c=[colors[i]], s=10, alpha=0.5)
        
        # Mark start
        ax.scatter(df['b'].iloc[0], df['mu'].iloc[0], 
                  marker='o', s=150, c=[colors[i]], edgecolors='black',
                  linewidths=2, zorder=10, label=f'Start: B₀={b_init:.2f}')
        
        # Mark end
        ax.scatter(df['b'].iloc[-1], df['mu'].iloc[-1],
                  marker='X', s=150, c=[colors[i]], edgecolors='black',
                  linewidths=2, zorder=10)
    
    ax.set_xlabel('Debt (B)', fontsize=12)
    ax.set_ylabel('Multiplier (μ)', fontsize=12)
    ax.set_xlim(config.b_min, config.b_max)
    ax.set_ylim(config.mu_min, config.mu_max)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  State-space plot saved to: {save_path}")


# =============================================================================
# MAIN API
# =============================================================================

def run_simulations(policy_net: nn.Module, value_net: nn.Module,
                    config: dict, device: torch.device,
                    T_sim: int = None) -> List[pd.DataFrame]:
    """
    Run simulations with multiple starting points and generate plots.
    
    Args:
        policy_net: Trained policy network
        value_net: Trained value network
        config: Configuration dictionary
        device: PyTorch device
        T_sim: Simulation length (overrides config)
    
    Returns:
        List of simulation DataFrames
    """
    print("\n>>> Running simulations...")
    
    # Set networks to eval mode
    policy_net.eval()
    value_net.eval()
    
    # Load config
    sim_config = SimConfig(config)
    if T_sim is not None:
        sim_config.T_sim = T_sim
    
    # Create simulator
    simulator = RamseySimulator(policy_net, value_net, sim_config, device)
    
    # Generate starting points
    b_range = sim_config.b_max - sim_config.b_min
    b_starts = [
        sim_config.b_min + 0.1 * b_range,
        sim_config.b_min + 0.3 * b_range,
        sim_config.b_min + 0.5 * b_range
    ]
    g_start_idx = 0
    
    # Run simulations
    sim_results = []
    with torch.no_grad():
        for b0 in b_starts:
            df = simulator.simulate(b0, g_start_idx)
            sim_results.append(df)
    
    # Generate plots
    plot_time_series(sim_results, 
                     save_path='figures/simulation_time_series.png',
                     title=f'Simulations (T={sim_config.T_sim})')
    
    plot_state_space_paths(sim_results, sim_config,
                           save_path='figures/simulation_state_space.png',
                           title='State-Space Trajectories')
    
    print(">>> Simulations complete.")
    return sim_results


# Legacy compatibility
def run_and_plot_simulations(lam_govt, value_govt, config, device, T_sim=100):
    """Legacy wrapper for run_simulations."""
    return run_simulations(lam_govt, value_govt, config, device, T_sim)


def load_config(config_file: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Running simulation module standalone...")
    
    # Load config
    config = load_config('config.json')
    sim_config = SimConfig(config)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create networks
    from value_module import create_policy_network, create_value_network, Config
    cfg = Config(config)
    
    policy_net = create_policy_network(cfg).to(device)
    value_net = create_value_network(cfg).to(device)
    
    # Load trained weights
    model_num = sim_config.model_number
    policy_path = f'models/trained_policy_nn_{model_num}.pth'
    value_path = f'models/trained_value_nn_{model_num}.pth'
    
    try:
        policy_net.load_state_dict(torch.load(policy_path, map_location=device))
        value_net.load_state_dict(torch.load(value_path, map_location=device))
        print(f"Loaded models: {policy_path}, {value_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train models first using the main dashboard.")
        exit(1)
    
    # Run simulations
    run_simulations(policy_net, value_net, config, device)
