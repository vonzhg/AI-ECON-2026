"""
Simulation Module for Heterogeneous Agent Ramsey Model.

This module provides functionality for:
1. Simulating equilibrium trajectories
2. Monte Carlo analysis
3. Ergodic distribution computation
4. Policy function analysis
5. Visualization and reporting

Called from dashboard.py after training completes.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime


def simulate_trajectory(model, initial_state, num_periods=50, verbose=False):
    """
    Simulate an equilibrium trajectory from an initial state.
    
    Args:
        model: Trained HAModel instance
        initial_state: torch.Tensor of shape (5,) or (1, 5) with (K, a^e, a^u, c^e, c^u)
        num_periods: Number of periods to simulate
        verbose: Print progress
        
    Returns:
        Dictionary containing trajectory data
    """
    model.eval()
    device = model.device
    
    if initial_state.dim() == 1:
        initial_state = initial_state.unsqueeze(0)
    
    trajectory = {
        'states': [],
        'controls': [],
        'prices': [],
        'welfare': [],
        'fb_residuals': [],
        'admissibility': []
    }
    
    current_state = initial_state.to(device)
    
    with torch.no_grad():
        for t in range(num_periods):
            trajectory['states'].append(current_state.cpu().numpy().flatten())
            
            out = model.forward_physics(current_state)
            
            if out is None:
                if verbose:
                    print(f"Simulation terminated at period {t} due to numerical issues")
                break
            
            # Extract controls
            raw_out = model.actor(current_state)
            n_e = (torch.sigmoid(raw_out[:, 0:1]) * (model.n_max - model.n_min) + model.n_min)
            c_prime_e = torch.exp(raw_out[:, 1:2]) * model.c_scale
            c_prime_u = torch.exp(raw_out[:, 2:3]) * model.c_scale
            
            trajectory['controls'].append([n_e.item(), c_prime_e.item(), c_prime_u.item()])
            
            # Prices
            Q = out['physics']['Q']
            c_e = current_state[:, 3:4]
            w_hat = (n_e ** model.gamma) * (c_e ** model.sigma)
            trajectory['prices'].append([Q.item(), w_hat.item()])
            
            # Welfare and residuals
            trajectory['welfare'].append(out['welfare'].item())
            fb_e, fb_u = out['fb_residuals']
            trajectory['fb_residuals'].append([fb_e.item(), fb_u.item()])
            
            # Admissibility
            adm_score = model.compute_admissibility(out['physics'])
            trajectory['admissibility'].append(adm_score.mean().item())
            
            current_state = out['next_state']
    
    for key in trajectory:
        trajectory[key] = np.array(trajectory[key])
    
    return trajectory


def generate_random_states(num, config, device):
    """Generate random states within bounds."""
    sb = config['state_bounds']
    K = torch.rand(num, 1, device=device) * (sb['K_max'] - sb['K_min']) + sb['K_min']
    ae = torch.rand(num, 1, device=device) * (sb['a_max'] - sb['a_min']) + sb['a_min']
    au = torch.rand(num, 1, device=device) * (sb['a_max'] - sb['a_min']) + sb['a_min']
    ce = torch.rand(num, 1, device=device) * (sb['c_max'] - sb['c_min']) + sb['c_min']
    cu = torch.rand(num, 1, device=device) * (sb['c_max'] - sb['c_min']) + sb['c_min']
    return torch.cat([K, ae, au, ce, cu], dim=1)


def run_monte_carlo(model, boundary, config, device, 
                    num_trajectories=100, num_periods=50, seed=42):
    """
    Run Monte Carlo simulation with multiple trajectories.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    print(f"\n>> Monte Carlo: {num_trajectories} trajectories, {num_periods} periods")
    
    trajectories = []
    successful = 0
    sb = config['state_bounds']
    
    for i in range(num_trajectories):
        # Sample initial state from learned boundary
        if boundary.admissible_points is not None and len(boundary.admissible_points) > 10:
            idx = np.random.randint(len(boundary.admissible_points))
            base_point = boundary.admissible_points[idx]
            
            initial_state = torch.tensor([
                base_point[0], base_point[1], base_point[2],
                np.random.uniform(sb['c_min'], sb['c_max']),
                np.random.uniform(sb['c_min'], sb['c_max'])
            ], dtype=torch.float32, device=device)
        else:
            initial_state = generate_random_states(1, config, device).squeeze(0)
        
        traj = simulate_trajectory(model, initial_state, num_periods)
        
        if len(traj['states']) == num_periods:
            trajectories.append(traj)
            successful += 1
        
        if (i + 1) % 20 == 0:
            print(f"   Completed {i+1}/{num_trajectories} ({successful} successful)")
    
    print(f"   Done: {successful}/{num_trajectories} successful")
    return trajectories


def compute_ergodic_distribution(trajectories, burn_in=10):
    """Compute ergodic distribution statistics."""
    all_states, all_controls, all_welfare = [], [], []
    
    for traj in trajectories:
        if len(traj['states']) > burn_in:
            all_states.append(traj['states'][burn_in:])
            all_controls.append(traj['controls'][burn_in:])
            all_welfare.append(traj['welfare'][burn_in:])
    
    if len(all_states) == 0:
        return None
    
    all_states = np.vstack(all_states)
    all_controls = np.vstack(all_controls)
    all_welfare = np.concatenate(all_welfare)
    
    state_names = ['K', 'a_e', 'a_u', 'c_e', 'c_u']
    control_names = ['n_e', 'c_prime_e', 'c_prime_u']
    
    return {
        'states': {
            'mean': dict(zip(state_names, np.mean(all_states, axis=0))),
            'std': dict(zip(state_names, np.std(all_states, axis=0))),
            'min': dict(zip(state_names, np.min(all_states, axis=0))),
            'max': dict(zip(state_names, np.max(all_states, axis=0))),
            'raw': all_states
        },
        'controls': {
            'mean': dict(zip(control_names, np.mean(all_controls, axis=0))),
            'std': dict(zip(control_names, np.std(all_controls, axis=0))),
        },
        'welfare': {
            'mean': np.mean(all_welfare),
            'std': np.std(all_welfare),
        },
        'n_observations': len(all_welfare)
    }


def analyze_policy_functions(model, config, device, grid_resolution=30):
    """Analyze learned policy functions over a grid."""
    model.eval()
    sb = config['state_bounds']
    
    mean_K = (sb['K_max'] + sb['K_min']) / 2
    mean_c = (sb['c_max'] + sb['c_min']) / 2
    
    a_vals = np.linspace(sb['a_min'], sb['a_max'], grid_resolution)
    ae_grid, au_grid = np.meshgrid(a_vals, a_vals)
    n_points = grid_resolution ** 2
    
    states = torch.zeros(n_points, 5, device=device)
    states[:, 0] = mean_K
    states[:, 1] = torch.tensor(ae_grid.flatten())
    states[:, 2] = torch.tensor(au_grid.flatten())
    states[:, 3] = mean_c
    states[:, 4] = mean_c
    
    with torch.no_grad():
        raw_out = model.actor(states)
        n_e = (torch.sigmoid(raw_out[:, 0]) * (model.n_max - model.n_min) + model.n_min)
        c_prime_e = torch.exp(raw_out[:, 1]) * model.c_scale
        c_prime_u = torch.exp(raw_out[:, 2]) * model.c_scale
        values = model.critic(states)
    
    return {
        'ae_grid': ae_grid,
        'au_grid': au_grid,
        'n_e': n_e.cpu().numpy().reshape(grid_resolution, grid_resolution),
        'c_prime_e': c_prime_e.cpu().numpy().reshape(grid_resolution, grid_resolution),
        'c_prime_u': c_prime_u.cpu().numpy().reshape(grid_resolution, grid_resolution),
        'value': values.cpu().numpy().reshape(grid_resolution, grid_resolution),
        'fixed_K': mean_K,
        'fixed_c': mean_c
    }


def plot_trajectories(trajectories, save_dir):
    """Plot sample trajectories."""
    if len(trajectories) == 0:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    n_plot = min(5, len(trajectories))
    colors = plt.cm.viridis(np.linspace(0, 1, n_plot))
    
    labels = ['K', 'aᵉ', 'aᵘ', 'cᵉ', 'cᵘ', 'Welfare']
    
    for i in range(n_plot):
        traj = trajectories[i]
        t_vals = np.arange(len(traj['states']))
        
        for j in range(5):
            axes[j // 3, j % 3].plot(t_vals, traj['states'][:, j], color=colors[i], alpha=0.7)
        axes[1, 2].plot(t_vals, traj['welfare'], color=colors[i], alpha=0.7)
    
    for j in range(6):
        ax = axes[j // 3, j % 3]
        ax.set_xlabel('Period')
        ax.set_title(labels[j])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/simulation_trajectories.png", dpi=150)
    plt.close()


def plot_distributions(stats, save_dir):
    """Plot ergodic distributions."""
    if stats is None:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    labels = ['K', 'aᵉ', 'aᵘ', 'cᵉ', 'cᵘ']
    keys = ['K', 'a_e', 'a_u', 'c_e', 'c_u']
    
    for j in range(5):
        ax = axes[j // 3, j % 3]
        ax.hist(stats['states']['raw'][:, j], bins=50, density=True, alpha=0.7, color='steelblue')
        ax.axvline(stats['states']['mean'][keys[j]], color='red', linestyle='--', label='Mean')
        ax.set_xlabel(labels[j])
        ax.set_title(f'Distribution: {labels[j]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[1, 2].axis('off')
    axes[1, 2].text(0.5, 0.5, 
                   f"N = {stats['n_observations']}\n"
                   f"Mean Welfare = {stats['welfare']['mean']:.4f}\n"
                   f"Std Welfare = {stats['welfare']['std']:.4f}",
                   ha='center', va='center', fontsize=12, transform=axes[1, 2].transAxes)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/simulation_distributions.png", dpi=150)
    plt.close()


def plot_policy_functions(policy_data, save_dir):
    """Plot policy function contours."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    data = [
        ('n_e', 'Labor Supply nᵉ', 'viridis'),
        ('c_prime_e', "Future Cons c'ᵉ", 'plasma'),
        ('c_prime_u', "Future Cons c'ᵘ", 'plasma'),
        ('value', 'Value Function V(s)', 'RdYlGn')
    ]
    
    for idx, (key, title, cmap) in enumerate(data):
        ax = axes[idx // 2, idx % 2]
        im = ax.contourf(policy_data['ae_grid'], policy_data['au_grid'], 
                        policy_data[key], levels=20, cmap=cmap)
        ax.set_xlabel('aᵉ')
        ax.set_ylabel('aᵘ')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    
    plt.suptitle(f"Policy at K={policy_data['fixed_K']:.2f}, c={policy_data['fixed_c']:.2f}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/policy_functions.png", dpi=150)
    plt.close()


def generate_report(config, trajectories, stats, save_path):
    """Generate simulation report."""
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("  HETEROGENEOUS AGENT RAMSEY MODEL - SIMULATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        econ = config['economic_parameters']
        f.write("MODEL PARAMETERS\n" + "-"*40 + "\n")
        f.write(f"  β={econ['beta']}, α={econ['alpha']}, σ={econ['sigma']}, "
               f"γ={econ['gamma']}, δ={econ['delta']}\n\n")
        
        f.write("SIMULATION SUMMARY\n" + "-"*40 + "\n")
        f.write(f"  Trajectories: {len(trajectories)}\n")
        
        if stats:
            f.write(f"  Observations: {stats['n_observations']}\n\n")
            f.write("ERGODIC DISTRIBUTION\n" + "-"*40 + "\n")
            f.write(f"  {'Var':<8} {'Mean':>10} {'Std':>10}\n")
            for var in ['K', 'a_e', 'a_u', 'c_e', 'c_u']:
                f.write(f"  {var:<8} {stats['states']['mean'][var]:>10.4f} "
                       f"{stats['states']['std'][var]:>10.4f}\n")
            f.write(f"\n  Welfare: {stats['welfare']['mean']:.4f} ± {stats['welfare']['std']:.4f}\n")


def run_simulation(model, boundary, config, device, 
                   num_trajectories=None, num_periods=None, save_dir=None):
    """
    Main simulation function - called from dashboard after training.
    
    Parameters can be passed directly or read from config['simulation'].
    """
    # Read from config if not provided
    sim_config = config.get('simulation', {})
    num_trajectories = num_trajectories or sim_config.get('num_trajectories', 100)
    num_periods = num_periods or sim_config.get('num_periods', 50)
    save_dir = save_dir or sim_config.get('output_dir', 'results')
    burn_in = sim_config.get('burn_in', 10)
    seed = sim_config.get('random_seed', 42)
    
    print("\n" + "="*60)
    print("         POST-TRAINING SIMULATION")
    print("="*60)
    print(f"   Trajectories: {num_trajectories}")
    print(f"   Periods: {num_periods}")
    print(f"   Burn-in: {burn_in}")
    print(f"   Output: {save_dir}/")
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/figures", exist_ok=True)
    os.makedirs(f"{save_dir}/data", exist_ok=True)
    
    # 1. Monte Carlo
    print("\n>> Step 1: Monte Carlo Simulation")
    trajectories = run_monte_carlo(model, boundary, config, device,
                                   num_trajectories, num_periods, seed)
    
    # 2. Ergodic distribution
    print("\n>> Step 2: Ergodic Distribution")
    stats = compute_ergodic_distribution(trajectories, burn_in)
    if stats:
        print(f"   Mean Welfare: {stats['welfare']['mean']:.4f}")
        print(f"   Mean K: {stats['states']['mean']['K']:.4f}")
    
    # 3. Policy analysis
    print("\n>> Step 3: Policy Analysis")
    policy_data = analyze_policy_functions(model, config, device)
    
    # 4. Plots
    print("\n>> Step 4: Generating Plots")
    plot_trajectories(trajectories, f"{save_dir}/figures")
    plot_distributions(stats, f"{save_dir}/figures")
    plot_policy_functions(policy_data, f"{save_dir}/figures")
    
    # 5. Report
    print("\n>> Step 5: Generating Report")
    generate_report(config, trajectories, stats, f"{save_dir}/simulation_report.txt")
    
    # 6. Save data
    print("\n>> Step 6: Saving Data")
    with open(f"{save_dir}/data/trajectories.pkl", 'wb') as f:
        pickle.dump(trajectories, f)
    with open(f"{save_dir}/data/statistics.pkl", 'wb') as f:
        pickle.dump(stats, f)
    with open(f"{save_dir}/data/policy_data.pkl", 'wb') as f:
        pickle.dump(policy_data, f)
    
    print("\n" + "="*60)
    print(f"   Results saved to: {save_dir}/")
    print("="*60)
    
    return trajectories, stats, policy_data
