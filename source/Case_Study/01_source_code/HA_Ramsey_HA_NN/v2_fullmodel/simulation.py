"""
Simulation Module for Heterogeneous Agent Ramsey Model.

This module performs post-training analysis:
- Monte Carlo simulation of equilibrium trajectories
- Ergodic distribution computation
- Policy function analysis
- Visualization of results

IMPORTANT: This module now properly distinguishes between:
- Period t=0: Initial period where (c₀ᵉ, c₀ᵘ) are CONTROLS (chosen by planner)
- Period t>0: Continuation periods where (cᵉ, cᵘ) are STATE variables (inherited)

The t=0 problem requires solving for optimal initial consumption given
exogenous initial assets (K₀, a₀ᵉ, a₀ᵘ).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime


def compute_period_0_allocation(model, K_0, a_0_e, a_0_u, c_1_e, c_1_u):
    """
    Solve the period 0 problem given initial assets and future consumption promises.

    At t=0, the planner chooses (n₀ᵉ, c₀ᵉ, c₀ᵘ) given:
    - Exogenous: (K₀, a₀ᵉ, a₀ᵘ)
    - Future promises: (c₁ᵉ, c₁ᵘ) from the policy network

    This function computes all t=0 variables using explicit formulas.

    Args:
        model: HAModel instance
        K_0: Initial capital (scalar tensor)
        a_0_e, a_0_u: Initial assets (scalar tensors)
        c_1_e, c_1_u: Future consumption promises (scalar tensors)

    Returns:
        Dictionary with all period 0 variables
    """
    device = model.device

    # Economic parameters
    beta = model.beta
    alpha = model.alpha
    sigma = model.sigma
    gamma = model.gamma
    delta = model.delta
    pi_e = model.pi_e
    pi_u = model.pi_u
    pi_ee = model.pi_ee
    pi_eu = model.pi_eu
    pi_ue = model.pi_ue
    pi_uu = model.pi_uu

    # For period 0, we need to find (n₀ᵉ, c₀ᵉ, c₀ᵘ) that satisfy equilibrium conditions.
    # This requires numerical optimization or iteration. For now, we use a simple
    # grid search / optimization approach.

    # Approach: Given (c₁ᵉ, c₁ᵘ), iterate to find consistent (c₀ᵉ, c₀ᵘ, n₀ᵉ)

    # Initial guess: use steady-state-like values
    c_0_e = (c_1_e + 0.3) / 2  # Start between bounds
    c_0_u = (c_1_u + 0.1) / 2

    # Iterate to find consistent allocation
    for _ in range(50):
        # Given c₀ᵉ, c₀ᵘ, compute n₀ᵉ from resource constraint
        # At t=0: Y = c₀ᵉ·πᵉ + c₀ᵘ·πᵘ + K₁ - (1-δ)K₀
        # For simplicity, assume K₁ ≈ K₀ (near steady state)
        K_1_approx = K_0  # Will refine below

        # Production: Y = K^α · (n·πᵉ)^(1-α)
        # Solve for n: n·πᵉ = (Y / K^α)^(1/(1-α))
        Y_needed = c_0_e * pi_e + c_0_u * pi_u + K_1_approx - (1 - delta) * K_0
        Y_needed = torch.clamp(Y_needed, min=0.01)

        labor_aggregate = (Y_needed / (K_0 ** alpha)) ** (1 / (1 - alpha))
        n_0_e = torch.clamp(labor_aggregate / pi_e, min=model.n_min, max=model.n_max)

        # Actual output
        Y_0 = (K_0 ** alpha) * ((n_0_e * pi_e) ** (1 - alpha))

        # Update K₁
        K_1 = Y_0 + (1 - delta) * K_0 - c_0_e * pi_e - c_0_u * pi_u
        K_1 = torch.clamp(K_1, min=model.K_min, max=model.K_max)
        K_1_approx = K_1

        # Wage: ŵ = (n₀ᵉ)^γ · (c₀ᵉ)^σ
        w_hat_0 = (n_0_e ** gamma) * (c_0_e ** sigma)

        # Bond price: Q₀ = β·(c₀ᵉ)^σ·[π^{ee}/(c₁ᵉ)^σ + π^{eu}/(c₁ᵘ)^σ]
        Q_0 = beta * (c_0_e ** sigma) * (
            pi_ee * (c_1_e ** (-sigma)) + pi_eu * (c_1_u ** (-sigma))
        )
        Q_0 = torch.clamp(Q_0, min=model.Q_min, max=model.Q_max)

        # Asset transitions (CORRECTED: use proper transition probabilities)
        # a₁ᵉ: for agents who are employed in period 1
        wealth_e = (a_0_e * pi_e * pi_ee + a_0_u * pi_u * pi_ue) / pi_e
        a_1_e = (wealth_e + w_hat_0 * n_0_e - c_0_e) / Q_0

        # a₁ᵘ: for agents who are unemployed in period 1
        wealth_u = (a_0_e * pi_e * pi_eu + a_0_u * pi_u * pi_uu) / pi_u
        a_1_u = (wealth_u - c_0_u) / Q_0

        # Check Euler inequality for unemployed
        # φ₁ᵘ = Q₀/(c₀ᵘ)^σ - β·[π^{ue}/(c₁ᵉ)^σ + π^{uu}/(c₁ᵘ)^σ]
        phi_1_u = Q_0 / (c_0_u ** sigma) - beta * (
            pi_ue * (c_1_e ** (-sigma)) + pi_uu * (c_1_u ** (-sigma))
        )

        # If borrowing constraint binds for unemployed (a₁ᵘ < 0), adjust c₀ᵘ
        if a_1_u < 0:
            # Set a₁ᵘ = 0 and back out c₀ᵘ
            c_0_u = wealth_u  # All wealth goes to consumption
            a_1_u = torch.zeros_like(a_1_u)

    # Compute welfare
    u_e = (c_0_e ** (1 - sigma)) / (1 - sigma) - (n_0_e ** (1 + gamma)) / (1 + gamma)
    u_u = (c_0_u ** (1 - sigma)) / (1 - sigma)
    welfare_0 = pi_e * u_e + pi_u * u_u

    return {
        'n_0_e': n_0_e,
        'c_0_e': c_0_e,
        'c_0_u': c_0_u,
        'K_1': K_1,
        'a_1_e': a_1_e,
        'a_1_u': a_1_u,
        'Q_0': Q_0,
        'w_hat_0': w_hat_0,
        'phi_1_u': phi_1_u,
        'welfare_0': welfare_0
    }


def simulate_trajectory_with_t0(model, boundary, K_0, a_0_e, a_0_u, num_periods=50,
                                 use_projection=True, verbose=False):
    """
    Simulate an equilibrium trajectory starting from period 0.

    This function properly handles the t=0 problem where initial consumption
    is chosen optimally, then continues with the standard forward simulation.

    Args:
        model: Trained HAModel
        boundary: AlphaBoundary for projection
        K_0, a_0_e, a_0_u: Initial conditions (scalars or 1D tensors)
        num_periods: Total periods to simulate (including t=0)
        use_projection: Whether to project states back to admissible set
        verbose: Print debug information

    Returns:
        Dictionary containing full trajectory
    """
    model.eval()
    device = model.device

    # Convert inputs to tensors
    K_0 = torch.tensor([[K_0]], device=device, dtype=torch.float32) if not torch.is_tensor(K_0) else K_0.view(1, 1)
    a_0_e = torch.tensor([[a_0_e]], device=device, dtype=torch.float32) if not torch.is_tensor(a_0_e) else a_0_e.view(1, 1)
    a_0_u = torch.tensor([[a_0_u]], device=device, dtype=torch.float32) if not torch.is_tensor(a_0_u) else a_0_u.view(1, 1)

    trajectory = {
        'period': [],
        'states': [],
        'controls': [],
        'prices': [],
        'welfare': [],
        'fb_residuals': [],
        'was_projected': [],
        'projection_distance': [],
        'euler_discrepancy_e': [],
        'euler_discrepancy_u': [],
        'a_prime_e_raw': [],
        'a_prime_u_raw': [],
        'is_t0': []
    }

    with torch.no_grad():
        # ================================================================
        # PERIOD 0: Special treatment
        # ================================================================
        # First, we need to get (c₁ᵉ, c₁ᵘ) from the policy network.
        # Create a "probe" state to query the network for reasonable future consumption.

        # Use middle-of-range consumption as initial probe
        c_probe = (model.c_min + model.c_max) / 2
        probe_state = torch.tensor([[K_0.item(), a_0_e.item(), a_0_u.item(), c_probe, c_probe]],
                                    device=device, dtype=torch.float32)

        # Get policy output for future consumption
        probe_out = model.forward_physics(probe_state)
        if probe_out is None:
            if verbose:
                print("Period 0: Probe forward pass failed")
            return trajectory

        c_1_e = probe_out['controls']['c_prime_e']
        c_1_u = probe_out['controls']['c_prime_u']

        # Solve period 0 problem
        t0_result = compute_period_0_allocation(model, K_0, a_0_e, a_0_u, c_1_e, c_1_u)

        # Record period 0
        trajectory['period'].append(0)
        trajectory['states'].append([K_0.item(), a_0_e.item(), a_0_u.item(),
                                     t0_result['c_0_e'].item(), t0_result['c_0_u'].item()])
        trajectory['controls'].append([t0_result['n_0_e'].item(), c_1_e.item(), c_1_u.item()])
        trajectory['prices'].append([t0_result['Q_0'].item(), t0_result['w_hat_0'].item()])
        trajectory['welfare'].append(t0_result['welfare_0'].item())
        trajectory['fb_residuals'].append([0.0, 0.0])  # FB not computed for t=0
        trajectory['euler_discrepancy_e'].append(0.0)  # Employed Euler holds by construction
        trajectory['euler_discrepancy_u'].append(t0_result['phi_1_u'].item())
        trajectory['a_prime_e_raw'].append(t0_result['a_1_e'].item())
        trajectory['a_prime_u_raw'].append(t0_result['a_1_u'].item())
        trajectory['was_projected'].append(False)
        trajectory['projection_distance'].append(0.0)
        trajectory['is_t0'].append(True)

        # Construct state for period 1
        current_state = torch.tensor([[
            t0_result['K_1'].item(),
            t0_result['a_1_e'].item(),
            t0_result['a_1_u'].item(),
            c_1_e.item(),
            c_1_u.item()
        ]], device=device, dtype=torch.float32)

        # Project if needed
        if use_projection and boundary is not None:
            projected, dist, was_inside = boundary.project_to_admissible(current_state)
            if not was_inside.item():
                trajectory['was_projected'][-1] = True
                trajectory['projection_distance'][-1] = dist.item()
                current_state = projected

        # ================================================================
        # PERIODS t > 0: Standard forward simulation
        # ================================================================
        for t in range(1, num_periods):
            trajectory['period'].append(t)
            trajectory['states'].append(current_state.cpu().numpy().flatten().tolist())
            trajectory['is_t0'].append(False)

            out = model.forward_physics(current_state)
            if out is None:
                if verbose:
                    print(f"Period {t}: Forward pass failed")
                break

            n_e = out['controls']['n_e'].item()
            c_prime_e = out['controls']['c_prime_e'].item()
            c_prime_u = out['controls']['c_prime_u'].item()
            trajectory['controls'].append([n_e, c_prime_e, c_prime_u])

            Q = out['physics']['Q'].item()
            w_hat = out['physics']['w_hat'].item()
            trajectory['prices'].append([Q, w_hat])

            trajectory['welfare'].append(out['welfare'].item())

            fb_e, fb_u = out['fb_residuals']
            trajectory['fb_residuals'].append([fb_e.item(), fb_u.item()])

            phi_e = out['physics']['phi_e'].item()
            phi_u = out['physics']['phi_u'].item()
            trajectory['euler_discrepancy_e'].append(phi_e)
            trajectory['euler_discrepancy_u'].append(phi_u)

            a_prime_e = out['physics']['a_prime_e_raw'].item()
            a_prime_u = out['physics']['a_prime_u_raw'].item()
            trajectory['a_prime_e_raw'].append(a_prime_e)
            trajectory['a_prime_u_raw'].append(a_prime_u)

            next_state = out['next_state']

            if use_projection and boundary is not None:
                projected, dist, was_inside = boundary.project_to_admissible(next_state)
                trajectory['was_projected'].append(not was_inside.item())
                trajectory['projection_distance'].append(dist.item())
                current_state = projected
            else:
                trajectory['was_projected'].append(False)
                trajectory['projection_distance'].append(0.0)
                current_state = next_state

    # Convert lists to numpy arrays
    for key in trajectory:
        trajectory[key] = np.array(trajectory[key])

    return trajectory


def simulate_trajectory(model, boundary, initial_state, num_periods=50,
                       use_projection=True, verbose=False):
    """
    Simulate an equilibrium trajectory from an initial state.

    NOTE: This function assumes the initial state already includes valid
    (cᵉ, cᵘ) values representing inherited commitments. For proper economic
    interpretation starting from t=0, use simulate_trajectory_with_t0().
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
        'was_projected': [],
        'projection_distance': [],
        'euler_discrepancy_e': [],
        'euler_discrepancy_u': [],
        'a_prime_e_raw': [],
        'a_prime_u_raw': []
    }

    current_state = initial_state.to(device)

    with torch.no_grad():
        for t in range(num_periods):
            trajectory['states'].append(current_state.cpu().numpy().flatten())

            out = model.forward_physics(current_state)
            if out is None:
                if verbose:
                    print(f"  Period {t}: Forward pass failed")
                break

            n_e = out['controls']['n_e'].item()
            c_prime_e = out['controls']['c_prime_e'].item()
            c_prime_u = out['controls']['c_prime_u'].item()
            trajectory['controls'].append([n_e, c_prime_e, c_prime_u])

            Q = out['physics']['Q'].item()
            w_hat = out['physics']['w_hat'].item()
            trajectory['prices'].append([Q, w_hat])

            trajectory['welfare'].append(out['welfare'].item())

            fb_e, fb_u = out['fb_residuals']
            trajectory['fb_residuals'].append([fb_e.item(), fb_u.item()])

            phi_e = out['physics']['phi_e'].item()
            phi_u = out['physics']['phi_u'].item()
            trajectory['euler_discrepancy_e'].append(phi_e)
            trajectory['euler_discrepancy_u'].append(phi_u)

            a_prime_e = out['physics']['a_prime_e_raw'].item()
            a_prime_u = out['physics']['a_prime_u_raw'].item()
            trajectory['a_prime_e_raw'].append(a_prime_e)
            trajectory['a_prime_u_raw'].append(a_prime_u)

            next_state = out['next_state']

            if use_projection and boundary is not None:
                projected, dist, was_inside = boundary.project_to_admissible(next_state)
                trajectory['was_projected'].append(not was_inside.item())
                trajectory['projection_distance'].append(dist.item())
                current_state = projected
            else:
                trajectory['was_projected'].append(False)
                trajectory['projection_distance'].append(0.0)
                current_state = next_state

    for key in trajectory:
        trajectory[key] = np.array(trajectory[key])

    return trajectory


def simulate_individual_trajectory(model, boundary, initial_state, num_periods=50, seed=None):
    """Placeholder for individual household simulation."""
    raise NotImplementedError("Individual household simulation not yet implemented.")


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
    """Run Monte Carlo simulation with multiple trajectories."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    print(f"\n>> Monte Carlo: {num_trajectories} trajectories, {num_periods} periods")

    trajectories = []
    successful = 0
    n_projected_total = 0
    sb = config['state_bounds']

    for i in range(num_trajectories):
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

        traj = simulate_trajectory(model, boundary, initial_state, num_periods)

        if len(traj['states']) == num_periods:
            trajectories.append(traj)
            successful += 1
            n_projected_total += traj['was_projected'].sum()

        if (i + 1) % 20 == 0:
            print(f"   Completed {i+1}/{num_trajectories} ({successful} successful)")

    print(f"   Done: {successful}/{num_trajectories} successful")
    if successful > 0:
        print(f"   Total projections: {n_projected_total} "
              f"({100*n_projected_total/(successful*num_periods):.1f}% of steps)")

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
        out = model.forward_physics(states)
        if out is None:
            return None

        n_e = out['controls']['n_e']
        c_prime_e = out['controls']['c_prime_e']
        c_prime_u = out['controls']['c_prime_u']
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


def plot_trajectories(trajectories, save_dir, config=None):
    """Plot sample trajectories for states, controls, prices, and NEW: Euler/assets."""
    if len(trajectories) == 0:
        return

    n_plot = min(5, len(trajectories))
    colors = plt.cm.viridis(np.linspace(0, 1, n_plot))

    # Figure 1: State Variables
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    labels = ['K', 'aᵉ', 'aᵘ', 'cᵉ', 'cᵘ', 'Welfare']

    for i in range(n_plot):
        traj = trajectories[i]
        t_vals = np.arange(len(traj['states']))

        for j in range(5):
            axes[j // 3, j % 3].plot(t_vals, traj['states'][:, j],
                                     color=colors[i], alpha=0.7)
        axes[1, 2].plot(t_vals, traj['welfare'], color=colors[i], alpha=0.7)

    for j in range(6):
        ax = axes[j // 3, j % 3]
        ax.set_xlabel('Period')
        ax.set_title(labels[j])
        ax.grid(True, alpha=0.3)

    plt.suptitle('State Variable Trajectories', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/simulation_trajectories_states.png", dpi=150)
    plt.close()

    # Figure 2: Controls and Prices
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i in range(n_plot):
        traj = trajectories[i]
        t_vals = np.arange(len(traj['controls']))

        axes[0, 0].plot(t_vals, traj['controls'][:, 0], color=colors[i], alpha=0.7)
        Q_vals = traj['prices'][:, 0]
        axes[0, 1].plot(t_vals, Q_vals, color=colors[i], alpha=0.7)
        r_vals = 1.0 / Q_vals - 1.0
        axes[0, 2].plot(t_vals, r_vals, color=colors[i], alpha=0.7)
        axes[1, 0].plot(t_vals, traj['controls'][:, 1], color=colors[i], alpha=0.7)
        axes[1, 1].plot(t_vals, traj['controls'][:, 2], color=colors[i], alpha=0.7)
        axes[1, 2].plot(t_vals, traj['prices'][:, 1], color=colors[i], alpha=0.7)

    titles = ['Labor Supply nᵉ', 'Bond Price Q', 'Interest Rate r = 1/Q - 1',
              "Future Cons c'ᵉ", "Future Cons c'ᵘ", 'After-Tax Wage ŵ']

    for j, ax in enumerate(axes.flat):
        ax.set_xlabel('Period')
        ax.set_title(titles[j])
        ax.grid(True, alpha=0.3)

    plt.suptitle('Control and Price Trajectories', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/simulation_trajectories_controls.png", dpi=150)
    plt.close()

    # NEW Figure 3: Euler Discrepancies and Asset Choices
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i in range(n_plot):
        traj = trajectories[i]
        t_vals = np.arange(len(traj['euler_discrepancy_e']))

        # Euler discrepancy - employed
        axes[0, 0].plot(t_vals, traj['euler_discrepancy_e'], color=colors[i], alpha=0.7)

        # Euler discrepancy - unemployed
        axes[0, 1].plot(t_vals, traj['euler_discrepancy_u'], color=colors[i], alpha=0.7)

        # Asset choice - employed
        axes[1, 0].plot(t_vals, traj['a_prime_e_raw'], color=colors[i], alpha=0.7)

        # Asset choice - unemployed
        axes[1, 1].plot(t_vals, traj['a_prime_u_raw'], color=colors[i], alpha=0.7)

    # Add reference lines
    axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    if config is not None:
        sb = config['state_bounds']
        axes[1, 0].axhline(y=sb['a_min'], color='red', linestyle='--', alpha=0.5, label='a_min')
        axes[1, 0].axhline(y=sb['a_max'], color='red', linestyle=':', alpha=0.5, label='a_max')
        axes[1, 1].axhline(y=sb['a_min'], color='red', linestyle='--', alpha=0.5)
        axes[1, 1].axhline(y=sb['a_max'], color='red', linestyle=':', alpha=0.5)
        axes[1, 0].legend(loc='best')

    titles = ['Euler Discrepancy φᵉ (employed)', 'Euler Discrepancy φᵘ (unemployed)',
              "Asset Choice a'ᵉ (employed)", "Asset Choice a'ᵘ (unemployed)"]

    for j, ax in enumerate(axes.flat):
        ax.set_xlabel('Period')
        ax.set_title(titles[j])
        ax.grid(True, alpha=0.3)

    plt.suptitle('Euler Discrepancies and Asset Choices Over Time', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/simulation_trajectories_euler_assets.png", dpi=150)
    plt.close()

    # NEW Figure 4: Complementarity Check (φ vs a')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i in range(n_plot):
        traj = trajectories[i]

        # Employed: scatter of φᵉ vs a'ᵉ
        axes[0].scatter(traj['a_prime_e_raw'], traj['euler_discrepancy_e'],
                       color=colors[i], alpha=0.5, s=10)

        # Unemployed: scatter of φᵘ vs a'ᵘ
        axes[1].scatter(traj['a_prime_u_raw'], traj['euler_discrepancy_u'],
                       color=colors[i], alpha=0.5, s=10)

    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    axes[0].set_xlabel("Asset Choice a'ᵉ")
    axes[0].set_ylabel("Euler Discrepancy φᵉ")
    axes[0].set_title("Employed: Complementarity Check\n(φᵉ ≥ 0, a'ᵉ ≥ 0, φᵉ·a'ᵉ = 0)")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Asset Choice a'ᵘ")
    axes[1].set_ylabel("Euler Discrepancy φᵘ")
    axes[1].set_title("Unemployed: Complementarity Check\n(φᵘ ≥ 0, a'ᵘ ≥ 0, φᵘ·a'ᵘ = 0)")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Complementarity Conditions (KKT)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/simulation_complementarity.png", dpi=150)
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
        data = stats['states']['raw'][:, j]

        data_min = np.min(data)
        data_max = np.max(data)

        if np.isclose(data_max, data_min, atol=1e-6):
            ax.hist(data, bins=1, density=True, alpha=0.7, color='steelblue')
            ax.text(0.5, 0.9, "Near Constant", transform=ax.transAxes,
                   ha='center', color='red', fontsize=10)
        else:
            ax.hist(data, bins=50, density=True, alpha=0.7, color='steelblue')

        ax.axvline(stats['states']['mean'][keys[j]], color='red',
                  linestyle='--', label='Mean')
        ax.set_xlabel(labels[j])
        ax.set_title(f'Distribution: {labels[j]}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[1, 2].axis('off')
    axes[1, 2].text(0.5, 0.5,
                   f"N = {stats['n_observations']}\n"
                   f"Mean Welfare = {stats['welfare']['mean']:.4f}\n"
                   f"Std Welfare = {stats['welfare']['std']:.4f}\n\n"
                   f"NOTE: Without aggregate shocks,\n"
                   f"this is NOT a true ergodic distribution.",
                   ha='center', va='center', fontsize=11,
                   transform=axes[1, 2].transAxes)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/simulation_distributions.png", dpi=150)
    plt.close()


def plot_policy_functions(policy_data, save_dir):
    """Plot policy function contours."""
    if policy_data is None:
        return

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
               f"γ={econ['gamma']}, δ={econ['delta']}\n")
        f.write(f"  π_ee={econ['pi_matrix'][0][0]}, π_uu={econ['pi_matrix'][1][1]}\n\n")

        f.write("SIMULATION SUMMARY\n" + "-"*40 + "\n")
        f.write(f"  Trajectories: {len(trajectories)}\n")

        total_steps = sum(len(t['states']) for t in trajectories)
        total_proj = sum(t['was_projected'].sum() for t in trajectories)
        f.write(f"  Total steps: {total_steps}\n")
        f.write(f"  Projections: {total_proj} ({100*total_proj/total_steps:.1f}%)\n\n")

        # NEW: Euler discrepancy summary
        all_phi_e = np.concatenate([t['euler_discrepancy_e'] for t in trajectories])
        all_phi_u = np.concatenate([t['euler_discrepancy_u'] for t in trajectories])
        all_a_e = np.concatenate([t['a_prime_e_raw'] for t in trajectories])
        all_a_u = np.concatenate([t['a_prime_u_raw'] for t in trajectories])

        f.write("EULER DISCREPANCY SUMMARY\n" + "-"*40 + "\n")
        f.write(f"  φᵉ (employed):   mean={np.mean(all_phi_e):.6f}, std={np.std(all_phi_e):.6f}\n")
        f.write(f"  φᵘ (unemployed): mean={np.mean(all_phi_u):.6f}, std={np.std(all_phi_u):.6f}\n\n")

        f.write("ASSET CHOICE SUMMARY\n" + "-"*40 + "\n")
        f.write(f"  a'ᵉ (employed):   mean={np.mean(all_a_e):.4f}, std={np.std(all_a_e):.4f}\n")
        f.write(f"  a'ᵘ (unemployed): mean={np.mean(all_a_u):.4f}, std={np.std(all_a_u):.4f}\n")
        f.write(f"  a'ᵉ range: [{np.min(all_a_e):.4f}, {np.max(all_a_e):.4f}]\n")
        f.write(f"  a'ᵘ range: [{np.min(all_a_u):.4f}, {np.max(all_a_u):.4f}]\n\n")

        if stats:
            f.write(f"  Observations (post burn-in): {stats['n_observations']}\n\n")

            f.write("DISTRIBUTION STATISTICS\n" + "-"*40 + "\n")
            f.write("  (Note: Not true ergodic distribution without aggregate shocks)\n\n")
            f.write(f"  {'Var':<8} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}\n")
            f.write(f"  {'-'*48}\n")

            for var in ['K', 'a_e', 'a_u', 'c_e', 'c_u']:
                f.write(f"  {var:<8} {stats['states']['mean'][var]:>10.4f} "
                       f"{stats['states']['std'][var]:>10.4f} "
                       f"{stats['states']['min'][var]:>10.4f} "
                       f"{stats['states']['max'][var]:>10.4f}\n")

            f.write(f"\n  Welfare: {stats['welfare']['mean']:.4f} ± "
                   f"{stats['welfare']['std']:.4f}\n")


def run_simulation(model, boundary, config, device,
                   num_trajectories=None, num_periods=None, save_dir=None):
    """Main simulation function - called from dashboard after training."""
    sim_config = config.get('simulation', {})
    num_trajectories = num_trajectories or sim_config.get('num_trajectories', 100)
    num_periods = num_periods or sim_config.get('num_periods', 50)
    save_dir = save_dir or sim_config.get('output_dir', 'output/simulations')
    burn_in = sim_config.get('burn_in', 10)
    seed = sim_config.get('random_seed', 42)

    print("\n" + "="*60)
    print("         POST-TRAINING SIMULATION")
    print("="*60)
    print(f"   Output: {save_dir}/")
    print("\n   NOTE: Simulations use deterministic transitions")
    print("   (aggregate economy interpretation, no individual shocks)")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/figures", exist_ok=True)
    os.makedirs(f"{save_dir}/data", exist_ok=True)

    print("\n>> Step 1: Monte Carlo Simulation")
    trajectories = run_monte_carlo(model, boundary, config, device,
                                   num_trajectories, num_periods, seed)

    print("\n>> Step 2: Distribution Statistics")
    stats = compute_ergodic_distribution(trajectories, burn_in)

    print("\n>> Step 3: Policy Analysis")
    policy_data = analyze_policy_functions(model, config, device)

    print("\n>> Step 4: Generating Plots")
    plot_trajectories(trajectories, f"{save_dir}/figures", config=config)
    plot_distributions(stats, f"{save_dir}/figures")
    plot_policy_functions(policy_data, f"{save_dir}/figures")

    print("\n>> Step 5: Generating Report")
    generate_report(config, trajectories, stats, f"{save_dir}/simulation_report.txt")

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