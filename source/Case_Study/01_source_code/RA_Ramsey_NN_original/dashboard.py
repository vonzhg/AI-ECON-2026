"""
Main Training Dashboard for Ramsey Optimal Taxation Neural Network Solver.

This is the main entry point for training the policy and value networks
that solve the Ramsey optimal taxation problem. The training proceeds in
two phases:

1. WARMUP PHASE: Uses uniform sampling over the state space. This builds
   an initial understanding of the problem and collects trajectory data.

2. ADAPTIVE PHASE: Uses importance sampling weighted by admissibility scores.
   This focuses computation on the economically feasible region, improving
   efficiency and solution quality.

Usage:
    python dashboard.py

Configuration:
    All parameters are loaded from config.json. Key settings include:
    - num_iter: Total training iterations
    - warmup_iterations: Iterations before switching to adaptive sampling
    - use_txt_pretraining: Whether to pre-train from C++ solution data

Output:
    - models/trained_policy_nn_*.pth: Trained policy network
    - models/trained_value_nn_*.pth: Trained value network  
    - figures/: Diagnostic plots and visualizations

Authors: Zhigang Feng
Version: 2.0 (Streamlined)
"""

import torch
import torch.nn as nn
from datetime import datetime
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Local modules
import value_module
import adaptive_sampling
import simulation
import pretrain


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(config_file: str = 'config.json') -> dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_file: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    with open(config_file, 'r') as f:
        return json.load(f)


# =============================================================================
# NETWORK FACTORIES
# =============================================================================

def create_networks(config: dict, device: torch.device):
    """
    Create policy and value neural networks.
    
    Args:
        config: Configuration dictionary
        device: PyTorch device
    
    Returns:
        Tuple of (policy_net, value_net)
    """
    cfg = value_module.Config(config)
    
    policy_net = value_module.create_policy_network(cfg).to(device)
    value_net = value_module.create_value_network(cfg).to(device)
    
    # Store mu_min for legacy compatibility
    policy_net.mu_min = cfg.mu_min
    
    return policy_net, value_net


# =============================================================================
# SAMPLING UTILITIES
# =============================================================================

def sample_from_history(history: list, num_samples: int, config: dict,
                        device: torch.device) -> tuple:
    """
    Sample from accumulated trajectory history (used during warmup).
    
    Takes states from previous simulations and adds small perturbations
    to encourage exploration near known feasible regions.
    
    Args:
        history: List of state tensors from previous iterations
        num_samples: Number of samples to generate
        config: Configuration dictionary
        device: PyTorch device
    
    Returns:
        Tuple of (samples, None) - second element is None for warmup phase
    """
    # Get bounds
    penalty = config.get('penalty_params', {})
    b_min = penalty.get('b_min', -0.5)
    b_max = penalty.get('b_max', 3.5)
    
    bounds = config.get('state_bounds', config)
    mu_min = bounds.get('mu_min', config.get('mu_min', 1.27))
    mu_max = bounds.get('mu_max', config.get('mu_max', 2.51))
    
    if len(history) == 0:
        # No history yet - sample uniformly
        print("  No history available, using uniform sampling")
        samples = torch.zeros((num_samples, 3), device=device)
        samples[:, 0] = torch.rand(num_samples, device=device) * (b_max - b_min) + b_min
        samples[:, 1] = torch.rand(num_samples, device=device) * (mu_max - mu_min) + mu_min
        samples[:, 2] = torch.randint(0, 2, (num_samples,), device=device).float()
        return samples, None
    
    # Concatenate all historical data
    all_data = torch.cat(history, dim=0)
    
    # Filter to valid range
    mask = ((all_data[:, 0] >= b_min) & (all_data[:, 0] <= b_max) &
            (all_data[:, 1] >= mu_min) & (all_data[:, 1] <= mu_max))
    filtered = all_data[mask]
    
    if len(filtered) == 0:
        return sample_from_history([], num_samples, config, device)
    
    # Sample with replacement if needed
    n_available = filtered.shape[0]
    if n_available < num_samples:
        indices = torch.randint(0, n_available, (num_samples,), device=device)
    else:
        indices = torch.randperm(n_available, device=device)[:num_samples]
    
    samples = filtered[indices].clone()
    
    # Add perturbation noise (5% of range)
    noise_b = torch.randn(num_samples, device=device) * (b_max - b_min) * 0.05
    noise_mu = torch.randn(num_samples, device=device) * (mu_max - mu_min) * 0.05
    
    samples[:, 0] = torch.clamp(samples[:, 0] + noise_b, b_min, b_max)
    samples[:, 1] = torch.clamp(samples[:, 1] + noise_mu, mu_min, mu_max)
    
    # Randomly resample some g indices
    resample_mask = torch.rand(num_samples, device=device) < 0.2
    new_g = torch.randint(0, 2, (num_samples,), device=device).float()
    samples[resample_mask, 2] = new_g[resample_mask]
    
    return samples, None


# =============================================================================
# TRAINING LOOP
# =============================================================================

def run_training(config_file: str = 'config.json'):
    """
    Execute the main training loop.
    
    This function orchestrates the entire training process:
    1. Load configuration and setup
    2. Optional pre-training from C++ data
    3. Initialize adaptive sampling system
    4. Run warmup iterations (uniform sampling)
    5. Run adaptive iterations (importance sampling)
    6. Generate final diagnostics and simulations
    
    Args:
        config_file: Path to configuration file
    """
    start_time = datetime.now()
    
    # =================================================================
    # SETUP
    # =================================================================
    print("=" * 60)
    print("RAMSEY OPTIMAL TAXATION - NEURAL NETWORK SOLVER")
    print("=" * 60)
    
    # Load configuration
    config = load_config(config_file)
    
    # Extract key parameters
    iter_cfg = config.get('training_iterations', config)
    num_iter = iter_cfg.get('num_iter', config.get('num_iter', 10))
    warmup_iterations = iter_cfg.get('warmup_iterations', config.get('warmup_iterations', 5))
    history_size = iter_cfg.get('history_size', config.get('history_size', 5))
    
    sampling_cfg = config.get('sampling', config)
    num_samples = sampling_cfg.get('num_samples_value', config.get('num_samples_value', 5000))
    
    adaptive_cfg = config.get('adaptive_sampling', {})
    cache_update_freq = adaptive_cfg.get('cache_update_frequency', 1)
    cache_grid_size = adaptive_cfg.get('cache_grid_size', 5000)
    plot_freq = adaptive_cfg.get('plot_admissibility_frequency', 5)
    verbose_cpp = adaptive_cfg.get('verbose_cpp_analysis', False)
    
    io_cfg = config.get('model_io', config)
    use_pretraining = io_cfg.get('use_txt_pretraining', config.get('use_txt_pretraining', False))
    model_input = io_cfg.get('model_number_input', config.get('model_number_input', 101))
    model_output = io_cfg.get('model_number_output', config.get('model_number_output', 102))
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # Create networks
    policy_net, value_net = create_networks(config, device)
    
    # =================================================================
    # PRE-TRAINING / MODEL LOADING
    # =================================================================
    
    if use_pretraining:
        print("\n" + "=" * 60)
        print("PRE-TRAINING FROM C++ DATA")
        print("=" * 60)
        
        pretrain.run_pretraining(config_file)
        
        # Load pre-trained weights
        value_net.load_state_dict(torch.load(pretrain.VALUE_MODEL_PATH, map_location=device))
        policy_net.load_state_dict(torch.load(pretrain.POLICY_MODEL_PATH, map_location=device))
        print("Pre-trained models loaded successfully.")
        
    else:
        print("\n" + "=" * 60)
        print("LOADING EXISTING MODELS")
        print("=" * 60)
        
        value_path = f'models/trained_value_nn_{model_input}.pth'
        policy_path = f'models/trained_policy_nn_{model_input}.pth'
        
        try:
            value_net.load_state_dict(torch.load(value_path, map_location=device))
            policy_net.load_state_dict(torch.load(policy_path, map_location=device))
            print(f"Loaded models from: {model_input}")
        except FileNotFoundError:
            print(f"Models {model_input} not found. Starting from random initialization.")
    
    # =================================================================
    # INITIALIZE ADAPTIVE SAMPLING
    # =================================================================
    print("\n" + "=" * 60)
    print("INITIALIZING ADAPTIVE SAMPLING")
    print("=" * 60)
    
    scorer, sampler, visualizer = adaptive_sampling.initialize_adaptive_sampling(
        policy_net=policy_net,
        config=config,
        device=device
    )
    sampler.set_phase('warmup')
    
    print(f"\nTraining plan:")
    print(f"  - Total iterations: {num_iter}")
    print(f"  - Warmup (uniform): {warmup_iterations}")
    print(f"  - Adaptive (weighted): {num_iter - warmup_iterations}")
    
    # =================================================================
    # TRAINING LOOP
    # =================================================================
    print("\n" + "=" * 60)
    print("TRAINING ITERATIONS")
    print("=" * 60)
    
    # Storage
    trajectory_history = []
    policy_losses = []
    value_losses = []
    
    cfg = value_module.Config(config)
    
    for i_iter in range(num_iter):
        print(f"\n{'=' * 60}")
        print(f"ITERATION {i_iter + 1}/{num_iter}")
        print(f"{'=' * 60}")
        
        # --- Phase Transition ---
        if i_iter == warmup_iterations:
            print("\n" + "!" * 60)
            print("TRANSITIONING TO ADAPTIVE SAMPLING PHASE")
            print("!" * 60)
            
            sampler.set_phase('adaptive', total_adaptive_iters=(num_iter - warmup_iterations))
            
            print("\nBuilding initial admissibility cache...")
            adaptive_sampling.update_cache_periodic(scorer, sampler, n_grid=cache_grid_size)
            
            visualizer.plot_admissibility_heatmap(
                n_grid=50,
                save_path=f'figures/admissibility_heatmap_iter_{i_iter}.png',
                overlay_cpp_data=True,
                verbose_cpp_analysis=verbose_cpp
            )
        
        # --- Sampling ---
        print(f"\nStep 1: Sampling (Phase: {sampler.phase.upper()})")
        
        if sampler.phase == 'warmup':
            domain_samples, inadmissible_samples = sample_from_history(
                trajectory_history, num_samples, config, device
            )
        else:
            domain_samples, inadmissible_samples = sampler.sample_batch(
                batch_size=num_samples,
                policy_net=policy_net
            )
        
        print(f"  Samples: {domain_samples.shape[0]}")
        print(f"  B range: [{domain_samples[:, 0].min():.3f}, {domain_samples[:, 0].max():.3f}]")
        print(f"  μ range: [{domain_samples[:, 1].min():.3f}, {domain_samples[:, 1].max():.3f}]")
        
        if inadmissible_samples is not None:
            print(f"  Inadmissible samples for penalty training: {inadmissible_samples.shape[0]}")
        
        # --- Setup Trainer ---
        policy_net_old = copy.deepcopy(policy_net)
        value_module.model_number = model_output
        
        trainer = value_module.PolicyValueTrainer(
            policy_net, value_net, cfg, device, save_models=True
        )
        
        # --- Policy Training ---
        print(f"\nStep 2: Training policy network")
        policy_net = trainer.train_policy(
            domain_samples,
            all_losses=policy_losses,
            inadmissible_samples=inadmissible_samples
        )
        
        # --- Value Training ---
        print(f"\nStep 3: Training value network")
        value_net = trainer.train_value(
            domain_samples,
            all_losses=value_losses,
            inadmissible_samples=inadmissible_samples
        )
        
        # --- Collect Trajectory Data ---
        # Run simulation to get trajectory data for next iteration
        simulator = value_module.ValueSimulator(
            value_net, policy_net, policy_net_old, cfg, device
        )
        simulator.b_min = torch.tensor(scorer.b_min, device=device)
        simulator.b_max = torch.tensor(scorer.b_max, device=device)
        
        _, domain_data, _, _ = simulator.simulate_value(
            domain_samples, cfg.n_v_sim, apply_penalties=True
        )
        
        states = domain_data[:, :3]
        trajectory_history.append(states)
        
        if len(trajectory_history) > history_size:
            trajectory_history = trajectory_history[-history_size:]
        
        print(f"  Collected {states.shape[0]} trajectory states")
        print(f"  History: {len(trajectory_history)}/{history_size} batches")
        
        # --- Update Admissibility Cache ---
        if sampler.phase == 'adaptive':
            if (i_iter - warmup_iterations) % cache_update_freq == 0:
                print(f"\nStep 4: Updating admissibility cache")
                adaptive_sampling.update_cache_periodic(scorer, sampler, n_grid=cache_grid_size)
            
            # Periodic diagnostics
            if (i_iter - warmup_iterations) % plot_freq == 0:
                print(f"\nStep 5: Generating diagnostic plots")
                visualizer.plot_admissibility_heatmap(
                    n_grid=50,
                    save_path=f'figures/admissibility_heatmap_iter_{i_iter}.png',
                    overlay_cpp_data=True,
                    verbose_cpp_analysis=verbose_cpp
                )
                visualizer.plot_cache_distribution(
                    save_path=f'figures/cache_distribution_iter_{i_iter}.png'
                )
                sampler.plot_sampling_statistics()
        
        sampler.increment_iteration()
    
    # =================================================================
    # POST-TRAINING ANALYSIS
    # =================================================================
    print("\n" + "=" * 60)
    print("POST-TRAINING ANALYSIS")
    print("=" * 60)
    
    # Plot losses
    print("\n>>> Plotting training losses...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    if policy_losses:
        ax1.plot(policy_losses)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Value')
        ax1.set_title('Policy Training: Value over Time')
        if warmup_iterations > 0:
            epochs_per_iter = cfg.num_epochs_p
            ax1.axvline(x=warmup_iterations * epochs_per_iter, color='red', 
                       linestyle='--', label='Adaptive Start')
            ax1.legend()
    
    if value_losses:
        ax2.plot(value_losses)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MSE Loss')
        ax2.set_title('Value Training: MSE Loss')
        ax2.set_yscale('log')
        if warmup_iterations > 0:
            epochs_per_iter = cfg.num_epochs_v
            ax2.axvline(x=warmup_iterations * epochs_per_iter, color='red',
                       linestyle='--', label='Adaptive Start')
            ax2.legend()
    
    plt.tight_layout()
    plt.savefig('figures/training_losses.png', dpi=150)
    plt.close(fig)
    print("  Saved: figures/training_losses.png")
    
    # Final admissibility analysis
    print("\n>>> Final admissibility analysis...")
    visualizer.plot_admissibility_heatmap(
        n_grid=50,
        save_path='figures/admissibility_heatmap_final.png',
        overlay_cpp_data=True,
        verbose_cpp_analysis=verbose_cpp
    )
    visualizer.plot_cache_distribution(save_path='figures/cache_distribution_final.png')
    
    if sampler.phase == 'adaptive':
        sampler.plot_sampling_statistics(save_path='figures/adaptive_sampling_stats_final.png')
        stats = sampler.get_statistics()
        if stats['mean_A']:
            print(f"\n>>> Adaptive Sampling Summary:")
            print(f"  Final mean A score: {stats['mean_A'][-1]:.3f}")
            print(f"  Strongly admissible: {stats['frac_strongly_admissible'][-1]:.1%}")
            print(f"  Inadmissible: {stats['frac_inadmissible'][-1]:.1%}")
    
    # Plot final functions
    print("\n>>> Plotting final policy and value surfaces...")
    thresholds = config.get('admissibility_thresholds', {})
    threshold = thresholds.get('sampling_threshold', 0.85)
    
    plotter = value_module.SurfacePlotter(
        policy_net, value_net, cfg, device,
        scorer=scorer, admissibility_threshold=threshold
    )
    plotter.generate_data(15)
    plotter.create_plot(
        save_path='figures/final_surface_plots.png',
        title_suffix=' (Final Trained)'
    )
    
    # =================================================================
    # SIMULATIONS
    # =================================================================
    print("\n" + "=" * 60)
    print("RUNNING SIMULATIONS")
    print("=" * 60)
    
    # Update config with learned bounds
    final_config = config.copy()
    if 'penalty_params' not in final_config:
        final_config['penalty_params'] = {}
    final_config['penalty_params']['b_min'] = scorer.b_min
    final_config['penalty_params']['b_max'] = scorer.b_max
    
    policy_net.eval()
    value_net.eval()
    
    with torch.no_grad():
        simulation.run_simulations(policy_net, value_net, final_config, device)
    
    # =================================================================
    # SUMMARY
    # =================================================================
    end_time = datetime.now()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nDuration: {end_time - start_time}")
    print(f"Total iterations: {num_iter}")
    print(f"  Warmup: {warmup_iterations}")
    print(f"  Adaptive: {num_iter - warmup_iterations}")
    print(f"Final debt bounds: [{scorer.b_min:.3f}, {scorer.b_max:.3f}]")
    
    if torch.cuda.is_available():
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
    
    print("\nOutput files:")
    print(f"  models/trained_policy_nn_{model_output}.pth")
    print(f"  models/trained_value_nn_{model_output}.pth")
    print("  figures/*.png")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_training()
