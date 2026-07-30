# dashboard_v2.py
"""
Deep Ramsey Algorithm - Main Training Dashboard
VERSION 2: Streamlined and aligned with document specification.

Training loop:
1. Warmup phase: Uniform/history sampling
2. Adaptive phase: Boundary-focused sampling with two-level fixed-point
   - Inner loop: Boundary refinement (fixed policy)
   - Outer loop: Policy/value training → boundary update
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from datetime import datetime
import copy

# Import v2 modules
import Ramsey_RA_adaptive_sampling_v2 as adaptive
import Ramsey_RA_value_module_v2 as value_module
import Ramsey_RA_simulation_module_v2 as simulation


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    """Main training loop implementing two-level fixed-point algorithm."""
    
    start_time = datetime.now()
    
    # --- Load Config ---
    config = load_config('config_v2.json')
    
    # --- Device Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- Build Networks ---
    econ = config['economic_parameters']
    model_cfg = config['model']
    
    policy_hidden = model_cfg['policy_network']['hidden_layers']
    value_hidden = model_cfg['value_network']['hidden_layers']
    
    policy_net = value_module.build_network(3, 2, policy_hidden).to(device)
    value_net = value_module.build_network(3, 1, value_hidden).to(device)
    
    print(f"Policy network: 3 -> {policy_hidden} -> 2")
    print(f"Value network: 3 -> {value_hidden} -> 1")
    
    # --- Try Loading Existing Models ---
    model_num_in = model_cfg['model_number_input']
    policy_path = f'models/trained_policy_nn_{model_num_in}.pth'
    value_path = f'models/trained_value_nn_{model_num_in}.pth'
    
    try:
        policy_net.load_state_dict(torch.load(policy_path, map_location=device))
        value_net.load_state_dict(torch.load(value_path, map_location=device))
        print(f"Loaded models from: {policy_path}, {value_path}")
    except FileNotFoundError:
        print("No existing models found, starting from random initialization")
    
    # --- Initialize Adaptive Sampling ---
    scorer, sampler, visualizer = adaptive.initialize_adaptive_sampling(
        policy_net, config, device)
    
    # --- Initialize Trainer ---
    trainer = value_module.RamseyTrainer(policy_net, value_net, config, device)
    
    # --- Training Parameters ---
    train_cfg = config['training']
    num_iterations = train_cfg['num_iterations']
    warmup_iterations = train_cfg['warmup_iterations']
    
    sampling_cfg = config['sampling']
    num_samples = sampling_cfg['num_samples']
    cache_update_freq = sampling_cfg['cache_update_frequency']
    cache_grid_size = sampling_cfg['cache_grid_size']
    
    diag_cfg = config['diagnostics']
    plot_freq = diag_cfg['plot_frequency']
    
    # History for warmup sampling
    history = []
    history_size = sampling_cfg['history_size']
    
    # Loss tracking
    policy_losses = []
    value_losses = []
    
    print("\n" + "=" * 60)
    print("TRAINING PLAN")
    print(f"  Total iterations: {num_iterations}")
    print(f"  Warmup iterations: {warmup_iterations}")
    print(f"  Adaptive iterations: {num_iterations - warmup_iterations}")
    print(f"  Samples per iteration: {num_samples}")
    print("=" * 60)
    
    # ==========================================================================
    # OUTER FIXED-POINT LOOP
    # ==========================================================================
    
    for iteration in range(num_iterations):
        print(f"\n{'=' * 60}")
        print(f"ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'=' * 60}")
        
        # --- Phase Transition ---
        if iteration == warmup_iterations:
            print("\n" + "!" * 60)
            print("TRANSITIONING TO ADAPTIVE PHASE")
            print("!" * 60)
            
            sampler.set_phase('adaptive')
            
            # Initial boundary refinement
            print("\n[Phase Transition] Running initial boundary refinement...")
            adaptive.refine_boundaries(scorer, sampler, config, cache_grid_size)
            
            # Initial visualization
            visualizer.plot_heatmap(save_path=f'figures/heatmap_iter_{iteration}.png')
            visualizer.plot_score_distribution(save_path=f'figures/distribution_iter_{iteration}.png')
        
        # --- Sampling ---
        print(f"\n[Sampling] Phase: {sampler.phase.upper()}")
        
        if sampler.phase == 'warmup':
            good_samples, bad_samples = sampler.sample_from_history(history, num_samples)
        else:
            good_samples, bad_samples = sampler.sample_adaptive(num_samples)
        
        print(f"  Good samples: {good_samples.shape[0]}")
        if bad_samples is not None:
            print(f"  Bad samples: {bad_samples.shape[0]}")
        
        # --- Update Trainer Bounds ---
        trainer.update_bounds(scorer.b_min, scorer.b_max)
        
        # --- Policy Training ---
        print("\n[Training] Policy network...")
        policy_net = trainer.train_policy(good_samples, bad_samples)
        
        # --- Value Training ---
        print("\n[Training] Value network...")
        value_net = trainer.train_value(good_samples, bad_samples)
        
        # --- Collect Simulation Data for History ---
        print("\n[History] Collecting simulation data...")
        n_sim = config['training']['value']['n_sim_periods']
        sim_data = trainer.collect_simulation_data(good_samples, n_sim)
        history.append(sim_data)
        if len(history) > history_size:
            history = history[-history_size:]
        print(f"  History: {len(history)}/{history_size} batches")
        
        # --- Boundary Refinement (Adaptive Phase) ---
        if sampler.phase == 'adaptive':
            if (iteration - warmup_iterations) % cache_update_freq == 0:
                print("\n[Boundary] Running refinement (inner fixed-point)...")
                adaptive.refine_boundaries(scorer, sampler, config, cache_grid_size)
        
        # --- Diagnostics ---
        if sampler.phase == 'adaptive' and (iteration - warmup_iterations) % plot_freq == 0:
            print("\n[Diagnostics] Generating plots...")
            visualizer.plot_heatmap(save_path=f'figures/heatmap_iter_{iteration}.png')
            visualizer.plot_score_distribution(save_path=f'figures/distribution_iter_{iteration}.png')
        
        sampler.increment_iteration()
    
    # ==========================================================================
    # POST-TRAINING
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("POST-TRAINING")
    print("=" * 60)
    
    # Save models
    model_num_out = model_cfg['model_number_output']
    os.makedirs('models', exist_ok=True)
    
    if diag_cfg['save_models']:
        torch.save(policy_net.state_dict(), f'models/trained_policy_nn_{model_num_out}.pth')
        torch.save(value_net.state_dict(), f'models/trained_value_nn_{model_num_out}.pth')
        print(f"Models saved: trained_*_nn_{model_num_out}.pth")
    
    # Final visualization
    print("\n[Final] Generating final plots...")
    visualizer.plot_heatmap(save_path='figures/heatmap_final.png')
    visualizer.plot_score_distribution(save_path='figures/distribution_final.png')
    
    # Policy/value surfaces
    pv_viz = value_module.PolicyValueVisualizer(policy_net, value_net, config, device)
    pv_viz.plot_surfaces(save_path='figures/surfaces_final.png', title_suffix=' (Final)')
    
    # Run simulations
    print("\n[Final] Running simulations...")
    simulation.run_and_plot_simulations(policy_net, value_net, config, device)
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Duration: {duration}")
    print(f"  Final debt bounds: [{scorer.b_min:.3f}, {scorer.b_max:.3f}]")
    print("=" * 60)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    # Ensure output directories exist
    os.makedirs('figures', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    main()
