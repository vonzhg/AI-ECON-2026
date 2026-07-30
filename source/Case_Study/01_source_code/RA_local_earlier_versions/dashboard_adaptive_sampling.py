# dashboard_adaptive_sampling.py
# Enhanced with Two-Stage Adaptive Sampling Strategy

import torch
import torch.nn as nn
from datetime import datetime
import copy
import Ramsey_RA_value_module
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import Ramsey_RA_simulation_module
import os
import pretrain_from_txt

# --- NEW: Import adaptive sampling module ---
import Ramsey_RA_adaptive_sampling as adaptive_sampling


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


# Load configuration
config = load_config("config.json")

# Extract configuration parameters
num_iter = config['num_iter']
penalty_params = config.get('penalty_params', {})
b_min = penalty_params.get('b_min', -0.5)
b_max = penalty_params.get('b_max', 3.5)
mu_min = config['mu_min']
mu_max = config['mu_max']
num_samples_value = config['num_samples_value']
num_epochs_v = config['num_epochs_v']
lr_value = config['lr_value']
batch_size_v = config['batch_size_v']
n_v_sim = config['n_v_sim']
num_epochs_p = config['num_epochs_p']
lr_policy = config['lr_policy']
batch_size_p = config['batch_size_p']
n_p_sim = config['n_p_sim']
num_worker = config['num_worker']
zagg_vec = config['zagg_vec']
pi_zagg = config['pi_zagg']
history_size = config.get('history_size', 5)
model_number = config['model_number_input']
n1_p = config['n1_p']
n2_p = config['n2_p']
n1_v = config['n1_v']
n2_v = config['n2_v']
warmup_iterations = config.get('warmup_iterations', 3)
cache_update_frequency = config.get('cache_update_frequency', 1)
cache_grid_size = config.get('cache_grid_size', 1000)
plot_admissibility_frequency = config.get('plot_admissibility_frequency', 2)
verbose_cpp_analysis = config.get('verbose_cpp_analysis', True)
USE_TXT_PRETRAINING = config.get('use_txt_pretraining', False)

start_time = datetime.now()

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# Store simulated data from previous iterations
simulated_data_history = []

# Define networks
n_input, n_output = 3, 2
lam_govt = nn.Sequential(nn.Linear(n_input, n1_p),
                         nn.ReLU(),
                         nn.Linear(n1_p, n2_p),
                         nn.ReLU(),
                         nn.Linear(n2_p, n_output)).to(device)
lam_govt.mu_min = mu_min
n_input, n_output = 3, 1
value_govt = nn.Sequential(nn.Linear(n_input, n1_v),
                           nn.ReLU(),
                           nn.Linear(n1_v, n2_v),
                           nn.ReLU(),
                           nn.Linear(n2_v, n_output)).to(device)


def sample_from_previous_data(previous_data, num_samples, device):
    """
    Sample from previous simulated data with optional perturbation.
    MODIFIED: Now returns (samples, None) to match sampler output format.
    """
    if len(previous_data) == 0:
        print("No previous data available. Sampling uniformly from bounds.")
        samples = torch.zeros((num_samples, 3), device=device)
        samples[:, 0] = torch.rand(num_samples, device=device) * (b_max - b_min) + b_min
        samples[:, 1] = torch.rand(num_samples, device=device) * (mu_max - mu_min) + mu_min
        g_indices = torch.randint(0, 2, (num_samples,), device=device)
        samples[:, 2] = g_indices.float()
        return samples, None  # Return (samples, None)

    all_data = torch.cat(previous_data, dim=0)
    mask = (all_data[:, 0] >= b_min) & (all_data[:, 0] <= b_max) & \
           (all_data[:, 1] >= mu_min) & (all_data[:, 1] <= mu_max)
    filtered_data = all_data[mask]

    if len(filtered_data) == 0:
        print("All previous data out of bounds. Sampling uniformly.")
        return sample_from_previous_data([], num_samples, device)

    n_available = filtered_data.shape[0]
    if n_available < num_samples:
        indices = torch.randint(0, n_available, (num_samples,), device=device)
    else:
        indices = torch.randperm(n_available, device=device)[:num_samples]
    samples = filtered_data[indices].clone()
    noise_std_b = (b_max - b_min) * 0.05
    noise_std_lambda = (mu_max - mu_min) * 0.05
    samples[:, 0] += torch.randn(num_samples, device=device) * noise_std_b
    samples[:, 1] += torch.randn(num_samples, device=device) * noise_std_lambda
    samples[:, 0] = torch.clamp(samples[:, 0], b_min, b_max)
    samples[:, 1] = torch.clamp(samples[:, 1], mu_min, mu_max)
    resample_mask = torch.rand(num_samples, device=device) < 0.2
    g_indices = torch.randint(0, 2, (num_samples,), device=device)
    samples[resample_mask, 2] = g_indices[resample_mask].float()

    return samples, None  # Return (samples, None)


# ... (Pre-training and Model Loading is unchanged) ...
all_policy_losses = []
all_value_losses = []
value_govt_pretrained_comparison = None
lam_govt_pretrained_comparison = None

if USE_TXT_PRETRAINING:
    print("=" * 60);
    print("========== Running Pre-training from TXT file ==========");
    print("=" * 60)
    pretrain_from_txt.run_pretraining()
    PRETRAIN_VALUE_PATH = pretrain_from_txt.VALUE_MODEL_PATH
    PRETRAIN_POLICY_PATH = pretrain_from_txt.POLICY_MODEL_PATH
    try:
        value_govt.load_state_dict(torch.load(PRETRAIN_VALUE_PATH, map_location=device))
        lam_govt.load_state_dict(torch.load(PRETRAIN_POLICY_PATH, map_location=device))
        print("\n...Pre-training complete. Loaded pre-trained models into memory.")
        value_govt_pretrained_comparison = copy.deepcopy(value_govt)
        lam_govt_pretrained_comparison = copy.deepcopy(lam_govt)
        print("\n========== Plotting Initial Pre-trained Functions ==========")
        plot_policy_pretrained = Ramsey_RA_value_module.policy_equm_funcs(
            lam_govt_pretrained_comparison, value_govt_pretrained_comparison, device=device
        )
        plot_policy_pretrained.generate_data(10)
        plot_policy_pretrained.create_plot(
            save_filename='figures/pretrained_surface_plots.png',
            title_suffix=' (Pre-trained from TXT)'
        )
    except FileNotFoundError:
        print(f"ERROR: Pre-training ran, but could not find saved models.")
        print("...Starting from random initialization.")
        USE_TXT_PRETRAINING = False
else:
    print("=" * 60);
    print("========== Loading Models from Config Input File ==========");
    print("=" * 60)
    model_number = config['model_number_input']
    PRETRAIN_VALUE_PATH = f'models/trained_value_nn_{model_number}.pth'
    PRETRAIN_POLICY_PATH = f'models/trained_policy_nn_{model_number}.pth'
    try:
        value_govt.load_state_dict(torch.load(PRETRAIN_VALUE_PATH, map_location=device))
        lam_govt.load_state_dict(torch.load(PRETRAIN_POLICY_PATH, map_location=device))
        print(f"...Loaded existing models from '...{model_number}.pth'")
    except FileNotFoundError:
        print(f"...Could not find models '...{model_number}.pth'.")
        print("...Starting from random initialization.")

# =============================================================================
# INITIALIZE ADAPTIVE SAMPLING SYSTEM
# =============================================================================
print("\n" + "=" * 60);
print("INITIALIZING ADAPTIVE SAMPLING SYSTEM");
print("=" * 60)
scorer, sampler, visualizer = adaptive_sampling.initialize_adaptive_sampling(
    policy_net=lam_govt,
    config=config,
    device=device
)
sampler.set_phase('warmup')
print(f"\nTraining Plan:")
print(f"  Warmup iterations: {warmup_iterations} (uniform sampling)")
print(f"  Adaptive iterations: {num_iter - warmup_iterations} (reweighted sampling)")
print(f"  Cache updates every: {cache_update_frequency} iteration(s)")
print(f"  Admissibility plots every: {plot_admissibility_frequency} iteration(s)")

# =============================================================================
# TRAINING ITERATIONS
# =============================================================================

for i_iter in range(num_iter):

    print(f'\n{"=" * 60}');
    print(f'ITERATION {i_iter + 1}/{num_iter}');
    print(f'{"=" * 60}')

    # --- Phase Transition ---
    if i_iter == warmup_iterations:
        print("\n" + "!" * 60);
        print("!!! TRANSITIONING TO ADAPTIVE SAMPLING PHASE !!!");
        print("!" * 60)
        sampler.set_phase('adaptive', total_adaptive_iters=(num_iter - warmup_iterations))
        print("\n>>> Building initial admissibility cache...")
        adaptive_sampling.update_cache_periodic(scorer, sampler, n_grid=cache_grid_size)
        print("\n>>> Plotting admissibility with C++ data overlay for comparison...")
        visualizer.plot_admissibility_heatmap(
            n_grid=50,
            save_path=f'figures/admissibility_heatmap_iter_{i_iter}.png',
            overlay_cpp_data=True,
            cpp_data_file='policy_v6_out_61.txt',
            verbose_cpp_analysis=verbose_cpp_analysis
        )
        visualizer.plot_cache_distribution(save_path=f'figures/cache_distribution_iter_{i_iter}.png')

    # --- Sampling ---
    print(f"\nStep 1: Sampling (Phase: {sampler.phase.upper()})")

    # --- MODIFIED: Receive two batches ---
    inadmissible_samples = None  # Ensure it's defined
    if sampler.phase == 'warmup':
        domain_samples, inadmissible_samples = sample_from_previous_data(
            simulated_data_history,
            num_samples=num_samples_value,
            device=device
        )
    else:
        domain_samples, inadmissible_samples = sampler.sample_batch(
            batch_size=num_samples_value,
            policy_net=lam_govt
        )

    print(f"  Generated {domain_samples.shape[0]} main samples")
    if inadmissible_samples is not None:
        print(f"  Generated {inadmissible_samples.shape[0]} inadmissible samples for explicit training")
    print(f"  B range: [{domain_samples[:, 0].min():.3f}, {domain_samples[:, 0].max():.3f}]")
    print(f"  λ range: [{domain_samples[:, 1].min():.3f}, {domain_samples[:, 1].max():.3f}]")
    g_counts = [(domain_samples[:, 2] == i).sum().item() for i in range(2)]
    print(f"  G counts: [G=0: {g_counts[0]}, G=1: {g_counts[1]}]")

    # --- Model Setup ---
    lam_govt_old = copy.deepcopy(lam_govt)
    Ramsey_RA_value_module.model_number = config['model_number_output']
    decision_trainer = Ramsey_RA_value_module.equm_trainer(
        num_epochs_v, num_epochs_p, lr_value, lr_policy,
        batch_size_v, num_worker, lam_govt, lam_govt_old,
        value_govt,
        device=device, i_save=1
    )
    x_print = 0
    if i_iter == num_iter - 1:
        x_print = 1

    # --- Policy Training ---
    print(f"\nStep 2: Training policy network")
    x_i_ind = 0

    # --- MODIFIED: Pass inadmissible_samples to the trainer ---
    lam_govt = decision_trainer.policy_train(
        domain_samples,  # Pass the "good" samples
        zagg_vec,
        pi_zagg,
        x_i_ind,
        n_p_sim,
        x_print,
        all_losses_list=all_policy_losses,
        inadmissible_samples=inadmissible_samples  # Pass the "bad" samples
    )

    # --- Value Training ---
    print(f"\nStep 3: Training value network and collecting simulation data")
    x_i_ind = 1

    current_penalty_params = config['penalty_params'].copy()
    current_penalty_params['b_min'] = scorer.b_min
    current_penalty_params['b_max'] = scorer.b_max

    compute_value = Ramsey_RA_value_module.define_objective(
        value_govt, lam_govt, lam_govt_old,
        penalty_params=current_penalty_params,
        device=device
    )

    # --- MODIFIED: Pass inadmissible_samples and v_threshold to the trainer ---
    value_govt = decision_trainer.value_train(
        compute_value,
        domain_samples,  # Pass the "good" samples
        all_losses_list=all_value_losses,
        inadmissible_samples=inadmissible_samples,  # Pass the "bad" samples
        v_threshold=config['v_threshold']  # Pass the "bad" value target
    )

    # Collect simulated trajectories
    _, domain_data, _, _ = compute_value.obj_sim_value(
        domain_samples, zagg_vec, pi_zagg, x_i_ind, n_v_sim, x_print
    )
    states_only = domain_data[:, 0:3]
    simulated_data_history.append(states_only)
    if len(simulated_data_history) > history_size:
        simulated_data_history = simulated_data_history[-history_size:]
    print(f"  Collected {states_only.shape[0]} state samples for next iteration")
    print(f"  History now contains {len(simulated_data_history)}/{history_size} batches")

    # --- Update Cache (Adaptive Phase Only) ---
    if sampler.phase == 'adaptive' and (i_iter - warmup_iterations) % cache_update_frequency == 0:
        print(f"\nStep 4: Updating admissibility cache (iteration {i_iter})")
        adaptive_sampling.update_cache_periodic(scorer, sampler, n_grid=cache_grid_size)

    # --- Increment Adaptive Sampler ---
    sampler.increment_iteration()

    # --- Periodic Diagnostics ---
    if sampler.phase == 'adaptive' and (i_iter - warmup_iterations) % plot_admissibility_frequency == 0:
        print(f"\nStep 5: Generating diagnostic plots (iteration {i_iter})")
        visualizer.plot_admissibility_heatmap(
            n_grid=50,
            save_path=f'figures/admissibility_heatmap_iter_{i_iter}.png',
            overlay_cpp_data=True,
            cpp_data_file='policy_v6_out_61.txt',
            verbose_cpp_analysis=verbose_cpp_analysis
        )

        print("       Generating boundary diagnostic plot...")
        visualizer.plot_boundary_diagnostics(
            save_path=f'figures/boundary_diagnostics_iter_{i_iter}.png'
        )

        visualizer.plot_cache_distribution(
            save_path=f'figures/cache_distribution_iter_{i_iter}.png'
        )
        sampler.plot_sampling_statistics(
            save_path='figures/adaptive_sampling_stats.png'
        )

# ... (Post-training Analysis and Simulation are unchanged) ...
print("\n" + "=" * 60);
print("POST-TRAINING ANALYSIS");
print("=" * 60)
print("\n>>> Plotting training losses...")
fig_losses = plt.figure(figsize=(12, 6))
ax1 = fig_losses.add_subplot(1, 2, 1)
if all_policy_losses:
    ax1.plot(all_policy_losses)
ax1.set_xlabel('Total Epochs (all iterations)');
ax1.set_ylabel('Avg. Value (V)')
ax1.set_title('Policy Training: Avg. Value (V) over Time')
if warmup_iterations > 0 and all_policy_losses:
    warmup_epoch = warmup_iterations * num_epochs_p
    ax1.axvline(x=warmup_epoch, color='red', linestyle='--', linewidth=2, label='Adaptive Phase Start')
    ax1.legend()
ax2 = fig_losses.add_subplot(1, 2, 2)
if all_value_losses:
    ax2.plot(all_value_losses)
ax2.set_xlabel('Total Epochs (all iterations)');
ax2.set_ylabel('Avg. MSE Loss')
ax2.set_title('Value Training: Avg. MSE Loss over Time');
ax2.set_yscale('log')
if warmup_iterations > 0 and all_value_losses:
    warmup_epoch = warmup_iterations * num_epochs_v
    ax2.axvline(x=warmup_epoch, color='red', linestyle='--', linewidth=2, label='Adaptive Phase Start')
    ax2.legend()
plt.tight_layout();
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/all_losses_plot.png');
plt.close(fig_losses)
print("    Saved: figures/all_losses_plot.png")

print("\n>>> Final admissibility analysis WITH C++ data overlay...")
visualizer.plot_admissibility_heatmap(
    n_grid=50,
    save_path='figures/admissibility_heatmap_final.png',
    overlay_cpp_data=True,
    cpp_data_file='policy_v6_out_61.txt',
    verbose_cpp_analysis=verbose_cpp_analysis
)
visualizer.plot_cache_distribution(
    save_path='figures/cache_distribution_final.png'
)

if sampler.phase == 'adaptive':
    sampler.plot_sampling_statistics(
        save_path='figures/adaptive_sampling_stats_final.png'
    )
    stats = sampler.get_statistics()
    if len(stats['mean_A']) > 0:
        print("\n>>> Adaptive Sampling Summary:")
        print(f"    Final mean A score: {stats['mean_A'][-1]:.3f}")
        print(f"    Final distribution:")
        print(f"      Strongly admissible: {stats['frac_strongly_admissible'][-1]:.1%}")
        # print(f"      Transition: {stats['frac_transition'][-1]:.1%}") # No longer relevant
        print(f"      Inadmissible: {stats['frac_inadmissible'][-1]:.1%}")
        print(f"    Final Dynamic Debt Bounds (b_min, b_max): [{scorer.b_min:.3f}, {scorer.b_max:.3f}]")

print("\n>>> Plotting final policy and value functions...")
final_admissibility_threshold = sampler.admissibility_thresholds

plot_policy_final = Ramsey_RA_value_module.policy_equm_funcs(
    lam_govt,
    value_govt,
    scorer,  # Pass the final trained scorer
    final_admissibility_threshold,
    device=device
)
plot_policy_final.generate_data(10)
plot_policy_final.create_plot(
    save_filename='figures/final_surface_plots.png',
    title_suffix=' (Final Trained with Adaptive Sampling)'
)
if value_govt_pretrained_comparison is not None:
    print("\n>>> Re-plotting pre-trained functions for comparison...")
    plot_policy_pretrained.create_plot(
        save_filename='figures/pretrained_surface_plots.png',
        title_suffix=' (Pre-trained from TXT)'
    )

print("\n" + "=" * 60);
print("RUNNING SIMULATIONS");
print("=" * 60)
lam_govt.eval();
value_govt.eval()
final_config = config.copy()
final_config['penalty_params']['b_min'] = scorer.b_min
final_config['penalty_params']['b_max'] = scorer.b_max
final_config['b_min'] = scorer.b_min  # For simulation module
final_config['b_max'] = scorer.b_max  # For simulation module
with torch.no_grad():
    Ramsey_RA_simulation_module.run_and_plot_simulations(lam_govt, value_govt, final_config, device)

end_time = datetime.now()
if torch.cuda.is_available():
    print(f"torch.cuda.memory_allocated: {torch.cuda.memory_allocated(0) / 1024 / 1024:.2f} MB")
print(f"torch.cuda.memory_reserved: {torch.cuda.memory_reserved(0) / 1024 / 1024:.2f} MB")
print(f"torch.cuda.max_memory_reserved: {torch.cuda.max_memory_reserved(0) / 1024 / 1024:.2f} MB")
print(f'Duration: {end_time - start_time}')
print(f'Warmup iterations: {warmup_iterations}')
print(f'Adaptive iterations: {num_iter - warmup_iterations}')
print(f'Total iterations: {num_iter}')
print(f'Final Dynamic Debt Bounds (b_min, b_max): [{scorer.b_min:.3f}, {scorer.b_max:.3f}]')
print("=" * 60)