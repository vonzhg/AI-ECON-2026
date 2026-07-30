"""
Main Dashboard for Heterogeneous Agent Ramsey Solver.
Revised: Ensures all outputs go to the 'output' folder.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os

from ha_model import HAModel
from boundary import AlphaBoundary
from visualization import HAVisualizer
from simulation import run_simulation

torch.autograd.set_detect_anomaly(True)

# --- CONFIGURATION ---
OUTPUT_ROOT = "output"
MODELS_DIR = os.path.join(OUTPUT_ROOT, "models")
FIGURES_DIR = os.path.join(OUTPUT_ROOT, "figures")
RESULTS_DIR = os.path.join(OUTPUT_ROOT, "results")

# Ensure directories exist
for d in [OUTPUT_ROOT, MODELS_DIR, FIGURES_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)


def load_config_json(config_file='config.json'):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with open(config_file, 'r') as f:
        return json.load(f)


def print_training_flow(config):
    n_batch = config['training']['batches_per_phase']
    n_bound = config['boundary']['n_boundary_samples']

    print("\n" + "="*60)
    print("               TRAINING FLOW")
    print("="*60)
    print(f"├── 1. DOMAIN DISCOVERY ({n_bound} samples -> α-Shape)")
    print(f"├── 2. TRAIN CRITIC ({n_batch} batches)")
    print(f"└── 3. TRAIN ACTOR ({n_batch} batches)")
    print("="*60 + "\n")


def generate_random_states(num, config, device):
    """Generate random states within bounds."""
    sb = config['state_bounds']
    K = torch.rand(num, 1, device=device) * (sb['K_max'] - sb['K_min']) + sb['K_min']
    ae = torch.rand(num, 1, device=device) * (sb['a_max'] - sb['a_min']) + sb['a_min']
    au = torch.rand(num, 1, device=device) * (sb['a_max'] - sb['a_min']) + sb['a_min']
    ce = torch.rand(num, 1, device=device) * (sb['c_max'] - sb['c_min']) + sb['c_min']
    cu = torch.rand(num, 1, device=device) * (sb['c_max'] - sb['c_min']) + sb['c_min']
    return torch.cat([K, ae, au, ce, cu], dim=1)


def sample_from_boundary(num_samples, boundary, config, device):
    """Sample states from learned α-shape interior."""
    sb = config['state_bounds']

    if boundary.admissible_points is not None and len(boundary.admissible_points) > 10:
        n_base = min(len(boundary.admissible_points), num_samples * 2)
        indices = np.random.choice(len(boundary.admissible_points), n_base, replace=True)
        base_points = boundary.admissible_points[indices]

        # Add perturbations
        scale = 0.05
        # Perturb all 5 dimensions
        perturbation = np.random.randn(n_base, 5) * scale
        candidates_np = base_points + perturbation

        # Clip to hard bounds
        candidates_np[:, 0] = np.clip(candidates_np[:, 0], sb['K_min'], sb['K_max'])
        candidates_np[:, 1] = np.clip(candidates_np[:, 1], sb['a_min'], sb['a_max'])
        candidates_np[:, 2] = np.clip(candidates_np[:, 2], sb['a_min'], sb['a_max'])
        candidates_np[:, 3] = np.clip(candidates_np[:, 3], sb['c_min'], sb['c_max'])
        candidates_np[:, 4] = np.clip(candidates_np[:, 4], sb['c_min'], sb['c_max'])

        candidates = torch.tensor(candidates_np, dtype=torch.float32, device=device)

        # Verify admissibility
        is_valid = boundary.is_admissible(candidates).squeeze()
        valid = candidates[is_valid]

        if len(valid) >= num_samples:
            return valid[:num_samples]

        extra = generate_random_states(num_samples - len(valid), config, device)
        return torch.cat([valid, extra], dim=0)[:num_samples] if len(valid) > 0 else extra

    return generate_random_states(num_samples, config, device)


def train():
    """Main training function."""
    config = load_config_json()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print_training_flow(config)

    # Initialize with output directory passed to Visualizer
    model = HAModel(config, device)
    boundary = AlphaBoundary(config)
    viz = HAVisualizer(config, device, save_dir=FIGURES_DIR)

    opt_actor = optim.Adam(model.actor.parameters(), lr=config['training']['lr_actor'])
    opt_critic = optim.Adam(model.critic.parameters(), lr=config['training']['lr_critic'])

    lambda_fb = config['fischer_burmeister']['lambda_initial']
    lambda_max = config['fischer_burmeister']['lambda_max']

    actor_losses, critic_losses, fb_penalties = [], [], []

    num_iters = config['training']['num_iterations']
    batches = config['training']['batches_per_phase']
    epochs = config['training']['epochs_per_batch']
    batch_size = config['training']['batch_size']
    rollout_steps = config['training']['rollout_steps']
    plot_freq = config['training'].get('plot_frequency', 5)
    n_boundary = config['boundary']['n_boundary_samples']

    # ==================== TRAINING LOOP ====================
    for iteration in range(num_iters):
        print(f"\n--- Iteration {iteration+1}/{num_iters} ---")

        # --- Domain Discovery ---
        print("Updating Boundary...")
        with torch.no_grad():
            candidates = generate_random_states(n_boundary, config, device)
            out = model.forward_physics(candidates)
            if out is not None:
                # Pass boundary for recursive check
                scores = model.compute_admissibility(out['physics'], boundary=boundary)

                boundary.update(out['next_state'], scores, threshold=0.9)
                stats = boundary.get_boundary_stats()
                print(f"   {stats['n_points']} pts, {stats['n_alpha_simplices']}/{stats['n_simplices']} α-simplices")

        # Visualize
        if (iteration + 1) % plot_freq == 0 and out is not None:
            is_good = (scores > 0.9).flatten()
            if is_good.sum() > 0:
                viz.plot_boundary_3d(boundary, iteration + 1, sample_points=out['next_state'][is_good])
                viz.plot_boundary_2d(boundary, iteration + 1, sample_points=out['next_state'][is_good])

        model.train()

        # --- Train Critic ---
        print(">> Critic")
        avg_c = 0
        for _ in range(batches):
            batch_s = sample_from_boundary(batch_size, boundary, config, device)
            for _ in range(epochs):
                with torch.no_grad():
                    curr = batch_s.clone()
                    reward = torch.zeros(batch_size, 1, device=device)
                    for t in range(rollout_steps):
                        out = model.forward_physics(curr)
                        if out is None: break
                        reward += (model.beta ** t) * out['welfare']
                        curr = out['next_state']
                    target = reward + (model.beta ** rollout_steps) * model.critic(curr)

                loss = nn.MSELoss()(model.critic(batch_s), target.detach())
                if torch.isnan(loss): return

                opt_critic.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 1.0)
                opt_critic.step()
            avg_c += loss.item()
        critic_losses.append(avg_c / batches)

        # --- Train Actor ---
        print(">> Actor")
        avg_a, avg_fb = 0, 0
        for _ in range(batches):
            batch_s = sample_from_boundary(batch_size, boundary, config, device)
            for _ in range(epochs):
                curr = batch_s.clone()
                reward = torch.zeros(batch_size, 1, device=device)
                fb_sum = torch.zeros(batch_size, 1, device=device)

                for t in range(rollout_steps):
                    out = model.forward_physics(curr)
                    if out is None: break
                    reward += (model.beta ** t) * out['welfare']
                    fb_e, fb_u = out['fb_residuals']
                    fb_sum += fb_e**2 + fb_u**2
                    curr = out['next_state']

                if out is None: continue

                value = reward + (model.beta ** rollout_steps) * model.critic(curr)
                loss = -value.mean() + lambda_fb * fb_sum.mean()

                if torch.isnan(loss): continue

                opt_actor.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 0.5)
                opt_actor.step()

            avg_a += loss.item()
            avg_fb += fb_sum.mean().item()

        actor_losses.append(avg_a / batches)
        fb_penalties.append(avg_fb / batches)

        if fb_penalties[-1] > config['fischer_burmeister']['penalty_threshold']:
            lambda_fb = min(lambda_fb * config['fischer_burmeister']['rho'], lambda_max)

        print(f"Critic: {critic_losses[-1]:.4f} | Actor: {actor_losses[-1]:.4f} | "
              f"FB: {fb_penalties[-1]:.4f} | λ: {lambda_fb:.1f}")

    # ==================== SAVE & VISUALIZE ====================
    viz.plot_losses(actor_losses, critic_losses)
    viz.plot_fb_penalty(fb_penalties)

    # Save Model to 'output/models'
    file_path = os.path.join(MODELS_DIR, "ha_model_final.pth")
    torch.save(model.state_dict(), file_path)
    print(f"Model saved successfully to: {file_path}")

    print("\n" + "="*60)
    print("              TRAINING COMPLETE")
    print("="*60)

    # ==================== POST-TRAINING SIMULATION ====================
    sim_config = config.get('simulation', {})
    if sim_config.get('run_after_training', True):
        run_simulation(
            model, boundary, config, device,
            num_trajectories=sim_config.get('num_trajectories', 100),
            num_periods=sim_config.get('num_periods', 50),
            save_dir=RESULTS_DIR
        )

    return model, boundary


if __name__ == "__main__":
    train()