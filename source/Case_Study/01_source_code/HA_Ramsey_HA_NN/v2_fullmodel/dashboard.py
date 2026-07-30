"""
Main Dashboard for Heterogeneous Agent Ramsey Solver.

This module orchestrates the training process:
1. Model versioning (load/save with version numbers)
2. Two-level fixed-point iteration:
   - Level 1: Domain stabilization (update α-shape boundary)
   - Level 2: Policy optimization (actor-critic updates)
3. Hard projection to admissible set during rollouts

Key Algorithm (from Document Section 4.6):
- Sample s_0 uniformly from hypercube H
- Filter to keep s_0 where s_1 = f(s_0, π(s_0)) ∈ S_α
- During rollout: if s_{t+1} ∉ S_α, PROJECT back and add penalty
- Train actor to maximize welfare minus penalties
- Train critic via TD learning with target network

IMPORTANT DESIGN DECISIONS:
1. Hard projection: When s_{t+1} is outside S_α, we replace it with the
   nearest admissible point. This keeps the rollout economically meaningful.
2. Projection penalty: We add λ_shape × distance² to the actor loss.
   This encourages the policy to stay within the admissible set.
3. Hypercube penalty: We ALSO penalize violations of the box constraints
   on assets. This provides gradient signal even when S_α is small/empty.

PENALTY WEIGHT SCHEDULES:
- lambda_fb: ADAPTIVE (increases when FB violations exceed threshold)
- lambda_bound: FIXED (hypercube violation, needed for early training)
- lambda_shape: FIXED (α-shape projection penalty)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import json
import os
import subprocess
import torch.nn.functional as F
from datetime import datetime
import time

from ha_model import HAModel
from boundary import AlphaBoundary
from visualization import HAVisualizer
from simulation import run_simulation

# Enable anomaly detection for debugging NaN gradients
torch.autograd.set_detect_anomaly(True)

# ==================== Directory Setup ====================
OUTPUT_ROOT = "output"
MODELS_DIR = os.path.join(OUTPUT_ROOT, "models")
FIGURES_DIR = os.path.join(OUTPUT_ROOT, "figures")
RESULTS_DIR = os.path.join(OUTPUT_ROOT, "simulations")

# Create directories WITHOUT deleting existing contents
for d in [OUTPUT_ROOT, MODELS_DIR, FIGURES_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ==================== File Protection ====================

def check_existing_output_files(config):
    """
    Check if output files already exist and warn user.

    Returns:
        bool: True if safe to proceed, False if should abort
    """
    version_config = config.get('versioning', {})
    output_version = version_config.get('output_version', 'latest')

    model_path = get_model_path(output_version)
    boundary_path = get_boundary_path(output_version)

    existing_files = []
    if os.path.exists(model_path):
        existing_files.append(model_path)
    if os.path.exists(boundary_path):
        existing_files.append(boundary_path)

    if existing_files:
        print("\n" + "!"*60)
        print("   WARNING: Output files already exist!")
        print("!"*60)
        for f in existing_files:
            # Get file modification time
            mtime = os.path.getmtime(f)
            mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            size_kb = os.path.getsize(f) / 1024
            print(f"   - {f}")
            print(f"     Modified: {mtime_str}, Size: {size_kb:.1f} KB")
        print("\n   These files will be OVERWRITTEN at the end of training.")
        print("   To avoid this:")
        print(f"   1. Change 'output_version' in config.json (currently: '{output_version}')")
        print(f"   2. Or backup the existing files manually")
        print("!"*60 + "\n")

        # In non-interactive mode, proceed with warning
        # In interactive mode, you could add input() here to confirm
        return True

    return True


def backup_existing_files(config):
    """
    Create backups of existing output files before overwriting.

    Backups are named with timestamp: filename.YYYYMMDD_HHMMSS.bak
    """
    version_config = config.get('versioning', {})
    output_version = version_config.get('output_version', 'latest')

    model_path = get_model_path(output_version)
    boundary_path = get_boundary_path(output_version)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    backed_up = []
    for filepath in [model_path, boundary_path]:
        if os.path.exists(filepath):
            backup_path = f"{filepath}.{timestamp}.bak"
            import shutil
            shutil.copy2(filepath, backup_path)
            backed_up.append((filepath, backup_path))

    if backed_up:
        print("\n   Backed up existing files:")
        for orig, bak in backed_up:
            print(f"   - {orig} -> {bak}")

    return backed_up


# ==================== System Check ====================

def check_gpu():
    """Check GPU availability and print nvidia-smi output."""
    print("\n" + "="*60)
    print("   SYSTEM CHECK")
    print("="*60)

    # PyTorch CUDA check
    if torch.cuda.is_available():
        print(f"   PyTorch CUDA: Available")
        print(f"   Device count: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
        print(f"   Device name: {torch.cuda.get_device_name(0)}")

        # Get total memory
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   Total GPU Memory: {total_mem:.1f} GB")
    else:
        print("   PyTorch CUDA: Not available (using CPU)")

    # nvidia-smi
    print("\n   nvidia-smi output:")
    print("   " + "-"*50)
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # Print with indentation
            for line in result.stdout.split('\n')[:20]:  # First 20 lines
                print(f"   {line}")
        else:
            print("   nvidia-smi failed or not available")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("   nvidia-smi not found (no NVIDIA GPU or drivers not installed)")

    print("="*60 + "\n")


def log_gpu_memory(prefix=""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        print(f"   {prefix}GPU Memory: {allocated:.2f}GB allocated, "
              f"{reserved:.2f}GB reserved, {max_allocated:.2f}GB peak")


# ==================== Configuration ====================

def load_config_json(config_file='config.json'):
    """Load configuration from JSON file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with open(config_file, 'r') as f:
        return json.load(f)


def get_model_path(version):
    """Get the file path for a model version."""
    return os.path.join(MODELS_DIR, f"ha_model_v{version}.pth")


def get_boundary_path(version):
    """Get the file path for a boundary version."""
    return os.path.join(MODELS_DIR, f"boundary_v{version}.pkl")


def load_model_if_exists(model, config, device):
    """
    Load model weights from file if input_version is specified and file exists.

    Returns:
        bool: True if model was loaded, False if starting fresh
    """
    version_config = config.get('versioning', {})
    input_version = version_config.get('input_version')

    if input_version is None:
        print("   No input version specified. Starting with random weights.")
        return False

    model_path = get_model_path(input_version)

    if not os.path.exists(model_path):
        print(f"   Model file not found: {model_path}")
        print("   Starting with random weights.")
        return False

    try:
        checkpoint = torch.load(model_path, map_location=device)

        # Handle different checkpoint formats
        if 'actor' in checkpoint:
            # New format with separate components
            model.actor.load_state_dict(checkpoint['actor'])
            model.critic.load_state_dict(checkpoint['critic'])
            if 'critic_target' in checkpoint:
                model.critic_target.load_state_dict(checkpoint['critic_target'])
            else:
                model.hard_update_critic_target()
        else:
            # Old format with full model state
            model.load_state_dict(checkpoint)
            model.hard_update_critic_target()

        print(f"   Loaded model from: {model_path}")
        return True

    except Exception as e:
        print(f"   Failed to load model: {e}")
        print("   Starting with random weights.")
        return False


def load_boundary_if_exists(boundary, config):
    """
    Load boundary state from file if input_version is specified.

    Returns:
        bool: True if boundary was loaded, False if starting fresh
    """
    version_config = config.get('versioning', {})
    input_version = version_config.get('input_version')

    if input_version is None:
        return False

    boundary_path = get_boundary_path(input_version)
    return boundary.load(boundary_path)


def save_model_and_boundary(model, boundary, config):
    """Save model and boundary with output version number."""
    version_config = config.get('versioning', {})
    output_version = version_config.get('output_version', 'latest')

    model_path = get_model_path(output_version)
    boundary_path = get_boundary_path(output_version)

    # Save model components
    torch.save({
        'actor': model.actor.state_dict(),
        'critic': model.critic.state_dict(),
        'critic_target': model.critic_target.state_dict(),
        'config': config,
        'timestamp': datetime.now().isoformat()
    }, model_path)
    print(f"   Model saved to: {model_path}")

    # Save boundary
    boundary.save(boundary_path)


# ==================== Data Generation ====================

def generate_candidates_uniform(num, config, device):
    """
    Sample states uniformly from the hypercube H = [s_min, s_max].

    This is the FALLBACK method used when:
    - No boundary object is available
    - Boundary has insufficient admissible points

    For smarter sampling that focuses on the admissible region,
    use boundary.sample_candidates() instead.

    Args:
        num: Number of samples
        config: Configuration dict
        device: Torch device

    Returns:
        Tensor (num, 5): Sampled states
    """
    sb = config['state_bounds']
    K = torch.rand(num, 1, device=device) * (sb['K_max'] - sb['K_min']) + sb['K_min']
    ae = torch.rand(num, 1, device=device) * (sb['a_max'] - sb['a_min']) + sb['a_min']
    au = torch.rand(num, 1, device=device) * (sb['a_max'] - sb['a_min']) + sb['a_min']
    ce = torch.rand(num, 1, device=device) * (sb['c_max'] - sb['c_min']) + sb['c_min']
    cu = torch.rand(num, 1, device=device) * (sb['c_max'] - sb['c_min']) + sb['c_min']
    return torch.cat([K, ae, au, ce, cu], dim=1)


def generate_candidates(boundary, num, config, device):
    """
    Sample candidate states using the boundary's configured sampling method.

    This is the PREFERRED method that uses smart sampling:
    - Method 0 ("uniform"): Full hypercube (fallback)
    - Method A ("expanded_shape"): Centroid-based α-shape expansion
    - Method B ("gaussian"): Gaussian perturbation of admissible points

    Args:
        boundary: AlphaBoundary object
        num: Number of samples
        config: Configuration dict
        device: Torch device

    Returns:
        Tensor (num, 5): Sampled states
    """
    return boundary.sample_candidates(num, device=device)


# ==================== Level 1: Domain Stabilization ====================

def run_level1_domain_stabilization(model, boundary, config, device, iteration_idx):
    """
    Level 1 (Inner Loop): Domain Consistency.

    Per Document Section 4.6:
    1. Sample candidates using configured method (uniform/expanded_shape/gaussian)
    2. Compute s_1 = f(s_0, π(s_0)) using current policy
    3. Evaluate admissibility A(s_0) based on whether s_1 ∈ S_α
    4. Update α-shape boundary with points where A(s_0) ≥ threshold

    ADAPTIVE SAMPLING:
    Instead of fixed inner_steps with fixed samples, this function uses a
    do-while loop to keep sampling until we accumulate at least
    `min_boundary_samples` valid points for the α-shape estimation.

    Config parameters:
    - min_boundary_samples: Target number of valid points for boundary update
    - batch_sample_size: How many candidates to sample each loop iteration

    Returns:
        valid_points: Tensor of valid next-states for visualization
    """
    # Hyperparameters for adaptive sampling
    min_boundary = config['boundary'].get('min_boundary_samples', 10000)
    batch_sample_size = config['boundary'].get('batch_boundary_sample_size', 100000)
    threshold = config['boundary']['admissibility_threshold']
    sampling_method = config['boundary'].get('sampling_method', 'uniform')
    expansion_percent = config['boundary'].get('expansion_percent', 0.20)

    print(f"\n>> Level 1: Domain Stabilization (adaptive sampling)")
    print(f"     Target: {min_boundary} valid points for boundary")
    print(f"     Batch size: {batch_sample_size} candidates per round")
    print(f"     Sampling method: {sampling_method}", end="")
    if sampling_method != 'uniform':
        print(f" (expansion: {expansion_percent*100:.0f}%)")
    else:
        print()

    # Accumulator for valid points
    all_valid_states = []
    all_valid_scores = []
    total_candidates = 0
    total_valid = 0
    round_num = 0
    valid_points_for_plot = None

    model.eval()
    with torch.no_grad():
        # Do-while loop: keep sampling until we have enough valid points
        while True:
            round_num += 1

            # Step 1: Sample candidates using configured method
            candidates = generate_candidates(boundary, batch_sample_size, config, device)
            total_candidates += batch_sample_size

            # Step 2: Forward pass to get next states
            out = model.forward_physics(candidates)
            if out is None:
                print(f"     Round {round_num}: Forward pass failed (NaN), retrying...")
                continue

            # Step 3: Compute admissibility scores
            # A(s_0) depends on whether s_1 is in current S_α
            scores = model.compute_admissibility(out['physics'], boundary=boundary)

            # Step 4: Filter valid points
            valid_mask = (scores >= threshold).squeeze()
            n_valid_this_round = valid_mask.sum().item()
            total_valid += n_valid_this_round
            pass_rate = 100 * n_valid_this_round / batch_sample_size

            # Accumulate valid points
            if n_valid_this_round > 0:
                valid_states = candidates[valid_mask]
                valid_scores_batch = scores[valid_mask]
                all_valid_states.append(valid_states)
                all_valid_scores.append(valid_scores_batch)
                # Keep for plotting
                valid_points_for_plot = out['next_state'][valid_mask]

            print(f"     Round {round_num}: {n_valid_this_round}/{batch_sample_size} valid "
                  f"({pass_rate:.1f}%) | Accumulated: {total_valid}/{min_boundary}")

            # Check if we have enough
            if total_valid >= min_boundary:
                break

            # Safety check: warn if pass rate is very low
            if round_num >= 5 and total_valid < min_boundary * 0.1:
                print(f"     WARNING: Very low pass rate after {round_num} rounds. "
                      f"Consider adjusting parameters.")

    # Concatenate all valid data
    if len(all_valid_states) == 0:
        print("   ! No valid points collected for boundary update.")
        return None

    combined_states = torch.cat(all_valid_states, dim=0)
    combined_scores = torch.cat(all_valid_scores, dim=0)

    # Trim to exactly min_boundary if we overshot
    if len(combined_states) > min_boundary:
        indices = torch.randperm(len(combined_states))[:min_boundary]
        combined_states = combined_states[indices]
        combined_scores = combined_scores[indices]

    overall_pass_rate = 100 * total_valid / total_candidates
    print(f"   Summary: Collected {len(combined_states)} points from {total_candidates} "
          f"candidates ({overall_pass_rate:.1f}% pass rate) in {round_num} rounds")

    # Update boundary with all accumulated valid points
    boundary.update(combined_states, combined_scores, threshold=threshold)

    # Log boundary statistics
    stats = boundary.get_boundary_stats()
    print(f"   Boundary: {stats['n_points']} points, "
          f"{stats['n_alpha_simplices']}/{stats['n_simplices']} α-simplices")

    return valid_points_for_plot


# ==================== Level 2: Dataset Preparation ====================

def prepare_level2_dataset(model, boundary, config, device):
    """
    Prepare training dataset D_train for Level 2 (policy optimization).

    D_train = {s_0 ∈ S_cand | A(s_0) ≥ A_admissible}

    ADAPTIVE SAMPLING:
    Instead of sampling a fixed number of candidates, this function uses a
    do-while loop to keep sampling until we accumulate at least
    `min_admissible_samples` admissible points. This ensures:
    - Training always has enough data regardless of pass rate
    - Efficient when pass rate is high (stops early)
    - Robust when pass rate is low (keeps sampling)

    The α-shape boundary is taken as GIVEN during this stage.

    Config parameters:
    - min_admissible_samples: Target number of admissible points to collect
    - batch_sample_size: How many candidates to sample each loop iteration

    Returns:
        DataLoader or None if insufficient valid data after many attempts
    """
    # Hyperparameters for adaptive sampling
    min_admissible = config['training'].get('min_admissible_samples', 50000)
    batch_sample_size = config['training'].get('batch_sample_size', 100000)
    threshold = config['boundary']['admissibility_threshold']
    batch_size = config['training']['batch_size']
    sampling_method = config['boundary'].get('sampling_method', 'uniform')

    print(f"   Generating Level 2 Data (adaptive sampling):")
    print(f"     Target: {min_admissible} admissible points")
    print(f"     Batch size: {batch_sample_size} candidates per round")
    print(f"     Sampling method: {sampling_method}")

    model.eval()

    # Accumulator for valid points
    all_valid_data = []
    total_candidates = 0
    total_valid = 0
    round_num = 0

    with torch.no_grad():
        # Do-while loop: keep sampling until we have enough admissible points
        while True:
            round_num += 1

            # Sample a batch of candidates
            candidates = generate_candidates(boundary, batch_sample_size, config, device)
            total_candidates += batch_sample_size

            # Forward pass to compute next states
            out = model.forward_physics(candidates)
            if out is None:
                print(f"     Round {round_num}: Forward pass failed (NaN), retrying...")
                continue

            # Compute admissibility scores (α-shape is taken as given)
            scores = model.compute_admissibility(out['physics'], boundary=boundary)
            valid_mask = (scores >= threshold).squeeze()
            valid_data = candidates[valid_mask]

            n_valid_this_round = len(valid_data)
            total_valid += n_valid_this_round
            pass_rate = 100 * n_valid_this_round / batch_sample_size

            # Accumulate valid points
            if n_valid_this_round > 0:
                all_valid_data.append(valid_data)

            print(f"     Round {round_num}: {n_valid_this_round}/{batch_sample_size} valid "
                  f"({pass_rate:.1f}%) | Accumulated: {total_valid}/{min_admissible}")

            # Check if we have enough
            if total_valid >= min_admissible:
                break

            # Safety check: warn if pass rate is very low
            if round_num >= 5 and total_valid < min_admissible * 0.1:
                print(f"     WARNING: Very low pass rate after {round_num} rounds. "
                      f"Consider adjusting parameters.")

    # Concatenate all valid data
    if len(all_valid_data) == 0:
        print("   ! No valid data collected.")
        return None

    combined_valid_data = torch.cat(all_valid_data, dim=0)

    # Trim to exactly min_admissible if we overshot
    if len(combined_valid_data) > min_admissible:
        # Random subset to avoid bias toward later rounds
        indices = torch.randperm(len(combined_valid_data))[:min_admissible]
        combined_valid_data = combined_valid_data[indices]

    overall_pass_rate = 100 * total_valid / total_candidates
    print(f"   Summary: Collected {len(combined_valid_data)} points from {total_candidates} "
          f"candidates ({overall_pass_rate:.1f}% overall pass rate) in {round_num} rounds")

    if len(combined_valid_data) < batch_size:
        print("   ! Not enough valid data to form a batch.")
        return None

    return DataLoader(TensorDataset(combined_valid_data), batch_size=batch_size, shuffle=True)


# ==================== Loss Tracking ====================

class LossTracker:
    """
    Track losses at multiple granularities for visualization and debugging.

    Tracks BOTH:
    - Weighted penalties (what the optimizer sees)
    - Raw penalties (actual violation magnitudes, for interpretability)

    This allows plotting raw violations to see if constraints are being satisfied,
    independent of the lambda weighting.

    NEW: Also tracks Euler discrepancies and asset choices for each type.
    """

    def __init__(self):
        # Per-iteration averages (WEIGHTED - used in optimization)
        self.iter_critic = []
        self.iter_actor_total = []
        self.iter_actor_val = []
        self.iter_actor_fb = []
        self.iter_actor_bound = []
        self.iter_actor_shape = []

        # Per-iteration averages (RAW - actual violation magnitudes)
        self.iter_raw_fb = []        # Mean |Φ|² (unweighted)
        self.iter_raw_bound = []     # Mean bound violation² (unweighted)
        self.iter_raw_shape = []     # Mean projection distance² (unweighted)

        # NEW: Euler discrepancy, FB residual, and asset tracking (per iteration)
        self.iter_phi_e = []         # Mean Euler discrepancy (employed)
        self.iter_phi_u = []         # Mean Euler discrepancy (unemployed)
        self.iter_fb_e = []          # Mean raw FB residual (employed), before squaring
        self.iter_fb_u = []          # Mean raw FB residual (unemployed), before squaring
        self.iter_a_prime_e = []     # Mean next-period assets (employed)
        self.iter_a_prime_u = []     # Mean next-period assets (unemployed)

        # Per-epoch within current iteration
        self.epoch_critic = []
        self.epoch_actor_total = []
        self.epoch_actor_val = []
        self.epoch_actor_fb = []
        self.epoch_actor_bound = []
        self.epoch_actor_shape = []

        # Raw values for current iteration
        self.epoch_raw_fb = []
        self.epoch_raw_bound = []
        self.epoch_raw_shape = []

        # NEW: Epoch-level Euler/FB/asset tracking
        self.epoch_phi_e = []
        self.epoch_phi_u = []
        self.epoch_fb_e = []         # Raw FB residuals (before squaring)
        self.epoch_fb_u = []
        self.epoch_a_prime_e = []
        self.epoch_a_prime_u = []

        # Diagnostics
        self.fb_violations = []
        self.Q_stats = []
        self.projection_stats = []
        self.lambda_fb_history = []  # Track lambda_fb over iterations

        # NEW: Timing tracking
        self.iter_timing = []  # List of dicts with timing info per iteration

    def start_iteration(self):
        """Reset epoch-level accumulators for new iteration."""
        self.epoch_critic = []
        self.epoch_actor_total = []
        self.epoch_actor_val = []
        self.epoch_actor_fb = []
        self.epoch_actor_bound = []
        self.epoch_actor_shape = []
        self.epoch_raw_fb = []
        self.epoch_raw_bound = []
        self.epoch_raw_shape = []
        # NEW
        self.epoch_phi_e = []
        self.epoch_phi_u = []
        self.epoch_fb_e = []
        self.epoch_fb_u = []
        self.epoch_a_prime_e = []
        self.epoch_a_prime_u = []

    def log_critic_epoch(self, loss):
        self.epoch_critic.append(loss)

    def log_actor_epoch(self, val, fb, bound, shape, total,
                        raw_fb=None, raw_bound=None, raw_shape=None,
                        phi_e=None, phi_u=None, fb_e=None, fb_u=None,
                        a_prime_e=None, a_prime_u=None):
        """
        Log actor losses for one epoch.

        Args:
            val, fb, bound, shape, total: Weighted loss components
            raw_fb, raw_bound, raw_shape: Raw (unweighted) penalty values
            phi_e, phi_u: Euler discrepancies for employed/unemployed
            fb_e, fb_u: Raw FB residuals (before squaring) for employed/unemployed
            a_prime_e, a_prime_u: Next-period asset choices
        """
        self.epoch_actor_val.append(val)
        self.epoch_actor_fb.append(fb)
        self.epoch_actor_bound.append(bound)
        self.epoch_actor_shape.append(shape)
        self.epoch_actor_total.append(total)

        # Raw values (if provided)
        if raw_fb is not None:
            self.epoch_raw_fb.append(raw_fb)
        if raw_bound is not None:
            self.epoch_raw_bound.append(raw_bound)
        if raw_shape is not None:
            self.epoch_raw_shape.append(raw_shape)

        # NEW: Euler discrepancies, FB residuals, and assets
        if phi_e is not None:
            self.epoch_phi_e.append(phi_e)
        if phi_u is not None:
            self.epoch_phi_u.append(phi_u)
        if fb_e is not None:
            self.epoch_fb_e.append(fb_e)
        if fb_u is not None:
            self.epoch_fb_u.append(fb_u)
        if a_prime_e is not None:
            self.epoch_a_prime_e.append(a_prime_e)
        if a_prime_u is not None:
            self.epoch_a_prime_u.append(a_prime_u)

    def end_iteration(self, fb_violation=None, Q_mean=None, Q_std=None,
                      n_projected=None, mean_proj_dist=None, lambda_fb=None,
                      timing=None):
        """Compute iteration averages from epoch data."""
        if self.epoch_critic:
            self.iter_critic.append(np.mean(self.epoch_critic))
        if self.epoch_actor_total:
            self.iter_actor_total.append(np.mean(self.epoch_actor_total))
            self.iter_actor_val.append(np.mean(self.epoch_actor_val))
            self.iter_actor_fb.append(np.mean(self.epoch_actor_fb))
            self.iter_actor_bound.append(np.mean(self.epoch_actor_bound))
            self.iter_actor_shape.append(np.mean(self.epoch_actor_shape))

        # Raw penalties
        if self.epoch_raw_fb:
            self.iter_raw_fb.append(np.mean(self.epoch_raw_fb))
        if self.epoch_raw_bound:
            self.iter_raw_bound.append(np.mean(self.epoch_raw_bound))
        if self.epoch_raw_shape:
            self.iter_raw_shape.append(np.mean(self.epoch_raw_shape))

        # NEW: Euler discrepancies, FB residuals, and assets
        if self.epoch_phi_e:
            self.iter_phi_e.append(np.mean(self.epoch_phi_e))
        if self.epoch_phi_u:
            self.iter_phi_u.append(np.mean(self.epoch_phi_u))
        if self.epoch_fb_e:
            self.iter_fb_e.append(np.mean(self.epoch_fb_e))
        if self.epoch_fb_u:
            self.iter_fb_u.append(np.mean(self.epoch_fb_u))
        if self.epoch_a_prime_e:
            self.iter_a_prime_e.append(np.mean(self.epoch_a_prime_e))
        if self.epoch_a_prime_u:
            self.iter_a_prime_u.append(np.mean(self.epoch_a_prime_u))

        if fb_violation is not None:
            self.fb_violations.append(fb_violation)
        if Q_mean is not None:
            self.Q_stats.append((Q_mean, Q_std))
        if n_projected is not None:
            self.projection_stats.append((n_projected, mean_proj_dist))
        if lambda_fb is not None:
            self.lambda_fb_history.append(lambda_fb)

        # NEW: Timing
        if timing is not None:
            self.iter_timing.append(timing)

    def get_iter_metrics(self):
        """Return dict for visualization module (weighted penalties)."""
        return {
            'val': self.iter_actor_val,
            'fb': self.iter_actor_fb,
            'bound': self.iter_actor_bound,
            'shape': self.iter_actor_shape,
            'total': self.iter_actor_total
        }

    def get_raw_metrics(self):
        """Return dict of raw (unweighted) penalty values."""
        return {
            'fb': self.iter_raw_fb,
            'bound': self.iter_raw_bound,
            'shape': self.iter_raw_shape
        }

    def get_euler_asset_metrics(self):
        """Return dict of Euler discrepancies, FB residuals, and asset choices."""
        return {
            'phi_e': self.iter_phi_e,
            'phi_u': self.iter_phi_u,
            'fb_e': self.iter_fb_e,
            'fb_u': self.iter_fb_u,
            'a_prime_e': self.iter_a_prime_e,
            'a_prime_u': self.iter_a_prime_u
        }

    def get_timing_metrics(self):
        """Return list of timing dicts per iteration."""
        return self.iter_timing

    def get_epoch_metrics(self):
        """Return epoch-level data for detailed plots."""
        return {
            'critic': self.epoch_critic,
            'actor_val': self.epoch_actor_val,
            'actor_fb': self.epoch_actor_fb,
            'actor_bound': self.epoch_actor_bound,
            'actor_shape': self.epoch_actor_shape,
            'actor_total': self.epoch_actor_total
        }


# ==================== Main Training Function ====================

def train():
    """
    Main training loop implementing the two-level fixed-point iteration.

    ALGORITHM OVERVIEW:

    For each global iteration:
        LEVEL 1 (Domain Stabilization):
            - Sample candidates from hypercube
            - Filter to those where next-state is admissible
            - Update α-shape boundary

        LEVEL 2 (Policy Optimization):
            - Prepare dataset from admissible starting points
            - Train critic via TD learning (with target network)
            - Train actor to maximize welfare minus penalties

            ROLLOUT DETAILS (Issue #3 & #4):
            - For each step t in rollout:
                - Compute s_{t+1} = f(s_t, π(s_t))
                - Check if s_{t+1} ∈ S_α
                - If NOT: project to nearest admissible point
                - Add projection distance to penalty
                - Continue rollout from PROJECTED state
    """
    # ==================== System Check ====================
    check_gpu()

    # ==================== Setup ====================
    config = load_config_json()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==================== Check for Existing Files ====================
    # Warn if output files already exist (will be overwritten)
    check_existing_output_files(config)

    # Optional: Create backups of existing files
    # Uncomment the next line to enable automatic backups
    # backup_existing_files(config)

    print("="*60)
    print("   HETEROGENEOUS AGENT RAMSEY SOLVER")
    print("="*60)
    print(f"   Device: {device}")
    print(f"   Q bounds: [{config['admissibility']['Q_min']}, "
          f"{config['admissibility']['Q_max']}]")

    # Initialize components
    model = HAModel(config, device)
    boundary = AlphaBoundary(config)

    # Get version for figure naming
    version_prefix = config.get('versioning', {}).get('output_version', None)
    viz = HAVisualizer(config, device, save_dir=FIGURES_DIR, version_prefix=version_prefix)
    tracker = LossTracker()

    # ==================== Issue #1: Model Versioning ====================
    print("\n>> Checking for existing model...")
    model_loaded = load_model_if_exists(model, config, device)
    boundary_loaded = load_boundary_if_exists(boundary, config)

    if model_loaded:
        print("   Resuming training from checkpoint.")
    else:
        print("   Starting fresh training.")

    # Optimizers
    opt_actor = optim.Adam(model.actor.parameters(), lr=config['training']['lr_actor'])
    opt_critic = optim.Adam(model.critic.parameters(), lr=config['training']['lr_critic'])

    # ==================== Training Hyperparameters ====================
    num_global_iters = config['training']['num_iterations']
    rollout_steps = config['training']['rollout_steps']
    epochs_per_iter = config['training'].get('epochs_per_batch', 3)

    # Penalty weights
    lambda_bound = config['training'].get('lambda_boundary', 10.0)  # Hypercube violation
    lambda_shape = config['training'].get('lambda_shape', 5.0)      # α-shape projection
    lambda_fb = config['fischer_burmeister']['lambda_initial']
    lambda_max = config['fischer_burmeister']['lambda_max']
    rho = config['fischer_burmeister']['rho']

    # Target network update
    target_update_freq = config['training'].get('target_update_freq', 1)
    tau = config['training'].get('tau', 0.005)

    print(f"\n   Training Configuration:")
    print(f"   - Iterations: {num_global_iters}")
    print(f"   - Epochs/Iter: {epochs_per_iter}")
    print(f"   - Rollout steps: {rollout_steps}")
    print(f"   - λ_fb: {lambda_fb} (max {lambda_max})")
    print(f"   - λ_bound (hypercube): {lambda_bound}")
    print(f"   - λ_shape (α-shape): {lambda_shape}")
    print("="*60)

    # ==================== Global Training Loop ====================
    for global_iter in range(num_global_iters):
        print(f"\n{'='*60}")
        print(f"  ITERATION {global_iter+1}/{num_global_iters}")
        print(f"{'='*60}")

        # NEW: Start timing for this iteration
        iter_timing = {
            'level1': 0.0,
            'data_prep': 0.0,
            'critic_training': 0.0,
            'actor_training': 0.0,
            'visualization': 0.0,
            'total': 0.0
        }
        iter_start_time = time.time()

        tracker.start_iteration()

        # ============================================================
        # LEVEL 1: DOMAIN STABILIZATION
        # ============================================================
        level1_start = time.time()
        valid_plot_points = run_level1_domain_stabilization(
            model, boundary, config, device, global_iter
        )
        iter_timing['level1'] = time.time() - level1_start

        # ============================================================
        # LEVEL 2: POLICY OPTIMIZATION
        # ============================================================
        print("\n>> Level 2: Policy Optimization")

        data_prep_start = time.time()
        train_loader = prepare_level2_dataset(model, boundary, config, device)
        iter_timing['data_prep'] = time.time() - data_prep_start

        if train_loader is None:
            print("   Skipping optimization (no valid data)")
            iter_timing['total'] = time.time() - iter_start_time
            tracker.end_iteration(timing=iter_timing)
            continue

        model.train()

        # Accumulators for diagnostics
        iter_fb_violation = 0
        iter_Q_vals = []
        iter_n_projected = 0

        # --------------------------------------------------------
        # PHASE A: CRITIC UPDATE
        # --------------------------------------------------------
        critic_start = time.time()
        print(f"   > Training Critic ({epochs_per_iter} epochs)...")

        for epoch in range(epochs_per_iter):
            epoch_critic_loss = 0
            n_batches = 0

            for (batch_s,) in train_loader:
                batch_s = batch_s.to(device)
                n_batches += 1

                # Compute TD target using TARGET network
                with torch.no_grad():
                    curr = batch_s.clone()
                    reward = torch.zeros(curr.size(0), 1, device=device)

                    for t in range(rollout_steps):
                        out = model.forward_physics(curr)
                        if out is None:
                            break
                        reward += (model.beta ** t) * out['welfare']

                        # Get next state (with projection for critic target)
                        next_state = out['next_state']
                        projected, _, _ = boundary.project_to_admissible(next_state)
                        curr = projected

                    # Bootstrap value from TARGET critic
                    target = reward + (model.beta ** rollout_steps) * model.critic_target(curr)

                # Critic loss: MSE to TD target
                pred = model.critic(batch_s)
                loss_c = F.mse_loss(pred, target.detach())

                opt_critic.zero_grad()
                loss_c.backward()
                torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 1.0)
                opt_critic.step()

                epoch_critic_loss += loss_c.item()

            avg_epoch_loss = epoch_critic_loss / max(n_batches, 1)
            tracker.log_critic_epoch(avg_epoch_loss)

        # Soft update target network
        if (global_iter + 1) % target_update_freq == 0:
            model.soft_update_critic_target(tau=tau)

        iter_timing['critic_training'] = time.time() - critic_start

        # --------------------------------------------------------
        # PHASE B: ACTOR UPDATE
        # --------------------------------------------------------
        actor_start = time.time()
        print(f"   > Training Actor ({epochs_per_iter} epochs)...")

        for epoch in range(epochs_per_iter):
            epoch_val = 0
            epoch_fb = 0
            epoch_bound = 0
            epoch_shape = 0
            epoch_total = 0
            epoch_raw_fb = 0
            epoch_raw_bound = 0
            epoch_raw_shape = 0
            # Euler/FB/asset accumulators
            epoch_phi_e = 0
            epoch_phi_u = 0
            epoch_fb_e = 0
            epoch_fb_u = 0
            epoch_a_prime_e = 0
            epoch_a_prime_u = 0
            n_batches = 0

            for (batch_s,) in train_loader:
                batch_s = batch_s.to(device)
                n_batches += 1
                batch_size = batch_s.size(0)

                curr = batch_s.clone()
                reward_sum = torch.zeros(batch_size, 1, device=device)
                fb_sum = torch.zeros(batch_size, 1, device=device)
                bound_penalty_sum = torch.zeros(batch_size, 1, device=device)
                shape_penalty_sum = torch.zeros(batch_size, 1, device=device)

                valid_rollout = True
                for t in range(rollout_steps):
                    out = model.forward_physics(curr)
                    if out is None:
                        valid_rollout = False
                        break

                    # Accumulate discounted welfare
                    reward_sum += (model.beta ** t) * out['welfare']

                    # Fischer-Burmeister residuals
                    fb_e, fb_u = out['fb_residuals']
                    fb_sum += fb_e**2 + fb_u**2

                    # Hypercube penalty
                    ae_raw = out['physics']['a_prime_e_raw']
                    au_raw = out['physics']['a_prime_u_raw']
                    p_ae = F.relu(ae_raw - model.a_max) + F.relu(model.a_min - ae_raw)
                    p_au = F.relu(au_raw - model.a_max) + F.relu(model.a_min - au_raw)
                    bound_penalty_sum += (p_ae**2 + p_au**2)

                    # Track first step diagnostics only
                    if t == 0:
                        iter_Q_vals.append(out['physics']['Q'].mean().item())
                        epoch_phi_e += out['physics']['phi_e'].mean().item()
                        epoch_phi_u += out['physics']['phi_u'].mean().item()
                        epoch_fb_e += fb_e.mean().item()
                        epoch_fb_u += fb_u.mean().item()
                        epoch_a_prime_e += ae_raw.mean().item()
                        epoch_a_prime_u += au_raw.mean().item()

                    # HARD PROJECTION TO S_α
                    next_state = out['next_state']
                    projected_state, proj_dist, was_inside = boundary.project_to_admissible(next_state)

                    shape_penalty_sum += proj_dist**2
                    iter_n_projected += (~was_inside).sum().item()

                    curr = projected_state

                if not valid_rollout:
                    continue

                # Value from critic
                value = reward_sum + (model.beta ** rollout_steps) * model.critic(curr)

                # ACTOR LOSS
                raw_fb = fb_sum.mean()
                raw_bound = bound_penalty_sum.mean()
                raw_shape = shape_penalty_sum.mean()

                loss_val = -value.mean()
                loss_fb = lambda_fb * raw_fb
                loss_bound = lambda_bound * raw_bound
                loss_shape = lambda_shape * raw_shape

                loss_actor = loss_val + loss_fb + loss_bound + loss_shape

                opt_actor.zero_grad()
                loss_actor.backward()
                torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 0.5)
                opt_actor.step()

                # Accumulate stats
                epoch_val += loss_val.item()
                epoch_fb += loss_fb.item()
                epoch_bound += loss_bound.item()
                epoch_shape += loss_shape.item()
                epoch_total += loss_actor.item()
                epoch_raw_fb += raw_fb.item()
                epoch_raw_bound += raw_bound.item()
                epoch_raw_shape += raw_shape.item()

            # Log epoch averages
            if n_batches > 0:
                tracker.log_actor_epoch(
                    epoch_val / n_batches,
                    epoch_fb / n_batches,
                    epoch_bound / n_batches,
                    epoch_shape / n_batches,
                    epoch_total / n_batches,
                    raw_fb=epoch_raw_fb / n_batches,
                    raw_bound=epoch_raw_bound / n_batches,
                    raw_shape=epoch_raw_shape / n_batches,
                    phi_e=epoch_phi_e / n_batches,
                    phi_u=epoch_phi_u / n_batches,
                    fb_e=epoch_fb_e / n_batches,
                    fb_u=epoch_fb_u / n_batches,
                    a_prime_e=epoch_a_prime_e / n_batches,
                    a_prime_u=epoch_a_prime_u / n_batches
                )
                iter_fb_violation += epoch_raw_fb / n_batches

        iter_timing['actor_training'] = time.time() - actor_start

        # ========================================================
        # End of Iteration Bookkeeping
        # ========================================================
        iter_fb_violation /= max(epochs_per_iter, 1)
        Q_mean = np.mean(iter_Q_vals) if iter_Q_vals else 0
        Q_std = np.std(iter_Q_vals) if iter_Q_vals else 0

        tracker.end_iteration(
            fb_violation=iter_fb_violation,
            Q_mean=Q_mean,
            Q_std=Q_std,
            n_projected=iter_n_projected,
            mean_proj_dist=0.0,  # Simplified - removed per-step tracking for performance
            lambda_fb=lambda_fb,  # Track lambda_fb history
            timing=iter_timing
        )

        # Adaptive FB penalty update
        if iter_fb_violation > config['fischer_burmeister']['penalty_threshold']:
            lambda_fb = min(lambda_fb * rho, lambda_max)
            print(f"   [Penalty] Increased λ_fb to {lambda_fb:.2f}")

        # Print summary
        metrics = tracker.get_iter_metrics()
        euler_metrics = tracker.get_euler_asset_metrics()

        if metrics['total']:
            print(f"\n   Summary:")
            print(f"     Critic Loss: {tracker.iter_critic[-1]:.4f}")
            print(f"     Actor Total: {metrics['total'][-1]:.4f}")
            print(f"       - Value:   {metrics['val'][-1]:.4f}")
            print(f"       - FB:      {metrics['fb'][-1]:.4f}")
            print(f"       - Bound:   {metrics['bound'][-1]:.4f}")
            print(f"       - Shape:   {metrics['shape'][-1]:.4f}")
            print(f"     Q: {Q_mean:.4f} ± {Q_std:.4f}")
            print(f"     Projections: {iter_n_projected} points")

            # NEW: Print Euler discrepancies, FB residuals, and assets
            if euler_metrics['phi_e']:
                print(f"\n     Euler Discrepancies (φ) - raw Euler eq. violation:")
                print(f"       - φᵉ (employed):   {euler_metrics['phi_e'][-1]:.6f}")
                print(f"       - φᵘ (unemployed): {euler_metrics['phi_u'][-1]:.6f}")
            if euler_metrics.get('fb_e'):
                print(f"     FB Residuals (Φ) - complementarity condition:")
                print(f"       - Φᵉ (employed):   {euler_metrics['fb_e'][-1]:.6f}")
                print(f"       - Φᵘ (unemployed): {euler_metrics['fb_u'][-1]:.6f}")
            if euler_metrics['a_prime_e']:
                print(f"     Asset Choices (a'):")
                print(f"       - a'ᵉ (employed):   {euler_metrics['a_prime_e'][-1]:.4f}")
                print(f"       - a'ᵘ (unemployed): {euler_metrics['a_prime_u'][-1]:.4f}")

            # Log GPU memory usage
            log_gpu_memory(prefix="")

        # NEW: Print timing summary
        iter_timing['total'] = time.time() - iter_start_time
        print(f"\n     Timing (seconds):")
        print(f"       - Level 1 (Domain):    {iter_timing['level1']:.2f}s")
        print(f"       - Data Preparation:    {iter_timing['data_prep']:.2f}s")
        print(f"       - Critic Training:     {iter_timing['critic_training']:.2f}s")
        print(f"       - Actor Training:      {iter_timing['actor_training']:.2f}s")
        print(f"       - Total Iteration:     {iter_timing['total']:.2f}s")

        # Visualization
        viz_start = time.time()
        if (global_iter + 1) % config['training']['plot_frequency'] == 0:
            viz.plot_losses(metrics, tracker.iter_critic, tracker.get_raw_metrics(),
                           euler_asset_metrics=euler_metrics,
                           timing_metrics=tracker.get_timing_metrics())
            viz.plot_boundary_3d(boundary, global_iter + 1, sample_points=valid_plot_points)
            viz.plot_epoch_losses(tracker.get_epoch_metrics(), global_iter + 1)
        iter_timing['visualization'] = time.time() - viz_start

    # ==================== Save & Finish ====================
    print("\n" + "="*60)
    print("   TRAINING COMPLETE")
    print("="*60)

    # Issue #1: Save with version number
    save_model_and_boundary(model, boundary, config)

    # Post-training simulation
    if config['simulation']['run_after_training']:
        run_simulation(model, boundary, config, device, save_dir=RESULTS_DIR)


if __name__ == "__main__":
    train()