# Ramsey_RA_adaptive_sampling_v2.py
"""
Adaptive Sampling Strategy for Ramsey Problem with Endogenous Feasible Set.
VERSION 2: Aligned with document specification.

Key changes from v1:
1. Power barrier scoring function (not piecewise linear)
2. Vectorized batch score computation
3. Corrected A_mu computation (min over branches, not max)
4. Proportional buffer widths
5. Consolidated thresholds (tau_high, tau_low only)
6. Streamlined code structure
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import pandas as pd
from scipy.interpolate import interp1d


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def power_barrier(x: torch.Tensor, x_min: float, x_max: float, 
                  delta: float, kappa: float = 4.0) -> torch.Tensor:
    """
    Distance-based power barrier function.
    
    S(x) = [1 - (dist(x, [x_min, x_max]) / delta)^kappa]^+
    
    Properties:
    - Returns 1.0 when x is strictly within [x_min, x_max]
    - Smoothly decays to 0.0 as x moves delta distance outside bounds
    - Differentiable everywhere except at the zero-crossing
    
    Args:
        x: Input tensor of values to score
        x_min, x_max: Feasible interval bounds
        delta: Buffer width (score reaches 0 at distance delta from bounds)
        kappa: Sharpness parameter (higher = sharper transition)
    
    Returns:
        Tensor of scores in [0, 1]
    """
    # Distance to feasible set: max(0, x_min - x, x - x_max)
    dist_low = torch.clamp(x_min - x, min=0.0)
    dist_high = torch.clamp(x - x_max, min=0.0)
    dist = torch.maximum(dist_low, dist_high)
    
    # Power barrier with clamping
    score = torch.clamp(1.0 - (dist / delta) ** kappa, min=0.0, max=1.0)
    return score


def power_barrier_np(x: np.ndarray, x_min: float, x_max: float,
                     delta: float, kappa: float = 4.0) -> np.ndarray:
    """NumPy version of power barrier for non-tensor operations."""
    dist_low = np.maximum(0.0, x_min - x)
    dist_high = np.maximum(0.0, x - x_max)
    dist = np.maximum(dist_low, dist_high)
    score = np.clip(1.0 - (dist / delta) ** kappa, 0.0, 1.0)
    return score


# =============================================================================
# ADMISSIBILITY SCORER
# =============================================================================

class AdmissibilityScorer:
    """
    Computes admissibility scores for states using vectorized operations.
    
    Score components:
    - A_tau: Tax feasibility of current state
    - A_mu: Policy safety (both shock branches within bounds)
    - A_debt: Next-period debt sustainability (both shock branches)
    
    Global score: A(s) = w_tau * A_tau + w_mu * A_mu + w_debt * A_debt
    """

    def __init__(self, policy_net: nn.Module, config: dict, device: torch.device):
        self.policy_net = policy_net
        self.device = device
        
        # Economic parameters
        econ = config['economic_parameters']
        self.mu_min = econ['mu_min']
        self.mu_max = econ['mu_max']
        self.beta = econ['beta']
        self.gamma_l = econ['gamma_l']
        self.zagg_vec = torch.tensor(econ['zagg_vec'], device=device, dtype=torch.float32).squeeze()
        self.pi_zagg = torch.tensor(econ['pi_zagg'], device=device, dtype=torch.float32)
        self.n_shocks = len(econ['zagg_vec'])
        
        # Feasibility bounds
        bounds = config['feasibility_bounds']
        self.tau_min = bounds['tau_min']
        self.tau_max = bounds['tau_max']
        self.b_min_initial = bounds['b_min_initial']
        self.b_max_initial = bounds['b_max_initial']
        self.b_min = self.b_min_initial
        self.b_max = self.b_max_initial
        
        # Scoring parameters
        scoring = config['scoring']
        self.kappa = scoring['kappa']
        self.w_tau = scoring['weights']['w_tau']
        self.w_mu = scoring['weights']['w_mu']
        self.w_debt = scoring['weights']['w_debt']
        
        # Proportional buffer widths
        buffers = scoring['buffer_proportions']
        self.delta_tau = buffers['delta_tau_prop'] * (self.tau_max - self.tau_min)
        self.delta_mu = buffers['delta_mu_prop'] * (self.mu_max - self.mu_min)
        self.delta_debt_prop = buffers['delta_debt_prop']  # Applied dynamically
        
        # Boundary learning parameters
        boundary = config['boundary_learning']
        self.use_dynamic_bounds = boundary['use_dynamic_bounds']
        self.boundary_method = boundary['method']
        self.quantile_alpha = boundary['quantile_alpha']
        self.n_mu_bins = boundary['n_mu_bins']
        self.min_points_per_bin = boundary['min_points_per_bin']
        self.fallback_band = boundary['fallback_band']
        
        # State-dependent boundary functions: b_min(mu, g), b_max(mu, g)
        self._init_boundary_funcs()
        
        # Cache for scores (optional, for visualization)
        self.cache = {}

    def _init_boundary_funcs(self):
        """Initialize boundary functions to constant initial bounds."""
        self.bound_funcs = {}
        for g in range(self.n_shocks):
            self.bound_funcs[g] = {
                'min': lambda mu, b=self.b_min_initial: b,
                'max': lambda mu, b=self.b_max_initial: b
            }

    def get_debt_bounds(self, mu: torch.Tensor, g_idx: torch.Tensor) -> tuple:
        """
        Get state-dependent debt bounds for given (mu, g) pairs.
        
        Args:
            mu: Tensor of multiplier values [N]
            g_idx: Tensor of shock indices [N]
        
        Returns:
            (b_min, b_max): Tensors of bounds [N]
        """
        N = mu.shape[0]
        b_min = torch.zeros(N, device=self.device)
        b_max = torch.zeros(N, device=self.device)
        
        mu_np = mu.cpu().numpy()
        g_np = g_idx.cpu().numpy().astype(int)
        
        for i in range(N):
            g = int(g_np[i])
            if g not in self.bound_funcs:
                g = 0
            b_min[i] = self.bound_funcs[g]['min'](mu_np[i])
            b_max[i] = self.bound_funcs[g]['max'](mu_np[i])
        
        # Fallback for collapsed bounds
        collapsed = b_max <= b_min + 1e-6
        if collapsed.any():
            mid = (b_min[collapsed] + b_max[collapsed]) / 2
            b_min[collapsed] = mid - self.fallback_band
            b_max[collapsed] = mid + self.fallback_band
        
        return b_min, b_max

    def compute_score_batch(self, states: torch.Tensor) -> torch.Tensor:
        """
        Vectorized computation of admissibility scores.
        
        Args:
            states: Tensor of shape [N, 3] with columns (B, mu, g_idx)
        
        Returns:
            Tensor of scores [N] in [0, 1]
        """
        self.policy_net.eval()
        N = states.shape[0]
        
        with torch.no_grad():
            B = states[:, 0]
            mu = states[:, 1]
            g_idx = states[:, 2].long()
            
            # Get g values for each state
            g_val = self.zagg_vec[g_idx]
            
            # === 1. TAX FEASIBILITY (A_tau) ===
            c = 1.0 / mu
            x = c + g_val  # total resources used
            l = 1.0 - x    # leisure
            tau = 1.0 - self.gamma_l * c / (l + 1e-8)
            
            A_tau = power_barrier(tau, self.tau_min, self.tau_max, 
                                  self.delta_tau, self.kappa)
            
            # === 2. POLICY SAFETY (A_mu) ===
            # Get policy outputs for both shock branches
            state_input = torch.stack([B, mu, g_val], dim=1)
            policy_logits = self.policy_net(state_input)
            
            mu_next_g0 = torch.sigmoid(policy_logits[:, 0]) * (self.mu_max - self.mu_min) + self.mu_min
            mu_next_g1 = torch.sigmoid(policy_logits[:, 1]) * (self.mu_max - self.mu_min) + self.mu_min
            
            # Score each branch, take minimum (safe in BOTH futures)
            A_mu_g0 = power_barrier(mu_next_g0, self.mu_min, self.mu_max,
                                    self.delta_mu, self.kappa)
            A_mu_g1 = power_barrier(mu_next_g1, self.mu_min, self.mu_max,
                                    self.delta_mu, self.kappa)
            A_mu = torch.minimum(A_mu_g0, A_mu_g1)
            
            # === 3. DEBT SUSTAINABILITY (A_debt) ===
            # Compute expected next-period multiplier and bond price
            E_mu_next = (self.pi_zagg[g_idx, 0] * mu_next_g0 + 
                         self.pi_zagg[g_idx, 1] * mu_next_g1)
            q = self.beta * E_mu_next / mu
            
            # Compute next-period debt (clamped tau for calculation)
            tau_clamped = torch.clamp(tau, self.tau_min, self.tau_max)
            B_next = (B + g_val - tau_clamped * x) / (q + 1e-8)
            
            # Check debt sustainability for BOTH future shock branches
            # Get bounds for future states (mu', g')
            b_min_g0, b_max_g0 = self.get_debt_bounds(mu_next_g0, torch.zeros_like(g_idx))
            b_min_g1, b_max_g1 = self.get_debt_bounds(mu_next_g1, torch.ones_like(g_idx))
            
            # Proportional buffer based on current bound width
            delta_debt_g0 = self.delta_debt_prop * (b_max_g0 - b_min_g0 + 1e-6)
            delta_debt_g1 = self.delta_debt_prop * (b_max_g1 - b_min_g1 + 1e-6)
            
            A_debt_g0 = power_barrier(B_next, b_min_g0, b_max_g0, delta_debt_g0, self.kappa)
            A_debt_g1 = power_barrier(B_next, b_min_g1, b_max_g1, delta_debt_g1, self.kappa)
            A_debt = torch.minimum(A_debt_g0, A_debt_g1)
            
            # === AGGREGATE SCORE ===
            score = self.w_tau * A_tau + self.w_mu * A_mu + self.w_debt * A_debt
            
        return score

    def compute_score(self, B: float, mu: float, g_idx: int) -> float:
        """Single-state score computation (wrapper for batch version)."""
        states = torch.tensor([[B, mu, g_idx]], device=self.device, dtype=torch.float32)
        return self.compute_score_batch(states).item()

    def update_cache(self, states: torch.Tensor, scores: torch.Tensor = None):
        """Update score cache (for visualization)."""
        if scores is None:
            scores = self.compute_score_batch(states)
        
        for i in range(states.shape[0]):
            key = (round(states[i, 0].item(), 4),
                   round(states[i, 1].item(), 4),
                   int(states[i, 2].item()))
            self.cache[key] = scores[i].item()

    def clear_cache(self):
        """Clear score cache and reset boundary functions."""
        self.cache = {}
        self._init_boundary_funcs()

    def update_boundary_functions(self, admissible_states: torch.Tensor):
        """
        Update state-dependent debt bounds from admissible points.
        
        Uses quantile-based estimation within mu bins to handle outliers.
        """
        if not self.use_dynamic_bounds:
            return
        
        if admissible_states.shape[0] < 100:
            print(f"  [Boundary] Skipping update: insufficient points ({admissible_states.shape[0]})")
            return
        
        # Convert to numpy for binning
        data = admissible_states.cpu().numpy()
        df = pd.DataFrame(data, columns=['b', 'mu', 'g'])
        
        mu_bins = np.linspace(self.mu_min, self.mu_max, self.n_mu_bins + 1)
        
        for g_val in range(self.n_shocks):
            df_g = df[df['g'] == g_val]
            if len(df_g) < 50:
                continue
            
            df_g = df_g.copy()
            df_g['mu_bin'] = pd.cut(df_g['mu'], bins=mu_bins, labels=False, include_lowest=True)
            
            # Compute quantile bounds per bin
            grouped = df_g.groupby('mu_bin')['b']
            
            if self.boundary_method == 'quantile':
                stats_min = grouped.quantile(self.quantile_alpha)
                stats_max = grouped.quantile(1.0 - self.quantile_alpha)
            else:
                stats_min = grouped.min()
                stats_max = grouped.max()
            
            stats_count = grouped.count()
            
            # Build interpolation points
            x_points, y_min, y_max = [], [], []
            
            for bin_idx in range(self.n_mu_bins):
                if bin_idx in stats_count.index and stats_count[bin_idx] >= self.min_points_per_bin:
                    mu_center = (mu_bins[bin_idx] + mu_bins[bin_idx + 1]) / 2
                    x_points.append(mu_center)
                    y_min.append(stats_min[bin_idx])
                    y_max.append(stats_max[bin_idx])
            
            if len(x_points) < 3:
                print(f"  [Boundary] G={g_val}: insufficient bin coverage, keeping previous bounds")
                continue
            
            # Extend to edges
            if x_points[0] > self.mu_min:
                x_points.insert(0, self.mu_min)
                y_min.insert(0, y_min[0])
                y_max.insert(0, y_max[0])
            if x_points[-1] < self.mu_max:
                x_points.append(self.mu_max)
                y_min.append(y_min[-1])
                y_max.append(y_max[-1])
            
            # Create interpolation functions
            try:
                self.bound_funcs[g_val]['min'] = interp1d(x_points, y_min, kind='linear', 
                                                          fill_value='extrapolate')
                self.bound_funcs[g_val]['max'] = interp1d(x_points, y_max, kind='linear',
                                                          fill_value='extrapolate')
                
                # Update global bounds (for reference)
                self.b_min = min(self.b_min, min(y_min))
                self.b_max = max(self.b_max, max(y_max))
                
                print(f"  [Boundary] G={g_val}: updated from {len(x_points)} bins, "
                      f"b ∈ [{min(y_min):.3f}, {max(y_max):.3f}]")
            except Exception as e:
                print(f"  [Boundary] G={g_val}: interpolation failed: {e}")


# =============================================================================
# ADAPTIVE SAMPLER
# =============================================================================

class AdaptiveSampler:
    """
    Adaptive sampling strategy with boundary-focused exploration.
    
    Partitions state space into:
    - Safe: A(s) > tau_high (used for optimality training)
    - Boundary: tau_low <= A(s) <= tau_high (oversampled)
    - Infeasible: A(s) < tau_low (used for penalty training)
    """

    def __init__(self, scorer: AdmissibilityScorer, config: dict, device: torch.device):
        self.scorer = scorer
        self.device = device
        
        # Bounds
        bounds = config['feasibility_bounds']
        self.b_min = bounds['b_min_initial']
        self.b_max = bounds['b_max_initial']
        self.mu_min = config['economic_parameters']['mu_min']
        self.mu_max = config['economic_parameters']['mu_max']
        self.n_shocks = len(config['economic_parameters']['zagg_vec'])
        
        # Thresholds
        thresholds = config['scoring']['thresholds']
        self.tau_high = thresholds['tau_high']
        self.tau_low = thresholds['tau_low']
        
        # Sampling parameters
        sampling = config['sampling']
        self.candidate_multiplier = sampling['candidate_multiplier']
        self.perturbation_std = sampling['perturbation_std']
        
        # State
        self.phase = 'warmup'
        self.iteration = 0

    def set_phase(self, phase: str):
        """Set sampling phase ('warmup' or 'adaptive')."""
        self.phase = phase
        self.iteration = 0
        print(f"[Sampler] Phase set to: {phase.upper()}")

    def sample_uniform(self, n: int, use_initial_bounds: bool = True) -> torch.Tensor:
        """Generate uniform samples over state space."""
        samples = torch.zeros((n, 3), device=self.device)
        
        if use_initial_bounds:
            b_lo, b_hi = self.b_min, self.b_max
        else:
            b_lo, b_hi = self.scorer.b_min, self.scorer.b_max
        
        samples[:, 0] = torch.rand(n, device=self.device) * (b_hi - b_lo) + b_lo
        samples[:, 1] = torch.rand(n, device=self.device) * (self.mu_max - self.mu_min) + self.mu_min
        samples[:, 2] = torch.randint(0, self.n_shocks, (n,), device=self.device).float()
        
        return samples

    def sample_from_history(self, history: list, n: int) -> tuple:
        """Sample from simulation history with perturbation."""
        if len(history) == 0:
            return self.sample_uniform(n), None
        
        all_data = torch.cat(history, dim=0)
        
        # Filter to valid range
        mask = ((all_data[:, 0] >= self.b_min) & (all_data[:, 0] <= self.b_max) &
                (all_data[:, 1] >= self.mu_min) & (all_data[:, 1] <= self.mu_max))
        valid_data = all_data[mask]
        
        if len(valid_data) == 0:
            return self.sample_uniform(n), None
        
        # Sample with replacement if needed
        indices = torch.randint(0, len(valid_data), (n,), device=self.device)
        samples = valid_data[indices].clone()
        
        # Add perturbation
        noise_b = torch.randn(n, device=self.device) * self.perturbation_std * (self.b_max - self.b_min)
        noise_mu = torch.randn(n, device=self.device) * self.perturbation_std * (self.mu_max - self.mu_min)
        
        samples[:, 0] = torch.clamp(samples[:, 0] + noise_b, self.b_min, self.b_max)
        samples[:, 1] = torch.clamp(samples[:, 1] + noise_mu, self.mu_min, self.mu_max)
        
        # Occasionally resample shock
        resample_mask = torch.rand(n, device=self.device) < 0.2
        samples[resample_mask, 2] = torch.randint(0, self.n_shocks, 
                                                   (resample_mask.sum(),), device=self.device).float()
        
        return samples, None

    def sample_adaptive(self, n: int) -> tuple:
        """
        Adaptive sampling: oversample high-score regions.
        
        Returns:
            (safe_samples, fail_samples): Tensors for training
        """
        n_candidates = n * self.candidate_multiplier
        candidates = self.sample_uniform(n_candidates, use_initial_bounds=False)
        
        # Compute scores
        scores = self.scorer.compute_score_batch(candidates)
        
        # Assign weights: high weight for safe, low for others
        weights = torch.where(scores > self.tau_high, 
                              torch.ones_like(scores),
                              torch.full_like(scores, 1e-4))
        
        # Normalize and sample
        weights = weights / weights.sum()
        
        try:
            indices = torch.multinomial(weights, n, replacement=False)
        except RuntimeError:
            indices = torch.randperm(n_candidates, device=self.device)[:n]
        
        safe_samples = candidates[indices]
        
        # Collect fail samples
        fail_mask = scores < self.tau_low
        fail_candidates = candidates[fail_mask]
        
        if fail_candidates.shape[0] > 0:
            n_fail = min(n, fail_candidates.shape[0])
            fail_indices = torch.randperm(fail_candidates.shape[0], device=self.device)[:n_fail]
            fail_samples = fail_candidates[fail_indices]
        else:
            fail_samples = None
        
        return safe_samples, fail_samples

    def sample_batch(self, n: int, history: list = None) -> tuple:
        """Main sampling interface."""
        if self.phase == 'warmup':
            return self.sample_from_history(history or [], n)
        else:
            return self.sample_adaptive(n)

    def increment_iteration(self):
        self.iteration += 1


# =============================================================================
# BOUNDARY REFINEMENT
# =============================================================================

def refine_boundaries(scorer: AdmissibilityScorer, sampler: AdaptiveSampler, 
                      config: dict, n_grid: int = 2000):
    """
    Iterative boundary refinement (inner fixed-point).
    
    For fixed policy network, iterate:
    1. Score grid points with current bounds
    2. Identify admissible points
    3. Update bounds from admissible distribution
    4. Repeat until stable
    """
    n_steps = config['boundary_learning']['n_refinement_steps']
    tau_high = config['scoring']['thresholds']['tau_high']
    
    print(f"\n[Boundary Refinement] Starting with {n_steps} iterations...")
    
    # Generate fixed reference grid
    grid = sampler.sample_uniform(n_grid, use_initial_bounds=True)
    
    for step in range(n_steps):
        # Score with current bounds
        scores = scorer.compute_score_batch(grid)
        
        # Identify admissible points
        admissible_mask = scores > tau_high
        admissible_states = grid[admissible_mask]
        n_admissible = admissible_states.shape[0]
        
        # Update bounds
        prev_b_min = scorer.b_min
        scorer.update_boundary_functions(admissible_states)
        
        print(f"  Step {step + 1}/{n_steps}: {n_admissible} admissible points, "
              f"b_min: {prev_b_min:.3f} -> {scorer.b_min:.3f}")
    
    # Final synchronization: re-score with final bounds
    print("  [Boundary Refinement] Final synchronization...")
    final_scores = scorer.compute_score_batch(grid)
    scorer.update_cache(grid, final_scores)
    
    print(f"  [Boundary Refinement] Complete. Final bounds: [{scorer.b_min:.3f}, {scorer.b_max:.3f}]")


# =============================================================================
# VISUALIZATION
# =============================================================================

class AdmissibilityVisualizer:
    """Visualization tools for admissibility scores and boundaries."""

    def __init__(self, scorer: AdmissibilityScorer, config: dict, device: torch.device):
        self.scorer = scorer
        self.device = device
        
        bounds = config['feasibility_bounds']
        econ = config['economic_parameters']
        
        self.b_min = bounds['b_min_initial']
        self.b_max = bounds['b_max_initial']
        self.mu_min = econ['mu_min']
        self.mu_max = econ['mu_max']
        self.zagg_vec = econ['zagg_vec']
        
        thresholds = config['scoring']['thresholds']
        self.tau_high = thresholds['tau_high']
        self.tau_low = thresholds['tau_low']

    def plot_heatmap(self, n_grid: int = 50, save_path: str = 'figures/admissibility_heatmap.png'):
        """Plot admissibility score heatmaps for each shock state."""
        b_vals = np.linspace(self.b_min, self.b_max, n_grid)
        mu_vals = np.linspace(self.mu_min, self.mu_max, n_grid)
        B_grid, Mu_grid = np.meshgrid(b_vals, mu_vals)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Admissibility Scores (Power Barrier)', fontsize=14)
        
        for g_idx, (ax, g_name) in enumerate(zip(axes, ['Low G', 'High G'])):
            # Prepare states tensor
            states = torch.zeros((n_grid * n_grid, 3), device=self.device)
            states[:, 0] = torch.tensor(B_grid.ravel(), device=self.device)
            states[:, 1] = torch.tensor(Mu_grid.ravel(), device=self.device)
            states[:, 2] = g_idx
            
            # Compute scores
            scores = self.scorer.compute_score_batch(states).cpu().numpy()
            A_grid = scores.reshape(n_grid, n_grid)
            
            # Plot
            im = ax.contourf(B_grid, Mu_grid, A_grid, levels=np.linspace(0, 1, 21), cmap='RdYlGn')
            plt.colorbar(im, ax=ax)
            
            # Plot dynamic bounds if available
            if self.scorer.use_dynamic_bounds:
                mu_line = np.linspace(self.mu_min, self.mu_max, 100)
                b_min_line = [self.scorer.bound_funcs[g_idx]['min'](m) for m in mu_line]
                b_max_line = [self.scorer.bound_funcs[g_idx]['max'](m) for m in mu_line]
                ax.plot(b_min_line, mu_line, 'k--', lw=2, label='Debt bounds')
                ax.plot(b_max_line, mu_line, 'k--', lw=2)
            
            ax.set_xlabel('Debt (B)')
            ax.set_ylabel('Multiplier (μ)')
            ax.set_title(f'Shock State: {g_name}')
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[Visualization] Heatmap saved to: {save_path}")

    def plot_score_distribution(self, save_path: str = 'figures/score_distribution.png'):
        """Plot distribution of cached scores."""
        if len(self.scorer.cache) == 0:
            print("[Visualization] Cache empty, skipping distribution plot")
            return
        
        scores = list(self.scorer.cache.values())
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax = axes[0]
        ax.hist(scores, bins=50, color='steelblue', edgecolor='black', range=(0, 1))
        ax.axvline(self.tau_low, color='red', ls='--', label=f'τ_low={self.tau_low}')
        ax.axvline(self.tau_high, color='green', ls='--', label=f'τ_high={self.tau_high}')
        ax.set_xlabel('Admissibility Score')
        ax.set_ylabel('Count')
        ax.set_title('Score Distribution')
        ax.legend()
        
        # Pie chart
        ax = axes[1]
        n_safe = sum(1 for s in scores if s > self.tau_high)
        n_fail = sum(1 for s in scores if s < self.tau_low)
        n_boundary = len(scores) - n_safe - n_fail
        
        ax.pie([n_safe, n_boundary, n_fail], 
               labels=['Safe', 'Boundary', 'Fail'],
               colors=['green', 'orange', 'red'],
               autopct='%1.1f%%')
        ax.set_title('State Classification')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[Visualization] Distribution saved to: {save_path}")


# =============================================================================
# INITIALIZATION HELPER
# =============================================================================

def initialize_adaptive_sampling(policy_net: nn.Module, config: dict, 
                                  device: torch.device) -> tuple:
    """
    Initialize all adaptive sampling components.
    
    Returns:
        (scorer, sampler, visualizer)
    """
    scorer = AdmissibilityScorer(policy_net, config, device)
    sampler = AdaptiveSampler(scorer, config, device)
    visualizer = AdmissibilityVisualizer(scorer, config, device)
    
    print("\n" + "=" * 60)
    print("ADAPTIVE SAMPLING INITIALIZED (v2)")
    print(f"  Scoring: Power barrier (κ={scorer.kappa})")
    print(f"  Buffers: δ_τ={scorer.delta_tau:.4f}, δ_μ={scorer.delta_mu:.4f}, "
          f"δ_B={scorer.delta_debt_prop:.0%} of |Ω_B|")
    print(f"  Thresholds: τ_high={sampler.tau_high}, τ_low={sampler.tau_low}")
    print(f"  Dynamic bounds: {scorer.use_dynamic_bounds}")
    print("=" * 60)
    
    return scorer, sampler, visualizer
