"""
Adaptive Sampling Module for Ramsey Optimal Taxation Problem.

This module implements an adaptive sampling strategy that focuses computational
resources on the economically feasible region of the state space. It uses
alpha-shapes (generalized convex hulls) to detect the boundary of the admissible
set and employs buffered soft boundaries for smooth sampling transitions.

Key Components:
    - AdmissibilityScorer: Computes feasibility scores for state-action pairs
    - AdaptiveSampler: Generates samples biased toward admissible regions
    - AdmissibilityVisualizer: Creates diagnostic plots for monitoring

Mathematical Background:
    The admissibility score A(B, λ, g) ∈ [0, 1] combines three components:
    1. Tax feasibility: τ ∈ [τ_min, τ_max]
    2. Policy safety: μ' ∈ [μ_min, μ_max]
    3. Debt sustainability: B' stays within learned feasible set

    Score aggregation: A = w_τ·A_τ + w_μ·A_μ + w_B·A_B

Authors: Zhigang Feng
Version: 2.0 (Streamlined)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import pandas as pd
from typing import Dict, Tuple, Optional, List

# Optional geometry libraries for alpha-shape boundary detection
try:
    import alphashape
    from shapely.geometry import Point, Polygon, MultiPolygon
    HAS_GEOMETRY = True
except ImportError:
    HAS_GEOMETRY = False
    print("WARNING: 'alphashape' and 'shapely' not found. Install via: pip install alphashape shapely")


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def power_barrier(dist: torch.Tensor, delta: float, kappa: float = 4.0) -> torch.Tensor:
    """
    Compute a smooth distance-based barrier function.
    
    Creates a soft boundary that smoothly transitions from 1 (inside) to 0 (outside)
    over a buffer zone of width delta.
    
    Args:
        dist: Distance from boundary (0 = on boundary, positive = outside)
        delta: Width of the buffer zone where score decays
        kappa: Exponent controlling decay sharpness (higher = sharper)
    
    Returns:
        Score in [0, 1]: S(d) = max(0, 1 - (d/δ)^κ)
    
    Mathematical Form:
        S(d) = [1 - (d/δ)^κ]_+
        
        Properties:
        - S(0) = 1 (on boundary)
        - S(δ) = 0 (at buffer edge)
        - S'(0) = 0 (smooth at boundary)
    """
    delta = max(delta, 1e-6)  # Avoid division by zero
    return torch.clamp(1.0 - (dist / delta) ** kappa, min=0.0, max=1.0)


# =============================================================================
# ADMISSIBILITY SCORER
# =============================================================================

class AdmissibilityScorer:
    """
    Computes admissibility scores for (B, μ, g) state points.
    
    The scorer evaluates whether a state leads to economically feasible outcomes
    by checking tax rates, future multipliers, and debt sustainability.
    
    Attributes:
        policy_net: Neural network that outputs μ'(g=0) and μ'(g=1) given state
        device: PyTorch device (CPU/CUDA)
        boundary_polygons: Dict[int, Polygon] mapping shock index to feasible region
        cache: Dict storing precomputed scores for efficiency
        
    Configuration Parameters (from config):
        - Economic: beta, gamma_l, mu_min, mu_max, zagg_vec, pi_zagg
        - Bounds: tau_min, tau_max, b_min, b_max
        - Buffers: delta_tau, delta_mu, delta_geo_norm (computed from buffer_parameters)
        - Weights: w_tau, w_mu, w_debt (from admissibility_weights)
    """
    
    def __init__(self, policy_net: nn.Module, config: dict, device: torch.device):
        """
        Initialize the admissibility scorer.
        
        Args:
            policy_net: Trained policy network π(B, μ, g) → (μ'_g0, μ'_g1)
            config: Configuration dictionary with all parameters
            device: PyTorch device for computation
        """
        self.policy_net = policy_net
        self.device = device
        
        # === Load Economic Parameters ===
        econ = config.get('economic_parameters', config)
        self.beta = econ.get('beta', config.get('beta', 0.9))
        self.gamma_l = econ.get('gamma_l', config.get('gamma_l', 0.3))
        
        zagg = econ.get('zagg_vec', config.get('zagg_vec'))
        self.zagg_vec = torch.tensor(zagg, device=device, dtype=torch.float32).squeeze()
        
        pi = econ.get('pi_zagg', config.get('pi_zagg'))
        self.pi_zagg = torch.tensor(pi, device=device, dtype=torch.float32)
        
        # === Load State Bounds ===
        bounds = config.get('state_bounds', config)
        self.mu_min = bounds.get('mu_min', config.get('mu_min', 1.27))
        self.mu_max = bounds.get('mu_max', config.get('mu_max', 2.51))
        
        # === Load Penalty/Constraint Bounds ===
        penalty = config.get('penalty_params', {})
        self.tau_min = penalty.get('tau_min', 0.0)
        self.tau_max = penalty.get('tau_max', 1.0)
        self.b_min_initial = penalty.get('b_min', -0.5)
        self.b_max_initial = penalty.get('b_max', 3.5)
        
        # Dynamic bounds (updated by alpha-shape learning)
        self.b_min = self.b_min_initial
        self.b_max = self.b_max_initial
        
        # === Load Scoring Parameters ===
        scoring = config.get('scoring_parameters', {})
        self.use_dynamic_bounds = scoring.get('use_dynamic_debt_bounds', True)
        
        alpha_config = config.get('alpha_shape', {})
        self.alpha_param = alpha_config.get('alpha_param', 
                                            scoring.get('alpha_shape_parameter', 1.5))
        self.min_points_hull = alpha_config.get('min_points_for_hull', 50)
        self.min_points_shock = alpha_config.get('min_points_per_shock', 10)
        self.jitter_std = alpha_config.get('jitter_std', 0.0001)
        
        # === Load Buffer Parameters ===
        buffer_cfg = config.get('buffer_parameters', {})
        self.kappa = buffer_cfg.get('kappa', 4.0)
        
        # Compute buffer widths as fractions of variable ranges
        tau_range = self.tau_max - self.tau_min
        mu_range = self.mu_max - self.mu_min
        
        self.delta_tau = buffer_cfg.get('tau_buffer_fraction', 0.1) * tau_range
        self.delta_mu = buffer_cfg.get('mu_buffer_fraction', 0.05) * mu_range
        self.delta_geo_norm = buffer_cfg.get('debt_buffer_fraction', 0.1)  # Normalized space
        
        # === Load Admissibility Weights ===
        weights = config.get('admissibility_weights', {})
        self.w_tau = weights.get('w_tau', 0.4)
        self.w_mu = weights.get('w_lambda', 0.0)
        self.w_debt = weights.get('w_debt', 0.6)
        
        # === Initialize Boundaries and Cache ===
        self.boundary_polygons: Dict[int, Polygon] = {}
        self._init_default_boundaries()
        self.cache: Dict[Tuple, float] = {}
        
        self._print_initialization_summary()
    
    def _print_initialization_summary(self):
        """Print summary of scorer configuration."""
        print(f"    [AdmissibilityScorer] Initialized with buffer zones:")
        print(f"      - Tau buffer:  {self.delta_tau:.4f} (range: {self.tau_max - self.tau_min:.2f})")
        print(f"      - Debt buffer: {self.delta_geo_norm:.4f} (normalized)")
        print(f"      - Mu buffer:   {self.delta_mu:.4f} (range: {self.mu_max - self.mu_min:.2f})")
        print(f"      - Weights: τ={self.w_tau:.2f}, μ={self.w_mu:.2f}, B={self.w_debt:.2f}")
    
    def _init_default_boundaries(self):
        """Initialize boundaries as full rectangular domain (before learning)."""
        if not HAS_GEOMETRY:
            return
        # Normalized box [0,1] x [0,1]
        box = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        for g in range(len(self.zagg_vec)):
            self.boundary_polygons[g] = box
    
    # --- Normalization Helpers ---
    def normalize_b(self, b: torch.Tensor) -> np.ndarray:
        """Normalize debt B from [b_min, b_max] to [0, 1]."""
        b_np = b.cpu().numpy() if torch.is_tensor(b) else b
        return (b_np - self.b_min_initial) / (self.b_max_initial - self.b_min_initial)
    
    def normalize_mu(self, mu: torch.Tensor) -> np.ndarray:
        """Normalize multiplier μ from [μ_min, μ_max] to [0, 1]."""
        mu_np = mu.cpu().numpy() if torch.is_tensor(mu) else mu
        return (mu_np - self.mu_min) / (self.mu_max - self.mu_min)
    
    def denormalize_b(self, b_norm: float) -> float:
        """Convert normalized B back to original scale."""
        return b_norm * (self.b_max_initial - self.b_min_initial) + self.b_min_initial
    
    def denormalize_mu(self, mu_norm: float) -> float:
        """Convert normalized μ back to original scale."""
        return mu_norm * (self.mu_max - self.mu_min) + self.mu_min
    
    def compute_geo_score_batch(self, b_tensor: torch.Tensor, mu_tensor: torch.Tensor,
                                 g_idx_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute geometric admissibility scores based on alpha-shape boundaries.
        
        Points inside the learned feasible set get score 1.0, points outside
        decay smoothly based on distance to the boundary.
        
        Args:
            b_tensor: Debt values (N,)
            mu_tensor: Multiplier values (N,)
            g_idx_tensor: Shock state indices (N,)
        
        Returns:
            Tensor of scores (N,) in [0, 1]
        """
        if not HAS_GEOMETRY:
            return torch.ones_like(b_tensor)
        
        N = b_tensor.shape[0]
        scores = torch.zeros(N, device=self.device)
        
        # Convert to normalized coordinates
        b_norm = self.normalize_b(b_tensor)
        mu_norm = self.normalize_mu(mu_tensor)
        g_np = g_idx_tensor.cpu().numpy().astype(int)
        
        for g in range(len(self.zagg_vec)):
            mask = (g_np == g)
            if not np.any(mask):
                continue
            
            poly = self.boundary_polygons.get(g)
            indices = np.where(mask)[0]
            
            if poly is None or poly.is_empty:
                scores[indices] = 1.0  # No boundary yet → allow exploration
                continue
            
            for idx in indices:
                pt = Point(b_norm[idx], mu_norm[idx])
                
                if poly.contains(pt):
                    scores[idx] = 1.0  # Inside → full score
                else:
                    # Outside → decay based on distance
                    dist = poly.distance(pt)
                    d_tensor = torch.tensor(dist, device=self.device, dtype=torch.float32)
                    scores[idx] = power_barrier(d_tensor, self.delta_geo_norm, self.kappa).item()
        
        return scores
    
    def compute_score_batch(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute admissibility scores for a batch of states.
        
        The score combines three components:
        1. Tax feasibility: Is the implied tax rate in [τ_min, τ_max]?
        2. Policy safety: Are the predicted μ' values in bounds?
        3. Debt sustainability: Does B' stay in the feasible set?
        
        Args:
            states: Tensor of shape (N, 3) with columns [B, μ, g_idx]
        
        Returns:
            Tensor of scores (N,) with A(B, μ, g) ∈ [0, 1]
        """
        self.policy_net.eval()
        
        with torch.no_grad():
            B = states[:, 0]
            mu = states[:, 1]
            g_idx = states[:, 2].long()
            g_val = self.zagg_vec[g_idx]
            
            # === 1. TAX FEASIBILITY ===
            # From FOC: c = 1/μ, x = c + g, l = 1 - x
            # Tax rate: τ = 1 - γ_l · c / l
            c = 1.0 / mu
            x = c + g_val
            l = 1.0 - x
            tau = 1.0 - self.gamma_l * c / (l + 1e-8)
            
            # Distance to feasible tax range
            dist_tau_low = torch.clamp(self.tau_min - tau, min=0.0)
            dist_tau_high = torch.clamp(tau - self.tau_max, min=0.0)
            tau_dist = dist_tau_low + dist_tau_high
            
            A_tau = power_barrier(tau_dist, self.delta_tau, self.kappa)
            
            # === 2. POLICY SAFETY ===
            # Get policy network predictions
            state_input = torch.stack([B, mu, g_val], dim=1)
            net_dtype = next(self.policy_net.parameters()).dtype
            state_input = state_input.to(dtype=net_dtype)
            policy_logits = self.policy_net(state_input)
            
            # Convert logits to μ' values
            mu_next_g0 = torch.sigmoid(policy_logits[:, 0]) * (self.mu_max - self.mu_min) + self.mu_min
            mu_next_g1 = torch.sigmoid(policy_logits[:, 1]) * (self.mu_max - self.mu_min) + self.mu_min
            
            # Check if μ' hits upper bound (indicates infeasibility)
            dist_mu_g0 = torch.clamp(mu_next_g0 - (self.mu_max - self.delta_mu), min=0.0)
            dist_mu_g1 = torch.clamp(mu_next_g1 - (self.mu_max - self.delta_mu), min=0.0)
            
            A_mu = torch.minimum(
                power_barrier(dist_mu_g0, self.delta_mu, self.kappa),
                power_barrier(dist_mu_g1, self.delta_mu, self.kappa)
            )
            
            # === 3. DEBT SUSTAINABILITY ===
            # Compute next-period debt: B' = (B + g - τx) / q
            E_mu_next = self.pi_zagg[g_idx, 0] * mu_next_g0 + self.pi_zagg[g_idx, 1] * mu_next_g1
            q = self.beta * E_mu_next / mu
            tau_clamped = torch.clamp(tau, self.tau_min, self.tau_max)
            B_next = (B + g_val - tau_clamped * x) / (q + 1e-8)
            
            # Check geometric feasibility for both possible future shocks
            score_g0 = self.compute_geo_score_batch(B_next, mu_next_g0, torch.zeros_like(g_idx))
            score_g1 = self.compute_geo_score_batch(B_next, mu_next_g1, torch.ones_like(g_idx))
            
            A_debt = torch.minimum(score_g0, score_g1)
            
            # === AGGREGATE SCORE ===
            score = self.w_tau * A_tau + self.w_mu * A_mu + self.w_debt * A_debt
        
        return score
    
    def compute_score(self, B: float, mu: float, g_idx: int) -> float:
        """Compute admissibility score for a single state point."""
        states = torch.tensor([[B, mu, g_idx]], device=self.device, dtype=torch.float32)
        return self.compute_score_batch(states).item()
    
    def update_cache(self, states: torch.Tensor, scores: Optional[torch.Tensor] = None):
        """Store precomputed scores in cache for efficiency."""
        if scores is None:
            scores = self.compute_score_batch(states)
        for i in range(states.shape[0]):
            key = (round(states[i, 0].item(), 4),
                   round(states[i, 1].item(), 4),
                   int(states[i, 2].item()))
            self.cache[key] = scores[i].item()
    
    def get_cached_score(self, B, mu, g_idx) -> Optional[float]:
        """Retrieve cached score if available."""
        B_r = round(B.item() if torch.is_tensor(B) else B, 4)
        mu_r = round(mu.item() if torch.is_tensor(mu) else mu, 4)
        g_int = int(g_idx.item() if torch.is_tensor(g_idx) else g_idx)
        return self.cache.get((B_r, mu_r, g_int))
    
    def clear_cache(self):
        """Clear the score cache."""
        self.cache = {}
    
    def update_boundary_alphashape(self, admissible_states: torch.Tensor):
        """
        Update feasible set boundaries using alpha-shapes.
        
        Alpha-shapes are a generalization of convex hulls that can capture
        non-convex feasible regions. The alpha parameter controls how tightly
        the boundary fits the data.
        
        Args:
            admissible_states: Tensor (N, 3) of states with high admissibility scores
        """
        if not self.use_dynamic_bounds or not HAS_GEOMETRY:
            return
        
        if admissible_states.shape[0] < self.min_points_hull:
            print(f"    Skipping alpha-shape: insufficient points ({admissible_states.shape[0]})")
            return
        
        data = admissible_states.cpu().numpy()
        df = pd.DataFrame(data, columns=['b', 'mu', 'g'])
        
        for g_val in range(len(self.zagg_vec)):
            df_g = df[df['g'] == g_val]
            if len(df_g) < self.min_points_shock:
                continue
            
            # Normalize to [0,1] for numerical stability
            b_norm = self.normalize_b(df_g['b'].values)
            mu_norm = self.normalize_mu(df_g['mu'].values)
            points = np.column_stack([b_norm, mu_norm])
            
            # Add small jitter to handle duplicate points
            jitter = np.random.normal(0, self.jitter_std, points.shape)
            points_jittered = points + jitter
            
            try:
                # Use convex hull for sparse data, alpha-shape otherwise
                if len(points) < self.min_points_hull:
                    hull = alphashape.alphashape(points_jittered, 0.0)  # 0 = convex hull
                else:
                    hull = alphashape.alphashape(points_jittered, self.alpha_param)
                
                if hull.is_empty:
                    hull = alphashape.alphashape(points_jittered, 0.0)  # Fallback
                
                self.boundary_polygons[g_val] = hull
                
                # Update dynamic debt bounds
                self.b_min = min(self.b_min, df_g['b'].min())
                self.b_max = max(self.b_max, df_g['b'].max())
                
            except Exception as e:
                print(f"    Alpha-shape failed for g={g_val}: {e}")


# =============================================================================
# ADAPTIVE SAMPLER
# =============================================================================

class AdaptiveSampler:
    """
    Generates samples biased toward the admissible region.
    
    During warmup: Uses uniform sampling over the full state space.
    During adaptive phase: Uses importance sampling with weights proportional
    to admissibility scores, focusing computation on feasible states.
    
    Attributes:
        scorer: AdmissibilityScorer instance for computing weights
        phase: Current sampling phase ('warmup' or 'adaptive')
        sampling_stats: Dict tracking sampling statistics over iterations
    """
    
    def __init__(self, scorer: AdmissibilityScorer, config: dict, device: torch.device):
        """
        Initialize the adaptive sampler.
        
        Args:
            scorer: Configured AdmissibilityScorer
            config: Configuration dictionary
            device: PyTorch device
        """
        self.scorer = scorer
        self.device = device
        
        # State space bounds
        penalty = config.get('penalty_params', {})
        self.b_min_initial = penalty.get('b_min', -0.5)
        self.b_max_initial = penalty.get('b_max', 3.5)
        
        bounds = config.get('state_bounds', config)
        self.mu_min = bounds.get('mu_min', config.get('mu_min', 1.27))
        self.mu_max = bounds.get('mu_max', config.get('mu_max', 2.51))
        
        econ = config.get('economic_parameters', config)
        self.n_shocks = len(econ.get('zagg_vec', config.get('zagg_vec')))
        
        # Admissibility thresholds
        thresholds = config.get('admissibility_thresholds', {})
        self.threshold_strong = thresholds.get('strong_admissible', 0.85)
        self.threshold_inad = thresholds.get('inadmissible', 0.5)
        self.sampling_threshold = thresholds.get('sampling_threshold', 0.85)
        
        # Sampling parameters
        adaptive_cfg = config.get('adaptive_sampling', {})
        self.candidate_multiplier = adaptive_cfg.get('candidate_multiplier', 10)
        
        # State
        self.phase = 'warmup'
        self.iteration = 0
        self.total_adaptive_iters = 0
        self.sampling_stats = defaultdict(list)
        
        print(f">>> AdaptiveSampler initialized (buffered geometry).")
    
    def set_phase(self, phase: str, total_adaptive_iters: int = None):
        """
        Set the sampling phase.
        
        Args:
            phase: 'warmup' for uniform sampling, 'adaptive' for weighted sampling
            total_adaptive_iters: Total number of adaptive iterations (for scheduling)
        """
        self.phase = phase
        if phase == 'adaptive' and total_adaptive_iters is not None:
            self.total_adaptive_iters = total_adaptive_iters
            self.iteration = 0
        print(f"\n>>> Sampling phase set to: {phase.upper()}")
    
    def _sample_uniform(self, batch_size: int, use_initial_bounds: bool = True) -> torch.Tensor:
        """Generate uniform samples over the state space."""
        samples = torch.zeros((batch_size, 3), device=self.device)
        
        b_lo = self.b_min_initial if use_initial_bounds else self.scorer.b_min
        b_hi = self.b_max_initial if use_initial_bounds else self.scorer.b_max
        
        samples[:, 0] = torch.rand(batch_size, device=self.device) * (b_hi - b_lo) + b_lo
        samples[:, 1] = torch.rand(batch_size, device=self.device) * (self.mu_max - self.mu_min) + self.mu_min
        samples[:, 2] = torch.randint(0, self.n_shocks, (batch_size,), device=self.device).float()
        
        return samples
    
    def sample_batch(self, batch_size: int, policy_net: nn.Module = None
                     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate a batch of samples.
        
        Args:
            batch_size: Number of samples to generate
            policy_net: Policy network (unused, kept for API compatibility)
        
        Returns:
            Tuple of (admissible_samples, inadmissible_samples)
            inadmissible_samples is None during warmup
        """
        if self.phase == 'warmup':
            return self._sample_uniform(batch_size, use_initial_bounds=True), None
        else:
            return self._sample_adaptive(batch_size)
    
    def _sample_adaptive(self, batch_size: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate samples using importance sampling weighted by admissibility.
        
        Strategy:
        1. Generate many candidate samples uniformly
        2. Compute admissibility scores for all candidates
        3. Sample with replacement, weighted by scores
        4. Also return highly inadmissible samples for explicit training
        """
        n_candidates = batch_size * self.candidate_multiplier
        candidates = self._sample_uniform(n_candidates)
        scores = self.scorer.compute_score_batch(candidates)
        
        # Importance weights: admissible → weight 1, inadmissible → small weight
        weights = torch.where(
            scores > self.sampling_threshold,
            torch.ones_like(scores),
            torch.full_like(scores, 1e-4)
        )
        weights = weights / weights.sum()
        
        # Sample admissible states
        try:
            indices = torch.multinomial(weights, batch_size, replacement=False)
        except RuntimeError:
            # Fallback if weights are degenerate
            indices = torch.randperm(n_candidates, device=self.device)[:batch_size]
        
        admissible_samples = candidates[indices]
        
        # Also collect inadmissible samples for explicit penalty training
        fail_mask = scores < self.threshold_inad
        fail_candidates = candidates[fail_mask]
        inadmissible_samples = None
        
        if fail_candidates.shape[0] > 0:
            n_fail = min(batch_size, fail_candidates.shape[0])
            fail_indices = torch.randperm(fail_candidates.shape[0], device=self.device)[:n_fail]
            inadmissible_samples = fail_candidates[fail_indices]
        
        # Track statistics
        sel_scores = scores[indices].cpu().numpy()
        self.sampling_stats['iteration'].append(self.iteration)
        self.sampling_stats['mean_A'].append(np.mean(sel_scores))
        self.sampling_stats['frac_strongly_admissible'].append(np.mean(sel_scores > self.threshold_strong))
        self.sampling_stats['frac_inadmissible'].append(np.mean(sel_scores < self.threshold_inad))
        
        return admissible_samples, inadmissible_samples
    
    def increment_iteration(self):
        """Increment iteration counter (adaptive phase only)."""
        if self.phase == 'adaptive':
            self.iteration += 1
    
    def get_statistics(self) -> dict:
        """Return sampling statistics dictionary."""
        return dict(self.sampling_stats)
    
    def plot_sampling_statistics(self, save_path: str = 'figures/adaptive_sampling_stats.png'):
        """Generate and save sampling statistics plots."""
        if len(self.sampling_stats['iteration']) == 0:
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        iters = self.sampling_stats['iteration']
        
        # Mean admissibility score
        ax = axes[0]
        ax.plot(iters, self.sampling_stats['mean_A'], 'b-', linewidth=2)
        ax.set_ylabel('Mean A Score')
        ax.set_title('Average Admissibility Score over Iterations')
        ax.grid(True, alpha=0.3)
        
        # Fraction breakdown
        ax = axes[1]
        ax.plot(iters, self.sampling_stats['frac_strongly_admissible'], 'g-', 
                linewidth=2, label='Strongly Admissible')
        ax.plot(iters, self.sampling_stats['frac_inadmissible'], 'r-', 
                linewidth=2, label='Inadmissible')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Fraction')
        ax.set_title('Sample Distribution by Category')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"    Sampling statistics saved to: {save_path}")


# =============================================================================
# VISUALIZATION
# =============================================================================

class AdmissibilityVisualizer:
    """
    Creates diagnostic visualizations for admissibility analysis.
    
    Provides heatmaps, boundary plots, and distribution visualizations
    to monitor the adaptive sampling process.
    """
    
    def __init__(self, scorer: AdmissibilityScorer, config: dict, device: torch.device):
        """
        Initialize visualizer.
        
        Args:
            scorer: AdmissibilityScorer instance
            config: Configuration dictionary
            device: PyTorch device
        """
        self.scorer = scorer
        self.device = device
        
        # Bounds for plotting
        penalty = config.get('penalty_params', {})
        self.b_min = penalty.get('b_min', -0.5)
        self.b_max = penalty.get('b_max', 3.5)
        
        bounds = config.get('state_bounds', config)
        self.mu_min = bounds.get('mu_min', config.get('mu_min', 1.27))
        self.mu_max = bounds.get('mu_max', config.get('mu_max', 2.51))
        
        # Thresholds for coloring
        thresholds = config.get('admissibility_thresholds', {})
        self.threshold_strong = thresholds.get('strong_admissible', 0.85)
        self.threshold_inad = thresholds.get('inadmissible', 0.5)
        
        # Visualization settings
        viz_cfg = config.get('visualization', {})
        self.dpi = viz_cfg.get('figure_dpi', 150)
    
    def plot_admissibility_heatmap(self, n_grid: int = 50,
                                    save_path: str = 'figures/admissibility_heatmap.png',
                                    overlay_cpp_data: bool = False,
                                    cpp_data_file: str = 'policy_v6_out_61.txt',
                                    verbose_cpp_analysis: bool = True):
        """
        Generate heatmap of admissibility scores over state space.
        
        Args:
            n_grid: Number of grid points per dimension
            save_path: Output file path
            overlay_cpp_data: If True, overlay C++ solution points
            cpp_data_file: Path to C++ output file
            verbose_cpp_analysis: Print detailed statistics
        """
        b_vals = np.linspace(self.b_min, self.b_max, n_grid)
        mu_vals = np.linspace(self.mu_min, self.mu_max, n_grid)
        B_grid, Mu_grid = np.meshgrid(b_vals, mu_vals)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Admissibility Scores by Shock State', fontsize=14)
        
        for g_idx, (ax, g_name) in enumerate(zip(axes, ['Low G', 'High G'])):
            # Create state grid
            states = torch.zeros((n_grid * n_grid, 3), device=self.device)
            states[:, 0] = torch.tensor(B_grid.ravel(), device=self.device)
            states[:, 1] = torch.tensor(Mu_grid.ravel(), device=self.device)
            states[:, 2] = g_idx
            
            # Compute scores
            scores = self.scorer.compute_score_batch(states).cpu().numpy()
            A_grid = scores.reshape(n_grid, n_grid)
            
            # Plot heatmap
            im = ax.contourf(B_grid, Mu_grid, A_grid, levels=np.linspace(0, 1, 21), cmap='RdYlGn')
            plt.colorbar(im, ax=ax)
            
            # Plot alpha-shape boundary
            if HAS_GEOMETRY and g_idx in self.scorer.boundary_polygons:
                poly = self.scorer.boundary_polygons[g_idx]
                self._plot_polygon(ax, poly)
            
            ax.set_xlabel('Debt (B)')
            ax.set_ylabel('Multiplier (μ)')
            ax.set_title(f'Shock State: {g_name}')
            
            # Overlay C++ data if requested
            if overlay_cpp_data:
                self._overlay_cpp_data(ax, g_idx, cpp_data_file)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi)
        plt.close(fig)
        print(f"    Admissibility heatmap saved to: {save_path}")
    
    def _plot_polygon(self, ax, poly):
        """Helper to plot polygon boundary on axes."""
        def plot_single(p):
            x, y = p.exterior.xy
            x_denorm = [self.scorer.denormalize_b(xi) for xi in x]
            y_denorm = [self.scorer.denormalize_mu(yi) for yi in y]
            ax.plot(x_denorm, y_denorm, 'k-', lw=2)
        
        if isinstance(poly, Polygon):
            plot_single(poly)
        elif isinstance(poly, MultiPolygon):
            for p in poly.geoms:
                plot_single(p)
    
    def _overlay_cpp_data(self, ax, g_idx: int, cpp_data_file: str):
        """Overlay C++ solution points on plot."""
        try:
            df = pd.read_csv(cpp_data_file, header=None, sep=r'\s+')
            df = df[df.iloc[:, 3] != -500]  # Filter flag values
            df_g = df[df.iloc[:, 2] == g_idx]
            if len(df_g) > 0:
                ax.scatter(df_g.iloc[:, 0], df_g.iloc[:, 1], c='blue', s=5, alpha=0.3, label='C++ Data')
        except Exception:
            pass  # Silently skip if file not found
    
    def plot_cache_distribution(self, save_path: str = 'figures/cache_distribution.png'):
        """Plot distribution of cached admissibility scores."""
        if len(self.scorer.cache) == 0:
            return
        
        scores = list(self.scorer.cache.values())
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax = axes[0]
        ax.hist(scores, bins=50, color='steelblue', edgecolor='black', range=(0, 1))
        ax.axvline(self.threshold_strong, color='green', linestyle='--', label=f'Strong ({self.threshold_strong})')
        ax.axvline(self.threshold_inad, color='red', linestyle='--', label=f'Inadmissible ({self.threshold_inad})')
        ax.set_xlabel('Admissibility Score')
        ax.set_ylabel('Count')
        ax.set_title('Score Distribution')
        ax.legend()
        
        # Pie chart
        ax = axes[1]
        n_safe = sum(1 for s in scores if s > self.threshold_strong)
        n_fail = sum(1 for s in scores if s < self.threshold_inad)
        n_trans = len(scores) - n_safe - n_fail
        
        ax.pie([n_safe, n_trans, n_fail], 
               labels=['Admissible', 'Transition', 'Inadmissible'],
               colors=['green', 'orange', 'red'],
               autopct='%1.1f%%')
        ax.set_title('Score Categories')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi)
        plt.close(fig)
        print(f"    Cache distribution saved to: {save_path}")
    
    def plot_boundary_diagnostics(self, save_path: str = 'figures/boundary_diagnostics.png'):
        """Placeholder for boundary diagnostic plots."""
        pass  # Can be extended for detailed boundary analysis


# =============================================================================
# MODULE API
# =============================================================================

def initialize_adaptive_sampling(policy_net: nn.Module, config: dict, device: torch.device
                                  ) -> Tuple[AdmissibilityScorer, AdaptiveSampler, AdmissibilityVisualizer]:
    """
    Initialize the complete adaptive sampling system.
    
    Args:
        policy_net: Trained policy neural network
        config: Configuration dictionary
        device: PyTorch device
    
    Returns:
        Tuple of (scorer, sampler, visualizer)
    """
    scorer = AdmissibilityScorer(policy_net, config, device)
    sampler = AdaptiveSampler(scorer, config, device)
    visualizer = AdmissibilityVisualizer(scorer, config, device)
    return scorer, sampler, visualizer


def update_cache_periodic(scorer: AdmissibilityScorer, sampler: AdaptiveSampler, 
                          n_grid: int = 5000, n_refinement_steps: int = 3):
    """
    Periodically update alpha-shape boundaries and score cache.
    
    Uses iterative refinement: sample → score → update boundary → repeat.
    
    Args:
        scorer: AdmissibilityScorer to update
        sampler: AdaptiveSampler for generating samples
        n_grid: Number of samples per refinement step
        n_refinement_steps: Number of refinement iterations
    """
    print(f"\n>>> Updating alpha-shape boundaries ({n_grid} samples)...")
    
    grid = sampler._sample_uniform(n_grid, use_initial_bounds=True)
    
    for step in range(n_refinement_steps):
        scores = scorer.compute_score_batch(grid)
        admissible_mask = scores > sampler.threshold_strong
        admissible_states = grid[admissible_mask]
        n_admissible = admissible_states.shape[0]
        
        scorer.update_boundary_alphashape(admissible_states)
        print(f"    Refinement {step + 1}/{n_refinement_steps}: {n_admissible} admissible points")
    
    # Update cache with final scores
    final_scores = scorer.compute_score_batch(grid)
    scorer.clear_cache()
    scorer.update_cache(grid, final_scores)
    print("    >>> Boundary update complete.")
