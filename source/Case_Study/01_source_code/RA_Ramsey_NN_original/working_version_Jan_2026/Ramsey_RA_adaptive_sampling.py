# Ramsey_RA_adaptive_sampling.py
"""
Adaptive Sampling Strategy for Ramsey Problem with Endogenous Feasible Set.
VERSION: BUFFERED ALPHA SHAPE (Soft Geometric Boundaries)
UPDATED: Buffer sizes fixed to 5% of variable ranges (Tau & B).

Changes:
1. Tau Buffer: Calculated as 5% of (tau_max - tau_min).
2. Debt Buffer: Calculated as 5% of the normalized B range (0.05).
3. Scoring: Retains Buffered Alpha Shape logic (Inside=1.0, Buffer=Decay, Outside=0.0).
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import pandas as pd
import json

# Try importing geometry libraries
try:
    import alphashape
    from shapely.geometry import Point, Polygon, MultiPolygon

    HAS_GEOMETRY = True
except ImportError:
    HAS_GEOMETRY = False
    print("WARNING: 'alphashape' and 'shapely' libraries not found.")
    print("Please install them via: pip install alphashape shapely")


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def power_barrier(dist: torch.Tensor, delta: float, kappa: float = 4.0) -> torch.Tensor:
    """
    Distance-based power barrier function.
    S(d) = [1 - (d / delta)^kappa]^+
    """
    # Avoid division by zero
    delta = max(delta, 1e-6)
    score = torch.clamp(1.0 - (dist / delta) ** kappa, min=0.0, max=1.0)
    return score


# =============================================================================
# ADMISSIBILITY SCORER (BUFFERED GEOMETRY - 5% RULE)
# =============================================================================

class AdmissibilityScorer:
    def __init__(self, policy_net: nn.Module, config: dict, device: torch.device):
        self.policy_net = policy_net
        self.device = device

        # Economic parameters
        self.mu_min = config['mu_min']
        self.mu_max = config['mu_max']
        self.beta = config['beta']
        self.gamma_l = config['gamma_l']
        self.zagg_vec = torch.tensor(config['zagg_vec'], device=device, dtype=torch.float32).squeeze()
        self.pi_zagg = torch.tensor(config['pi_zagg'], device=device, dtype=torch.float32)

        # Bounds and Penalties
        penalty_params = config.get('penalty_params', {})
        self.tau_min = penalty_params.get('tau_min', 0.0)
        self.tau_max = penalty_params.get('tau_max', 1.0)
        self.b_min_initial = penalty_params.get('b_min', -0.5)
        self.b_max_initial = penalty_params.get('b_max', 3.5)

        self.b_min = self.b_min_initial
        self.b_max = self.b_max_initial

        # Scoring parameters
        scoring_config = config.get('scoring_parameters', {})
        self.kappa = scoring_config.get('kappa', 4.0)

        weights = config.get('admissibility_weights', {'w_tau': 0.33, 'w_mu': 0.33, 'w_debt': 0.34})
        self.w_tau = weights.get('w_tau', 0.33)
        self.w_mu = weights.get('w_lambda', 0.33)
        self.w_debt = weights.get('w_debt', 0.34)

        # --- CALCULATE BUFFERS (5% of Range) ---
        # 1. Tau Buffer
        tau_range = self.tau_max - self.tau_min
        self.delta_tau = 0.1 * tau_range

        # 2. Debt Buffer (Normalized Geometry)
        # Since we normalize B to [0,1] for geometric checks, the range is 1.0.
        # Therefore, 5% of the range is simply 0.05.
        self.delta_geo_norm = 0.1

        # 3. Mu Buffer (Keep default or config, but consistent logic)
        mu_range = self.mu_max - self.mu_min
        self.delta_mu = 0.05 * mu_range  # Also applying 5% rule for consistency

        print(f"    [Scorer] Buffers set to 5% of range:")
        print(f"      - Tau Delta: {self.delta_tau:.4f} (Range: {tau_range:.2f})")
        print(f"      - Debt Delta: {self.delta_geo_norm:.4f} (Normalized)")
        print(f"      - Mu Delta:  {self.delta_mu:.4f} (Range: {mu_range:.2f})")

        # Boundary learning parameters
        self.use_dynamic_bounds = scoring_config.get('use_dynamic_debt_bounds', True)
        self.alpha_param = scoring_config.get('alpha_shape_parameter', 1.5)

        self.boundary_polygons = {}
        self._init_default_boundaries()
        self.cache = {}

    def _init_default_boundaries(self):
        """Initialize boundaries as the full rectangular initial domain."""
        if not HAS_GEOMETRY: return
        # Normalized box (0,0) to (1,1)
        box = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        for g in range(len(self.zagg_vec)):
            self.boundary_polygons[g] = box

    # --- Normalization Helpers ---
    def normalize_b(self, b):
        return (b - self.b_min_initial) / (self.b_max_initial - self.b_min_initial)

    def normalize_mu(self, mu):
        return (mu - self.mu_min) / (self.mu_max - self.mu_min)

    def denormalize_b(self, b_norm):
        return b_norm * (self.b_max_initial - self.b_min_initial) + self.b_min_initial

    def denormalize_mu(self, mu_norm):
        return mu_norm * (self.mu_max - self.mu_min) + self.mu_min

    def compute_geo_score_batch(self, b_tensor: torch.Tensor, mu_tensor: torch.Tensor,
                                g_idx_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute BUFFERED geometric score.
        - Inside hull: 1.0
        - Outside hull: Decays based on distance (Power Barrier)
        """
        if not HAS_GEOMETRY: return torch.ones_like(b_tensor)

        N = b_tensor.shape[0]
        scores = torch.zeros(N, device=self.device)

        b_norm = self.normalize_b(b_tensor).cpu().numpy()
        mu_norm = self.normalize_mu(mu_tensor).cpu().numpy()
        g_np = g_idx_tensor.cpu().numpy().astype(int)

        for g in range(len(self.zagg_vec)):
            mask = (g_np == g)
            if not np.any(mask): continue

            poly = self.boundary_polygons.get(g)
            indices = np.where(mask)[0]

            if poly is None or poly.is_empty:
                # If no boundary yet, allow exploration (score 1.0)
                scores[indices] = 1.0
                continue

            for idx in indices:
                pt = Point(b_norm[idx], mu_norm[idx])

                if poly.contains(pt):
                    # Strictly inside -> Score 1.0
                    scores[idx] = 1.0
                else:
                    # Outside -> Calculate distance
                    dist = poly.distance(pt)
                    # Decay score based on buffer width
                    # Convert distance to tensor for power_barrier
                    d_tensor = torch.tensor(dist, device=self.device, dtype=torch.float32)
                    scores[idx] = power_barrier(d_tensor, self.delta_geo_norm, self.kappa).item()

        return scores

    def compute_score_batch(self, states: torch.Tensor) -> torch.Tensor:
        self.policy_net.eval()

        with torch.no_grad():
            B = states[:, 0]
            mu = states[:, 1]
            g_idx = states[:, 2].long()
            g_val = self.zagg_vec[g_idx]

            # === 1. TAX FEASIBILITY ===
            c = 1.0 / mu
            x = c + g_val
            l = 1.0 - x
            tau = 1.0 - self.gamma_l * c / (l + 1e-8)

            dist_tau_low = torch.clamp(self.tau_min - tau, min=0.0)
            dist_tau_high = torch.clamp(tau - self.tau_max, min=0.0)
            tau_dist = dist_tau_low + dist_tau_high

            # Use calculated 5% buffer
            A_tau = power_barrier(tau_dist, self.delta_tau, self.kappa)

            # === 2. POLICY SAFETY ===
            state_input = torch.stack([B, mu, g_val], dim=1)
            net_dtype = next(self.policy_net.parameters()).dtype
            state_input = state_input.to(dtype=net_dtype)
            policy_logits = self.policy_net(state_input)

            mu_next_g0 = torch.sigmoid(policy_logits[:, 0]) * (self.mu_max - self.mu_min) + self.mu_min
            mu_next_g1 = torch.sigmoid(policy_logits[:, 1]) * (self.mu_max - self.mu_min) + self.mu_min

            dist_mu_g0 = torch.clamp(mu_next_g0 - (self.mu_max - self.delta_mu), min=0.0)
            dist_mu_g1 = torch.clamp(mu_next_g1 - (self.mu_max - self.delta_mu), min=0.0)
            A_mu = torch.minimum(
                power_barrier(dist_mu_g0, self.delta_mu, self.kappa),
                power_barrier(dist_mu_g1, self.delta_mu, self.kappa)
            )

            # === 3. DEBT SUSTAINABILITY (BUFFERED GEOMETRY) ===
            E_mu_next = (self.pi_zagg[g_idx, 0] * mu_next_g0 + self.pi_zagg[g_idx, 1] * mu_next_g1)
            q = self.beta * E_mu_next / mu
            tau_clamped = torch.clamp(tau, self.tau_min, self.tau_max)
            B_next = (B + g_val - tau_clamped * x) / (q + 1e-8)

            # Compute Buffered Scores for both futures
            score_g0 = self.compute_geo_score_batch(B_next, mu_next_g0, torch.zeros_like(g_idx))
            score_g1 = self.compute_geo_score_batch(B_next, mu_next_g1, torch.ones_like(g_idx))

            A_debt = torch.minimum(score_g0, score_g1)

            # === AGGREGATE ===
            score = self.w_tau * A_tau + self.w_mu * A_mu + self.w_debt * A_debt

        return score

    def compute_score(self, B: float, mu: float, g_idx: int) -> float:
        if isinstance(B, float):
            states = torch.tensor([[B, mu, g_idx]], device=self.device, dtype=torch.float32)
            return self.compute_score_batch(states).item()
        else:
            states = torch.stack([B, mu, g_idx]).unsqueeze(0)
            return self.compute_score_batch(states).item()

    def update_cache(self, states: torch.Tensor, scores: torch.Tensor = None):
        if scores is None:
            scores = self.compute_score_batch(states)
        for i in range(states.shape[0]):
            key = (round(states[i, 0].item(), 4),
                   round(states[i, 1].item(), 4),
                   int(states[i, 2].item()))
            self.cache[key] = scores[i].item()

    def get_cached_score(self, B, lam, g_idx):
        B_rounded = round(B.item() if torch.is_tensor(B) else B, 4)
        lam_rounded = round(lam.item() if torch.is_tensor(lam) else lam, 4)
        g_idx_int = int(g_idx.item() if torch.is_tensor(g_idx) else g_idx)
        key = (B_rounded, lam_rounded, g_idx_int)
        return self.cache.get(key, None)

    def clear_cache(self):
        self.cache = {}

    def update_boundary_alphashape(self, admissible_states: torch.Tensor):
        if not self.use_dynamic_bounds or not HAS_GEOMETRY: return

        if admissible_states.shape[0] < 20:
            print(f"    Skipping Alpha Shape: insufficient points ({admissible_states.shape[0]})")
            return

        data = admissible_states.cpu().numpy()
        df = pd.DataFrame(data, columns=['b', 'mu', 'g'])

        for g_val in range(len(self.zagg_vec)):
            df_g = df[df['g'] == g_val]
            if len(df_g) < 10: continue

            b_norm = self.normalize_b(df_g['b'].values)
            mu_norm = self.normalize_mu(df_g['mu'].values)
            points = np.column_stack([b_norm, mu_norm])

            jitter = np.random.normal(0, 0.0001, points.shape)
            points_jittered = points + jitter

            try:
                if len(points) < 50:
                    hull = alphashape.alphashape(points_jittered, 0.0)
                else:
                    hull = alphashape.alphashape(points_jittered, self.alpha_param)

                if hull.is_empty:
                    hull = alphashape.alphashape(points_jittered, 0.0)

                self.boundary_polygons[g_val] = hull
                self.b_min = min(self.b_min, df_g['b'].min())
                self.b_max = max(self.b_max, df_g['b'].max())
            except Exception as e:
                print(f"    Alpha Shape failed for G={g_val}: {e}")


# =============================================================================
# ADAPTIVE SAMPLER (Standard)
# =============================================================================

class AdaptiveSampler:
    def __init__(self, scorer: AdmissibilityScorer, config: dict, device: torch.device):
        self.scorer = scorer
        self.device = device

        penalty_params = config.get('penalty_params', {})
        self.b_min_initial = penalty_params.get('b_min', -0.5)
        self.b_max_initial = penalty_params.get('b_max', 3.5)
        self.mu_min = config['mu_min']
        self.mu_max = config['mu_max']
        self.n_shocks = len(config['zagg_vec'])

        thresholds = config.get('admissibility_thresholds', {})
        self.threshold_strong = thresholds.get('strong_admissible', 0.85)
        self.threshold_inad = thresholds.get('inadmissible', 0.5)
        self.admissibility_thresholds = thresholds.get('admissibility_thresholds', 0.85)

        self.candidate_multiplier = 10
        self.phase = 'warmup'
        self.iteration = 0
        self.total_adaptive_iters = 0
        self.sampling_stats = defaultdict(list)
        print(f">>> AdaptiveSampler initialized (Buffered 5% Geometry).")

    def set_phase(self, phase, total_adaptive_iters=None):
        self.phase = phase
        if phase == 'adaptive' and total_adaptive_iters is not None:
            self.total_adaptive_iters = total_adaptive_iters
            self.iteration = 0
        print(f"\n>>> Sampling phase set to: {phase.upper()}")

    def _sample_uniform(self, batch_size, use_initial_bounds=True):
        samples = torch.zeros((batch_size, 3), device=self.device)
        b_lo, b_hi = self.b_min_initial, self.b_max_initial
        samples[:, 0] = torch.rand(batch_size, device=self.device) * (b_hi - b_lo) + b_lo
        samples[:, 1] = torch.rand(batch_size, device=self.device) * (self.mu_max - self.mu_min) + self.mu_min
        samples[:, 2] = torch.randint(0, self.n_shocks, (batch_size,), device=self.device).float()
        return samples

    def sample_batch(self, batch_size, policy_net=None):
        if self.phase == 'warmup':
            samples = self._sample_uniform(batch_size, use_initial_bounds=True)
            return samples, None
        else:
            return self._sample_adaptive(batch_size)

    def _sample_adaptive(self, batch_size):
        n_candidates = batch_size * self.candidate_multiplier
        candidates = self._sample_uniform(n_candidates)
        scores = self.scorer.compute_score_batch(candidates)

        weights = torch.where(scores > self.admissibility_thresholds,
                              torch.ones_like(scores),
                              torch.full_like(scores, 1e-4))
        weights = weights / weights.sum()

        try:
            indices = torch.multinomial(weights, batch_size, replacement=False)
        except RuntimeError:
            indices = torch.randperm(n_candidates, device=self.device)[:batch_size]

        admissible_samples = candidates[indices]

        fail_mask = scores < self.threshold_inad
        fail_candidates = candidates[fail_mask]
        inadmissible_samples = None
        if fail_candidates.shape[0] > 0:
            n_fail = min(batch_size, fail_candidates.shape[0])
            fail_indices = torch.randperm(fail_candidates.shape[0], device=self.device)[:n_fail]
            inadmissible_samples = fail_candidates[fail_indices]

        # Stats
        sel_scores = scores[indices].cpu().numpy()
        self.sampling_stats['iteration'].append(self.iteration)
        self.sampling_stats['mean_A'].append(np.mean(sel_scores))
        self.sampling_stats['frac_strongly_admissible'].append(np.mean(sel_scores > self.threshold_strong))
        self.sampling_stats['frac_inadmissible'].append(np.mean(sel_scores < self.threshold_inad))

        return admissible_samples, inadmissible_samples

    def increment_iteration(self):
        if self.phase == 'adaptive': self.iteration += 1

    def get_statistics(self):
        return dict(self.sampling_stats)

    def plot_sampling_statistics(self, save_path='figures/adaptive_sampling_stats.png'):
        if len(self.sampling_stats['iteration']) == 0: return
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        iters = self.sampling_stats['iteration']
        ax = axes[0, 0]
        ax.plot(iters, self.sampling_stats['mean_A'], 'b-')
        ax.set_title('Average A Score')
        ax = axes[0, 1]
        ax.plot(iters, self.sampling_stats['frac_strongly_admissible'], 'g-', label='Strong')
        ax.plot(iters, self.sampling_stats['frac_inadmissible'], 'r-', label='Inadmissible')
        ax.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)


# =============================================================================
# VISUALIZATION
# =============================================================================

class AdmissibilityVisualizer:
    def __init__(self, scorer, config, device):
        self.scorer = scorer
        self.device = device
        penalty_params = config.get('penalty_params', {})
        self.b_min = penalty_params.get('b_min', -0.5)
        self.b_max = penalty_params.get('b_max', 3.5)
        self.mu_min = config['mu_min']
        self.mu_max = config['mu_max']
        self.threshold_strong = config.get('admissibility_thresholds', {}).get('strong_admissible', 0.85)
        self.threshold_inad = config.get('admissibility_thresholds', {}).get('inadmissible', 0.5)

    def plot_admissibility_heatmap(self, n_grid=50, save_path='figures/admissibility_heatmap.png',
                                   overlay_cpp_data=False, cpp_data_file='policy_v6_out_61.txt',
                                   verbose_cpp_analysis=True):
        b_vals = np.linspace(self.b_min, self.b_max, n_grid)
        mu_vals = np.linspace(self.mu_min, self.mu_max, n_grid)
        B_grid, Mu_grid = np.meshgrid(b_vals, mu_vals)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Admissibility Scores (Buffered Geometry 5%)', fontsize=14)

        for g_idx, (ax, g_name) in enumerate(zip(axes, ['Low G', 'High G'])):
            states = torch.zeros((n_grid * n_grid, 3), device=self.device)
            states[:, 0] = torch.tensor(B_grid.ravel(), device=self.device)
            states[:, 1] = torch.tensor(Mu_grid.ravel(), device=self.device)
            states[:, 2] = g_idx
            scores = self.scorer.compute_score_batch(states).cpu().numpy()
            A_grid = scores.reshape(n_grid, n_grid)

            im = ax.contourf(B_grid, Mu_grid, A_grid, levels=np.linspace(0, 1, 21), cmap='RdYlGn')
            plt.colorbar(im, ax=ax)

            # Plot the Polygon Boundary
            if HAS_GEOMETRY and g_idx in self.scorer.boundary_polygons:
                poly = self.scorer.boundary_polygons[g_idx]

                def plot_poly(p):
                    x, y = p.exterior.xy
                    x_denorm = [self.scorer.denormalize_b(xi) for xi in x]
                    y_denorm = [self.scorer.denormalize_mu(yi) for yi in y]
                    ax.plot(x_denorm, y_denorm, 'k-', lw=2)

                if isinstance(poly, Polygon):
                    plot_poly(poly)
                elif isinstance(poly, MultiPolygon):
                    for p in poly.geoms: plot_poly(p)

            ax.set_title(f'Shock State: {g_name}')
            if overlay_cpp_data:
                try:
                    df = pd.read_csv(cpp_data_file, header=None, sep='\s+')
                    df = df[(df.iloc[:, 3] != -500)]
                    df_g = df[df.iloc[:, 2] == g_idx]
                    if len(df_g) > 0: ax.scatter(df_g.iloc[:, 0], df_g.iloc[:, 1], c='blue', s=5, alpha=0.3)
                except:
                    pass

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Admissibility heatmap saved to: {save_path}")

    def plot_cache_distribution(self, save_path='figures/cache_distribution.png'):
        if len(self.scorer.cache) == 0: return
        scores = list(self.scorer.cache.values())
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax = axes[0]
        ax.hist(scores, bins=50, color='steelblue', edgecolor='black', range=(0, 1))
        ax.set_title('Score Histogram')
        ax = axes[1]
        n_safe = sum(1 for s in scores if s > self.threshold_strong)
        n_fail = sum(1 for s in scores if s < self.threshold_inad)
        n_bound = len(scores) - n_safe - n_fail
        ax.pie([n_safe, n_bound, n_fail], labels=['Safe', 'Bound', 'Fail'], colors=['green', 'orange', 'red'],
               autopct='%1.1f%%')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    def plot_boundary_diagnostics(self, save_path):
        pass


# =============================================================================
# INITIALIZATION & API
# =============================================================================

def initialize_adaptive_sampling(policy_net: nn.Module, config: dict, device: torch.device):
    scorer = AdmissibilityScorer(policy_net, config, device)
    sampler = AdaptiveSampler(scorer, config, device)
    visualizer = AdmissibilityVisualizer(scorer, config, device)
    return scorer, sampler, visualizer


def update_cache_periodic(scorer: AdmissibilityScorer, sampler: AdaptiveSampler, n_grid: int = 5000):
    print(f"\n>>> Updating Alpha Shape Boundaries ({n_grid} samples)...")
    n_refinement_steps = 3
    grid = sampler._sample_uniform(n_grid, use_initial_bounds=True)
    for step in range(n_refinement_steps):
        scores = scorer.compute_score_batch(grid)
        admissible_mask = scores > sampler.threshold_strong
        admissible_states = grid[admissible_mask]
        n_admissible = admissible_states.shape[0]
        scorer.update_boundary_alphashape(admissible_states)
        print(f"    Refinement {step + 1}/{n_refinement_steps}: {n_admissible} admissible points found.")
    final_scores = scorer.compute_score_batch(grid)
    scorer.clear_cache()
    scorer.update_cache(grid, final_scores)
    print("    >>> Alpha Shape update complete.")