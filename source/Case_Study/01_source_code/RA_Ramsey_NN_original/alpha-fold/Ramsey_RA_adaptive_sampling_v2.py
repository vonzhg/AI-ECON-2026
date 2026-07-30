# Ramsey_RA_adaptive_sampling_v2.py
"""
Adaptive Sampling Strategy for Ramsey Problem with Endogenous Feasible Set.
VERSION 3: ALPHA SHAPE (CONCAVE HULL) REVISION

Changes from v2:
1. Replaced 1D quantile boundary estimation with 2D Alpha Shapes (Concave Hulls).
2. Implemented geometric distance-to-polygon scoring for A_debt.
3. Added coordinate normalization to handle scale differences between B and mu.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import pandas as pd

# New dependencies for Alpha Shapes
try:
    import alphashape
    from shapely.geometry import Point, Polygon, MultiPolygon
    from shapely.ops import nearest_points
    HAS_GEOMETRY = True
except ImportError:
    HAS_GEOMETRY = False
    print("WARNING: alphashape/shapely not found. Reverting to simple bounds.")


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def power_barrier(dist: torch.Tensor, delta: float, kappa: float = 4.0) -> torch.Tensor:
    """
    Distance-based power barrier function.
    S(d) = [1 - (d / delta)^kappa]^+
    """
    # Score is 1.0 if distance is 0 (inside), decays as distance increases
    score = torch.clamp(1.0 - (dist / delta) ** kappa, min=0.0, max=1.0)
    return score


# =============================================================================
# ADMISSIBILITY SCORER (ALPHA SHAPE VERSION)
# =============================================================================

class AdmissibilityScorer:
    """
    Computes admissibility scores using Alpha Shapes for domain geometry.
    """

    def __init__(self, policy_net: nn.Module, config: dict, device: torch.device):
        self.policy_net = policy_net
        self.device = device
        
        if not HAS_GEOMETRY:
            raise ImportError("Please install 'alphashape' and 'shapely' to use this version.")

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
        self.weights = scoring['weights']
        
        # Buffer widths
        buffers = scoring['buffer_proportions']
        self.delta_tau = buffers['delta_tau_prop'] * (self.tau_max - self.tau_min)
        self.delta_mu = buffers['delta_mu_prop'] * (self.mu_max - self.mu_min)
        # For geometry, delta is defined in normalized [0,1] space
        self.delta_geo_norm = buffers['delta_debt_prop'] 
        
        # Boundary learning parameters
        self.boundary_config = config['boundary_learning']
        self.use_dynamic_bounds = self.boundary_config['use_dynamic_bounds']
        
        # Alpha parameter for hull tightness (0=convex hull, higher=tighter concave)
        # Can be tuned or added to config. 
        self.alpha_param = 2.5 

        # Store boundary polygons (shapely objects) for each shock g
        self.boundary_polygons = {} 
        self._init_default_boundaries()
        
        # Cache for scores
        self.cache = {}

    def _init_default_boundaries(self):
        """Initialize boundaries as the full rectangular initial domain."""
        # Create a rectangular box for each g
        p_min = (self.normalize_b(self.b_min_initial), self.normalize_mu(self.mu_min))
        p_max = (self.normalize_b(self.b_max_initial), self.normalize_mu(self.mu_max))
        
        box = Polygon([
            (p_min[0], p_min[1]), 
            (p_max[0], p_min[1]), 
            (p_max[0], p_max[1]), 
            (p_min[0], p_max[1])
        ])
        
        for g in range(self.n_shocks):
            self.boundary_polygons[g] = box

    # --- Normalization Helpers (Critical for Geometry) ---
    def normalize_b(self, b):
        return (b - self.b_min_initial) / (self.b_max_initial - self.b_min_initial)

    def normalize_mu(self, mu):
        return (mu - self.mu_min) / (self.mu_max - self.mu_min)
    
    def denormalize_b(self, b_norm):
        return b_norm * (self.b_max_initial - self.b_min_initial) + self.b_min_initial

    def get_geo_distance_batch(self, b_tensor: torch.Tensor, mu_tensor: torch.Tensor, g_idx_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute distance from points to the learned Alpha Shape polygons.
        Returns 0 if inside, distance > 0 if outside.
        OPERATES IN NORMALIZED [0,1] SPACE.
        """
        N = b_tensor.shape[0]
        distances = torch.zeros(N, device=self.device)
        
        # Normalize inputs
        b_norm = self.normalize_b(b_tensor).cpu().numpy()
        mu_norm = self.normalize_mu(mu_tensor).cpu().numpy()
        g_np = g_idx_tensor.cpu().numpy().astype(int)
        
        # Iterate by shock to batch shapely calls (optimization)
        for g in range(self.n_shocks):
            mask = (g_np == g)
            if not np.any(mask):
                continue
            
            poly = self.boundary_polygons.get(g)
            indices = np.where(mask)[0]
            
            # If no polygon exists (e.g. initialization failed), assume infinite distance or safe?
            # We assume initialized to full box, so it exists.
            
            for idx in indices:
                pt = Point(b_norm[idx], mu_norm[idx])
                
                if poly.contains(pt):
                    d = 0.0
                else:
                    # Shapely distance is Euclidean distance to nearest point on boundary
                    d = poly.distance(pt)
                
                distances[idx] = float(d)
                
        return distances

    def compute_score_batch(self, states: torch.Tensor) -> torch.Tensor:
        """Vectorized computation of admissibility scores."""
        self.policy_net.eval()
        
        with torch.no_grad():
            B = states[:, 0]
            mu = states[:, 1]
            g_idx = states[:, 2].long()
            g_val = self.zagg_vec[g_idx]
            
            # === 1. TAX FEASIBILITY (A_tau) ===
            c = 1.0 / mu
            x = c + g_val
            l = 1.0 - x
            tau = 1.0 - self.gamma_l * c / (l + 1e-8)
            
            # Simple 1D barrier for scalar constraints
            tau_dist = torch.maximum(torch.zeros_like(tau), self.tau_min - tau) + \
                       torch.maximum(torch.zeros_like(tau), tau - self.tau_max)
            A_tau = power_barrier(tau_dist, self.delta_tau, self.kappa)
            
            # === 2. POLICY SAFETY (A_mu) ===
            state_input = torch.stack([B, mu, g_val], dim=1)
            policy_logits = self.policy_net(state_input)
            
            mu_next_g0 = torch.sigmoid(policy_logits[:, 0]) * (self.mu_max - self.mu_min) + self.mu_min
            mu_next_g1 = torch.sigmoid(policy_logits[:, 1]) * (self.mu_max - self.mu_min) + self.mu_min
            
            # 1D barrier for mu
            mu_dist_g0 = torch.maximum(torch.zeros_like(mu_next_g0), self.mu_min - mu_next_g0) + \
                         torch.maximum(torch.zeros_like(mu_next_g0), mu_next_g0 - self.mu_max)
            mu_dist_g1 = torch.maximum(torch.zeros_like(mu_next_g1), self.mu_min - mu_next_g1) + \
                         torch.maximum(torch.zeros_like(mu_next_g1), mu_next_g1 - self.mu_max)
            
            A_mu_g0 = power_barrier(mu_dist_g0, self.delta_mu, self.kappa)
            A_mu_g1 = power_barrier(mu_dist_g1, self.delta_mu, self.kappa)
            A_mu = torch.minimum(A_mu_g0, A_mu_g1)
            
            # === 3. DEBT SUSTAINABILITY (A_debt) - GEOMETRIC ===
            # Compute next B
            E_mu_next = (self.pi_zagg[g_idx, 0] * mu_next_g0 + self.pi_zagg[g_idx, 1] * mu_next_g1)
            q = self.beta * E_mu_next / mu
            tau_clamped = torch.clamp(tau, self.tau_min, self.tau_max)
            B_next = (B + g_val - tau_clamped * x) / (q + 1e-8)
            
            # Calculate distance to Polygon for both futures
            # g0 future
            dist_g0 = self.get_geo_distance_batch(B_next, mu_next_g0, torch.zeros_like(g_idx))
            A_debt_g0 = power_barrier(dist_g0, self.delta_geo_norm, self.kappa)
            
            # g1 future
            dist_g1 = self.get_geo_distance_batch(B_next, mu_next_g1, torch.ones_like(g_idx))
            A_debt_g1 = power_barrier(dist_g1, self.delta_geo_norm, self.kappa)
            
            A_debt = torch.minimum(A_debt_g0, A_debt_g1)
            
            # === AGGREGATE ===
            score = self.weights['w_tau'] * A_tau + \
                    self.weights['w_mu'] * A_mu + \
                    self.weights['w_debt'] * A_debt
            
        return score

    def compute_score(self, B: float, mu: float, g_idx: int) -> float:
        """Wrapper for single input."""
        states = torch.tensor([[B, mu, g_idx]], device=self.device, dtype=torch.float32)
        return self.compute_score_batch(states).item()

    def update_cache(self, states: torch.Tensor, scores: torch.Tensor = None):
        if scores is None:
            scores = self.compute_score_batch(states)
        for i in range(states.shape[0]):
            key = (round(states[i, 0].item(), 4), round(states[i, 1].item(), 4), int(states[i, 2].item()))
            self.cache[key] = scores[i].item()

    def update_boundary_alphashape(self, admissible_states: torch.Tensor):
        """
        Update 2D boundaries using Alpha Shapes (Concave Hull).
        """
        if not self.use_dynamic_bounds:
            return
        
        if admissible_states.shape[0] < 200:
            print(f"  [Boundary] Skipping Alpha Shape: insufficient points ({admissible_states.shape[0]})")
            return
            
        # Move to CPU numpy
        data = admissible_states.cpu().numpy()
        df = pd.DataFrame(data, columns=['b', 'mu', 'g'])
        
        for g_val in range(self.n_shocks):
            df_g = df[df['g'] == g_val]
            if len(df_g) < 100:
                continue
                
            # Normalize points before shape fitting
            # (x, y) = (b_norm, mu_norm)
            points = np.column_stack([
                self.normalize_b(df_g['b'].values),
                self.normalize_mu(df_g['mu'].values)
            ])
            
            # Add some jitter to prevent collinearity issues in Qhull
            jitter = np.random.normal(0, 0.001, points.shape)
            points_jittered = points + jitter
            
            try:
                # Generate Alpha Shape
                # alpha=0 -> convex hull
                # alpha>0 -> concave hull (tighter fit)
                # We start with a moderate alpha. If it fragments too much, fallback to convex hull.
                
                # Option 1: Constant alpha
                hull = alphashape.alphashape(points_jittered, self.alpha_param)
                
                # Check validity: Hull should be a Polygon or MultiPolygon, and not empty
                if hull.is_empty:
                    print(f"  [Boundary] G={g_val}: Alpha shape empty, keeping old.")
                    continue
                
                if isinstance(hull, MultiPolygon):
                    # If fragmented, we might want to take the largest chunk 
                    # or keep it if we believe the domain is disconnected.
                    # For stability, let's take the Convex Hull of the Alpha Shape if it's too fragmented?
                    # Or just use it. Let's use it, but check area.
                    pass
                
                self.boundary_polygons[g_val] = hull
                
                # Calculate coverage stats
                area = hull.area # Area in normalized units [0,1]
                print(f"  [Boundary] G={g_val}: Alpha Shape updated. Norm Area={area:.3f}")
                
            except Exception as e:
                print(f"  [Boundary] G={g_val}: Alpha Shape failed ({e}), keeping old.")


# =============================================================================
# ADAPTIVE SAMPLER (UNCHANGED INTERFACE)
# =============================================================================

class AdaptiveSampler:
    # ... (Same as v2, uses the updated Scorer transparently) ...
    def __init__(self, scorer: AdmissibilityScorer, config: dict, device: torch.device):
        self.scorer = scorer
        self.device = device
        # Copy config params
        bounds = config['feasibility_bounds']
        self.b_min = bounds['b_min_initial']
        self.b_max = bounds['b_max_initial']
        self.mu_min = config['economic_parameters']['mu_min']
        self.mu_max = config['economic_parameters']['mu_max']
        self.n_shocks = len(config['economic_parameters']['zagg_vec'])
        thresholds = config['scoring']['thresholds']
        self.tau_high = thresholds['tau_high']
        self.tau_low = thresholds['tau_low']
        self.candidate_multiplier = config['sampling']['candidate_multiplier']
        self.perturbation_std = config['sampling']['perturbation_std']
        self.phase = 'warmup'
        self.iteration = 0

    def set_phase(self, phase: str):
        self.phase = phase
        self.iteration = 0

    def sample_uniform(self, n: int, use_initial_bounds: bool = True) -> torch.Tensor:
        # Same as v2
        samples = torch.zeros((n, 3), device=self.device)
        b_lo, b_hi = self.b_min, self.b_max
        samples[:, 0] = torch.rand(n, device=self.device) * (b_hi - b_lo) + b_lo
        samples[:, 1] = torch.rand(n, device=self.device) * (self.mu_max - self.mu_min) + self.mu_min
        samples[:, 2] = torch.randint(0, self.n_shocks, (n,), device=self.device).float()
        return samples

    def sample_from_history(self, history: list, n: int) -> tuple:
        # Same as v2
        if len(history) == 0: return self.sample_uniform(n), None
        all_data = torch.cat(history, dim=0)
        mask = ((all_data[:, 0] >= self.b_min) & (all_data[:, 0] <= self.b_max) &
                (all_data[:, 1] >= self.mu_min) & (all_data[:, 1] <= self.mu_max))
        valid_data = all_data[mask]
        if len(valid_data) == 0: return self.sample_uniform(n), None
        indices = torch.randint(0, len(valid_data), (n,), device=self.device)
        samples = valid_data[indices].clone()
        noise_b = torch.randn(n, device=self.device) * self.perturbation_std * (self.b_max - self.b_min)
        noise_mu = torch.randn(n, device=self.device) * self.perturbation_std * (self.mu_max - self.mu_min)
        samples[:, 0] = torch.clamp(samples[:, 0] + noise_b, self.b_min, self.b_max)
        samples[:, 1] = torch.clamp(samples[:, 1] + noise_mu, self.mu_min, self.mu_max)
        resample_mask = torch.rand(n, device=self.device) < 0.2
        samples[resample_mask, 2] = torch.randint(0, self.n_shocks, (resample_mask.sum(),), device=self.device).float()
        return samples, None

    def sample_adaptive(self, n: int) -> tuple:
        # Same as v2
        n_candidates = n * self.candidate_multiplier
        candidates = self.sample_uniform(n_candidates, use_initial_bounds=False)
        scores = self.scorer.compute_score_batch(candidates)
        weights = torch.where(scores > self.tau_high, torch.ones_like(scores), torch.full_like(scores, 1e-4))
        weights = weights / weights.sum()
        try:
            indices = torch.multinomial(weights, n, replacement=False)
        except RuntimeError:
            indices = torch.randperm(n_candidates, device=self.device)[:n]
        safe_samples = candidates[indices]
        fail_mask = scores < self.tau_low
        fail_candidates = candidates[fail_mask]
        if fail_candidates.shape[0] > 0:
            n_fail = min(n, fail_candidates.shape[0])
            fail_indices = torch.randperm(fail_candidates.shape[0], device=self.device)[:n_fail]
            fail_samples = fail_candidates[fail_indices]
        else:
            fail_samples = None
        return safe_samples, fail_samples

    def increment_iteration(self):
        self.iteration += 1


# =============================================================================
# BOUNDARY REFINEMENT (UPDATED)
# =============================================================================

def refine_boundaries(scorer: AdmissibilityScorer, sampler: AdaptiveSampler, 
                      config: dict, n_grid: int = 2000):
    """Refine Alpha Shape boundaries."""
    n_steps = config['boundary_learning']['n_refinement_steps']
    tau_high = config['scoring']['thresholds']['tau_high']
    
    print(f"\n[Boundary Refinement - Alpha Shape] Starting {n_steps} iterations...")
    
    # Use a denser grid for geometry learning
    grid = sampler.sample_uniform(n_grid * 2, use_initial_bounds=True)
    
    for step in range(n_steps):
        scores = scorer.compute_score_batch(grid)
        admissible_mask = scores > tau_high
        admissible_states = grid[admissible_mask]
        n_admissible = admissible_states.shape[0]
        
        # Update via Alpha Shape
        scorer.update_boundary_alphashape(admissible_states)
        
        print(f"  Step {step + 1}/{n_steps}: {n_admissible} admissible points. Polygons updated.")
    
    # Final sync
    final_scores = scorer.compute_score_batch(grid)
    scorer.update_cache(grid, final_scores)


# =============================================================================
# VISUALIZATION (UPDATED)
# =============================================================================

class AdmissibilityVisualizer:
    def __init__(self, scorer: AdmissibilityScorer, config: dict, device: torch.device):
        self.scorer = scorer
        self.device = device
        bounds = config['feasibility_bounds']
        self.b_min = bounds['b_min_initial']
        self.b_max = bounds['b_max_initial']
        self.mu_min = config['economic_parameters']['mu_min']
        self.mu_max = config['economic_parameters']['mu_max']
        
    def plot_heatmap(self, n_grid: int = 50, save_path: str = 'figures/admissibility_heatmap.png'):
        if not HAS_GEOMETRY: return
        
        b_vals = np.linspace(self.b_min, self.b_max, n_grid)
        mu_vals = np.linspace(self.mu_min, self.mu_max, n_grid)
        B_grid, Mu_grid = np.meshgrid(b_vals, mu_vals)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Admissibility Scores (Alpha Shape)', fontsize=14)
        
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
            if g_idx in self.scorer.boundary_polygons:
                poly = self.scorer.boundary_polygons[g_idx]
                # Convert normalized polygon back to data coords for plotting
                if isinstance(poly, Polygon):
                    x, y = poly.exterior.xy
                    x_denorm = [self.scorer.denormalize_b(xi) for xi in x]
                    y_denorm = [yi * (self.mu_max - self.mu_min) + self.mu_min for yi in y]
                    ax.plot(x_denorm, y_denorm, 'k-', lw=2, label='Alpha Hull')
                elif isinstance(poly, MultiPolygon):
                    for p in poly.geoms:
                        x, y = p.exterior.xy
                        x_denorm = [self.scorer.denormalize_b(xi) for xi in x]
                        y_denorm = [yi * (self.mu_max - self.mu_min) + self.mu_min for yi in y]
                        ax.plot(x_denorm, y_denorm, 'k-', lw=2)
            
            ax.set_xlabel('Debt (B)')
            ax.set_ylabel('Multiplier (μ)')
            ax.set_title(f'Shock State: {g_name}')
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    def plot_score_distribution(self, save_path: str = 'figures/score_distribution.png'):
        # Same as v2
        if len(self.scorer.cache) == 0: return
        scores = list(self.scorer.cache.values())
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax = axes[0]
        ax.hist(scores, bins=50, color='steelblue', edgecolor='black', range=(0, 1))
        ax.set_title('Score Distribution')
        ax = axes[1]
        n_safe = sum(1 for s in scores if s > 0.7)
        n_fail = sum(1 for s in scores if s < 0.3)
        n_bound = len(scores) - n_safe - n_fail
        ax.pie([n_safe, n_bound, n_fail], labels=['Safe', 'Bound', 'Fail'], autopct='%1.1f%%')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_adaptive_sampling(policy_net: nn.Module, config: dict, device: torch.device):
    scorer = AdmissibilityScorer(policy_net, config, device)
    sampler = AdaptiveSampler(scorer, config, device)
    visualizer = AdmissibilityVisualizer(scorer, config, device)
    return scorer, sampler, visualizer