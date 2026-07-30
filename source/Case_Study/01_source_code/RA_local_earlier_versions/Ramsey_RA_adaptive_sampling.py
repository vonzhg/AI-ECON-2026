# Ramsey_RA_adaptive_sampling.py
"""
Adaptive Sampling Strategy for Ramsey Problem with Endogenous Feasible Set.
REFINED: Implements State-Dependent (Lambda & Shock specific) Debt Bounds.
LOGIC:
1. Admissibility = Tau safe AND b_next safe in Future(g=0) AND b_next safe in Future(g=1).
2. Boundaries: Computed from admissible points using configurable method (Min/Max or Quantile).
3. Fallback: If bounds collapse (max <= min), enforcing a minimal 'band_safe' window around the midpoint.
4. Consistency: Cache is re-scored after boundary updates to ensure Heatmap/Scatter plot match.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import pandas as pd
from scipy.interpolate import interp1d
import json


def _compute_piecewise_score(value, params, score_at_medium):
    """
    Computes a soft score based on a 6-point piecewise linear function.
    """
    x_points = [
        params['hard_bound_low'],
        params['medium_bound_low'],
        params['safe_bound_low'],
        params['safe_bound_high'],
        params['medium_bound_high'],
        params['hard_bound_high']
    ]
    y_points = [
        0.0,
        score_at_medium,
        1.0,
        1.0,
        score_at_medium,
        0.0
    ]

    # Safety: Ensure monotonic increasing for interpolation to prevent errors
    for i in range(len(x_points) - 1):
        if x_points[i + 1] < x_points[i]:
            x_points[i + 1] = x_points[i] + 1e-6

    score = np.interp(value, x_points, y_points)
    return float(score)


class AdmissibilityScorer:
    """
    Computes and caches admissibility scores for states.
    FEATURES: State-Dependent Debt Bounds b_min(lam, g) and b_max(lam, g).
    """

    def __init__(self, policy_net, config, device):
        self.policy_net = policy_net
        self.device = device
        self.cache = {}
        self.mu_min = config['mu_min']
        self.mu_max = config['mu_max']
        self.zagg_vec = torch.tensor(config['zagg_vec'], device=device).squeeze()
        self.pi_zagg = torch.tensor(config['pi_zagg'], device=device, dtype=torch.float32)
        self.beta = config['beta']
        self.gamma_l = config['gamma_l']

        self.n_lam_bin = config['n_lam_bin']

        weights = config.get('admissibility_weights',
                             {'w_lambda': 0.333, 'w_tau': 0.333, 'w_debt': 0.334})
        self.w_lambda = weights.get('w_lambda', 0.333)
        self.w_tau = weights.get('w_tau', 0.333)
        self.w_debt = weights.get('w_debt', 0.334)

        scoring_config = config.get('scoring_parameters', {})
        self.score_at_medium = scoring_config.get('score_at_medium_bound', 0.2)

        # --- Boundary Calculation Parameters ---
        # 'quantile' or 'min_max'
        self.boundary_method = scoring_config.get('boundary_method', 'quantile')
        self.boundary_quantile_low = scoring_config.get('boundary_quantile_low', 0.05)
        self.boundary_quantile_high = scoring_config.get('boundary_quantile_high', 0.95)

        # --- Global Hard Bounds (Safety Net) ---
        self.tau_hard_bound_low = scoring_config.get('tau_hard_bound_low', -0.1)
        self.tau_hard_bound_high = scoring_config.get('tau_hard_bound_high', 5.0)
        self.debt_hard_bound_low = scoring_config.get('debt_hard_bound_low', -2.0)
        self.debt_hard_bound_high = scoring_config.get('debt_hard_bound_high', 10.0)

        penalty_params = config.get('penalty_params', {})
        self.tau_min = penalty_params.get('tau_min', 0.0)
        self.tau_max = penalty_params.get('tau_max', 1.0)

        # Bands
        self.tau_band = scoring_config.get('tau_band', 0.02)
        self.tau_band_safe = scoring_config.get('tau_band_safe', 0.1)
        self.b_band = scoring_config.get('b_band', 0.1)
        self.b_band_safe = scoring_config.get('b_band_safe', 0.1)
        self.mu_band = scoring_config.get('mu_band', 0.1)

        # --- Initial Global Bounds ---
        self.b_min = penalty_params.get('b_min', -0.5)
        self.b_max = penalty_params.get('b_max', 3.5)
        self.initial_b_min = self.b_min
        self.initial_b_max = self.b_max

        self.use_dynamic_debt_bounds = scoring_config.get('use_dynamic_debt_bounds', False)
        self.dynamic_debt_percentile = scoring_config.get('dynamic_debt_percentile', 5)  # Legacy parameter?

        # --- STATE DEPENDENT BOUNDS STORAGE ---
        self.bound_funcs = {
            0: {'min': None, 'max': None},
            1: {'min': None, 'max': None}
        }
        self._reset_bound_funcs()

    def _reset_bound_funcs(self):
        """Resets bounds to global constant functions."""
        for g in [0, 1]:
            self.bound_funcs[g]['min'] = lambda x: self.initial_b_min
            self.bound_funcs[g]['max'] = lambda x: self.initial_b_max

    def get_dynamic_bounds(self, lam, g_idx):
        """
        Retrieves b_min and b_max for a specific lambda and g.
        IMPLEMENTS COLLAPSE LOGIC: If bounds cross/collapse, enforce a minimal safety band.
        """
        g_int = int(g_idx)
        if g_int not in [0, 1]: g_int = 0

        # Clamp lambda to valid range
        lam_val = float(lam)
        lam_val = max(self.mu_min, min(self.mu_max, lam_val))

        # Get raw interpolated values
        b_min_local = float(self.bound_funcs[g_int]['min'](lam_val))
        b_max_local = float(self.bound_funcs[g_int]['max'](lam_val))

        # --- FALLBACK LOGIC ---
        # If the region is effectively closed (or inverted), open a small window
        # around the crossing point to allow the sampler to find new valid points.
        if b_max_local <= b_min_local + 1e-6:
            mid = (b_max_local + b_min_local) / 2.0
            b_min_local = mid - self.b_band_safe
            b_max_local = mid + self.b_band_safe

        return b_min_local, b_max_local

    def compute_score(self, B, lam, g_idx):
        # Force deterministic evaluation to remove dropout noise discrepancy
        self.policy_net.eval()

        with torch.no_grad():
            g_idx_int = int(g_idx.item())
            g_val = self.zagg_vec[g_idx_int]

            state_input = torch.tensor([[B.item(), lam.item(), g_val.item()]],
                                       device=self.device, dtype=torch.float32)

            policy_logits = self.policy_net(state_input)
            lam_plus_g0 = torch.sigmoid(policy_logits[:, 0]) * (self.mu_max - self.mu_min) + self.mu_min
            lam_plus_g1 = torch.sigmoid(policy_logits[:, 1]) * (self.mu_max - self.mu_min) + self.mu_min
            max_lam_plus = torch.max(lam_plus_g0, lam_plus_g1)

            # --- 1. Lambda Score (Global Cap) ---
            effective_max = self.mu_max - self.mu_band
            if max_lam_plus.item() >= effective_max:
                A_lambda = 0.0
            else:
                A_lambda = 1.0 - (max_lam_plus.item() / effective_max)
            A_lambda = max(0.0, A_lambda)

            # --- 2. Tau Score (Current Feasibility) ---
            c0 = 1.0 / lam.item()
            x0 = c0 + g_val.item()
            tau0 = 1.0 - self.gamma_l * c0 / (1.0 - x0 + 1e-8)

            tau_params = {
                'hard_bound_low': self.tau_hard_bound_low,
                'medium_bound_low': self.tau_min - self.tau_band,
                'safe_bound_low': self.tau_min + self.tau_band_safe,
                'safe_bound_high': self.tau_max - self.tau_band_safe,
                'medium_bound_high': self.tau_max + self.tau_band,
                'hard_bound_high': self.tau_hard_bound_high
            }
            A_tau = _compute_piecewise_score(tau0, tau_params, self.score_at_medium)

            # --- 3. Debt Score (Future Sustainability) ---
            e_mu_next = self.pi_zagg[g_idx_int, 0] * lam_plus_g0 + \
                        self.pi_zagg[g_idx_int, 1] * lam_plus_g1
            q0 = self.beta * e_mu_next / lam
            b_next = (B + g_val - tau0 * x0) / q0

            # === CHECK FUTURE STATE 0 (g'=0) ===
            lam_next_0 = lam_plus_g0.item()
            # This calls get_dynamic_bounds which handles the collapse logic
            b_min_0, b_max_0 = self.get_dynamic_bounds(lam_next_0, 0)

            debt_params_0 = {
                'hard_bound_low': self.debt_hard_bound_low,
                'medium_bound_low': b_min_0 - self.b_band,
                'safe_bound_low': b_min_0 + self.b_band_safe,
                'safe_bound_high': b_max_0 - self.b_band_safe,
                'medium_bound_high': b_max_0 + self.b_band,
                'hard_bound_high': self.debt_hard_bound_high
            }
            score_b_g0 = _compute_piecewise_score(b_next.item(), debt_params_0, self.score_at_medium)

            # === CHECK FUTURE STATE 1 (g'=1) ===
            lam_next_1 = lam_plus_g1.item()
            # This calls get_dynamic_bounds which handles the collapse logic
            b_min_1, b_max_1 = self.get_dynamic_bounds(lam_next_1, 1)

            debt_params_1 = {
                'hard_bound_low': self.debt_hard_bound_low,
                'medium_bound_low': b_min_1 - self.b_band,
                'safe_bound_low': b_min_1 + self.b_band_safe,
                'safe_bound_high': b_max_1 - self.b_band_safe,
                'medium_bound_high': b_max_1 + self.b_band,
                'hard_bound_high': self.debt_hard_bound_high
            }
            score_b_g1 = _compute_piecewise_score(b_next.item(), debt_params_1, self.score_at_medium)

            # === AGGREGATION: SAFE IN BOTH FUTURES ===
            A_debt = min(score_b_g0, score_b_g1)

            final_score = (self.w_lambda * A_lambda) + \
                          (self.w_tau * A_tau) + \
                          (self.w_debt * A_debt)

        return final_score

    def compute_score_batch(self, states):
        N = states.shape[0]
        scores = torch.zeros(N, device=self.device)
        with torch.no_grad():
            for i in range(N):
                B, lam, g_idx = states[i]
                scores[i] = self.compute_score(B, lam, g_idx)
        return scores

    def update_cache(self, states, scores=None):
        if scores is None:
            scores = self.compute_score_batch(states)
        for i in range(states.shape[0]):
            B = round(states[i, 0].item(), 4)
            lam = round(states[i, 1].item(), 4)
            g_idx = int(states[i, 2].item())
            key = (B, lam, g_idx)
            self.cache[key] = scores[i].item()

    def get_cached_score(self, B, lam, g_idx):
        B_rounded = round(B.item() if torch.is_tensor(B) else B, 4)
        lam_rounded = round(lam.item() if torch.is_tensor(lam) else lam, 4)
        g_idx_int = int(g_idx.item() if torch.is_tensor(g_idx) else g_idx)
        key = (B_rounded, lam_rounded, g_idx_int)
        return self.cache.get(key, None)

    def clear_cache(self):
        self.cache = {}
        self._reset_bound_funcs()

    def update_dynamic_debt_bounds(self, admissible_keys):
        if not self.use_dynamic_debt_bounds:
            return

        if len(admissible_keys) < 200:
            print(f"  > Dynamic bounds update skipped (Insufficient admissible data: {len(admissible_keys)})")
            return

        df = pd.DataFrame(admissible_keys, columns=['b', 'lam', 'g'])

        num_bins = self.n_lam_bin
        lam_bins = np.linspace(self.mu_min, self.mu_max, num_bins + 1)
        updates_made = False

        print(f"  > Computing dynamic bounds using method: {self.boundary_method.upper()}")

        for g_val in [0, 1]:
            df_g = df[df['g'] == g_val]
            if len(df_g) < 50: continue

            df_g = df_g.copy()
            df_g['lam_bin'] = pd.cut(df_g['lam'], bins=lam_bins, labels=False, include_lowest=True)

            # --- BOUNDARY CALCULATION LOGIC ---
            grouped = df_g.groupby('lam_bin')['b']

            if self.boundary_method == 'quantile':
                # ROBUST METHOD: Use quantiles to ignore outliers (Green Noise)
                stats_min = grouped.quantile(self.boundary_quantile_low)
                stats_max = grouped.quantile(self.boundary_quantile_high)
                stats_count = grouped.count()
            else:
                # LEGACY METHOD: Use absolute Min/Max (Sensitive to outliers)
                agg_stats = grouped.agg(['min', 'max', 'count'])
                stats_min = agg_stats['min']
                stats_max = agg_stats['max']
                stats_count = agg_stats['count']

            x_points = []
            y_min_points = []
            y_max_points = []

            for bin_idx in range(num_bins):
                # Check if bin exists in stats and has enough data
                if bin_idx in stats_count.index and stats_count[bin_idx] > 3:
                    lam_center = (lam_bins[bin_idx] + lam_bins[bin_idx + 1]) / 2.0

                    raw_min = stats_min[bin_idx]
                    raw_max = stats_max[bin_idx]

                    # Apply Hard Bounds (Absolute safety net)
                    safe_min = max(raw_min, self.debt_hard_bound_low)
                    safe_max = min(raw_max, self.debt_hard_bound_high)

                    x_points.append(lam_center)
                    y_min_points.append(safe_min)
                    y_max_points.append(safe_max)

            if len(x_points) < 3:
                print(f"  > Warning: Not enough lambda-coverage for G={g_val}. Keeping previous bounds.")
                continue

            # Extend to edges to prevent extrapolation error
            if x_points[0] > self.mu_min:
                x_points.insert(0, self.mu_min)
                y_min_points.insert(0, y_min_points[0])
                y_max_points.insert(0, y_max_points[0])
            if x_points[-1] < self.mu_max:
                x_points.append(self.mu_max)
                y_min_points.append(y_min_points[-1])
                y_max_points.append(y_max_points[-1])

            try:
                f_min = interp1d(x_points, y_min_points, kind='linear', fill_value="extrapolate")
                f_max = interp1d(x_points, y_max_points, kind='linear', fill_value="extrapolate")
                self.bound_funcs[g_val]['min'] = f_min
                self.bound_funcs[g_val]['max'] = f_max
                updates_made = True

                # Update scalar proxy for general info
                if g_val == 0:
                    self.b_min = np.min(y_min_points)
                    self.b_max = np.max(y_max_points)

            except Exception as e:
                print(f"  > Error fitting boundary curves for G={g_val}: {e}")

        if updates_made:
            print(f"  >>> Dynamic state-dependent debt bounds updated.")


class AdaptiveSampler:
    def __init__(self, b_range, lam_range, g_indices, scorer, config, device):
        self.b_min_initial, self.b_max_initial = b_range
        self.lam_min, self.lam_max = lam_range
        self.g_indices = g_indices
        self.scorer = scorer
        self.device = device
        thresholds = config.get('admissibility_thresholds', {})
        self.threshold_strong = thresholds.get('strong_admissible', 0.7)
        self.threshold_inad = thresholds.get('inadmissible', 0.3)
        self.admissibility_thresholds = thresholds.get('admissibility_thresholds', 0.9)
        self.phase = 'warmup'
        self.iteration = 0
        self.total_adaptive_iters = 0
        self.sampling_stats = defaultdict(list)
        print(f"\n>>> AdaptiveSampler initialized.")

    def set_phase(self, phase, total_adaptive_iters=None):
        self.phase = phase
        if phase == 'adaptive' and total_adaptive_iters is not None:
            self.total_adaptive_iters = total_adaptive_iters
            self.iteration = 0
        print(f"\n>>> Sampling phase set to: {phase.upper()}")

    def sampling_weight(self, A, iteration, total_iters):
        if self.phase == 'warmup': return 1.0
        if A > self.admissibility_thresholds:
            return 1.0
        else:
            return 0.0001

    def sample_batch(self, batch_size, policy_net=None):
        if self.phase == 'warmup':
            samples = self._sample_uniform(batch_size, use_initial_bounds=True)
            return samples, None
        else:
            return self._sample_adaptive(batch_size, policy_net)

    def _sample_uniform(self, batch_size, use_initial_bounds=False):
        samples = torch.zeros((batch_size, 3), device=self.device)
        if use_initial_bounds or not self.scorer.use_dynamic_debt_bounds:
            b_min_to_use = self.b_min_initial
            b_max_to_use = self.b_max_initial
        else:
            b_min_to_use = self.scorer.b_min
            b_max_to_use = self.scorer.b_max
        samples[:, 0] = torch.rand(batch_size, device=self.device) * (b_max_to_use - b_min_to_use) + b_min_to_use
        samples[:, 1] = torch.rand(batch_size, device=self.device) * (self.lam_max - self.lam_min) + self.lam_min
        samples[:, 2] = torch.randint(0, 2, (batch_size,), device=self.device).float()
        return samples

    def _sample_adaptive(self, batch_size, policy_net):
        n_candidates = batch_size * 10
        candidates = self._sample_uniform(n_candidates, use_initial_bounds=False)
        weights = torch.zeros(n_candidates, device=self.device)
        scores = torch.zeros(n_candidates, device=self.device)
        for i in range(n_candidates):
            B, lam, g_idx = candidates[i]
            A = self.scorer.get_cached_score(B, lam, g_idx)
            if A is None: A = self.scorer.compute_score(B, lam, g_idx)
            scores[i] = A
            weights[i] = self.sampling_weight(A, self.iteration, self.total_adaptive_iters)
        weights_sum = weights.sum()
        if weights_sum <= 0:
            weights = torch.ones(n_candidates, device=self.device) / n_candidates
        else:
            weights = weights / weights_sum
        try:
            selected_indices = torch.multinomial(weights, batch_size, replacement=False)
        except RuntimeError:
            selected_indices = torch.randperm(n_candidates, device=self.device)[:batch_size]
        admissible_samples = candidates[selected_indices]
        inadmissible_mask = (scores < self.threshold_inad)
        inadmissible_candidates = candidates[inadmissible_mask]
        inadmissible_samples = None
        if inadmissible_candidates.shape[0] > 0:
            num_to_sample = min(batch_size, inadmissible_candidates.shape[0])
            inad_indices = torch.randperm(inadmissible_candidates.shape[0], device=self.device)[:num_to_sample]
            inadmissible_samples = inadmissible_candidates[inad_indices]
        selected_scores = scores[selected_indices].cpu().numpy()
        self.sampling_stats['iteration'].append(self.iteration)
        self.sampling_stats['mean_A'].append(np.mean(selected_scores))
        self.sampling_stats['frac_strongly_admissible'].append(
            sum(s > self.threshold_strong for s in selected_scores) / len(
                selected_scores) if selected_scores.size > 0 else 0
        )
        self.sampling_stats['frac_transition'].append(0)
        self.sampling_stats['frac_inadmissible'].append(
            sum(s < self.threshold_inad for s in selected_scores) / len(
                selected_scores) if selected_scores.size > 0 else 0
        )
        return admissible_samples, inadmissible_samples

    def increment_iteration(self):
        if self.phase == 'adaptive': self.iteration += 1

    def get_statistics(self):
        return dict(self.sampling_stats)

    def plot_sampling_statistics(self, save_path='figures/adaptive_sampling_stats.png'):
        if len(self.sampling_stats['iteration']) == 0: return
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Adaptive Sampling Statistics', fontsize=16)
        iters = self.sampling_stats['iteration']
        ax = axes[0, 0]
        ax.plot(iters, self.sampling_stats['mean_A'], 'b-', linewidth=2)
        ax.set_title('Average A Score')
        ax.grid(True, alpha=0.3)
        ax = axes[0, 1]
        ax.plot(iters, self.sampling_stats['frac_strongly_admissible'], 'g-', label='Strong')
        ax.plot(iters, self.sampling_stats['frac_inadmissible'], 'r-', label='Inadmissible')
        ax.set_title('Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax = axes[1, 0]
        strong_frac = np.array(self.sampling_stats['frac_strongly_admissible'])
        inad_frac = np.array(self.sampling_stats['frac_inadmissible'])
        ax.fill_between(iters, 0, strong_frac, color='green', alpha=0.3, label='Strong')
        ax.fill_between(iters, strong_frac, strong_frac + inad_frac, color='red', alpha=0.3, label='Inadmissible')
        ax.set_title('Stacked Distribution')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)


class AdmissibilityVisualizer:
    def __init__(self, scorer, b_range, lam_range, zagg_vec, config, device):
        self.scorer = scorer
        self.b_min_initial, self.b_max_initial = b_range
        self.lam_min, self.lam_max = lam_range
        self.zagg_vec = zagg_vec
        self.device = device
        self.cpp_data = None
        self.threshold_strong = config.get('admissibility_thresholds', {}).get('strong_admissible', 0.7)
        self.threshold_inad = config.get('admissibility_thresholds', {}).get('inadmissible', 0.3)

    def load_cpp_data(self, filepath='policy_v6_out_61.txt'):
        if self.cpp_data is not None: return self.cpp_data
        try:
            import pandas as pd
            col_names = ['b0', 'mu0', 'g_index', 'v0', 'b1', 'mu1_g0', 'mu1_g1', 'c', 'l', 'tau', 'q']
            df = pd.read_csv(filepath, header=None, sep='\s+', names=col_names)
            df = df[(df != -500).all(axis=1)]
            self.cpp_data = df
            return df
        except:
            return None

    def plot_admissibility_heatmap(self, n_grid=50, save_path='figures/admissibility_heatmap.png',
                                   overlay_cpp_data=False, cpp_data_file='policy_v6_out_61.txt',
                                   verbose_cpp_analysis=True):
        cpp_df = None
        if overlay_cpp_data: cpp_df = self.load_cpp_data(cpp_data_file)

        b_vals = np.linspace(self.b_min_initial, self.b_max_initial, n_grid)
        lam_vals = np.linspace(self.lam_min, self.lam_max, n_grid)
        B_grid, Lam_grid = np.meshgrid(b_vals, lam_vals)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle('Admissibility Heatmap (State-Dependent Bounds)', fontsize=16)

        for g_idx, (ax, g_name) in enumerate(zip(axes, ['Low G', 'High G'])):
            A_grid = np.zeros_like(B_grid)
            for i in range(n_grid):
                for j in range(n_grid):
                    B = torch.tensor(B_grid[i, j], device=self.device)
                    lam = torch.tensor(Lam_grid[i, j], device=self.device)
                    g = torch.tensor(g_idx, device=self.device)
                    A_grid[i, j] = self.scorer.compute_score(B, lam, g)

            im = ax.contourf(B_grid, Lam_grid, A_grid, levels=np.linspace(0.0, 1.0, 21), cmap='RdYlGn')

            if self.scorer.use_dynamic_debt_bounds:
                lam_line = np.linspace(self.lam_min, self.lam_max, 100)
                bounds = [self.scorer.get_dynamic_bounds(l, g_idx) for l in lam_line]
                b_min_line = [b[0] for b in bounds]
                b_max_line = [b[1] for b in bounds]
                ax.plot(b_min_line, lam_line, 'k--', linewidth=2, label='Dyn. b_min(λ)')
                ax.plot(b_max_line, lam_line, 'k--', linewidth=2, label='Dyn. b_max(λ)')

            if cpp_df is not None:
                df_g = cpp_df[cpp_df['g_index'] == g_idx]
                if len(df_g) > 0:
                    ax.scatter(df_g['b0'], df_g['mu0'], c='blue', s=15, alpha=0.6,
                               edgecolors='darkblue', marker='o', label='C++ Data')

            ax.set_xlabel('B (Debt)')
            ax.set_ylabel('λ (Multiplier)')
            ax.set_title(f'State: {g_name}')
            plt.colorbar(im, ax=ax)
            ax.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Admissibility heatmap saved to: {save_path}")

    def plot_boundary_diagnostics(self, save_path='figures/boundary_diagnostics.png'):
        if len(self.scorer.cache) == 0:
            print("Cannot plot boundary diagnostics: Cache is empty.")
            return

        data = []
        for (b, lam, g), score in self.scorer.cache.items():
            data.append({'b': b, 'lam': lam, 'g': g, 'score': score})
        df = pd.DataFrame(data)

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('Boundary Diagnostics: Data Envelope vs Computed Bounds', fontsize=16)
        lam_line = np.linspace(self.lam_min, self.lam_max, 200)

        for g_idx, (ax, g_name) in enumerate(zip(axes, ['Low G', 'High G'])):
            df_g = df[df['g'] == g_idx]

            # Plot Points
            inad = df_g[df_g['score'] < self.threshold_inad]
            if len(inad) > 0:
                ax.scatter(inad['b'], inad['lam'], c='red', s=5, alpha=0.3, label='Inadmissible')

            adm = df_g[df_g['score'] > self.threshold_strong]
            if len(adm) > 0:
                ax.scatter(adm['b'], adm['lam'], c='green', s=15, alpha=0.6, label='Strong Admissible')

            # Plot Computed Envelope
            if self.scorer.use_dynamic_debt_bounds:
                bounds = [self.scorer.get_dynamic_bounds(l, g_idx) for l in lam_line]
                b_min_line = [b[0] for b in bounds]
                b_max_line = [b[1] for b in bounds]
                ax.plot(b_min_line, lam_line, 'k--', linewidth=3, label='Computed Envelope')
                ax.plot(b_max_line, lam_line, 'k--', linewidth=3)

            ax.set_title(f"State: {g_name} (N_adm={len(adm)})")
            ax.set_xlabel("Debt (B)")
            ax.set_ylabel("Lambda")
            ax.set_xlim(self.scorer.initial_b_min - 0.5, self.scorer.initial_b_max + 0.5)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Boundary diagnostics plot saved to: {save_path}")

    def plot_cache_distribution(self, save_path='figures/cache_distribution.png'):
        if len(self.scorer.cache) == 0: return
        scores = list(self.scorer.cache.values())
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        ax.hist(scores, bins=50, color='steelblue', edgecolor='black', range=(0, 1))
        ax.axvline(x=self.threshold_inad, color='red', linestyle='--')
        ax.axvline(x=self.threshold_strong, color='green', linestyle='--')
        ax.set_title('Score Histogram')
        ax = axes[1]
        if len(scores) > 0:
            s_adm = sum(s > self.threshold_strong for s in scores) / len(scores)
            s_inad = sum(s < self.threshold_inad for s in scores) / len(scores)
            s_trans = 1.0 - s_adm - s_inad
            ax.pie([s_adm, s_trans, s_inad], labels=['Strong', 'Trans', 'Inad'],
                   colors=['green', 'orange', 'red'], autopct='%1.1f%%')
        ax.set_title('Cache Composition')
        plt.savefig(save_path)
        plt.close(fig)


def initialize_adaptive_sampling(policy_net, config, device):
    if not hasattr(policy_net, 'mu_min'):
        policy_net.mu_min = config['mu_min']

    scorer = AdmissibilityScorer(policy_net, config, device)
    initial_b_range = (scorer.b_min, scorer.b_max)

    sampler = AdaptiveSampler(
        b_range=initial_b_range,
        lam_range=(config['mu_min'], config['mu_max']),
        g_indices=[0, 1],
        scorer=scorer,
        config=config,
        device=device
    )

    visualizer = AdmissibilityVisualizer(
        scorer=scorer,
        b_range=initial_b_range,
        lam_range=(config['mu_min'], config['mu_max']),
        zagg_vec=config['zagg_vec'],
        config=config,
        device=device
    )

    print("\n" + "=" * 60)
    print("Adaptive Sampling Initialized (STATE-DEPENDENT BOUNDS)")
    print(f"Boundary Method: {scorer.boundary_method.upper()}")
    if scorer.boundary_method == 'quantile':
        print(f"Percentiles: {scorer.boundary_quantile_low * 100}% - {scorer.boundary_quantile_high * 100}%")
    print("=" * 60)
    return scorer, sampler, visualizer


def update_cache_periodic(scorer, sampler, n_grid=1000):
    print(f"\n>>> Updating admissibility cache with {n_grid} grid samples (Iterative Refinement)...")

    # 1. Generate a FIXED grid of candidates to evaluate
    # We use a larger multiplier here to ensure good coverage for the boundary definition
    # We use initial bounds to ensure we scan the whole potentially valid space
    grid_samples = sampler._sample_uniform(n_grid * 2, use_initial_bounds=True)

    # 2. Iterative Refinement Loop
    # We loop to find the fixed point where Bounds match the Admissibility implied by those Bounds.
    n_refinement_steps = 2

    for step in range(n_refinement_steps):
        # a. Clear cache to ensure no stale points influence the bounds
        scorer.clear_cache()

        # b. Compute scores using the CURRENT bounds (which tighten every step)
        scores = scorer.compute_score_batch(grid_samples)

        # c. Populate cache
        scorer.update_cache(grid_samples, scores)

        # d. Identify admissible points
        admissible_keys = [k for k, v in scorer.cache.items() if v > sampler.threshold_strong]

        # e. Update Bounds (The envelope tightens based on the current valid set)
        prev_b_min = scorer.b_min
        scorer.update_dynamic_debt_bounds(admissible_keys)

        print(
            f"    Refinement step {step + 1}/{n_refinement_steps}: {len(admissible_keys)} admissible points (Global b_min: {prev_b_min:.3f} -> {scorer.b_min:.3f})")

    # 3. FINAL SYNC: Re-score all points against the finalized bounds
    # This is crucial because points that were "green" in step 2 might effectively become "red"
    # once the bounds tighten in the final update. We want the scatter plot to reflect the final reality.
    print("    >>> Synchronizing cache with final bounds for plotting consistency...")

    # Extract all keys currently in the cache
    keys = list(scorer.cache.keys())
    if len(keys) > 0:
        # Convert keys (tuples) back to a tensor batch
        states_tensor = torch.tensor(keys, device=scorer.device, dtype=torch.float32)

        # Re-compute scores using the FINAL bounds from step 3
        new_scores = scorer.compute_score_batch(states_tensor)

        # Update the cache with these new scores
        scorer.update_cache(states_tensor, new_scores)

    print("    >>> Cache and Boundaries fully synchronized.")