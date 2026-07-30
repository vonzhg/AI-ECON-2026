"""
Visualization Module for Heterogeneous Agent Model.

Provides plotting functions for:
- Training progress (losses over iterations)
- Epoch-level loss details
- Boundary visualization (3D projections, 2D slices)
- Policy function surfaces
- Diagnostic plots (Q distribution, FB violations, projection stats)
- NEW: Euler discrepancy and asset choice tracking
- NEW: Timing analysis
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import os


class HAVisualizer:
    """
    Visualization utilities for the HA Ramsey model.

    All plots are saved to the specified save_dir.
    Existing files are NOT deleted - new plots overwrite only their specific files.

    To preserve all plots across runs, either:
    1. Use different output directories per run
    2. Manually backup figures before re-running
    3. Set version_prefix to create versioned filenames
    """

    def __init__(self, config, device, save_dir='output/figures', version_prefix=None):
        self.config = config
        self.device = device
        self.save_dir = save_dir

        # Optional version prefix for all saved figures
        # e.g., version_prefix="v1.0.0" -> "v1.0.0_training_progress.png"
        self.version_prefix = version_prefix

        # Create directory without clearing existing contents
        os.makedirs(save_dir, exist_ok=True)

        # Log existing files (for awareness)
        existing = [f for f in os.listdir(save_dir) if f.endswith('.png')]
        if existing:
            print(f"   Note: {len(existing)} existing figures in {save_dir}/")

        sb = config['state_bounds']
        self.mean_c = (sb['c_max'] + sb['c_min']) / 2.0
        self.mean_K = (sb['K_max'] + sb['K_min']) / 2.0

    def _get_filepath(self, filename):
        """Get filepath with optional version prefix."""
        if self.version_prefix:
            filename = f"{self.version_prefix}_{filename}"
        return f"{self.save_dir}/{filename}"

    def plot_losses(self, actor_metrics, critic_losses, raw_metrics=None,
                   euler_asset_metrics=None, timing_metrics=None):
        """
        Plot iteration-level training progress.

        Creates a 3x3 grid showing:
        Row 1:
        - Critic TD loss
        - Actor total loss
        - Actor value component (-welfare)
        Row 2:
        - FB penalty (weighted + raw on right axis)
        - Bound penalty (weighted + raw on right axis)
        - Shape penalty (weighted + raw on right axis)
        Row 3 (NEW):
        - Euler discrepancies (φᵉ, φᵘ)
        - Asset choices (a'ᵉ, a'ᵘ)
        - Timing breakdown

        Args:
            actor_metrics: Dict with lists for 'val', 'fb', 'bound', 'shape', 'total'
            critic_losses: List of critic losses per iteration
            raw_metrics: Dict with lists for raw (unweighted) 'fb', 'bound', 'shape'
            euler_asset_metrics: Dict with 'phi_e', 'phi_u', 'a_prime_e', 'a_prime_u'
            timing_metrics: List of timing dicts per iteration
        """
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        iterations = range(1, len(critic_losses) + 1)

        # ==================== Row 1 ====================

        # 1. Critic Loss
        ax = axes[0, 0]
        ax.plot(iterations, critic_losses, 'o-', color='orange', linewidth=2, markersize=4)
        ax.set_title('Critic: TD Loss', fontsize=12)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('MSE Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # 2. Actor Total Loss
        ax = axes[0, 1]
        if actor_metrics['total']:
            n = len(actor_metrics['total'])
            ax.plot(range(1, n+1), actor_metrics['total'],
                   'o-', color='black', linewidth=2, markersize=4)
        ax.set_title('Actor: Total Loss', fontsize=12)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)

        # 3. Actor Value Component
        ax = axes[0, 2]
        if actor_metrics['val']:
            n = len(actor_metrics['val'])
            ax.plot(range(1, n+1), actor_metrics['val'],
                   'o-', color='blue', linewidth=2, markersize=4)
        ax.set_title('Actor: -Value (Welfare)', fontsize=12)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('-V(s)')
        ax.grid(True, alpha=0.3)

        # ==================== Row 2 ====================

        # 4. FB Penalty (weighted + raw)
        ax = axes[1, 0]
        if actor_metrics['fb']:
            n = len(actor_metrics['fb'])
            iters = range(1, n+1)
            line1, = ax.plot(iters, actor_metrics['fb'], 'o-', color='red',
                           linewidth=2, markersize=4, label='Weighted (λ×FB²)')
            ax.set_ylabel('Weighted Penalty', color='red')
            ax.tick_params(axis='y', labelcolor='red')

            # Raw penalty on right axis
            if raw_metrics and 'fb' in raw_metrics and len(raw_metrics['fb']) >= n:
                ax2 = ax.twinx()
                line2, = ax2.plot(iters, raw_metrics['fb'][:n], 's--', color='darkred',
                                linewidth=1.5, markersize=3, label='Raw FB²')
                ax2.set_ylabel('Raw |Φ|²', color='darkred')
                ax2.tick_params(axis='y', labelcolor='darkred')
                if any(v > 0 for v in raw_metrics['fb'][:n]):
                    ax2.set_yscale('log')
                ax.legend([line1, line2], ['Weighted', 'Raw'], loc='upper right')
        ax.set_title('FB Penalty (Equilibrium)', fontsize=12)
        ax.set_xlabel('Iteration')
        if actor_metrics['fb'] and any(v > 0 for v in actor_metrics['fb']):
            ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # 5. Bound Penalty (weighted + raw)
        ax = axes[1, 1]
        if actor_metrics['bound']:
            n = len(actor_metrics['bound'])
            iters = range(1, n+1)
            line1, = ax.plot(iters, actor_metrics['bound'], 'o-', color='purple',
                           linewidth=2, markersize=4, label='Weighted')
            ax.set_ylabel('Weighted Penalty', color='purple')
            ax.tick_params(axis='y', labelcolor='purple')

            if raw_metrics and 'bound' in raw_metrics and len(raw_metrics['bound']) >= n:
                ax2 = ax.twinx()
                line2, = ax2.plot(iters, raw_metrics['bound'][:n], 's--', color='indigo',
                                linewidth=1.5, markersize=3, label='Raw')
                ax2.set_ylabel('Raw Violation²', color='indigo')
                ax2.tick_params(axis='y', labelcolor='indigo')
                if any(v > 0 for v in raw_metrics['bound'][:n]):
                    ax2.set_yscale('log')
                ax.legend([line1, line2], ['Weighted', 'Raw'], loc='upper right')
        ax.set_title('Hypercube Penalty (Asset Bounds)', fontsize=12)
        ax.set_xlabel('Iteration')
        if actor_metrics['bound'] and any(v > 0 for v in actor_metrics['bound']):
            ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # 6. Shape Penalty (weighted + raw)
        ax = axes[1, 2]
        if 'shape' in actor_metrics and actor_metrics['shape']:
            n = len(actor_metrics['shape'])
            iters = range(1, n+1)
            line1, = ax.plot(iters, actor_metrics['shape'], 'o-', color='green',
                           linewidth=2, markersize=4, label='Weighted')
            ax.set_ylabel('Weighted Penalty', color='green')
            ax.tick_params(axis='y', labelcolor='green')

            if raw_metrics and 'shape' in raw_metrics and len(raw_metrics['shape']) >= n:
                ax2 = ax.twinx()
                line2, = ax2.plot(iters, raw_metrics['shape'][:n], 's--', color='darkgreen',
                                linewidth=1.5, markersize=3, label='Raw')
                ax2.set_ylabel('Raw Proj Dist²', color='darkgreen')
                ax2.tick_params(axis='y', labelcolor='darkgreen')
                if any(v > 0 for v in raw_metrics['shape'][:n]):
                    ax2.set_yscale('log')
                ax.legend([line1, line2], ['Weighted', 'Raw'], loc='upper right')
        ax.set_title('α-Shape Penalty (Projection)', fontsize=12)
        ax.set_xlabel('Iteration')
        if 'shape' in actor_metrics and actor_metrics['shape'] and any(v > 0 for v in actor_metrics['shape']):
            ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # ==================== Row 3 (NEW) ====================

        # 7. FB Residuals (Complementarity) - This is what should be ≈ 0
        ax = axes[2, 0]
        if euler_asset_metrics and euler_asset_metrics.get('fb_e'):
            n = len(euler_asset_metrics['fb_e'])
            iters = range(1, n+1)
            ax.plot(iters, euler_asset_metrics['fb_e'], 'o-', color='blue',
                   linewidth=2, markersize=4, label='Φᵉ (employed)')
            ax.plot(iters, euler_asset_metrics['fb_u'], 's-', color='red',
                   linewidth=2, markersize=4, label='Φᵘ (unemployed)')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.legend(loc='best')
        ax.set_title('FB Residuals (Φ) - should be ≈ 0', fontsize=12)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('FB Residual (raw)')
        ax.grid(True, alpha=0.3)

        # 8. Asset Choices with Euler discrepancy on secondary axis
        ax = axes[2, 1]
        if euler_asset_metrics and euler_asset_metrics.get('a_prime_e'):
            n = len(euler_asset_metrics['a_prime_e'])
            iters = range(1, n+1)

            # Primary axis: Asset choices
            line1, = ax.plot(iters, euler_asset_metrics['a_prime_e'], 'o-', color='blue',
                   linewidth=2, markersize=4, label="a'ᵉ")
            line2, = ax.plot(iters, euler_asset_metrics['a_prime_u'], 's-', color='red',
                   linewidth=2, markersize=4, label="a'ᵘ")

            # Add reference lines for bounds
            sb = self.config['state_bounds']
            ax.axhline(y=sb['a_min'], color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=2)
            ax.set_ylabel("Asset Choice (a')", color='blue')
            ax.tick_params(axis='y', labelcolor='blue')

            # Secondary axis: Euler discrepancies (for context)
            if euler_asset_metrics.get('phi_e'):
                ax2 = ax.twinx()
                line3, = ax2.plot(iters, euler_asset_metrics['phi_e'], '^--', color='lightblue',
                       linewidth=1, markersize=3, alpha=0.7, label='φᵉ')
                line4, = ax2.plot(iters, euler_asset_metrics['phi_u'], 'v--', color='lightcoral',
                       linewidth=1, markersize=3, alpha=0.7, label='φᵘ')
                ax2.set_ylabel('Euler Discrepancy (φ)', color='gray')
                ax2.tick_params(axis='y', labelcolor='gray')
                ax.legend([line1, line2, line3, line4],
                         ["a'ᵉ", "a'ᵘ", "φᵉ", "φᵘ"], loc='best', fontsize=8)
            else:
                ax.legend(loc='best')
        ax.set_title("Assets (a') & Euler Discrepancy (φ)", fontsize=12)
        ax.set_xlabel('Iteration')
        ax.grid(True, alpha=0.3)

        # 9. Timing Breakdown
        ax = axes[2, 2]
        if timing_metrics and len(timing_metrics) > 0:
            n = len(timing_metrics)
            iters = range(1, n+1)

            # Extract timing components
            level1_times = [t.get('level1', 0) for t in timing_metrics]
            data_prep_times = [t.get('data_prep', 0) for t in timing_metrics]
            critic_times = [t.get('critic_training', 0) for t in timing_metrics]
            actor_times = [t.get('actor_training', 0) for t in timing_metrics]
            total_times = [t.get('total', 0) for t in timing_metrics]

            # Stacked area plot
            ax.fill_between(iters, 0, level1_times, alpha=0.7, label='Level 1 (Domain)', color='blue')
            ax.fill_between(iters, level1_times,
                          np.array(level1_times) + np.array(data_prep_times),
                          alpha=0.7, label='Data Prep', color='green')
            ax.fill_between(iters,
                          np.array(level1_times) + np.array(data_prep_times),
                          np.array(level1_times) + np.array(data_prep_times) + np.array(critic_times),
                          alpha=0.7, label='Critic', color='orange')
            ax.fill_between(iters,
                          np.array(level1_times) + np.array(data_prep_times) + np.array(critic_times),
                          np.array(level1_times) + np.array(data_prep_times) + np.array(critic_times) + np.array(actor_times),
                          alpha=0.7, label='Actor', color='red')

            # Also plot total as a line
            ax.plot(iters, total_times, 'k--', linewidth=2, label='Total')
            ax.legend(loc='upper left', fontsize=8)
        ax.set_title('Timing Breakdown (seconds)', fontsize=12)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (s)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self._get_filepath("training_progress.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_epoch_losses(self, epoch_data, iteration):
        """
        Plot epoch-level losses within a single iteration.

        Useful for debugging convergence within each iteration.

        Args:
            epoch_data: Dict with lists for critic and actor component losses
            iteration: Current global iteration number
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Critic epochs
        ax = axes[0]
        if epoch_data['critic']:
            epochs = range(1, len(epoch_data['critic']) + 1)
            ax.plot(epochs, epoch_data['critic'], 'o-', color='orange', linewidth=2)
            ax.set_title(f'Critic Loss - Iteration {iteration}', fontsize=12)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE Loss')
            if any(v > 0 for v in epoch_data['critic']):
                ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

        # Actor epochs (all components)
        ax = axes[1]
        if epoch_data['actor_total']:
            epochs = range(1, len(epoch_data['actor_total']) + 1)
            ax.plot(epochs, epoch_data['actor_total'], 'o-', color='black',
                   linewidth=2, label='Total')
            ax.plot(epochs, epoch_data['actor_val'], 's--', color='blue',
                   linewidth=1.5, label='Value')
            ax.plot(epochs, epoch_data['actor_fb'], '^--', color='red',
                   linewidth=1.5, label='FB')
            ax.plot(epochs, epoch_data['actor_bound'], 'v--', color='purple',
                   linewidth=1.5, label='Bound')
            if 'actor_shape' in epoch_data and epoch_data['actor_shape']:
                ax.plot(epochs, epoch_data['actor_shape'], 'd--', color='green',
                       linewidth=1.5, label='Shape')
            ax.set_title(f'Actor Loss - Iteration {iteration}', fontsize=12)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self._get_filepath(f"epoch_losses_iter_{iteration}.png"), dpi=150)
        plt.close()

    def plot_boundary_3d(self, boundary, iteration, sample_points=None):
        """
        Visualize the α-shape boundary in 3D (K, a_e, a_u projection).

        Shows:
        - Gray points: Interior of α-shape
        - Green points: Valid sample points from training

        Note: This is a 3D projection of the 5D state space,
        with c^e and c^u fixed at their mean values.
        """
        sb = self.config['state_bounds']
        n_grid = 25

        k_vals = np.linspace(sb['K_min'], sb['K_max'], n_grid)
        a_vals = np.linspace(sb['a_min'], sb['a_max'], n_grid)

        K_g, ae_g, au_g = np.meshgrid(k_vals, a_vals, a_vals)
        N = K_g.size

        K_flat = torch.tensor(K_g.flatten(), dtype=torch.float32).unsqueeze(1)
        ae_flat = torch.tensor(ae_g.flatten(), dtype=torch.float32).unsqueeze(1)
        au_flat = torch.tensor(au_g.flatten(), dtype=torch.float32).unsqueeze(1)
        c_fixed = torch.full((N, 1), self.mean_c)

        query_states = torch.cat([K_flat, ae_flat, au_flat, c_fixed, c_fixed], dim=1)
        query_states = query_states.to(self.device)

        is_in = boundary.is_admissible(query_states).cpu().numpy().flatten()

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        interior_count = np.sum(is_in)
        if interior_count > 0:
            ax.scatter(K_g.flatten()[is_in], ae_g.flatten()[is_in], au_g.flatten()[is_in],
                       c='lightgray', alpha=0.15, s=8, label=f'α-Shape Interior ({interior_count})')

        if sample_points is not None:
            pts_np = sample_points.detach().cpu().numpy()
            if len(pts_np) > 1500:
                indices = np.random.choice(len(pts_np), 1500, replace=False)
                pts_np = pts_np[indices]

            ax.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2],
                       c='green', s=15, alpha=0.6, label=f'Valid Samples ({len(pts_np)})')

        ax.set_xlabel('Capital (K)', fontsize=11)
        ax.set_ylabel('Asset Emp (aᵉ)', fontsize=11)
        ax.set_zlabel('Asset Unemp (aᵘ)', fontsize=11)
        ax.set_title(f'Admissible Region (α-Shape) - Iteration {iteration}\n'
                     f'(c fixed at {self.mean_c:.2f})', fontsize=12)
        ax.legend(loc='upper left')

        ax.set_xlim([sb['K_min'], sb['K_max']])
        ax.set_ylim([sb['a_min'], sb['a_max']])
        ax.set_zlim([sb['a_min'], sb['a_max']])

        plt.savefig(self._get_filepath(f"boundary_3d_iter_{iteration}.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_boundary_2d_slices(self, boundary, iteration):
        """
        Plot 2D slices of the boundary at different K values.

        Shows the admissible region in (a^e, a^u) space for fixed K.
        """
        sb = self.config['state_bounds']
        n_grid = 50

        K_slices = [sb['K_min'] + 0.2, (sb['K_max'] + sb['K_min'])/2, sb['K_max'] - 0.2]
        a_vals = np.linspace(sb['a_min'], sb['a_max'], n_grid)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, K_val in enumerate(K_slices):
            ae_g, au_g = np.meshgrid(a_vals, a_vals)
            N = ae_g.size

            K_fixed = torch.full((N, 1), K_val)
            ae_flat = torch.tensor(ae_g.flatten(), dtype=torch.float32).unsqueeze(1)
            au_flat = torch.tensor(au_g.flatten(), dtype=torch.float32).unsqueeze(1)
            c_fixed = torch.full((N, 1), self.mean_c)

            query = torch.cat([K_fixed, ae_flat, au_flat, c_fixed, c_fixed], dim=1).to(self.device)
            is_in = boundary.is_admissible(query).cpu().numpy().reshape(n_grid, n_grid)

            ax = axes[idx]
            ax.contourf(ae_g, au_g, is_in.astype(float), levels=[0.5, 1.5],
                       colors=['lightblue'], alpha=0.5)
            ax.contour(ae_g, au_g, is_in.astype(float), levels=[0.5], colors=['blue'])
            ax.set_xlabel('aᵉ')
            ax.set_ylabel('aᵘ')
            ax.set_title(f'K = {K_val:.2f}')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Boundary Slices - Iteration {iteration}', fontsize=12)
        plt.tight_layout()
        plt.savefig(self._get_filepath(f"boundary_2d_iter_{iteration}.png"), dpi=150)
        plt.close()

    def plot_policy_surface(self, model, iteration):
        """
        Plot policy functions (n_e, c'_e, c'_u) over the asset space.
        """
        sb = self.config['state_bounds']
        n_grid = 40

        a_vals = np.linspace(sb['a_min'], sb['a_max'], n_grid)
        ae_g, au_g = np.meshgrid(a_vals, a_vals)
        N = ae_g.size

        states = torch.zeros(N, 5, device=self.device)
        states[:, 0] = self.mean_K
        states[:, 1] = torch.tensor(ae_g.flatten())
        states[:, 2] = torch.tensor(au_g.flatten())
        states[:, 3] = self.mean_c
        states[:, 4] = self.mean_c

        model.eval()
        with torch.no_grad():
            out = model.forward_physics(states)
            if out is None:
                return

            n_e = out['controls']['n_e'].cpu().numpy().reshape(n_grid, n_grid)
            c_prime_e = out['controls']['c_prime_e'].cpu().numpy().reshape(n_grid, n_grid)
            c_prime_u = out['controls']['c_prime_u'].cpu().numpy().reshape(n_grid, n_grid)
            Q = out['physics']['Q'].cpu().numpy().reshape(n_grid, n_grid)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        data = [
            (n_e, 'Labor Supply nᵉ', 'viridis'),
            (c_prime_e, "Future Cons c'ᵉ", 'plasma'),
            (c_prime_u, "Future Cons c'ᵘ", 'plasma'),
            (Q, 'Bond Price Q', 'coolwarm')
        ]

        for idx, (arr, title, cmap) in enumerate(data):
            ax = axes[idx // 2, idx % 2]
            im = ax.contourf(ae_g, au_g, arr, levels=20, cmap=cmap)
            ax.set_xlabel('aᵉ')
            ax.set_ylabel('aᵘ')
            ax.set_title(title)
            plt.colorbar(im, ax=ax)

        plt.suptitle(f'Policy Functions at K={self.mean_K:.2f}, c={self.mean_c:.2f} - Iter {iteration}')
        plt.tight_layout()
        plt.savefig(self._get_filepath(f"policy_surface_iter_{iteration}.png"), dpi=150)
        plt.close()

    def plot_diagnostics(self, tracker, iteration):
        """
        Plot diagnostic information: Q distribution, FB violations, projections.

        Args:
            tracker: LossTracker object with diagnostic data
            iteration: Current iteration number
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 1. FB violations over iterations
        ax = axes[0]
        if tracker.fb_violations:
            iters = range(1, len(tracker.fb_violations) + 1)
            ax.plot(iters, tracker.fb_violations, 'o-', color='red', linewidth=2)
            ax.axhline(y=self.config['fischer_burmeister']['penalty_threshold'],
                      color='gray', linestyle='--', label='Threshold')
            ax.set_title('FB Violation Over Training')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Mean |Φ|²')
            if any(v > 0 for v in tracker.fb_violations):
                ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 2. Q statistics
        ax = axes[1]
        if tracker.Q_stats:
            iters = range(1, len(tracker.Q_stats) + 1)
            means = [s[0] for s in tracker.Q_stats]
            stds = [s[1] for s in tracker.Q_stats]

            ax.errorbar(iters, means, yerr=stds, fmt='o-', color='blue',
                       capsize=3, linewidth=2)
            ax.axhline(y=self.config['admissibility']['Q_min'],
                      color='gray', linestyle='--',
                      label=f"Q_min={self.config['admissibility']['Q_min']}")
            ax.axhline(y=self.config['admissibility']['Q_max'],
                      color='gray', linestyle=':',
                      label=f"Q_max={self.config['admissibility']['Q_max']}")
            ax.set_title('Bond Price Q Over Training')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Q (mean ± std)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 3. Projection statistics
        ax = axes[2]
        if tracker.projection_stats:
            iters = range(1, len(tracker.projection_stats) + 1)
            n_projected = [s[0] for s in tracker.projection_stats]
            mean_dist = [s[1] for s in tracker.projection_stats]

            ax2 = ax.twinx()

            line1 = ax.bar(iters, n_projected, color='orange', alpha=0.7, label='N projected')
            line2, = ax2.plot(iters, mean_dist, 'o-', color='green', linewidth=2, label='Mean dist')

            ax.set_xlabel('Iteration')
            ax.set_ylabel('Number of Projections', color='orange')
            ax2.set_ylabel('Mean Projection Distance', color='green')
            ax.set_title('α-Shape Projections')

            ax.legend([line1, line2], ['N projected', 'Mean dist'], loc='upper right')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self._get_filepath(f"diagnostics_iter_{iteration}.png"), dpi=150)
        plt.close()