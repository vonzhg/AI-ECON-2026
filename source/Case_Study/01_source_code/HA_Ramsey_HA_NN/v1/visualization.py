"""
Visualization Module for Heterogeneous Agent Model.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import os

class HAVisualizer:
    # REVISED: Default save_dir to 'output/figures'
    def __init__(self, config, device, save_dir='output/figures'):
        self.config = config
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        sb = config['state_bounds']
        self.mean_c = (sb['c_max'] + sb['c_min']) / 2.0
        self.mean_K = (sb['K_max'] + sb['K_min']) / 2.0

    def plot_losses(self, actor_losses, critic_losses):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(actor_losses, label='Actor Loss', color='blue')
        plt.title('Actor Training')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(critic_losses, label='Critic Loss', color='orange')
        plt.title('Critic Training')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/training_progress.png", dpi=150)
        plt.close()

    def plot_fb_penalty(self, fb_penalties):
        plt.figure(figsize=(8, 5))
        plt.plot(fb_penalties, label='FB Penalty', color='red')
        plt.title('Fischer-Burmeister Penalty (Complementarity Violation)')
        plt.xlabel('Iteration')
        plt.ylabel('Penalty')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/fb_penalty.png", dpi=150)
        plt.close()

    def plot_boundary_3d(self, boundary, iteration, sample_points=None):
        sb = self.config['state_bounds']
        n_grid = 30
        k_vals = np.linspace(sb['K_min'], sb['K_max'], n_grid)
        a_vals = np.linspace(sb['a_min'], sb['a_max'], n_grid)

        K_g, ae_g, au_g = np.meshgrid(k_vals, a_vals, a_vals)
        N = K_g.size
        K_flat = torch.tensor(K_g.flatten(), dtype=torch.float32).unsqueeze(1)
        ae_flat = torch.tensor(ae_g.flatten(), dtype=torch.float32).unsqueeze(1)
        au_flat = torch.tensor(au_g.flatten(), dtype=torch.float32).unsqueeze(1)

        c_fixed = torch.full((N, 1), self.mean_c)
        query_states = torch.cat([K_flat, ae_flat, au_flat, c_fixed, c_fixed], dim=1).to(self.device)

        is_in = boundary.is_admissible(query_states).cpu().numpy().flatten()

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        if np.sum(is_in) > 0:
            ax.scatter(K_g.flatten()[is_in], ae_g.flatten()[is_in], au_g.flatten()[is_in],
                       c='lightgray', alpha=0.1, s=10, label='α-Shape Interior')

        if sample_points is not None:
            pts_np = sample_points.detach().cpu().numpy()
            if len(pts_np) > 1000:
                indices = np.random.choice(len(pts_np), 1000, replace=False)
                pts_np = pts_np[indices]

            ax.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2],
                       c='green', s=20, alpha=0.8, label='Valid Samples')

        ax.set_xlabel('Capital (K)')
        ax.set_ylabel('Asset Emp (aᵉ)')
        ax.set_zlabel('Asset Unemp (aᵘ)')
        ax.set_title(f'Admissible Region (α-Shape) - Iter {iteration}')
        ax.legend()

        plt.savefig(f"{self.save_dir}/boundary_3d_iter_{iteration}.png", dpi=150)
        plt.close()

    def plot_boundary_2d(self, boundary, iteration, sample_points=None):
        sb = self.config['state_bounds']
        n_grid = 50
        a_vals = np.linspace(sb['a_min'], sb['a_max'], n_grid)
        ae_g, au_g = np.meshgrid(a_vals, a_vals)

        N = ae_g.size
        ae_flat = torch.tensor(ae_g.flatten(), dtype=torch.float32).unsqueeze(1)
        au_flat = torch.tensor(au_g.flatten(), dtype=torch.float32).unsqueeze(1)

        c_fixed = torch.full((N, 1), self.mean_c)
        k_fixed = torch.full((N, 1), self.mean_K)

        query_states = torch.cat([k_fixed, ae_flat, au_flat, c_fixed, c_fixed], dim=1).to(self.device)
        is_in = boundary.is_admissible(query_states).cpu().numpy().flatten()

        plt.figure(figsize=(10, 8))
        Z = is_in.reshape(n_grid, n_grid).astype(int)
        plt.contourf(ae_g, au_g, Z, levels=[0.5, 1.0], colors=['#e6ffe6'], alpha=0.7)
        plt.contour(ae_g, au_g, Z, levels=[0.5], colors=['green'], linewidths=2)

        if sample_points is not None:
            pts_np = sample_points.detach().cpu().numpy()
            k_range = sb['K_max'] - sb['K_min']
            epsilon = k_range * 0.05
            mask = np.abs(pts_np[:, 0] - self.mean_K) < epsilon
            slice_points = pts_np[mask]

            if len(slice_points) > 0:
                plt.scatter(slice_points[:, 1], slice_points[:, 2],
                           c='black', s=15, alpha=0.5,
                           label=f'Samples (K ≈ {self.mean_K:.2f})')
                plt.legend(loc='upper right')

        plt.xlabel('Asset Employed (aᵉ)')
        plt.ylabel('Asset Unemployed (aᵘ)')
        plt.title(f'Admissible Slice (K={self.mean_K:.2f}, α-Shape) - Iter {iteration}')
        plt.grid(True, alpha=0.3)

        plt.savefig(f"{self.save_dir}/boundary_2d_iter_{iteration}.png", dpi=150)
        plt.close()