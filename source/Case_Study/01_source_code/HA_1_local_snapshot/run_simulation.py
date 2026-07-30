"""
Simulation Runner for Heterogeneous Agent Ramsey Model.

This script provides functionality to:
1. Train the model (or load a pre-trained model)
2. Simulate equilibrium paths from various initial conditions
3. Analyze policy functions and economic dynamics
4. Generate comprehensive visualization and reports

Usage:
    python run_simulation.py --mode train      # Train the model
    python run_simulation.py --mode simulate   # Run simulations with trained model
    python run_simulation.py --mode full       # Train and simulate
    python run_simulation.py --mode analyze    # Analyze a trained model
"""

import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import argparse
from datetime import datetime
import pickle

from ha_model import HAModel
from boundary import AlphaBoundary
from visualization import HAVisualizer
from dashboard import train, load_config_json, generate_random_states


class SimulationRunner:
    """
    Main simulation runner for the Heterogeneous Agent Ramsey Model.
    
    Implements the algorithm described in the document:
    - Two-level fixed-point iteration for policy and domain discovery
    - α-shape boundary learning for admissible set
    - Actor-critic training with Fischer-Burmeister complementarity handling
    """
    
    def __init__(self, config_file='config.json', output_dir='results'):
        """Initialize the simulation runner."""
        self.config = load_config_json(config_file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/figures", exist_ok=True)
        os.makedirs(f"{output_dir}/data", exist_ok=True)
        
        # Initialize model and boundary
        self.model = HAModel(self.config, self.device)
        self.boundary = AlphaBoundary(self.config)
        self.viz = HAVisualizer(self.config, self.device, save_dir=f"{output_dir}/figures")
        
        print(f"Simulation Runner initialized on device: {self.device}")
        self._print_config_summary()
    
    def _print_config_summary(self):
        """Print a summary of the configuration."""
        econ = self.config['economic_parameters']
        print("\n" + "="*60)
        print("           MODEL CONFIGURATION SUMMARY")
        print("="*60)
        print(f"Economic Parameters:")
        print(f"  β (discount)     = {econ['beta']}")
        print(f"  α (capital share)= {econ['alpha']}")
        print(f"  σ (risk aversion)= {econ['sigma']} {'(Log Utility)' if econ['sigma'] == 1.0 else '(CRRA)'}")
        print(f"  γ (inv. Frisch)  = {econ['gamma']}")
        print(f"  δ (depreciation) = {econ['delta']}")
        print(f"  π^e (emp. share) = {econ['pi_e']}")
        print(f"  Transition Matrix:")
        print(f"    [{econ['pi_matrix'][0][0]:.2f}, {econ['pi_matrix'][0][1]:.2f}]")
        print(f"    [{econ['pi_matrix'][1][0]:.2f}, {econ['pi_matrix'][1][1]:.2f}]")
        print("="*60 + "\n")
    
    def load_model(self, model_path='ha_model_final.pth'):
        """Load a pre-trained model."""
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded from {model_path}")
            return True
        else:
            print(f"Model file not found: {model_path}")
            return False
    
    def save_model(self, model_path=None):
        """Save the current model."""
        if model_path is None:
            model_path = f"{self.output_dir}/ha_model_trained.pth"
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    def simulate_trajectory(self, initial_state, num_periods=50, verbose=False):
        """
        Simulate an equilibrium trajectory from an initial state.
        
        Args:
            initial_state: torch.Tensor of shape (5,) or (1, 5) with (K, a^e, a^u, c^e, c^u)
            num_periods: Number of periods to simulate
            verbose: Print progress
            
        Returns:
            Dictionary containing trajectory data
        """
        self.model.eval()
        
        if initial_state.dim() == 1:
            initial_state = initial_state.unsqueeze(0)
        
        # Storage for trajectory
        trajectory = {
            'states': [],           # (K, a^e, a^u, c^e, c^u)
            'controls': [],         # (n^e, c'^e, c'^u)
            'prices': [],           # (Q, w_hat)
            'welfare': [],          # Period welfare
            'fb_residuals': [],     # (Φ^e, Φ^u)
            'admissibility': []     # Admissibility scores
        }
        
        current_state = initial_state.to(self.device)
        
        with torch.no_grad():
            for t in range(num_periods):
                # Store current state
                trajectory['states'].append(current_state.cpu().numpy().flatten())
                
                # Forward pass through model
                out = self.model.forward_physics(current_state)
                
                if out is None:
                    print(f"Simulation terminated at period {t} due to numerical issues")
                    break
                
                # Extract and store controls (from actor output)
                raw_out = self.model.actor(current_state)
                n_e = (torch.sigmoid(raw_out[:, 0:1]) * (self.model.n_max - self.model.n_min) + self.model.n_min)
                c_prime_e = torch.exp(raw_out[:, 1:2]) * self.model.c_scale
                c_prime_u = torch.exp(raw_out[:, 2:3]) * self.model.c_scale
                
                trajectory['controls'].append([
                    n_e.item(), 
                    c_prime_e.item(), 
                    c_prime_u.item()
                ])
                
                # Store prices
                Q = out['physics']['Q']
                K = current_state[:, 0:1]
                c_e = current_state[:, 3:4]
                w_hat = (n_e ** self.model.gamma) * (c_e ** self.model.sigma)
                trajectory['prices'].append([Q.item(), w_hat.item()])
                
                # Store welfare
                trajectory['welfare'].append(out['welfare'].item())
                
                # Store FB residuals
                fb_e, fb_u = out['fb_residuals']
                trajectory['fb_residuals'].append([fb_e.item(), fb_u.item()])
                
                # Compute and store admissibility
                adm_score = self.model.compute_admissibility(out['physics'])
                trajectory['admissibility'].append(adm_score.mean().item())
                
                # Advance to next state
                current_state = out['next_state']
                
                if verbose and (t + 1) % 10 == 0:
                    print(f"  Period {t+1}: K={current_state[0,0]:.3f}, "
                          f"Welfare={out['welfare'].item():.4f}")
        
        # Convert lists to numpy arrays
        for key in trajectory:
            trajectory[key] = np.array(trajectory[key])
        
        return trajectory
    
    def run_monte_carlo_simulation(self, num_trajectories=100, num_periods=50, 
                                    use_boundary=True, seed=None):
        """
        Run Monte Carlo simulation with multiple trajectories.
        
        Args:
            num_trajectories: Number of trajectories to simulate
            num_periods: Periods per trajectory
            use_boundary: Sample initial states from learned boundary
            seed: Random seed for reproducibility
            
        Returns:
            List of trajectory dictionaries
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        print(f"\nRunning Monte Carlo simulation: {num_trajectories} trajectories, {num_periods} periods each")
        
        trajectories = []
        successful = 0
        
        for i in range(num_trajectories):
            # Generate initial state
            if use_boundary and self.boundary.admissible_points is not None:
                # Sample from learned admissible region
                idx = np.random.randint(len(self.boundary.admissible_points))
                base_point = self.boundary.admissible_points[idx]
                
                sb = self.config['state_bounds']
                K0 = base_point[0]
                ae0 = base_point[1]
                au0 = base_point[2]
                ce0 = np.random.uniform(sb['c_min'], sb['c_max'])
                cu0 = np.random.uniform(sb['c_min'], sb['c_max'])
                
                initial_state = torch.tensor([K0, ae0, au0, ce0, cu0], 
                                            dtype=torch.float32, device=self.device)
            else:
                # Random initial state
                initial_state = generate_random_states(1, self.config, self.device).squeeze(0)
            
            # Simulate trajectory
            traj = self.simulate_trajectory(initial_state, num_periods)
            
            if len(traj['states']) == num_periods:
                trajectories.append(traj)
                successful += 1
            
            if (i + 1) % 20 == 0:
                print(f"  Completed {i+1}/{num_trajectories} trajectories ({successful} successful)")
        
        print(f"Monte Carlo complete: {successful}/{num_trajectories} successful trajectories")
        return trajectories
    
    def compute_ergodic_distribution(self, trajectories, burn_in=10):
        """
        Compute ergodic distribution statistics from simulation trajectories.
        
        Args:
            trajectories: List of trajectory dictionaries
            burn_in: Number of initial periods to discard
            
        Returns:
            Dictionary of distribution statistics
        """
        # Collect all states after burn-in
        all_states = []
        all_controls = []
        all_welfare = []
        
        for traj in trajectories:
            if len(traj['states']) > burn_in:
                all_states.append(traj['states'][burn_in:])
                all_controls.append(traj['controls'][burn_in:])
                all_welfare.append(traj['welfare'][burn_in:])
        
        if len(all_states) == 0:
            print("Warning: No valid trajectories for ergodic distribution")
            return None
        
        all_states = np.vstack(all_states)
        all_controls = np.vstack(all_controls)
        all_welfare = np.concatenate(all_welfare)
        
        # Compute statistics
        state_names = ['K', 'a_e', 'a_u', 'c_e', 'c_u']
        control_names = ['n_e', 'c_prime_e', 'c_prime_u']
        
        stats = {
            'states': {
                'mean': dict(zip(state_names, np.mean(all_states, axis=0))),
                'std': dict(zip(state_names, np.std(all_states, axis=0))),
                'min': dict(zip(state_names, np.min(all_states, axis=0))),
                'max': dict(zip(state_names, np.max(all_states, axis=0))),
            },
            'controls': {
                'mean': dict(zip(control_names, np.mean(all_controls, axis=0))),
                'std': dict(zip(control_names, np.std(all_controls, axis=0))),
            },
            'welfare': {
                'mean': np.mean(all_welfare),
                'std': np.std(all_welfare),
            },
            'n_observations': len(all_welfare)
        }
        
        return stats
    
    def analyze_policy_functions(self, grid_resolution=30):
        """
        Analyze the learned policy functions over a grid of states.
        
        Returns:
            Dictionary containing policy function evaluations
        """
        self.model.eval()
        sb = self.config['state_bounds']
        
        # Create 2D grid at mean values of other variables
        mean_K = (sb['K_max'] + sb['K_min']) / 2
        mean_c = (sb['c_max'] + sb['c_min']) / 2
        
        a_vals = np.linspace(sb['a_min'], sb['a_max'], grid_resolution)
        ae_grid, au_grid = np.meshgrid(a_vals, a_vals)
        
        n_points = grid_resolution ** 2
        
        # Create state tensor
        states = torch.zeros(n_points, 5, device=self.device)
        states[:, 0] = mean_K  # K
        states[:, 1] = torch.tensor(ae_grid.flatten())  # a^e
        states[:, 2] = torch.tensor(au_grid.flatten())  # a^u
        states[:, 3] = mean_c  # c^e
        states[:, 4] = mean_c  # c^u
        
        with torch.no_grad():
            # Get actor outputs
            raw_out = self.model.actor(states)
            
            n_e = (torch.sigmoid(raw_out[:, 0]) * (self.model.n_max - self.model.n_min) + self.model.n_min)
            c_prime_e = torch.exp(raw_out[:, 1]) * self.model.c_scale
            c_prime_u = torch.exp(raw_out[:, 2]) * self.model.c_scale
            
            # Get value function
            values = self.model.critic(states)
        
        policy_data = {
            'ae_grid': ae_grid,
            'au_grid': au_grid,
            'n_e': n_e.cpu().numpy().reshape(grid_resolution, grid_resolution),
            'c_prime_e': c_prime_e.cpu().numpy().reshape(grid_resolution, grid_resolution),
            'c_prime_u': c_prime_u.cpu().numpy().reshape(grid_resolution, grid_resolution),
            'value': values.cpu().numpy().reshape(grid_resolution, grid_resolution),
            'fixed_K': mean_K,
            'fixed_c': mean_c
        }
        
        return policy_data
    
    def plot_simulation_results(self, trajectories, save_prefix='simulation'):
        """
        Create comprehensive plots of simulation results.
        """
        if len(trajectories) == 0:
            print("No trajectories to plot")
            return
        
        # 1. Plot sample trajectories
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Select up to 5 trajectories to plot
        n_plot = min(5, len(trajectories))
        colors = plt.cm.viridis(np.linspace(0, 1, n_plot))
        
        state_names = ['K (Capital)', 'aᵉ (Assets Emp)', 'aᵘ (Assets Unemp)', 
                       'cᵉ (Cons Emp)', 'cᵘ (Cons Unemp)', 'Welfare']
        
        for i in range(n_plot):
            traj = trajectories[i]
            T = len(traj['states'])
            t_vals = np.arange(T)
            
            for j in range(5):
                axes[j // 3, j % 3].plot(t_vals, traj['states'][:, j], 
                                         color=colors[i], alpha=0.7)
            axes[1, 2].plot(t_vals, traj['welfare'], color=colors[i], alpha=0.7)
        
        for j in range(5):
            axes[j // 3, j % 3].set_xlabel('Period')
            axes[j // 3, j % 3].set_ylabel(state_names[j])
            axes[j // 3, j % 3].set_title(state_names[j])
            axes[j // 3, j % 3].grid(True, alpha=0.3)
        
        axes[1, 2].set_xlabel('Period')
        axes[1, 2].set_ylabel('Welfare')
        axes[1, 2].set_title('Period Welfare')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/{save_prefix}_trajectories.png", dpi=150)
        plt.close()
        
        # 2. Plot ergodic distributions
        stats = self.compute_ergodic_distribution(trajectories)
        if stats is None:
            return
        
        # Collect all data for histograms
        all_states = np.vstack([t['states'][10:] for t in trajectories if len(t['states']) > 10])
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        state_names_short = ['K', 'aᵉ', 'aᵘ', 'cᵉ', 'cᵘ']
        
        for j in range(5):
            ax = axes[j // 3, j % 3]
            ax.hist(all_states[:, j], bins=50, density=True, alpha=0.7, color='steelblue')
            ax.axvline(stats['states']['mean'][state_names_short[j].replace('ᵉ', '_e').replace('ᵘ', '_u')], 
                      color='red', linestyle='--', label='Mean')
            ax.set_xlabel(state_names_short[j])
            ax.set_ylabel('Density')
            ax.set_title(f'Ergodic Distribution: {state_names_short[j]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        axes[1, 2].axis('off')
        axes[1, 2].text(0.5, 0.5, f"N = {stats['n_observations']} observations\n"
                       f"Mean Welfare = {stats['welfare']['mean']:.4f}\n"
                       f"Std Welfare = {stats['welfare']['std']:.4f}",
                       ha='center', va='center', fontsize=12,
                       transform=axes[1, 2].transAxes)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/{save_prefix}_distributions.png", dpi=150)
        plt.close()
        
        print(f"Plots saved to {self.output_dir}/figures/")
    
    def plot_policy_functions(self, policy_data, save_prefix='policy'):
        """Plot the learned policy functions."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # n^e (labor supply)
        im0 = axes[0, 0].contourf(policy_data['ae_grid'], policy_data['au_grid'], 
                                   policy_data['n_e'], levels=20, cmap='viridis')
        axes[0, 0].set_xlabel('aᵉ (Assets Employed)')
        axes[0, 0].set_ylabel('aᵘ (Assets Unemployed)')
        axes[0, 0].set_title('Labor Supply nᵉ')
        plt.colorbar(im0, ax=axes[0, 0])
        
        # c'^e (future consumption employed)
        im1 = axes[0, 1].contourf(policy_data['ae_grid'], policy_data['au_grid'], 
                                   policy_data['c_prime_e'], levels=20, cmap='plasma')
        axes[0, 1].set_xlabel('aᵉ')
        axes[0, 1].set_ylabel('aᵘ')
        axes[0, 1].set_title("Future Consumption c'ᵉ")
        plt.colorbar(im1, ax=axes[0, 1])
        
        # c'^u (future consumption unemployed)
        im2 = axes[1, 0].contourf(policy_data['ae_grid'], policy_data['au_grid'], 
                                   policy_data['c_prime_u'], levels=20, cmap='plasma')
        axes[1, 0].set_xlabel('aᵉ')
        axes[1, 0].set_ylabel('aᵘ')
        axes[1, 0].set_title("Future Consumption c'ᵘ")
        plt.colorbar(im2, ax=axes[1, 0])
        
        # Value function
        im3 = axes[1, 1].contourf(policy_data['ae_grid'], policy_data['au_grid'], 
                                   policy_data['value'], levels=20, cmap='RdYlGn')
        axes[1, 1].set_xlabel('aᵉ')
        axes[1, 1].set_ylabel('aᵘ')
        axes[1, 1].set_title('Value Function V(s)')
        plt.colorbar(im3, ax=axes[1, 1])
        
        plt.suptitle(f"Policy Functions at K={policy_data['fixed_K']:.2f}, "
                    f"c={policy_data['fixed_c']:.2f}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/{save_prefix}_functions.png", dpi=150)
        plt.close()
    
    def generate_report(self, trajectories, stats, policy_data):
        """Generate a text report of simulation results."""
        report_path = f"{self.output_dir}/simulation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("    HETEROGENEOUS AGENT RAMSEY MODEL - SIMULATION REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model parameters
            f.write("-"*70 + "\n")
            f.write("MODEL PARAMETERS\n")
            f.write("-"*70 + "\n")
            econ = self.config['economic_parameters']
            f.write(f"  Discount factor (β):        {econ['beta']}\n")
            f.write(f"  Capital share (α):          {econ['alpha']}\n")
            f.write(f"  Risk aversion (σ):          {econ['sigma']}\n")
            f.write(f"  Inverse Frisch (γ):         {econ['gamma']}\n")
            f.write(f"  Depreciation (δ):           {econ['delta']}\n")
            f.write(f"  Employment share (π^e):     {econ['pi_e']}\n\n")
            
            # Simulation summary
            f.write("-"*70 + "\n")
            f.write("SIMULATION SUMMARY\n")
            f.write("-"*70 + "\n")
            f.write(f"  Number of trajectories:     {len(trajectories)}\n")
            if stats:
                f.write(f"  Observations (post burn-in):{stats['n_observations']}\n\n")
            
            # Ergodic distribution
            if stats:
                f.write("-"*70 + "\n")
                f.write("ERGODIC DISTRIBUTION\n")
                f.write("-"*70 + "\n")
                f.write("\n  State Variables:\n")
                f.write(f"    {'Variable':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}\n")
                f.write("    " + "-"*52 + "\n")
                for var in ['K', 'a_e', 'a_u', 'c_e', 'c_u']:
                    f.write(f"    {var:<12} {stats['states']['mean'][var]:>10.4f} "
                           f"{stats['states']['std'][var]:>10.4f} "
                           f"{stats['states']['min'][var]:>10.4f} "
                           f"{stats['states']['max'][var]:>10.4f}\n")
                
                f.write("\n  Control Variables:\n")
                f.write(f"    {'Variable':<12} {'Mean':>10} {'Std':>10}\n")
                f.write("    " + "-"*32 + "\n")
                for var in ['n_e', 'c_prime_e', 'c_prime_u']:
                    f.write(f"    {var:<12} {stats['controls']['mean'][var]:>10.4f} "
                           f"{stats['controls']['std'][var]:>10.4f}\n")
                
                f.write(f"\n  Welfare:\n")
                f.write(f"    Mean:  {stats['welfare']['mean']:.6f}\n")
                f.write(f"    Std:   {stats['welfare']['std']:.6f}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")
        
        print(f"Report saved to {report_path}")
    
    def run_full_analysis(self, train_model=True, num_trajectories=100, num_periods=50):
        """
        Run complete analysis pipeline.
        
        Args:
            train_model: Whether to train the model (False to load existing)
            num_trajectories: Number of Monte Carlo trajectories
            num_periods: Periods per trajectory
        """
        print("\n" + "="*70)
        print("     HETEROGENEOUS AGENT RAMSEY MODEL - FULL ANALYSIS")
        print("="*70 + "\n")
        
        # Step 1: Train or load model
        if train_model:
            print("Step 1: Training model...")
            train()  # Uses the dashboard train function
            self.load_model('ha_model_final.pth')
        else:
            print("Step 1: Loading pre-trained model...")
            if not self.load_model('ha_model_final.pth'):
                print("No model found. Please train first with --mode train")
                return
        
        # Step 2: Update boundary with trained model
        print("\nStep 2: Updating boundary with trained model...")
        with torch.no_grad():
            candidates = generate_random_states(5000, self.config, self.device)
            out = self.model.forward_physics(candidates)
            if out is not None:
                scores = self.model.compute_admissibility(out['physics'])
                self.boundary.update(out['next_state'], scores, threshold=0.9)
                stats = self.boundary.get_boundary_stats()
                print(f"  Boundary: {stats['n_points']} points, "
                      f"{stats['n_alpha_simplices']}/{stats['n_simplices']} α-simplices")
        
        # Step 3: Run Monte Carlo simulation
        print("\nStep 3: Running Monte Carlo simulation...")
        trajectories = self.run_monte_carlo_simulation(
            num_trajectories=num_trajectories,
            num_periods=num_periods,
            use_boundary=True,
            seed=42
        )
        
        # Step 4: Compute statistics
        print("\nStep 4: Computing ergodic distribution...")
        stats = self.compute_ergodic_distribution(trajectories)
        
        # Step 5: Analyze policy functions
        print("\nStep 5: Analyzing policy functions...")
        policy_data = self.analyze_policy_functions()
        
        # Step 6: Generate visualizations
        print("\nStep 6: Generating visualizations...")
        self.plot_simulation_results(trajectories)
        self.plot_policy_functions(policy_data)
        
        # Step 7: Generate report
        print("\nStep 7: Generating report...")
        self.generate_report(trajectories, stats, policy_data)
        
        # Step 8: Save data
        print("\nStep 8: Saving data...")
        with open(f"{self.output_dir}/data/trajectories.pkl", 'wb') as f:
            pickle.dump(trajectories, f)
        with open(f"{self.output_dir}/data/statistics.pkl", 'wb') as f:
            pickle.dump(stats, f)
        with open(f"{self.output_dir}/data/policy_data.pkl", 'wb') as f:
            pickle.dump(policy_data, f)
        
        print("\n" + "="*70)
        print("     ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nResults saved to: {self.output_dir}/")
        print(f"  - Figures: {self.output_dir}/figures/")
        print(f"  - Data:    {self.output_dir}/data/")
        print(f"  - Report:  {self.output_dir}/simulation_report.txt")
        
        return trajectories, stats, policy_data


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Heterogeneous Agent Ramsey Model Simulation Runner'
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        default='full',
        choices=['train', 'simulate', 'analyze', 'full'],
        help='Execution mode: train, simulate, analyze, or full'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--trajectories', 
        type=int, 
        default=100,
        help='Number of Monte Carlo trajectories'
    )
    parser.add_argument(
        '--periods', 
        type=int, 
        default=50,
        help='Number of periods per trajectory'
    )
    parser.add_argument(
        '--model-path', 
        type=str, 
        default='ha_model_final.pth',
        help='Path to model file (for loading)'
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = SimulationRunner(config_file=args.config, output_dir=args.output)
    
    if args.mode == 'train':
        print("Training model...")
        train()
        runner.save_model()
        
    elif args.mode == 'simulate':
        print("Running simulation with existing model...")
        if runner.load_model(args.model_path):
            trajectories = runner.run_monte_carlo_simulation(
                num_trajectories=args.trajectories,
                num_periods=args.periods
            )
            runner.plot_simulation_results(trajectories)
            stats = runner.compute_ergodic_distribution(trajectories)
            runner.generate_report(trajectories, stats, None)
        
    elif args.mode == 'analyze':
        print("Analyzing trained model...")
        if runner.load_model(args.model_path):
            policy_data = runner.analyze_policy_functions()
            runner.plot_policy_functions(policy_data)
        
    elif args.mode == 'full':
        print("Running full analysis pipeline...")
        runner.run_full_analysis(
            train_model=True,
            num_trajectories=args.trajectories,
            num_periods=args.periods
        )


if __name__ == "__main__":
    main()
