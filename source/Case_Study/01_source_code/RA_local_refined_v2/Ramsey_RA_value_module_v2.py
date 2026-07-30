# Ramsey_RA_value_module_v2.py
"""
Value and Policy Training Module for Deep Ramsey Algorithm.
VERSION 2: Aligned with document specification.

Key changes from v1:
1. Value training uses full Bellman target (includes continuation value)
2. Streamlined code structure
3. Configurable via JSON
4. Cleaner separation of concerns
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import os


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def inverse_sigmoid(mu: torch.Tensor, mu_min: float, mu_max: float, 
                    eps: float = 1e-7) -> torch.Tensor:
    """Convert value from [mu_min, mu_max] back to logit space."""
    scaled = (mu - mu_min) / (mu_max - mu_min)
    clamped = torch.clamp(scaled, eps, 1.0 - eps)
    return torch.log(clamped / (1.0 - clamped))


def build_network(input_dim: int, output_dim: int, hidden_layers: list, 
                  activation: str = 'relu') -> nn.Sequential:
    """Build feedforward network from config."""
    layers = []
    prev_dim = input_dim
    
    act_fn = nn.ReLU if activation == 'relu' else nn.Tanh
    
    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(act_fn())
        prev_dim = hidden_dim
    
    layers.append(nn.Linear(prev_dim, output_dim))
    
    return nn.Sequential(*layers)


# =============================================================================
# OBJECTIVE / SIMULATION
# =============================================================================

class SimulationObjective:
    """
    Computes simulated utility and value targets via policy rollout.
    
    Key fix: Value targets now include continuation value (Bellman-consistent).
    """

    def __init__(self, value_net: nn.Module, policy_net: nn.Module,
                 config: dict, device: torch.device):
        self.device = device
        self.value_net = value_net.to(device)
        self.policy_net = policy_net.to(device)
        
        # Economic parameters
        econ = config['economic_parameters']
        self.beta = econ['beta']
        self.gamma_l = econ['gamma_l']
        self.mu_min = econ['mu_min']
        self.mu_max = econ['mu_max']
        self.zagg_vec = torch.tensor(econ['zagg_vec'], device=device, dtype=torch.float32).squeeze()
        self.pi_zagg = torch.tensor(econ['pi_zagg'], device=device, dtype=torch.float32)
        
        # Bounds
        bounds = config['feasibility_bounds']
        self.tau_min = bounds['tau_min']
        self.tau_max = bounds['tau_max']
        self.b_min = bounds['b_min_initial']
        self.b_max = bounds['b_max_initial']
        
        # Soft penalties
        penalties = config['soft_penalties']
        self.l_eps = penalties['l_eps']
        self.tau_eps = penalties['tau_eps']
        self.b_eps = penalties['b_eps']

    def update_bounds(self, b_min: float, b_max: float):
        """Update debt bounds (called when dynamic bounds change)."""
        self.b_min = b_min
        self.b_max = b_max

    def simulate_value(self, states: torch.Tensor, n_sim: int, 
                       include_penalties: bool = True) -> tuple:
        """
        Simulate trajectories and compute value estimates.
        
        Args:
            states: Initial states [N, 3] = (B, mu, g_idx)
            n_sim: Number of periods to simulate
            include_penalties: Whether to include soft constraint penalties
        
        Returns:
            (mean_neg_value, domain_data, value_data, mean_pure_value)
            
            value_data: [M, 4] = (B, mu, g_val, V_target) where V_target includes
                        continuation value for Bellman consistency
        """
        N = states.shape[0]
        
        B = states[:, 0].unsqueeze(1)
        mu = states[:, 1].unsqueeze(1)
        g_idx = states[:, 2].unsqueeze(1).long()
        
        # Pre-generate shock sequence
        shock_seq = torch.zeros((N, n_sim + 1), dtype=torch.long, device=self.device)
        shock_seq[:, 0] = g_idx.squeeze()
        
        for t in range(1, n_sim + 1):
            current = shock_seq[:, t - 1]
            shock_seq[:, t] = torch.multinomial(self.pi_zagg[current], 1).squeeze()
        
        # Accumulate utility
        V_accum = torch.zeros_like(B)
        V_pure = torch.zeros_like(B)
        
        # Track validity for filtering
        is_valid = torch.ones(N, dtype=torch.bool, device=self.device)
        
        # Store first-period allocations for validation
        c0_check, tau0_check, B1_check = None, None, None
        
        for t in range(n_sim):
            g_t = self.zagg_vec[shock_seq[:, t]].unsqueeze(1)
            
            # Current allocations
            c = 1.0 / mu
            x = c + g_t
            l = 1.0 - x
            
            # === Compute penalties ===
            # Labor feasibility
            l_violation = torch.clamp(self.l_eps - l, min=0)
            l_penalty = l_violation / self.l_eps
            l_safe = torch.clamp(l, min=self.l_eps)
            
            # Tax feasibility
            tau_raw = 1.0 - self.gamma_l * c / (l_safe + 1e-8)
            tau_low_violation = torch.clamp(self.tau_min - tau_raw, min=0)
            tau_high_violation = torch.clamp(tau_raw - self.tau_max, min=0)
            tau_penalty = (tau_low_violation + tau_high_violation) / self.tau_eps
            tau = torch.clamp(tau_raw, self.tau_min, self.tau_max)
            
            # Get policy
            state_input = torch.cat([B, mu, g_t], dim=1)
            policy_logits = self.policy_net(state_input)
            mu_next_g0 = torch.sigmoid(policy_logits[:, 0:1]) * (self.mu_max - self.mu_min) + self.mu_min
            mu_next_g1 = torch.sigmoid(policy_logits[:, 1:2]) * (self.mu_max - self.mu_min) + self.mu_min
            
            # Select next mu based on realized shock
            next_shock = shock_seq[:, t + 1].unsqueeze(1)
            mu_next = torch.where(next_shock == 0, mu_next_g0, mu_next_g1)
            
            # Expected mu for bond pricing
            E_mu_next = (self.pi_zagg[shock_seq[:, t], 0].unsqueeze(1) * mu_next_g0 +
                         self.pi_zagg[shock_seq[:, t], 1].unsqueeze(1) * mu_next_g1)
            
            q = self.beta * E_mu_next / mu
            
            # Next period debt
            B_next_raw = (B + g_t - tau * x) / (q + 1e-8)
            
            # Debt penalty
            b_low_violation = torch.clamp(self.b_min - B_next_raw, min=0)
            b_high_violation = torch.clamp(B_next_raw - self.b_max, min=0)
            b_penalty = (b_low_violation + b_high_violation) / self.b_eps
            B_next = torch.clamp(B_next_raw, self.b_min, self.b_max)
            
            # Utility
            u_pure = torch.log(c) + self.gamma_l * torch.log(l_safe)
            u_penalized = u_pure - (l_penalty + tau_penalty + b_penalty) if include_penalties else u_pure
            
            V_accum = V_accum + (self.beta ** t) * u_penalized
            V_pure = V_pure + (self.beta ** t) * u_pure
            
            # Track validity (for filtering value training data)
            if t == 0:
                c0_check = c
                tau0_check = tau_raw
                B1_check = B_next_raw
                
                # Mark invalid states
                invalid = ((tau_raw < self.tau_min) | (tau_raw > self.tau_max) |
                          (B_next_raw < self.b_min) | (B_next_raw > self.b_max) |
                          (x >= 1.0))
                is_valid = is_valid & ~invalid.squeeze()
            
            # Update state
            B = B_next
            mu = mu_next
        
        # === CONTINUATION VALUE (KEY FIX) ===
        g_final = self.zagg_vec[shock_seq[:, n_sim]].unsqueeze(1)
        final_state = torch.cat([B, mu, g_final], dim=1)
        V_continuation = self.value_net(final_state)
        
        # Full Bellman target: sum of discounted utilities + discounted continuation
        V_full = V_accum + (self.beta ** n_sim) * V_continuation
        V_full_pure = V_pure + (self.beta ** n_sim) * V_continuation
        
        # === Prepare outputs ===
        # Domain data: all states with validity indicator
        validity_indicator = is_valid.float().unsqueeze(1)
        domain_data = torch.cat([states, validity_indicator], dim=1)
        
        # Value data: only valid states, with g_val (not g_idx) and FULL value target
        valid_mask = is_valid
        states_valid = states[valid_mask]
        V_target = V_full[valid_mask]
        
        if states_valid.shape[0] > 0:
            # Convert g_idx to g_val for network input
            g_vals = self.zagg_vec[states_valid[:, 2].long()].unsqueeze(1)
            value_data = torch.cat([
                states_valid[:, 0:2],  # B, mu
                g_vals,                 # g_val (not index)
                V_target                # Full Bellman target
            ], dim=1)
        else:
            value_data = torch.empty(0, 4, device=self.device)
        
        return (-V_full.mean(), domain_data, value_data, -V_full_pure.mean())


# =============================================================================
# TRAINER
# =============================================================================

class RamseyTrainer:
    """
    Trains policy and value networks for Deep Ramsey algorithm.
    
    Policy training:
    - Stage 1: Gradient ascent on simulated returns (optimality)
    - Stage 2: MSE fitting on good + bad samples (boundary marking)
    
    Value training:
    - MSE on simulated Bellman targets (good samples)
    - MSE on penalty value (bad samples)
    """

    def __init__(self, policy_net: nn.Module, value_net: nn.Module,
                 config: dict, device: torch.device):
        self.device = device
        self.policy_net = policy_net.to(device)
        self.value_net = value_net.to(device)
        
        # Store old policy for stability
        self.policy_net_old = copy.deepcopy(policy_net)
        
        # Config
        self.config = config
        econ = config['economic_parameters']
        self.mu_min = econ['mu_min']
        self.mu_max = econ['mu_max']
        self.zagg_vec = torch.tensor(econ['zagg_vec'], device=device, dtype=torch.float32).squeeze()
        self.pi_zagg = torch.tensor(econ['pi_zagg'], device=device, dtype=torch.float32)
        
        # Training params
        train_cfg = config['training']
        self.policy_cfg = train_cfg['policy']
        self.value_cfg = train_cfg['value']
        self.use_two_stage = train_cfg['use_two_stage_training']
        
        lr_cfg = train_cfg['lr_scheduler']
        self.lr_factor = lr_cfg['factor']
        self.lr_patience = lr_cfg['patience']
        
        # Simulation objective
        self.sim_objective = SimulationObjective(value_net, policy_net, config, device)

    def update_bounds(self, b_min: float, b_max: float):
        """Update bounds in simulation objective."""
        self.sim_objective.update_bounds(b_min, b_max)

    def train_policy(self, good_samples: torch.Tensor, 
                     bad_samples: torch.Tensor = None) -> nn.Module:
        """
        Two-stage policy training.
        
        Stage 1: Maximize simulated value on good samples
        Stage 2: MSE fitting with boundary marking on bad samples
        """
        cfg = self.policy_cfg
        n_sim = self.value_cfg['n_sim_periods']
        
        optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg['learning_rate'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', 
                                      factor=self.lr_factor, patience=self.lr_patience)
        
        # === STAGE 1: OPTIMALITY ===
        print("  [Policy] Stage 1: Gradient ascent on value...")
        
        dataset = TensorDataset(good_samples)
        loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)
        
        self.policy_net.train()
        
        with tqdm(total=cfg['num_epochs_stage1'], desc='Policy S1', leave=False) as pbar:
            for epoch in range(cfg['num_epochs_stage1']):
                epoch_value = 0.0
                
                for (batch,) in loader:
                    neg_value, _, _, _ = self.sim_objective.simulate_value(
                        batch, n_sim, include_penalties=True)
                    
                    optimizer.zero_grad()
                    neg_value.backward()
                    optimizer.step()
                    
                    epoch_value += -neg_value.item()
                
                avg_value = epoch_value / len(loader)
                scheduler.step(-avg_value)
                pbar.update(1)
                pbar.set_postfix({'V': f'{avg_value:.2f}'})
        
        # === STAGE 2: BOUNDARY MARKING ===
        if self.use_two_stage and bad_samples is not None and bad_samples.shape[0] > 0:
            print(f"  [Policy] Stage 2: MSE fitting ({bad_samples.shape[0]} bad samples)...")
            
            loss_fn = nn.MSELoss()
            
            # Prepare good samples: target = current output (preserve Stage 1)
            with torch.no_grad():
                g_vals_good = self.zagg_vec[good_samples[:, 2].long()]
                good_input = torch.stack([good_samples[:, 0], good_samples[:, 1], g_vals_good], dim=1)
                good_target = self.policy_net(good_input).detach()
            
            # Prepare bad samples: target = mu_max logit (boundary marker)
            mu_max_logit = inverse_sigmoid(
                torch.tensor(self.mu_max, device=self.device), self.mu_min, self.mu_max)
            
            g_vals_bad = self.zagg_vec[bad_samples[:, 2].long()]
            bad_input = torch.stack([bad_samples[:, 0], bad_samples[:, 1], g_vals_bad], dim=1)
            bad_target = torch.full((bad_samples.shape[0], 2), mu_max_logit, device=self.device)
            
            # Combined dataset
            combined_dataset = ConcatDataset([
                TensorDataset(good_input, good_target),
                TensorDataset(bad_input, bad_target)
            ])
            combined_loader = DataLoader(combined_dataset, batch_size=cfg['batch_size'], shuffle=True)
            
            # Fresh optimizer for Stage 2 (lower LR)
            optimizer_s2 = torch.optim.Adam(self.policy_net.parameters(), 
                                            lr=cfg['learning_rate'] * 0.1)
            
            with tqdm(total=cfg['num_epochs_stage2'], desc='Policy S2', leave=False) as pbar:
                for epoch in range(cfg['num_epochs_stage2']):
                    for x_batch, y_target in combined_loader:
                        y_pred = self.policy_net(x_batch)
                        loss = loss_fn(y_pred, y_target)
                        
                        optimizer_s2.zero_grad()
                        loss.backward()
                        optimizer_s2.step()
                    
                    pbar.update(1)
        
        # Update old policy
        self.policy_net_old = copy.deepcopy(self.policy_net)
        
        return self.policy_net

    def train_value(self, good_samples: torch.Tensor,
                    bad_samples: torch.Tensor = None) -> nn.Module:
        """
        Value network training with Bellman-consistent targets.
        
        Good samples: Target = simulated returns + continuation value
        Bad samples: Target = v_penalty (large negative)
        """
        cfg = self.value_cfg
        n_sim = cfg['n_sim_periods']
        v_penalty = cfg['v_penalty']
        
        optimizer = torch.optim.Adam(self.value_net.parameters(), lr=cfg['learning_rate'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                      factor=self.lr_factor, patience=self.lr_patience)
        loss_fn = nn.MSELoss()
        
        self.value_net.train()
        
        with tqdm(total=cfg['num_epochs'], desc='Value', leave=False) as pbar:
            for epoch in range(cfg['num_epochs']):
                # Regenerate targets periodically
                if epoch % cfg['num_epochs_draw'] == 0:
                    # Generate value targets for good samples
                    with torch.no_grad():
                        _, _, value_data, _ = self.sim_objective.simulate_value(
                            good_samples, n_sim, include_penalties=False)
                    
                    if value_data.shape[0] == 0:
                        print("  [Value] Warning: No valid samples, skipping epoch")
                        pbar.update(1)
                        continue
                    
                    # Good dataset: (B, mu, g_val) -> V_target
                    good_dataset = TensorDataset(value_data[:, 0:3], value_data[:, 3:4])
                    
                    # Bad dataset if available
                    if self.use_two_stage and bad_samples is not None and bad_samples.shape[0] > 0:
                        g_vals_bad = self.zagg_vec[bad_samples[:, 2].long()]
                        bad_input = torch.stack([bad_samples[:, 0], bad_samples[:, 1], g_vals_bad], dim=1)
                        bad_target = torch.full((bad_samples.shape[0], 1), v_penalty, device=self.device)
                        
                        bad_dataset = TensorDataset(bad_input, bad_target)
                        combined_dataset = ConcatDataset([good_dataset, bad_dataset])
                    else:
                        combined_dataset = good_dataset
                    
                    loader = DataLoader(combined_dataset, batch_size=cfg['batch_size'], shuffle=True)
                
                # Training step
                epoch_loss = 0.0
                for x_batch, y_target in loader:
                    y_pred = self.value_net(x_batch)
                    loss = loss_fn(y_pred, y_target)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(loader)
                scheduler.step(avg_loss)
                pbar.update(1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        return self.value_net

    def collect_simulation_data(self, samples: torch.Tensor, n_sim: int) -> torch.Tensor:
        """Collect state data from simulations for history."""
        with torch.no_grad():
            _, domain_data, _, _ = self.sim_objective.simulate_value(
                samples, n_sim, include_penalties=False)
        return domain_data[:, 0:3]  # Return states only


# =============================================================================
# VISUALIZATION
# =============================================================================

class PolicyValueVisualizer:
    """Visualize policy and value function surfaces."""

    def __init__(self, policy_net: nn.Module, value_net: nn.Module,
                 config: dict, device: torch.device):
        self.policy_net = policy_net
        self.value_net = value_net
        self.device = device
        
        econ = config['economic_parameters']
        bounds = config['feasibility_bounds']
        
        self.beta = econ['beta']
        self.gamma_l = econ['gamma_l']
        self.mu_min = econ['mu_min']
        self.mu_max = econ['mu_max']
        self.zagg_vec = torch.tensor(econ['zagg_vec'], device=device, dtype=torch.float32).squeeze()
        self.pi_zagg = torch.tensor(econ['pi_zagg'], device=device, dtype=torch.float32)
        
        self.b_min = bounds['b_min_initial']
        self.b_max = bounds['b_max_initial']
        self.tau_min = bounds['tau_min']
        self.tau_max = bounds['tau_max']

    def plot_surfaces(self, n_grid: int = 20, g_idx: int = 1,
                      save_path: str = 'figures/policy_value_surfaces.png',
                      title_suffix: str = ''):
        """Plot 3D surfaces for policy, value, tau, and B_next."""
        
        b_vals = np.linspace(self.b_min, self.b_max, n_grid)
        mu_vals = np.linspace(self.mu_min, self.mu_max, n_grid)
        B_grid, Mu_grid = np.meshgrid(b_vals, mu_vals)
        
        g_val = self.zagg_vec[g_idx].item()
        
        # Prepare input
        N = n_grid * n_grid
        inputs = torch.zeros((N, 3), device=self.device)
        inputs[:, 0] = torch.tensor(B_grid.ravel(), device=self.device)
        inputs[:, 1] = torch.tensor(Mu_grid.ravel(), device=self.device)
        inputs[:, 2] = g_val
        
        self.policy_net.eval()
        self.value_net.eval()
        
        with torch.no_grad():
            # Policy output
            policy_logits = self.policy_net(inputs)
            mu_next_g0 = torch.sigmoid(policy_logits[:, 0]) * (self.mu_max - self.mu_min) + self.mu_min
            mu_next_g1 = torch.sigmoid(policy_logits[:, 1]) * (self.mu_max - self.mu_min) + self.mu_min
            
            # Value
            values = self.value_net(inputs)[:, 0]
            
            # Implied tau
            c = 1.0 / inputs[:, 1]
            x = c + inputs[:, 2]
            tau = 1.0 - self.gamma_l * c / (1.0 - x + 1e-8)
            
            # Implied B_next
            E_mu_next = self.pi_zagg[g_idx, 0] * mu_next_g0 + self.pi_zagg[g_idx, 1] * mu_next_g1
            q = self.beta * E_mu_next / inputs[:, 1]
            tau_clamped = torch.clamp(tau, self.tau_min, self.tau_max)
            B_next = (inputs[:, 0] + inputs[:, 2] - tau_clamped * x) / (q + 1e-8)
        
        # Reshape
        Y_policy = mu_next_g0.cpu().numpy().reshape(n_grid, n_grid)
        Y_value = values.cpu().numpy().reshape(n_grid, n_grid)
        Y_tau = tau.cpu().numpy().reshape(n_grid, n_grid)
        Y_B_next = B_next.cpu().numpy().reshape(n_grid, n_grid)
        
        # Plot
        fig = plt.figure(figsize=(14, 12))
        
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.plot_surface(B_grid, Mu_grid, Y_policy, cmap='viridis')
        ax1.set_xlabel('B')
        ax1.set_ylabel('μ')
        ax1.set_zlabel("μ'(g=0)")
        ax1.set_title(f'Policy μ\'(g=0){title_suffix}')
        
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        ax2.plot_surface(B_grid, Mu_grid, Y_value, cmap='plasma')
        ax2.set_xlabel('B')
        ax2.set_ylabel('μ')
        ax2.set_zlabel('V')
        ax2.set_title(f'Value Function{title_suffix}')
        
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax3.plot_surface(B_grid, Mu_grid, Y_tau, cmap='coolwarm')
        ax3.set_xlabel('B')
        ax3.set_ylabel('μ')
        ax3.set_zlabel('τ')
        ax3.set_title(f'Implied Tax Rate{title_suffix}')
        
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        ax4.plot_surface(B_grid, Mu_grid, Y_B_next, cmap='cividis')
        ax4.set_xlabel('B')
        ax4.set_ylabel('μ')
        ax4.set_zlabel("B'")
        ax4.set_title(f'Next Period Debt{title_suffix}')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[Visualization] Surfaces saved to: {save_path}")
