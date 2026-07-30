"""
Value and Policy Network Training Module for Ramsey Optimal Taxation.

This module implements the core neural network training routines for solving
the Ramsey optimal taxation problem. It provides:

1. Value network training: Learns V(B, μ, g) via simulation-based regression
2. Policy network training: Learns μ'(B, μ, g) via policy gradient methods
3. Two-stage training: Optional refinement with explicit feasibility penalties

Key Classes:
    - Config: Central configuration manager
    - UniformSampler: Generates training samples over state space
    - ValueSimulator: Simulates utility trajectories for value function fitting
    - PolicyValueTrainer: Coordinates training of both networks
    - SurfacePlotter: Visualizes learned value and policy functions

Mathematical Background:
    The government's Bellman equation:
        V(B, μ, g) = max_{μ'(.)} U(c, l) + β·E[V(B', μ', g') | g]
    
    Subject to:
        - Budget constraint: B' = (B + g - τx) / q
        - Implementability: μ = 1/c (from household FOC)
        - Bond pricing: q = β·E[μ'] / μ

Authors: Zhigang Feng
Version: 2.0 (Streamlined)
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import copy
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import os
from typing import Optional, Tuple, List, Dict


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================

class Config:
    """
    Central configuration manager for the training module.
    
    Loads parameters from a config dictionary and provides typed access
    with sensible defaults. Handles both new structured format and legacy
    flat format for backwards compatibility.
    
    Attributes:
        raw: Original configuration dictionary
        
        Economic parameters:
            beta: Discount factor
            gamma_l: Leisure preference parameter
            zagg_vec: Government spending shock values
            pi_zagg: Shock transition matrix
        
        State bounds:
            mu_min, mu_max: Marginal utility bounds
            v_min, v_max: Value function bounds
        
        Penalty parameters:
            b_min, b_max: Debt bounds
            tau_min, tau_max: Tax rate bounds
            tau_eps, b_eps, l_eps: Penalty smoothing parameters
        
        Training parameters:
            num_epochs_p, num_epochs_v: Training epochs
            lr_policy, lr_value: Learning rates
            batch_size_p, batch_size_v: Batch sizes
            n_p_sim, n_v_sim: Simulation horizons
    """
    
    def __init__(self, config: dict):
        """
        Initialize configuration from dictionary.
        
        Args:
            config: Configuration dictionary (typically loaded from JSON)
        """
        self.raw = config
        
        # Economic parameters
        econ = config.get('economic_parameters', config)
        self.beta = econ.get('beta', config.get('beta', 0.9))
        self.gamma_l = econ.get('gamma_l', config.get('gamma_l', 0.3))
        self.zagg_vec = econ.get('zagg_vec', config.get('zagg_vec'))
        self.pi_zagg = econ.get('pi_zagg', config.get('pi_zagg'))
        
        # State bounds
        bounds = config.get('state_bounds', config)
        self.mu_min = bounds.get('mu_min', config.get('mu_min', 1.27))
        self.mu_max = bounds.get('mu_max', config.get('mu_max', 2.51))
        self.v_min = bounds.get('v_min', config.get('v_min', -25))
        self.v_max = bounds.get('v_max', config.get('v_max', 0))
        
        # Penalty parameters
        penalty = config.get('penalty_params', {})
        self.b_min = penalty.get('b_min', -0.5)
        self.b_max = penalty.get('b_max', 3.5)
        self.tau_min = penalty.get('tau_min', 0.0)
        self.tau_max = penalty.get('tau_max', 1.0)
        self.tau_eps = penalty.get('tau_eps', 0.001)
        self.b_eps = penalty.get('b_eps', 0.001)
        self.l_eps = penalty.get('l_eps', 0.01)
        
        # Policy training parameters
        p_train = config.get('policy_training', config)
        self.num_epochs_p = p_train.get('num_epochs_p', config.get('num_epochs_p', 10))
        self.num_epochs_p_stage2 = p_train.get('num_epochs_p_stage2', 
                                                config.get('num_epochs_p_stage2', 20))
        self.lr_policy = p_train.get('lr_policy', config.get('lr_policy', 0.0005))
        self.batch_size_p = p_train.get('batch_size_p', config.get('batch_size_p', 128))
        self.n_p_sim = p_train.get('n_p_sim', config.get('n_p_sim', 50))
        
        # Value training parameters
        v_train = config.get('value_training', config)
        self.num_epochs_v = v_train.get('num_epochs_v', config.get('num_epochs_v', 10))
        self.num_epochs_draw = v_train.get('num_epochs_draw', config.get('num_epochs_draw', 5))
        self.lr_value = v_train.get('lr_value', config.get('lr_value', 0.0002))
        self.batch_size_v = v_train.get('batch_size_v', config.get('batch_size_v', 128))
        self.n_v_sim = v_train.get('n_v_sim', config.get('n_v_sim', 50))
        self.v_threshold = v_train.get('v_threshold', config.get('v_threshold', -50))
        
        # Network architecture
        arch = config.get('network_architecture', config)
        self.n1_p = arch.get('n1_p', config.get('n1_p', 64))
        self.n2_p = arch.get('n2_p', config.get('n2_p', 64))
        self.n1_v = arch.get('n1_v', config.get('n1_v', 64))
        self.n2_v = arch.get('n2_v', config.get('n2_v', 64))
        
        # Learning rate scheduler
        lr_sched = config.get('learning_rate_scheduler', config)
        self.lr_factor = lr_sched.get('lr_factor', config.get('lr_factor', 0.99))
        self.lr_patience = lr_sched.get('lr_patience', config.get('lr_patience', 50))
        
        # Data loading
        data_cfg = config.get('data_loading', config)
        self.num_worker = data_cfg.get('num_worker', config.get('num_worker', 0))
        
        # Sampling
        sampling = config.get('sampling', config)
        self.num_samples_value = sampling.get('num_samples_value', 
                                              config.get('num_samples_value', 5000))
        self.num_samples_expand_p = sampling.get('num_samples_expand_p', 
                                                  config.get('num_samples_expand_p', 1.2))
        self.num_samples_expand_v = sampling.get('num_samples_expand_v', 
                                                  config.get('num_samples_expand_v', 1.25))
        
        # Model I/O
        io_cfg = config.get('model_io', config)
        self.model_number_input = io_cfg.get('model_number_input', 
                                             config.get('model_number_input', 101))
        self.model_number_output = io_cfg.get('model_number_output', 
                                              config.get('model_number_output', 102))
        
        # Scoring parameters
        scoring = config.get('scoring_parameters', {})
        self.use_two_stage_training = scoring.get('use_two_stage_training', False)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def inverse_sigmoid_scaler(mu: torch.Tensor, mu_min: float, mu_max: float, 
                           eps: float = 1e-7) -> torch.Tensor:
    """
    Convert bounded value to unbounded logit representation.
    
    This is the inverse of the sigmoid scaling used in the policy network:
        μ = sigmoid(logit) × (μ_max - μ_min) + μ_min
    
    Args:
        mu: Values in [mu_min, mu_max]
        mu_min: Lower bound
        mu_max: Upper bound
        eps: Small constant to avoid log(0)
    
    Returns:
        Logit values (unbounded): logit = log(p / (1-p)) where p = (μ - μ_min)/(μ_max - μ_min)
    """
    mu_scaled = (mu - mu_min) / (mu_max - mu_min)
    mu_scaled_clamped = torch.clamp(mu_scaled, eps, 1.0 - eps)
    logit = torch.log(mu_scaled_clamped / (1.0 - mu_scaled_clamped))
    return logit


def contains_nan(tensor: torch.Tensor) -> bool:
    """Check if tensor contains any NaN values."""
    return torch.isnan(tensor).any().item()


# =============================================================================
# SAMPLERS
# =============================================================================

class UniformSampler:
    """
    Generates uniformly distributed state samples.
    
    Samples (B, μ, g) where:
        - B ~ Uniform[b_min, b_max]
        - μ ~ Uniform[μ_min, μ_max]
        - g ~ Discrete{0, 1} (shock state index)
    
    Attributes:
        b_min, b_max: Current debt bounds (can be updated dynamically)
        mu_min, mu_max: Marginal utility bounds
        device: PyTorch device
    """
    
    def __init__(self, config: Config, device: torch.device):
        """
        Initialize sampler.
        
        Args:
            config: Configuration object with state bounds
            device: PyTorch device for tensor creation
        """
        self.b_min = config.b_min
        self.b_max = config.b_max
        self.mu_min = config.mu_min
        self.mu_max = config.mu_max
        self.device = device
    
    def update_b_range(self, b_min: float, b_max: float):
        """
        Update debt bounds for dynamic boundary learning.
        
        Args:
            b_min: New minimum debt level
            b_max: New maximum debt level
        """
        self.b_min = b_min
        self.b_max = b_max
    
    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """
        Generate uniform samples over state space.
        
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            Tensor of shape (num_samples, 3) with columns [B, μ, g_idx]
            
        Raises:
            ValueError: If num_samples <= 0
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        
        samples = torch.rand(num_samples, 3, dtype=torch.float32, device=self.device)
        
        # Scale to appropriate ranges
        samples[:, 0] = samples[:, 0] * (self.b_max - self.b_min) + self.b_min
        samples[:, 1] = samples[:, 1] * (self.mu_max - self.mu_min) + self.mu_min
        samples[:, 2] = (samples[:, 2] > 0.5).float()  # Discretize to {0, 1}
        
        return samples


# =============================================================================
# VALUE SIMULATOR
# =============================================================================

class ValueSimulator:
    """
    Simulates utility trajectories for value function estimation.
    
    Uses Monte Carlo simulation to compute:
        V(B₀, μ₀, g₀) ≈ Σᵢ βⁱ U(cᵢ, lᵢ) + β^T V(B_T, μ_T, g_T)
    
    The simulation follows the policy network for T periods, then uses
    the value network as a continuation value.
    
    Attributes:
        value_net: Value network V(B, μ, g) → ℝ
        policy_net: Policy network (B, μ, g) → (logit μ'_g0, logit μ'_g1)
        policy_net_old: Previous policy network (for stability)
        config: Configuration object
        device: PyTorch device
    """
    
    def __init__(self, value_net: nn.Module, policy_net: nn.Module, 
                 policy_net_old: nn.Module, config: Config, device: torch.device):
        """
        Initialize simulator.
        
        Args:
            value_net: Value network for continuation values
            policy_net: Current policy network
            policy_net_old: Previous policy network (unused but kept for API)
            config: Configuration object with economic parameters
            device: PyTorch device for computation
        """
        self.device = device
        self.value_net = value_net.to(device)
        self.policy_net = policy_net.to(device)
        self.policy_net_old = policy_net_old.to(device)
        self.config = config
        
        # Economic parameters
        self.beta = config.beta
        self.gamma_l = config.gamma_l
        self.mu_min = config.mu_min
        self.mu_max = config.mu_max
        
        # Penalty parameters as tensors
        self.l_eps = torch.tensor(config.l_eps, device=device)
        self.tau_eps = torch.tensor(config.tau_eps, device=device)
        self.b_eps = torch.tensor(config.b_eps, device=device)
        self.tau_min = torch.tensor(config.tau_min, device=device)
        self.tau_max = torch.tensor(config.tau_max, device=device)
        self.b_min = torch.tensor(config.b_min, device=device)
        self.b_max = torch.tensor(config.b_max, device=device)
        
        # Shock process as tensors
        self.zagg_vec = torch.tensor(config.zagg_vec, device=device)
        self.pi_zagg = torch.tensor(config.pi_zagg, device=device)
    
    def update_bounds(self, b_min: float, b_max: float):
        """
        Update debt bounds for dynamic boundary learning.
        
        Args:
            b_min: New minimum debt level
            b_max: New maximum debt level
        """
        self.b_min = torch.tensor(b_min, device=self.device)
        self.b_max = torch.tensor(b_max, device=self.device)
    
    def simulate_value(self, states: torch.Tensor, n_sim: int, 
                       apply_penalties: bool = True, verbose: bool = False
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate utility trajectories and compute value estimates.
        
        Args:
            states: Initial states (N, 3) with columns [B, μ, g_idx]
            n_sim: Number of simulation periods before using continuation
            apply_penalties: If True, apply constraint violation penalties
            verbose: If True, print diagnostic information
        
        Returns:
            Tuple containing:
                - mean_neg_value: Mean of -V (scalar, for minimization in policy gradient)
                - domain_data: States with feasibility flags (N, 4)
                - value_data: Valid (state, value) pairs for value training (M, 4)
                - mean_neg_value_pure: Mean of -V without penalties (scalar)
        """
        N = states.shape[0]
        
        # Extract initial state components
        B = states[:, 0].unsqueeze(1)
        mu = states[:, 1].unsqueeze(1)
        g_idx = states[:, 2].unsqueeze(1)
        
        # Simulate shock sequence
        g_sequence = self._simulate_shocks(g_idx.squeeze().long(), n_sim, N)
        
        # Initialize value accumulators
        V_sim = torch.zeros_like(B)
        V_sim_pure = torch.zeros_like(B)
        zero = torch.tensor(0.0, device=self.device)
        
        # Storage for first period (used for filtering)
        c0_chk, x0_chk, b1_chk, tau0_chk = None, None, None, None
        
        for t in range(n_sim):
            g_val = self.zagg_vec[g_sequence[:, t]]
            
            # === Compute allocations from μ ===
            c = 1.0 / mu  # Consumption from FOC
            x = c + g_val  # Total resources
            l_orig = 1.0 - x  # Leisure
            
            # Labor/leisure penalty (prevent negative leisure)
            l = torch.maximum(l_orig, zero + self.l_eps)
            l_penalty = (1.0 / self.l_eps) * torch.maximum(zero + self.l_eps - l_orig, zero)
            
            # === Compute tax rate ===
            tau_orig = 1.0 - self.gamma_l * c / (1.0 - x + 1e-8)
            
            # Tax rate penalty (keep in bounds)
            tau_penalty_low = (1.0 / self.tau_eps) * torch.maximum(self.tau_min - tau_orig, zero)
            tau_penalty_high = (1.0 / self.tau_eps) * torch.maximum(tau_orig - self.tau_max, zero)
            tau_penalty = tau_penalty_low + tau_penalty_high
            tau = torch.clamp(tau_orig, self.tau_min, self.tau_max)
            
            # === Get policy predictions ===
            state_input = torch.cat([B, mu, g_val], dim=1)
            policy_logits = self.policy_net(state_input.float())
            
            # Convert logits to μ' values
            mu_next_g0 = (torch.sigmoid(policy_logits[:, 0:1]) * 
                         (self.mu_max - self.mu_min) + self.mu_min)
            mu_next_g1 = (torch.sigmoid(policy_logits[:, 1:2]) * 
                         (self.mu_max - self.mu_min) + self.mu_min)
            
            # Select μ' based on realized shock
            mu_next = torch.where(g_sequence[:, t:t+1] == 0, mu_next_g0, mu_next_g1)
            
            # === Compute bond price and next-period debt ===
            g_idx_t = g_sequence[:, t]
            E_mu_next = (self.pi_zagg[g_idx_t, 0:1] * mu_next_g0 + 
                        self.pi_zagg[g_idx_t, 1:2] * mu_next_g1)
            q = self.beta * E_mu_next / mu
            
            B_next_orig = (B + g_val - tau * x) / q
            
            # Debt penalty (keep in bounds)
            b_penalty_low = torch.maximum(self.b_min - B_next_orig, zero)
            b_penalty_high = torch.maximum(B_next_orig - self.b_max, zero)
            b_penalty = (1.0 / self.b_eps) * (b_penalty_low + b_penalty_high)
            B_next = torch.clamp(B_next_orig, self.b_min, self.b_max)
            
            # === Accumulate utility ===
            U = torch.log(c) + self.gamma_l * torch.log(l)
            
            if apply_penalties:
                U_penalized = U - (l_penalty + tau_penalty + b_penalty)
            else:
                U_penalized = U
            
            V_sim = V_sim + (self.beta ** t) * U_penalized
            V_sim_pure = V_sim_pure + (self.beta ** t) * U
            
            # Store first period values for filtering
            if t == 0:
                c0_chk, x0_chk, b1_chk, tau0_chk = c, x, B_next, tau_orig
            
            # Update state for next period
            B = B_next
            mu = mu_next
        
        # === Add continuation value ===
        g_final = self.zagg_vec[g_sequence[:, n_sim]]
        state_final = torch.cat([B, mu, g_final], dim=1)
        V_continuation = self.value_net(state_final)
        
        V_total = V_sim + (self.beta ** n_sim) * V_continuation
        V_total_pure = V_sim_pure + (self.beta ** n_sim) * V_continuation
        
        # === Filter valid data for value training ===
        infeasible = (
            (tau0_chk <= self.tau_min) |
            (tau0_chk > self.tau_max) |
            (b1_chk < self.b_min) |
            (b1_chk > self.b_max) |
            (x0_chk >= 1.0)
        )
        feasible_flag = (~torch.any(infeasible, dim=1, keepdim=True)).float()
        
        # Create output datasets
        domain_data = torch.cat([states.detach(), feasible_flag.detach()], dim=1)
        
        feasible_mask = feasible_flag.squeeze() > 0
        if feasible_mask.any():
            value_data = torch.cat([
                states[feasible_mask].detach(),
                V_sim[feasible_mask].detach()
            ], dim=1)
        else:
            value_data = torch.empty(0, 4, device=self.device)
        
        return -torch.mean(V_total), domain_data, value_data, -torch.mean(V_total_pure)
    
    def _simulate_shocks(self, initial_g: torch.Tensor, n_sim: int, N: int) -> torch.Tensor:
        """
        Simulate Markov shock sequence.
        
        Args:
            initial_g: Initial shock states (N,) as long tensor
            n_sim: Number of periods to simulate
            N: Batch size
        
        Returns:
            Tensor of shape (N, n_sim + 1) with shock indices
        """
        g_sequence = torch.zeros((N, n_sim + 1), dtype=torch.long, device=self.device)
        g_sequence[:, 0] = initial_g
        
        for t in range(1, n_sim + 1):
            current_g = g_sequence[:, t - 1]
            transition_probs = self.pi_zagg[current_g]
            g_sequence[:, t] = torch.multinomial(transition_probs, 1).squeeze()
        
        return g_sequence


# =============================================================================
# TRAINER
# =============================================================================

class PolicyValueTrainer:
    """
    Coordinates training of policy and value networks.
    
    Training proceeds in two main steps per iteration:
    1. Policy training: Minimize E[-V] via gradient descent (policy gradient)
    2. Value training: Fit V(s) to simulated values via MSE regression
    
    Optional two-stage training adds explicit constraint penalties for
    states identified as inadmissible by the adaptive sampler.
    
    Attributes:
        policy_net: Policy network to train
        value_net: Value network to train  
        policy_net_old: Previous policy network (for stability)
        value_net_old: Previous value network (for recovery from NaN)
        config: Configuration object
        device: PyTorch device
        sampler: UniformSampler for generating training data
    """
    
    def __init__(self, policy_net: nn.Module, value_net: nn.Module,
                 config: Config, device: torch.device, save_models: bool = True):
        """
        Initialize trainer.
        
        Args:
            policy_net: Policy network π(B, μ, g) → (logit μ'_g0, logit μ'_g1)
            value_net: Value network V(B, μ, g) → ℝ
            config: Configuration object
            device: PyTorch device
            save_models: If True, save models after each training step
        """
        self.device = device
        self.policy_net = policy_net.to(device)
        self.value_net = value_net.to(device)
        self.policy_net_old = copy.deepcopy(policy_net)
        self.value_net_old = copy.deepcopy(value_net)
        self.config = config
        self.save_models = save_models
        
        # Create sampler for generating training data
        self.sampler = UniformSampler(config, device)
    
    def train_policy(self, states: torch.Tensor, 
                     all_losses: Optional[List[float]] = None,
                     inadmissible_samples: Optional[torch.Tensor] = None
                     ) -> nn.Module:
        """
        Train the policy network.
        
        Stage 1: Gradient descent on E[-V] (policy gradient method)
        Stage 2 (optional): MSE fitting to push inadmissible states toward bounds
        
        Args:
            states: Training states (N, 3) with columns [B, μ, g_idx]
            all_losses: List to append loss history (modified in place)
            inadmissible_samples: States to explicitly penalize (optional)
        
        Returns:
            Trained policy network
        """
        simulator = ValueSimulator(
            self.value_net, self.policy_net, self.policy_net_old,
            self.config, self.device
        )
        
        # === STAGE 1: Policy Gradient ===
        print("  Policy Training Stage 1: Gradient Descent...")
        
        optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.config.lr_policy)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', 
                                      factor=self.config.lr_factor, 
                                      patience=self.config.lr_patience)
        
        dataset = TensorDataset(states)
        loader = DataLoader(dataset, batch_size=self.config.batch_size_p, 
                           shuffle=True, num_workers=self.config.num_worker)
        
        with tqdm(total=self.config.num_epochs_p, desc='Policy (Grad)', leave=False) as pbar:
            for epoch in range(self.config.num_epochs_p):
                epoch_loss = 0.0
                
                for batch_x, in loader:
                    loss, _, _, loss_pure = simulator.simulate_value(
                        batch_x, self.config.n_p_sim, apply_penalties=True
                    )
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss_pure.item()
                    
                    # Check for NaN and reset if needed
                    if any(torch.isnan(p).any() for p in self.policy_net.parameters()):
                        self._reset_policy_network()
                
                pbar.update(1)
                avg_loss = epoch_loss / len(loader) if loader else 0
                if all_losses is not None:
                    all_losses.append(-avg_loss)
                scheduler.step(avg_loss)
        
        # === STAGE 2: Explicit Constraint Penalties (optional) ===
        if self.config.use_two_stage_training and inadmissible_samples is not None:
            self._train_policy_stage2(states, inadmissible_samples, optimizer)
        
        # Save and update old network
        self.policy_net_old = copy.deepcopy(self.policy_net)
        if self.save_models:
            self._save_policy()
        
        return self.policy_net
    
    def _train_policy_stage2(self, good_states: torch.Tensor, 
                              bad_states: torch.Tensor,
                              optimizer: torch.optim.Optimizer):
        """
        Stage 2: MSE fitting to explicitly penalize inadmissible states.
        
        Strategy:
        - For good states: fit to current network output (preserve behavior)
        - For bad states: fit to μ_max logit (push toward upper bound)
        
        Args:
            good_states: States with high admissibility (N, 3)
            bad_states: States with low admissibility (M, 3)
            optimizer: Optimizer to use for training
        """
        print("  Policy Training Stage 2: Constraint Refinement...")
        
        loss_fn = nn.MSELoss()
        g_values = torch.tensor(self.config.zagg_vec, device=self.device).squeeze()
        
        # Create good dataset (preserve current behavior)
        with torch.no_grad():
            g_idx_good = good_states[:, 2].long()
            good_input = good_states.clone()
            good_input[:, 2] = g_values[g_idx_good]
            good_targets = self.policy_net(good_input)
        
        good_dataset = TensorDataset(good_input, good_targets.detach())
        
        # Create bad dataset (push to upper boundary)
        mu_max_logit = inverse_sigmoid_scaler(
            torch.tensor(self.config.mu_max, device=self.device),
            self.config.mu_min, self.config.mu_max
        )
        bad_targets = torch.full((bad_states.shape[0], 2), mu_max_logit, device=self.device)
        
        g_idx_bad = bad_states[:, 2].long()
        bad_input = bad_states.clone()
        bad_input[:, 2] = g_values[g_idx_bad]
        
        bad_dataset = TensorDataset(bad_input, bad_targets)
        
        # Combined training
        combined = ConcatDataset([good_dataset, bad_dataset])
        loader = DataLoader(combined, batch_size=self.config.batch_size_p, shuffle=True)
        
        with tqdm(total=self.config.num_epochs_p_stage2, desc='Policy (MSE)', leave=False) as pbar:
            for epoch in range(self.config.num_epochs_p_stage2):
                for x_batch, y_target in loader:
                    y_pred = self.policy_net(x_batch)
                    loss = loss_fn(y_pred, y_target)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                pbar.update(1)
        
        print("  Policy refinement complete.")
    
    def train_value(self, states: torch.Tensor,
                    all_losses: Optional[List[float]] = None,
                    inadmissible_samples: Optional[torch.Tensor] = None
                    ) -> nn.Module:
        """
        Train the value network via MSE regression.
        
        Uses simulated utility trajectories as regression targets.
        Optionally assigns low value (v_threshold) to inadmissible states.
        
        Args:
            states: Training states (N, 3) with columns [B, μ, g_idx]
            all_losses: List to append loss history (modified in place)
            inadmissible_samples: States to assign low value (optional)
        
        Returns:
            Trained value network
        """
        print("  Value Training: MSE Regression...")
        
        simulator = ValueSimulator(
            self.value_net, self.policy_net, self.policy_net_old,
            self.config, self.device
        )
        
        optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.config.lr_value)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', 
                                      factor=self.config.lr_factor, patience=10)
        loss_fn = nn.MSELoss()
        
        g_values = torch.tensor(self.config.zagg_vec, device=self.device).squeeze()
        current_dataset = None
        loader = None
        
        with tqdm(total=self.config.num_epochs_v, desc='Value Training', leave=False) as pbar:
            for epoch in range(self.config.num_epochs_v):
                
                # Generate new training data periodically
                if epoch % self.config.num_epochs_draw == 0:
                    # Get good data from simulation
                    _, _, value_data, _ = simulator.simulate_value(
                        states, self.config.n_v_sim, apply_penalties=True
                    )
                    
                    if value_data.shape[0] == 0:
                        print("Warning: No valid value data generated.")
                        pbar.update(1)
                        continue
                    
                    # Convert g_idx to g_val for network input
                    good_input = value_data[:, :3].clone()
                    g_idx = value_data[:, 2].long()
                    good_input[:, 2] = g_values[g_idx]
                    good_targets = value_data[:, 3:4]
                    good_dataset = TensorDataset(good_input, good_targets)
                    
                    # Add bad dataset if two-stage training enabled
                    if (self.config.use_two_stage_training and 
                        inadmissible_samples is not None):
                        bad_targets = torch.full(
                            (inadmissible_samples.shape[0], 1),
                            self.config.v_threshold,
                            device=self.device, dtype=torch.float32
                        )
                        bad_input = inadmissible_samples.clone()
                        g_idx_bad = bad_input[:, 2].long()
                        bad_input[:, 2] = g_values[g_idx_bad]
                        bad_dataset = TensorDataset(bad_input, bad_targets)
                        
                        current_dataset = ConcatDataset([good_dataset, bad_dataset])
                    else:
                        current_dataset = good_dataset
                    
                    loader = DataLoader(current_dataset, 
                                       batch_size=self.config.batch_size_v, 
                                       shuffle=True)
                
                if loader is None:
                    pbar.update(1)
                    continue
                
                # Training step
                epoch_loss = 0.0
                for x_batch, y_batch in loader:
                    y_pred = self.value_net(x_batch)
                    loss = loss_fn(y_pred, y_batch)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    
                    # Check for NaN and recover if needed
                    if any(torch.isnan(p).any() for p in self.value_net.parameters()):
                        self.value_net.load_state_dict(self.value_net_old.state_dict())
                        print(f"NaN detected at epoch {epoch}. Resetting.")
                
                pbar.update(1)
                avg_loss = epoch_loss / len(loader) if loader else 0
                if all_losses is not None:
                    all_losses.append(avg_loss)
                scheduler.step(avg_loss)
        
        # Save and update old network
        self.value_net_old = copy.deepcopy(self.value_net)
        if self.save_models:
            self._save_value()
        
        return self.value_net
    
    def _reset_policy_network(self):
        """Reset policy network to random initialization."""
        for layer in self.policy_net:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        print("NaN detected. Resetting policy network.")
    
    def _save_policy(self):
        """Save policy network weights to file."""
        os.makedirs('models', exist_ok=True)
        path = f'models/trained_policy_nn_{self.config.model_number_output}.pth'
        torch.save(self.policy_net.state_dict(), path)
    
    def _save_value(self):
        """Save value network weights to file."""
        os.makedirs('models', exist_ok=True)
        path = f'models/trained_value_nn_{self.config.model_number_output}.pth'
        torch.save(self.value_net.state_dict(), path)


# =============================================================================
# VISUALIZATION
# =============================================================================

class SurfacePlotter:
    """
    Visualizes policy and value function surfaces.
    
    Creates 3D surface plots of:
    - Policy function: μ'(B, μ) - next-period multiplier
    - Value function: V(B, μ) - continuation value
    - Implied tax rate: τ(B, μ) - from FOC
    - Implied next-period debt: B'(B, μ) - from budget constraint
    
    Attributes:
        policy_net: Trained policy network
        value_net: Trained value network
        config: Configuration object
        device: PyTorch device
        scorer: Optional AdmissibilityScorer for filtering inadmissible regions
        admissibility_threshold: Score threshold for filtering
    """
    
    def __init__(self, policy_net: nn.Module, value_net: nn.Module,
                 config: Config, device: torch.device,
                 scorer=None, admissibility_threshold: float = None):
        """
        Initialize plotter.
        
        Args:
            policy_net: Trained policy network
            value_net: Trained value network
            config: Configuration object
            device: PyTorch device
            scorer: Optional AdmissibilityScorer for filtering plots
            admissibility_threshold: Score threshold for showing regions
        """
        self.policy_net = policy_net
        self.value_net = value_net
        self.config = config
        self.device = device
        self.scorer = scorer
        self.admissibility_threshold = admissibility_threshold
        
        # Data storage for plots
        self.X1 = None  # B grid
        self.X2 = None  # μ grid
        self.Y_policy = None
        self.Y_value = None
        self.Y_tau = None
        self.Y_b1 = None
    
    def generate_data(self, n: int, g_index: int = 1):
        """
        Generate surface data on a grid.
        
        Args:
            n: Grid resolution (n × n points)
            g_index: Shock state to plot (0 = low g, 1 = high g)
        """
        g_val = self.config.zagg_vec[g_index][0]
        pi_zagg = torch.tensor(self.config.pi_zagg, device=self.device)
        
        # Create meshgrid
        x1 = np.linspace(self.config.b_min, self.config.b_max, n)
        x2 = np.linspace(self.config.mu_min, self.config.mu_max, n)
        X1_m, X2_m = np.meshgrid(x1, x2)
        X3_m = np.ones_like(X1_m) * g_val
        
        # Flatten for network input
        inputs = np.column_stack([X1_m.ravel(), X2_m.ravel(), X3_m.ravel()])
        inputs_t = torch.from_numpy(inputs).float().to(self.device)
        
        # Extract components
        B = inputs_t[:, 0:1]
        mu = inputs_t[:, 1:2]
        g = inputs_t[:, 2:3]
        
        # Compute allocations
        c = 1.0 / mu
        x = c + g
        tau = 1.0 - self.config.gamma_l * c / (1.0 - x + 1e-8)
        
        # Get policy predictions
        policy_logits = self.policy_net(inputs_t)
        mu_next_g0 = (torch.sigmoid(policy_logits[:, 0:1]) * 
                     (self.config.mu_max - self.config.mu_min) + self.config.mu_min)
        mu_next_g1 = (torch.sigmoid(policy_logits[:, 1:2]) * 
                     (self.config.mu_max - self.config.mu_min) + self.config.mu_min)
        
        # Compute bond price and B'
        E_mu_next = pi_zagg[g_index, 0] * mu_next_g0 + pi_zagg[g_index, 1] * mu_next_g1
        q = self.config.beta * E_mu_next / mu
        tau_clamp = torch.clamp(tau, self.config.tau_min, self.config.tau_max)
        B_next = (B + g - tau_clamp * x) / q
        
        # Get value predictions
        Y_value = self.value_net(inputs_t)[:, 0]
        
        # Convert to numpy and reshape
        Y_policy_np = mu_next_g0.cpu().detach().numpy().reshape(n, n)
        Y_value_np = Y_value.cpu().detach().numpy().reshape(n, n)
        Y_tau_np = tau.cpu().detach().numpy().reshape(n, n)
        Y_b1_np = B_next.cpu().detach().numpy().reshape(n, n)
        
        # Optional: filter by admissibility score
        if self.scorer is not None and self.admissibility_threshold is not None:
            print(f"--- Filtering by admissibility (A > {self.admissibility_threshold}) ---")
            for i in range(n):
                for j in range(n):
                    B_val = X1_m[i, j]
                    mu_val = X2_m[i, j]
                    
                    A = self.scorer.compute_score(float(B_val), float(mu_val), g_index)
                    
                    if A <= self.admissibility_threshold:
                        Y_value_np[i, j] = np.nan
                        Y_policy_np[i, j] = np.nan
                        Y_tau_np[i, j] = np.nan
                        Y_b1_np[i, j] = np.nan
        
        # Store results
        self.X1 = X1_m
        self.X2 = X2_m
        self.Y_policy = Y_policy_np
        self.Y_value = Y_value_np
        self.Y_tau = Y_tau_np
        self.Y_b1 = Y_b1_np
    
    def create_plot(self, save_path: str = 'figures/surface_plots.png', 
                    title_suffix: str = ''):
        """
        Create and save 3D surface plots.
        
        Args:
            save_path: Output file path
            title_suffix: String to append to plot titles
        """
        print(f"--- Generating surface plots{title_suffix} ---")
        
        fig = plt.figure(figsize=(14, 12))
        
        # Policy surface
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.plot_surface(self.X1, self.X2, self.Y_policy, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('Debt (B)')
        ax1.set_ylabel('Multiplier (μ)')
        ax1.set_zlabel("μ'")
        ax1.set_title(f'Policy Surface{title_suffix}')
        
        # Value surface
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        ax2.plot_surface(self.X1, self.X2, self.Y_value, cmap='plasma', alpha=0.8)
        ax2.set_xlabel('Debt (B)')
        ax2.set_ylabel('Multiplier (μ)')
        ax2.set_zlabel('V')
        ax2.set_title(f'Value Surface{title_suffix}')
        
        # Tax rate surface
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax3.plot_surface(self.X1, self.X2, self.Y_tau, cmap='coolwarm', alpha=0.8)
        ax3.set_xlabel('Debt (B)')
        ax3.set_ylabel('Multiplier (μ)')
        ax3.set_zlabel('τ')
        ax3.set_title(f'Implied Tax Rate{title_suffix}')
        
        # Next-period debt surface
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        ax4.plot_surface(self.X1, self.X2, self.Y_b1, cmap='RdYlGn', alpha=0.8)
        ax4.set_xlabel('Debt (B)')
        ax4.set_ylabel('Multiplier (μ)')
        ax4.set_zlabel("B'")
        ax4.set_title(f'Next-Period Debt{title_suffix}')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"    Surface plots saved to: {save_path}")


# =============================================================================
# NETWORK FACTORY
# =============================================================================

def create_policy_network(config: Config) -> nn.Module:
    """
    Create policy network architecture.
    
    Network structure: Input(3) → Hidden(n1) → ReLU → Hidden(n2) → ReLU → Output(2)
    
    Input: (B, μ, g_val) - state variables
    Output: (logit_μ'_g0, logit_μ'_g1) - policy for each possible next shock
    
    Args:
        config: Configuration object
    
    Returns:
        Policy network module
    """
    return nn.Sequential(
        nn.Linear(3, config.n1_p),
        nn.ReLU(),
        nn.Linear(config.n1_p, config.n2_p),
        nn.ReLU(),
        nn.Linear(config.n2_p, 2)
    )


def create_value_network(config: Config) -> nn.Module:
    """
    Create value network architecture.
    
    Network structure: Input(3) → Hidden(n1) → ReLU → Hidden(n2) → ReLU → Output(1)
    
    Input: (B, μ, g_val) - state variables
    Output: V(B, μ, g) - value function
    
    Args:
        config: Configuration object
    
    Returns:
        Value network module
    """
    return nn.Sequential(
        nn.Linear(3, config.n1_v),
        nn.ReLU(),
        nn.Linear(config.n1_v, config.n2_v),
        nn.ReLU(),
        nn.Linear(config.n2_v, 1)
    )


# =============================================================================
# LEGACY COMPATIBILITY LAYER
# =============================================================================
# These classes/functions maintain backward compatibility with existing code

class UniformSamplerLegacy:
    """Legacy wrapper for UniformSampler (backward compatibility)."""
    
    def __init__(self, range_x1, range_x2, range_x3, device=None):
        self.range_x1 = range_x1
        self.range_x2 = range_x2
        self.range_x3 = range_x3
        self.device = device
        self.b_min = range_x1[0]
        self.b_max = range_x1[1]
    
    def update_b_range(self, b_min, b_max):
        self.b_min = b_min
        self.b_max = b_max
    
    def generate_samples(self, num_samples):
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        
        samples = torch.rand(num_samples, 3, dtype=torch.float32, device=self.device)
        samples[:, 0] = samples[:, 0] * (self.b_max - self.b_min) + self.b_min
        samples[:, 1] = samples[:, 1] * (self.range_x2[1] - self.range_x2[0]) + self.range_x2[0]
        samples[:, 2] = (samples[:, 2] > 0.5).float()
        return samples


class define_objective:
    """Legacy wrapper for ValueSimulator (backward compatibility)."""
    
    def __init__(self, x_value_govt, x_lam_govt, x_lam_govt_old, penalty_params, device=None):
        self.device = device
        self.x_value_govt = x_value_govt.to(device)
        self.x_lam_govt = x_lam_govt.to(device)
        self.x_lam_govt_old = x_lam_govt_old.to(device)
        
        # Load config for parameters
        import json
        with open('config.json', 'r') as f:
            config_dict = json.load(f)
        
        self.config = Config(config_dict)
        
        # Override with penalty_params
        self.b_min = torch.tensor(penalty_params.get('b_min', self.config.b_min), device=device)
        self.b_max = torch.tensor(penalty_params.get('b_max', self.config.b_max), device=device)
        self.tau_min = torch.tensor(penalty_params.get('tau_min', self.config.tau_min), device=device)
        self.tau_max = torch.tensor(penalty_params.get('tau_max', self.config.tau_max), device=device)
        self.l_eps = torch.tensor(penalty_params.get('l_eps', self.config.l_eps), device=device)
        self.tau_eps = torch.tensor(penalty_params.get('tau_eps', self.config.tau_eps), device=device)
        self.b_eps = torch.tensor(penalty_params.get('b_eps', self.config.b_eps), device=device)
    
    def obj_sim_value(self, x_batch, x_zagg_vec, x_pi_zagg, x_i_ind, x_n_sim, x_print):
        """Legacy simulation interface."""
        simulator = ValueSimulator(
            self.x_value_govt, self.x_lam_govt, self.x_lam_govt_old,
            self.config, self.device
        )
        simulator.b_min = self.b_min
        simulator.b_max = self.b_max
        
        return simulator.simulate_value(
            x_batch, x_n_sim, 
            apply_penalties=(x_i_ind == 0),
            verbose=(x_print == 1)
        )


class equm_trainer:
    """Legacy wrapper for PolicyValueTrainer (backward compatibility)."""
    
    def __init__(self, num_epochs_v, num_epochs_p, lr_v, lr_p, batch_size, 
                 n_worker, x_lam_govt, x_lam_govt_old, x_value_govt, 
                 device=None, i_save=0):
        import json
        with open('config.json', 'r') as f:
            config_dict = json.load(f)
        
        config = Config(config_dict)
        config.num_epochs_v = num_epochs_v
        config.num_epochs_p = num_epochs_p
        config.lr_value = lr_v
        config.lr_policy = lr_p
        config.batch_size_v = batch_size
        config.batch_size_p = batch_size
        config.num_worker = n_worker
        
        self.trainer = PolicyValueTrainer(
            x_lam_govt, x_value_govt, config, device, 
            save_models=(i_save == 1)
        )
        self.device = device
        self.config = config
        
        # Legacy attributes
        self.x_value_govt = x_value_govt
        self.x_lam_govt = x_lam_govt
        self.num_epochs_v = num_epochs_v
        self.num_epochs_p = num_epochs_p
    
    def policy_train(self, x_data, x_zagg_vec, x_pi_zagg, x_i_ind, x_n_sim, 
                     x_print, all_losses_list=None, inadmissible_samples=None):
        return self.trainer.train_policy(x_data, all_losses_list, inadmissible_samples)
    
    def value_train(self, compute_value, x_data, all_losses_list=None,
                    inadmissible_samples=None, v_threshold=None):
        return self.trainer.train_value(x_data, all_losses_list, inadmissible_samples)


class policy_equm_funcs:
    """Legacy wrapper for SurfacePlotter (backward compatibility)."""
    
    def __init__(self, x_policy, x_value, scorer=None, admissibility_threshold=None, device=None):
        import json
        with open('config.json', 'r') as f:
            config_dict = json.load(f)
        
        config = Config(config_dict)
        self.plotter = SurfacePlotter(x_policy, x_value, config, device, 
                                      scorer, admissibility_threshold)
        
        # Legacy attributes
        self.x_policy = x_policy
        self.x_value = x_value
        self.device = device
    
    def generate_data(self, n):
        self.plotter.generate_data(n)
        # Copy to legacy attributes
        self.X1 = self.plotter.X1
        self.X2 = self.plotter.X2
        self.Y_policy = self.plotter.Y_policy
        self.Y_value = self.plotter.Y_value
        self.Y_tau = self.plotter.Y_tau
        self.Y_b1 = self.plotter.Y_b1
    
    def create_plot(self, save_filename='figures/final_surface_plots.png', title_suffix=''):
        self.plotter.create_plot(save_filename, title_suffix)


# Legacy global variables for backward compatibility
def load_config(config_file):
    """Load configuration from JSON file."""
    import json
    with open(config_file, 'r') as f:
        return json.load(f)

        # Set model_number as module-level variable for legacy code
        model_number = 102
        """d to plot titles
        """
        print(f"--- Generating surface plots {title_suffix.strip()} ---")
        
        fig = plt.figure(figsize=(12, 12))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Policy surface
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.plot_surface(self.X1, self.X2, self.Y_policy, cmap='viridis')
        ax1.set_xlabel('B')
        ax1.set_ylabel('μ')
        ax1.set_zlabel("μ'")
        ax1.set_title('Policy Function' + title_suffix)
        
        # Value surface
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        ax2.plot_surface(self.X1, self.X2, self.Y_value, cmap='plasma')
        ax2.set_xlabel('B')
        ax2.set_ylabel('μ')
        ax2.set_zlabel('V')
        ax2.set_title('Value Function' + title_suffix)
        
        # Tax rate surface
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax3.plot_surface(self.X1, self.X2, self.Y_tau, cmap='coolwarm')
        ax3.set_xlabel('B')
        ax3.set_ylabel('μ')
        ax3.set_zlabel('τ')
        ax3.set_title('Implied Tax Rate' + title_suffix)
        
        # Next-period debt surface
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        ax4.plot_surface(self.X1, self.X2, self.Y_b1, cmap='cividis')
        ax4.set_xlabel('B')
        ax4.set_ylabel('μ')
        ax4.set_zlabel("B'")
        ax4.set_title('Next-Period Debt' + title_suffix)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Plot saved to: {save_path}")
