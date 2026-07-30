"""
Pre-training Module for Ramsey Optimal Taxation Networks.

This module initializes the neural networks using high-accuracy solution data
from a C++ solver. Pre-training provides a good starting point that significantly
accelerates convergence of the main training loop.

Workflow:
    1. Load solution data from C++ output file (policy_v6_out_61.txt)
    2. Filter out invalid/flagged data points
    3. Train value network via MSE regression on V(B, μ, g)
    4. Train policy network via MSE regression on logit(μ'(B, μ, g))
    5. Generate verification plots comparing actual vs. predicted surfaces

Data Format (C++ output):
    Columns: b0, mu0, g_index, v0, b1, mu1_g0, mu1_g1, c, l, tau, q
    - (b0, mu0, g_index): State variables
    - v0: Value function
    - (mu1_g0, mu1_g1): Policy function (μ' for each next-period shock)
    - (c, l, tau, q): Allocations and prices

Authors: Zhigang Feng
Version: 2.0 (Streamlined)
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from typing import Tuple, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

class PretrainConfig:
    """Configuration for pre-training from C++ data."""
    
    def __init__(self, config: dict):
        """
        Load pre-training configuration.
        
        Args:
            config: Configuration dictionary
        """
        # Model I/O
        io_cfg = config.get('model_io', config)
        self.txt_data_file = io_cfg.get('txt_data_file', 'policy_v6_out_61.txt')
        self.flag_value = io_cfg.get('flag_value', -500)
        
        # Pre-training hyperparameters
        pretrain = config.get('pretraining', {})
        self.epochs = pretrain.get('epochs', 1000)
        self.lr_value = pretrain.get('lr_value', 0.0005)
        self.lr_policy = pretrain.get('lr_policy', 0.0005)
        self.batch_size = pretrain.get('batch_size', 256)
        self.eps_scaler = pretrain.get('eps_scaler', 1e-7)
        
        # Network architecture
        arch = config.get('network_architecture', config)
        self.n1_p = arch.get('n1_p', config.get('n1_p', 64))
        self.n2_p = arch.get('n2_p', config.get('n2_p', 64))
        self.n1_v = arch.get('n1_v', config.get('n1_v', 64))
        self.n2_v = arch.get('n2_v', config.get('n2_v', 64))
        
        # Economic parameters (for g_value mapping)
        econ = config.get('economic_parameters', config)
        self.zagg_vec = econ.get('zagg_vec', config.get('zagg_vec'))
        
        # Output paths
        self.save_dir = 'models'
        self.figure_dir = 'figures'
        self.value_model_path = os.path.join(self.save_dir, 'pretrained_from_txt_value.pth')
        self.policy_model_path = os.path.join(self.save_dir, 'pretrained_from_txt_policy.pth')


# =============================================================================
# NETWORK FACTORIES
# =============================================================================

def create_policy_network(config: PretrainConfig) -> nn.Module:
    """
    Create policy network for pre-training.
    
    Architecture: (3) → (n1_p) → ReLU → (n2_p) → ReLU → (2)
    
    Args:
        config: Pre-training configuration
    
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


def create_value_network(config: PretrainConfig) -> nn.Module:
    """
    Create value network for pre-training.
    
    Architecture: (3) → (n1_v) → ReLU → (n2_v) → ReLU → (1)
    
    Args:
        config: Pre-training configuration
    
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
# UTILITY FUNCTIONS
# =============================================================================

def inverse_sigmoid_scaler(mu: torch.Tensor, mu_min: float, mu_max: float,
                           eps: float = 1e-7) -> torch.Tensor:
    """
    Convert bounded μ values to unbounded logit representation.
    
    The policy network outputs logits, which are converted to μ' via:
        μ' = sigmoid(logit) × (μ_max - μ_min) + μ_min
    
    This function inverts that transformation for training targets.
    
    Args:
        mu: Values in [mu_min, mu_max]
        mu_min: Lower bound
        mu_max: Upper bound
        eps: Small constant to avoid numerical issues
    
    Returns:
        Logit values (unbounded)
    """
    # Scale to (0, 1)
    mu_scaled = (mu - mu_min) / (mu_max - mu_min)
    mu_scaled_clamped = torch.clamp(mu_scaled, eps, 1.0 - eps)
    
    # Apply logit transformation
    logit = torch.log(mu_scaled_clamped / (1.0 - mu_scaled_clamped))
    return logit


# =============================================================================
# DATA LOADING
# =============================================================================

def load_cpp_data(config: PretrainConfig) -> Tuple[pd.DataFrame, float, float]:
    """
    Load and preprocess C++ solution data.
    
    Steps:
    1. Read whitespace-delimited text file
    2. Filter out rows with flag values (-500)
    3. Map g_index to actual g values
    4. Compute data-driven μ bounds
    
    Args:
        config: Pre-training configuration
    
    Returns:
        Tuple of (dataframe, mu_min, mu_max)
    
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If no valid data remains after filtering
    """
    col_names = ['b0', 'mu0', 'g_index', 'v0', 'b1', 'mu1_g0', 'mu1_g1',
                 'c', 'l', 'tau', 'q']
    
    print(f"Loading data from {config.txt_data_file}...")
    
    try:
        df = pd.read_csv(config.txt_data_file, header=None, 
                        delim_whitespace=True, names=col_names)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {config.txt_data_file}")
    
    # Filter flagged rows
    initial_rows = len(df)
    mask = (df != config.flag_value).all(axis=1)
    df = df[mask]
    filtered_rows = len(df)
    
    print(f"  Loaded {initial_rows} data points")
    print(f"  Dropped {initial_rows - filtered_rows} flagged rows")
    print(f"  Using {filtered_rows} valid data points")
    
    if filtered_rows == 0:
        raise ValueError("No valid data after filtering")
    
    # Map g_index to g_value
    g_values = [config.zagg_vec[0][0], config.zagg_vec[1][0]]
    df['g0'] = df['g_index'].map(lambda idx: g_values[int(idx)])
    
    # Compute data-driven μ bounds
    mu_cols = ['mu0', 'mu1_g0', 'mu1_g1']
    mu_min = df[mu_cols].min().min()
    mu_max = df[mu_cols].max().max()
    
    print(f"\n  Data-driven μ range: [{mu_min:.4f}, {mu_max:.4f}]")
    print("  SUGGESTION: Update config 'mu_min'/'mu_max' to match this range")
    
    return df, mu_min, mu_max


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def pretrain_value_network(value_net: nn.Module, df: pd.DataFrame,
                           config: PretrainConfig, device: torch.device
                           ) -> Tuple[nn.Module, list]:
    """
    Pre-train value network on C++ solution data.
    
    Trains via MSE regression: minimize ||V_θ(B, μ, g) - V_data||²
    
    Args:
        value_net: Value network to train
        df: DataFrame with columns ['b0', 'mu0', 'g0', 'v0']
        config: Pre-training configuration
        device: PyTorch device
    
    Returns:
        Tuple of (trained_network, loss_history)
    """
    # Prepare data
    X = df[['b0', 'mu0', 'g0']].values
    y = df[['v0']].values
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Training setup
    optimizer = torch.optim.Adam(value_net.parameters(), lr=config.lr_value)
    loss_fn = nn.MSELoss()
    loss_history = []
    
    print("\nTraining value network...")
    
    for epoch in tqdm(range(config.epochs), desc='Value Pre-training'):
        epoch_loss = 0.0
        
        for X_batch, y_batch in loader:
            y_pred = value_net(X_batch)
            loss = loss_fn(y_pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 100 == 0:
            tqdm.write(f"  Epoch {epoch+1}: Loss = {avg_loss:.6f}")
    
    return value_net, loss_history


def pretrain_policy_network(policy_net: nn.Module, df: pd.DataFrame,
                            mu_min: float, mu_max: float,
                            config: PretrainConfig, device: torch.device
                            ) -> Tuple[nn.Module, list]:
    """
    Pre-train policy network on C++ solution data.
    
    Trains via MSE regression on logit-transformed targets:
        minimize ||logit_θ(B, μ, g) - logit(μ'_data)||²
    
    Args:
        policy_net: Policy network to train
        df: DataFrame with columns ['b0', 'mu0', 'g0', 'mu1_g0', 'mu1_g1']
        mu_min: Lower bound for μ
        mu_max: Upper bound for μ
        config: Pre-training configuration
        device: PyTorch device
    
    Returns:
        Tuple of (trained_network, loss_history)
    """
    # Prepare data
    X = df[['b0', 'mu0', 'g0']].values
    y_raw = df[['mu1_g0', 'mu1_g1']].values
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_raw_tensor = torch.tensor(y_raw, dtype=torch.float32).to(device)
    
    # Convert targets to logits
    y_logit_tensor = inverse_sigmoid_scaler(y_raw_tensor, mu_min, mu_max, config.eps_scaler)
    
    dataset = TensorDataset(X_tensor, y_logit_tensor)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Training setup
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=config.lr_policy)
    loss_fn = nn.MSELoss()
    loss_history = []
    
    print("\nTraining policy network...")
    
    for epoch in tqdm(range(config.epochs), desc='Policy Pre-training'):
        epoch_loss = 0.0
        
        for X_batch, y_batch in loader:
            y_pred = policy_net(X_batch)
            loss = loss_fn(y_pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 100 == 0:
            tqdm.write(f"  Epoch {epoch+1}: Loss = {avg_loss:.6f}")
    
    return policy_net, loss_history


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_training_losses(v_losses: list, p_losses: list, save_path: str):
    """
    Plot training loss curves for both networks.
    
    Args:
        v_losses: Value network loss history
        p_losses: Policy network loss history
        save_path: Output file path
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(v_losses, 'b-', linewidth=1)
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Value Network Pre-training')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(p_losses, 'orange', linewidth=1)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss (Logit)')
    ax2.set_title('Policy Network Pre-training')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nLoss plot saved to: {save_path}")


def plot_verification_surfaces(value_net: nn.Module, policy_net: nn.Module,
                                df: pd.DataFrame, mu_min: float, mu_max: float,
                                config: PretrainConfig, device: torch.device):
    """
    Generate verification plots comparing actual vs. predicted surfaces.
    
    Creates 3×2 grids for each shock state showing:
    - Row 1: Value function (actual vs. predicted)
    - Row 2: Policy μ'(g=0) (actual vs. predicted)
    - Row 3: Policy μ'(g=1) (actual vs. predicted)
    
    Args:
        value_net: Trained value network
        policy_net: Trained policy network
        df: DataFrame with solution data
        mu_min, mu_max: μ bounds
        config: Pre-training configuration
        device: PyTorch device
    """
    print("\nGenerating verification plots...")
    
    value_net.eval()
    policy_net.eval()
    
    g_values = [config.zagg_vec[0][0], config.zagg_vec[1][0]]
    g_names = ['low_g', 'high_g']
    
    for g_val, g_name in zip(g_values, g_names):
        print(f"  Plotting for {g_name} state (g = {g_val})...")
        
        df_g = df[df['g0'] == g_val].copy()
        if len(df_g) == 0:
            print(f"    No data for g = {g_val}, skipping.")
            continue
        
        # Get raw data points
        X_b = df_g['b0'].values
        X_mu = df_g['mu0'].values
        points = np.column_stack([X_b, X_mu])
        
        # Actual values
        Z_val_actual = df_g['v0'].values
        Z_pol0_actual = df_g['mu1_g0'].values
        Z_pol1_actual = df_g['mu1_g1'].values
        
        # Predicted values
        X_tensor = torch.tensor(df_g[['b0', 'mu0', 'g0']].values, 
                               dtype=torch.float32).to(device)
        
        with torch.no_grad():
            Z_val_pred = value_net(X_tensor).cpu().numpy().squeeze()
            policy_logits = policy_net(X_tensor)
            Z_pol0_pred = (torch.sigmoid(policy_logits[:, 0]) * (mu_max - mu_min) + mu_min).cpu().numpy()
            Z_pol1_pred = (torch.sigmoid(policy_logits[:, 1]) * (mu_max - mu_min) + mu_min).cpu().numpy()
        
        # Create interpolation grid
        try:
            grid_x, grid_y = np.mgrid[X_b.min():X_b.max():100j, 
                                       X_mu.min():X_mu.max():100j]
        except ValueError:
            print(f"    Cannot create grid, skipping.")
            continue
        
        # Interpolate surfaces
        grid_val_actual = griddata(points, Z_val_actual, (grid_x, grid_y), method='cubic')
        grid_val_pred = griddata(points, Z_val_pred, (grid_x, grid_y), method='cubic')
        grid_pol0_actual = griddata(points, Z_pol0_actual, (grid_x, grid_y), method='cubic')
        grid_pol0_pred = griddata(points, Z_pol0_pred, (grid_x, grid_y), method='cubic')
        grid_pol1_actual = griddata(points, Z_pol1_actual, (grid_x, grid_y), method='cubic')
        grid_pol1_pred = griddata(points, Z_pol1_pred, (grid_x, grid_y), method='cubic')
        
        # Create figure
        fig, axes = plt.subplots(3, 2, figsize=(14, 18), subplot_kw={'projection': '3d'})
        fig.suptitle(f'Pre-training Verification (g = {g_val})', fontsize=16)
        
        def plot_surface(ax, grid_z, scatter_z, title):
            ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', alpha=0.7)
            ax.scatter(X_b, X_mu, scatter_z, c='red', s=3, alpha=0.3)
            ax.set_xlabel('B')
            ax.set_ylabel('μ')
            ax.set_title(title)
        
        plot_surface(axes[0, 0], grid_val_actual, Z_val_actual, 'Actual Value')
        plot_surface(axes[0, 1], grid_val_pred, Z_val_actual, 'Predicted Value (NN)')
        
        plot_surface(axes[1, 0], grid_pol0_actual, Z_pol0_actual, "Actual μ'(g'=0)")
        plot_surface(axes[1, 1], grid_pol0_pred, Z_pol0_actual, "Predicted μ'(g'=0)")
        
        plot_surface(axes[2, 0], grid_pol1_actual, Z_pol1_actual, "Actual μ'(g'=1)")
        plot_surface(axes[2, 1], grid_pol1_pred, Z_pol1_actual, "Predicted μ'(g'=1)")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = os.path.join(config.figure_dir, f'pretrain_verification_{g_name}.png')
        os.makedirs(config.figure_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"    Saved: {save_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_pretraining(config_file: str = 'config.json') -> Tuple[str, str]:
    """
    Execute full pre-training pipeline.
    
    Steps:
    1. Load configuration and data
    2. Create networks
    3. Train value network
    4. Train policy network
    5. Save models
    6. Generate verification plots
    
    Args:
        config_file: Path to configuration JSON file
    
    Returns:
        Tuple of (value_model_path, policy_model_path)
    """
    print("=" * 60)
    print("PRE-TRAINING FROM C++ SOLUTION DATA")
    print("=" * 60)
    
    # Load configuration
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    config = PretrainConfig(config_dict)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load data
    df, mu_min, mu_max = load_cpp_data(config)
    
    # Create networks
    value_net = create_value_network(config).to(device)
    policy_net = create_policy_network(config).to(device)
    
    # Train networks
    value_net, v_losses = pretrain_value_network(value_net, df, config, device)
    policy_net, p_losses = pretrain_policy_network(policy_net, df, mu_min, mu_max, config, device)
    
    # Save models
    os.makedirs(config.save_dir, exist_ok=True)
    torch.save(value_net.state_dict(), config.value_model_path)
    torch.save(policy_net.state_dict(), config.policy_model_path)
    
    print(f"\nModels saved:")
    print(f"  Value:  {config.value_model_path}")
    print(f"  Policy: {config.policy_model_path}")
    
    # Generate plots
    loss_plot_path = os.path.join(config.figure_dir, 'pretrain_losses.png')
    plot_training_losses(v_losses, p_losses, loss_plot_path)
    plot_verification_surfaces(value_net, policy_net, df, mu_min, mu_max, config, device)
    
    print("\n" + "=" * 60)
    print("PRE-TRAINING COMPLETE")
    print("=" * 60)
    
    return config.value_model_path, config.policy_model_path


# Legacy exports for backward compatibility
VALUE_MODEL_PATH = 'models/pretrained_from_txt_value.pth'
POLICY_MODEL_PATH = 'models/pretrained_from_txt_policy.pth'


if __name__ == "__main__":
    run_pretraining()
