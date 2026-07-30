# pretrain_from_txt.py
# This module reads data from the C++ output file to pre-train the models.
# --- MODIFIED: Added raw data scatter plots to verification surfaces ---

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# --- Configuration ---
# (You can adjust these hyperparameters)
CONFIG_FILE = 'config.json'
TXT_DATA_FILE = 'policy_v6_out_61.txt'  # The accurate data file
PRETRAIN_EPOCHS = 1000  # Increase epochs for better convergence
PRETRAIN_LR_VALUE = 0.0005  # Learning rate for Value Net
PRETRAIN_LR_POLICY = 0.0005  # Learning rate for Policy Net
PRETRAIN_BATCH_SIZE = 256
EPS_SCALER = 1e-7  # Small epsilon for inverse sigmoid
FLAG_VALUE = -500  # Value to filter out

# --- Define Output Paths ---
SAVE_DIR = 'models'
FIGURE_DIR = 'figures'
VALUE_MODEL_PATH = os.path.join(SAVE_DIR, 'pretrained_from_txt_value.pth')
POLICY_MODEL_PATH = os.path.join(SAVE_DIR, 'pretrained_from_txt_policy.pth')
LOSS_PLOT_FILE = os.path.join(FIGURE_DIR, 'pretrain_losses.png')
VERIFY_PLOT_LOW_G = os.path.join(FIGURE_DIR, 'pretrain_verification_low_g.png')
VERIFY_PLOT_HIGH_G = os.path.join(FIGURE_DIR, 'pretrain_verification_high_g.png')


# --- Load Config ---
def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


# --- Define Network Structures ---
def create_policy_network(config):
    n1_p = config['n1_p']
    n2_p = config['n2_p']
    n_input, n_output = 3, 2
    return nn.Sequential(nn.Linear(n_input, n1_p),
                         nn.ReLU(),
                         nn.Linear(n1_p, n2_p),
                         nn.ReLU(),
                         nn.Linear(n2_p, n_output))


def create_value_network(config):
    n1_v = config['n1_v']
    n2_v = config['n2_v']
    n_input, n_output = 3, 1
    return nn.Sequential(nn.Linear(n_input, n1_v),
                         nn.ReLU(),
                         nn.Linear(n1_v, n2_v),
                         nn.ReLU(),
                         nn.Linear(n2_v, n_output))


# --- Inverse Sigmoid (Logit) Function ---
def inverse_sigmoid_scaler(mu, mu_min, mu_max, eps=EPS_SCALER):
    """
    Converts a value 'mu' from [mu_min, mu_max] back to its
    unbounded logit representation.
    """
    # Scale mu to [0, 1]
    mu_scaled = (mu - mu_min) / (mu_max - mu_min)

    # Clamp to avoid log(0) or log(inf)
    mu_scaled_clamped = torch.clamp(mu_scaled, eps, 1.0 - eps)

    # Apply inverse sigmoid (logit)
    logit = torch.log(mu_scaled_clamped / (1.0 - mu_scaled_clamped))
    return logit


# --- Plotting Function ---
def plot_losses(v_losses, p_losses, save_path):
    """Saves a plot of the training losses."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Value Loss Plot
    ax1.plot(v_losses, label='Value Loss (MSE)', color='blue')
    ax1.set_title('Value Network Pre-training Loss')
    ax1.set_ylabel('Loss')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Policy Loss Plot
    ax2.plot(p_losses, label='Policy Loss (MSE on Logits)', color='orange')
    ax2.set_title('Policy Network Pre-training Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"\nLoss plot saved to {save_path}")


# --- Verification Plotting Function ---
def plot_verification_surfaces(value_govt, lam_govt, clean_df, mu_min, mu_max, config, device):
    """
    Plots the actual vs. predicted surfaces for verification.
    """
    print("Generating verification surface plots...")
    value_govt.eval()
    lam_govt.eval()

    g_values = [config['zagg_vec'][0][0], config['zagg_vec'][1][0]]
    g_names = ['low_g', 'high_g']
    plot_files = [VERIFY_PLOT_LOW_G, VERIFY_PLOT_HIGH_G]

    for g_val, g_name, plot_file in zip(g_values, g_names, plot_files):
        print(f"  Plotting for {g_name} state (g = {g_val})...")
        df_g = clean_df[clean_df['g0'] == g_val].copy()

        if len(df_g) == 0:
            print(f"  No data found for g = {g_val}. Skipping plot.")
            continue

        # --- 1. Get Inputs and Actual Outputs (Raw Data) ---
        X_b0 = df_g['b0'].values
        Y_mu0 = df_g['mu0'].values
        points = np.array([X_b0, Y_mu0]).T

        Z_val_actual = df_g['v0'].values
        Z_pol0_actual = df_g['mu1_g0'].values
        Z_pol1_actual = df_g['mu1_g1'].values

        # --- 2. Get Predicted Outputs (Raw Data) ---
        X_tensor = torch.tensor(df_g[['b0', 'mu0', 'g0']].values, dtype=torch.float32).to(device)

        with torch.no_grad():
            Z_val_pred_tensor = value_govt(X_tensor)
            Z_pol_logit_tensor = lam_govt(X_tensor)

        # Scale policy predictions from logits back to mu values
        Z_pol0_pred_tensor = torch.sigmoid(Z_pol_logit_tensor[:, 0]) * (mu_max - mu_min) + mu_min
        Z_pol1_pred_tensor = torch.sigmoid(Z_pol_logit_tensor[:, 1]) * (mu_max - mu_min) + mu_min

        # Move to CPU/Numpy
        Z_val_pred = Z_val_pred_tensor.cpu().numpy().squeeze()
        Z_pol0_pred = Z_pol0_pred_tensor.cpu().numpy().squeeze()
        Z_pol1_pred = Z_pol1_pred_tensor.cpu().numpy().squeeze()

        # --- 3. Create Interpolation Grid (for Surfaces) ---
        try:
            grid_x, grid_y = np.mgrid[X_b0.min():X_b0.max():100j, Y_mu0.min():Y_mu0.max():100j]
        except ValueError as e:
            print(f"  Error creating grid (not enough data points?): {e}. Skipping plot.")
            continue

        # Interpolate all 6 surfaces
        grid_Z_val_actual = griddata(points, Z_val_actual, (grid_x, grid_y), method='cubic')
        grid_Z_pol0_actual = griddata(points, Z_pol0_actual, (grid_x, grid_y), method='cubic')
        grid_Z_pol1_actual = griddata(points, Z_pol1_actual, (grid_x, grid_y), method='cubic')

        grid_Z_val_pred = griddata(points, Z_val_pred, (grid_x, grid_y), method='cubic')
        grid_Z_pol0_pred = griddata(points, Z_pol0_pred, (grid_x, grid_y), method='cubic')
        grid_Z_pol1_pred = griddata(points, Z_pol1_pred, (grid_x, grid_y), method='cubic')

        # --- 4. Create Plots (3x2 grid) ---
        fig, axes = plt.subplots(3, 2, figsize=(16, 20), subplot_kw={'projection': '3d'})
        fig.suptitle(f'Pre-training Verification (State g = {g_val})', fontsize=20)

        # --- Plot Value ---
        ax = axes[0, 0]
        ax.plot_surface(grid_x, grid_y, grid_Z_val_actual, cmap='viridis', alpha=0.7)
        ax.scatter(X_b0, Y_mu0, Z_val_actual, color='red', s=5, alpha=0.4, label='Actual Data')
        ax.set_title("Actual Value (Interpolated)")
        ax.set_xlabel('b0');
        ax.set_ylabel('mu0')

        ax = axes[0, 1]
        ax.plot_surface(grid_x, grid_y, grid_Z_val_pred, cmap='viridis', alpha=0.7)
        ax.scatter(X_b0, Y_mu0, Z_val_actual, color='red', s=5, alpha=0.4, label='Actual Data')
        ax.set_title("Predicted Value (NN)")
        ax.set_xlabel('b0');
        ax.set_ylabel('mu0')

        # --- Plot Policy (g=0) ---
        ax = axes[1, 0]
        ax.plot_surface(grid_x, grid_y, grid_Z_pol0_actual, cmap='viridis', alpha=0.7)
        ax.scatter(X_b0, Y_mu0, Z_pol0_actual, color='red', s=5, alpha=0.4, label='Actual Data')
        ax.set_title("Actual Policy (mu' for g=0)")
        ax.set_xlabel('b0');
        ax.set_ylabel('mu0')

        ax = axes[1, 1]
        ax.plot_surface(grid_x, grid_y, grid_Z_pol0_pred, cmap='viridis', alpha=0.7)
        ax.scatter(X_b0, Y_mu0, Z_pol0_actual, color='red', s=5, alpha=0.4, label='Actual Data')
        ax.set_title("Predicted Policy (mu' for g=0)")
        ax.set_xlabel('b0');
        ax.set_ylabel('mu0')

        # --- Plot Policy (g=1) ---
        ax = axes[2, 0]
        ax.plot_surface(grid_x, grid_y, grid_Z_pol1_actual, cmap='viridis', alpha=0.7)
        ax.scatter(X_b0, Y_mu0, Z_pol1_actual, color='red', s=5, alpha=0.4, label='Actual Data')
        ax.set_title("Actual Policy (mu' for g=1)")
        ax.set_xlabel('b0');
        ax.set_ylabel('mu0')

        ax = axes[2, 1]
        ax.plot_surface(grid_x, grid_y, grid_Z_pol1_pred, cmap='viridis', alpha=0.7)
        ax.scatter(X_b0, Y_mu0, Z_pol1_actual, color='red', s=5, alpha=0.4, label='Actual Data')
        ax.set_title("Predicted Policy (mu' for g=1)")
        ax.set_xlabel('b0');
        ax.set_ylabel('mu0')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(plot_file)
        plt.close(fig)
        print(f"  Verification plot saved to: {plot_file}")


# --- Main Pre-training Function ---
def run_pretraining():
    """
    Loads data, trains, and saves the pre-trained models.
    """
    print(f"--- Starting Pre-training from {TXT_DATA_FILE} ---")

    # 1. Load Config and Set Device
    config = load_config(CONFIG_FILE)
    zagg_vec = config['zagg_vec']

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 2. Load and Process Text Data
    print("Loading and processing text file...")
    col_names = ['b0', 'mu0', 'g_index', 'v0', 'b1', 'mu1_g0', 'mu1_g1',
                 'c', 'l', 'tau', 'q']

    try:
        df = pd.read_csv(TXT_DATA_FILE, header=None, delim_whitespace=True, names=col_names)
    except FileNotFoundError:
        print(f"ERROR: Could not find the data file: {TXT_DATA_FILE}")
        print("Please make sure it is in the same directory.")
        return
    except Exception as e:
        print(f"Error reading {TXT_DATA_FILE}: {e}")
        return

    # --- NEW: Filter out rows with the flag value ---
    initial_rows = len(df)
    # This creates a boolean mask for rows where *all* values are NOT the flag
    mask = (df != FLAG_VALUE).all(axis=1)
    df = df[mask]
    filtered_rows = len(df)
    rows_dropped = initial_rows - filtered_rows

    print(f"Loaded {initial_rows} data points.")
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows containing the {FLAG_VALUE} flag.")
    print(f"Using {filtered_rows} valid data points for pre-training.")
    # --- End of filtering ---

    if filtered_rows == 0:
        print("ERROR: No valid data remaining after filtering. Aborting.")
        return

    # --- TASK 1: Map g_index (0 or 1) to actual g_value ---
    g_values = [zagg_vec[0][0], zagg_vec[1][0]]
    df['g0'] = df['g_index'].map(lambda idx: g_values[int(idx)])
    print(f"Created 'g0' state variable from 'g_index' and 'zagg_vec'.")

    # --- TASK 2: Find data-driven mu_min and mu_max ---
    mu_cols = ['mu0', 'mu1_g0', 'mu1_g1']
    data_mu_min = df[mu_cols].min().min()
    data_mu_max = df[mu_cols].max().max()

    print("\n" + "=" * 40)
    print("  Data-Driven Multiplier Range Scan  ")
    print(f"  Min 'mu' found in data: {data_mu_min:.6f}")
    print(f"  Max 'mu' found in data: {data_mu_max:.6f}")
    print("  SUGGESTION: Update 'mu_min' and 'mu_max'")
    print("  in 'config.json' to this range")
    print("  (or slightly wider, e.g., [1.0, 4.6])")
    print("  Using this range for pre-training scaler.")
    print("=" * 40 + "\n")

    # Use these data-driven values for the scaler
    mu_min = data_mu_min
    mu_max = data_mu_max

    # 3. Prepare Data for Networks

    # --- Value Network Data ---
    # State is (b0, mu0, g0)
    X_v_data = df[['b0', 'mu0', 'g0']].values
    y_v_data = df[['v0']].values

    X_v_tensor = torch.tensor(X_v_data, dtype=torch.float32)
    y_v_tensor = torch.tensor(y_v_data, dtype=torch.float32)

    value_dataset = TensorDataset(X_v_tensor, y_v_tensor)
    value_loader = DataLoader(value_dataset, batch_size=PRETRAIN_BATCH_SIZE, shuffle=True)

    # --- Policy Network Data ---
    # State is (b0, mu0, g0)
    X_p_data = df[['b0', 'mu0', 'g0']].values
    # Target values are mu_next for g=0 and g=1
    y_p_data_raw = df[['mu1_g0', 'mu1_g1']].values

    X_p_tensor = torch.tensor(X_p_data, dtype=torch.float32)
    y_p_tensor_raw = torch.tensor(y_p_data_raw, dtype=torch.float32)

    # *** IMPORTANT: Convert targets to logits using data-driven min/max ***
    y_p_tensor_logit = inverse_sigmoid_scaler(y_p_tensor_raw, mu_min, mu_max)

    policy_dataset = TensorDataset(X_p_tensor, y_p_tensor_logit)
    policy_loader = DataLoader(policy_dataset, batch_size=PRETRAIN_BATCH_SIZE, shuffle=True)

    # 4. Initialize Networks, Optimizers, Loss
    value_govt = create_value_network(config).to(device)
    lam_govt = create_policy_network(config).to(device)

    optimizer_v = torch.optim.Adam(value_govt.parameters(), lr=PRETRAIN_LR_VALUE)
    optimizer_p = torch.optim.Adam(lam_govt.parameters(), lr=PRETRAIN_LR_POLICY)

    loss_fn = nn.MSELoss()

    # --- TASK 3: Store losses for plotting ---
    v_loss_history = []
    p_loss_history = []

    # 5. Run Training Loop
    print(f"Starting training for {PRETRAIN_EPOCHS} epochs...")
    for epoch in tqdm(range(PRETRAIN_EPOCHS), desc='Pre-training'):
        value_govt.train()
        lam_govt.train()

        v_epoch_loss = 0.0
        p_epoch_loss = 0.0

        # --- Train Value Net ---
        for X_batch, y_batch in value_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_pred = value_govt(X_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer_v.zero_grad()
            loss.backward()
            optimizer_v.step()
            v_epoch_loss += loss.item()

        # --- Train Policy Net ---
        for X_batch, y_batch_logit in policy_loader:
            X_batch, y_batch_logit = X_batch.to(device), y_batch_logit.to(device)

            y_pred_logit = lam_govt(X_batch)
            loss = loss_fn(y_pred_logit, y_batch_logit)

            optimizer_p.zero_grad()
            loss.backward()
            optimizer_p.step()
            p_epoch_loss += loss.item()

        # Store average epoch losses
        avg_v_loss = v_epoch_loss / len(value_loader) if len(value_loader) > 0 else 0
        avg_p_loss = p_epoch_loss / len(policy_loader) if len(policy_loader) > 0 else 0
        v_loss_history.append(avg_v_loss)
        p_loss_history.append(avg_p_loss)

        if (epoch + 1) % 50 == 0:
            tqdm.write(f"Epoch {epoch + 1}/{PRETRAIN_EPOCHS} | "
                       f"V Loss: {avg_v_loss:.6f} | "
                       f"P Loss: {avg_p_loss:.6f}")

    # 6. Save Models
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(value_govt.state_dict(), VALUE_MODEL_PATH)
    torch.save(lam_govt.state_dict(), POLICY_MODEL_PATH)

    print("\n--- Pre-training complete ---")
    print(f"Value model saved to: {VALUE_MODEL_PATH}")
    print(f"Policy model saved to: {POLICY_MODEL_PATH}")

    # 7. Plot and Save Losses
    plot_losses(v_loss_history, p_loss_history, LOSS_PLOT_FILE)

    # 8. --- NEW: Plot Verification Surfaces ---
    plot_verification_surfaces(value_govt, lam_govt, df, mu_min, mu_max, config, device)


if __name__ == "__main__":
    run_pretraining()