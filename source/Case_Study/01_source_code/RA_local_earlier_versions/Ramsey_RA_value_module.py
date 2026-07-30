# Ramsey_RA_value_module.py
# REFINED: Added Two-Stage Training Logic for Policy and Value
# REFINED: Loads all hyperparameters from config, including num_epochs_p_stage2

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.optimize import minimize_scalar
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import pdb
import torch.autograd.profiler as profiler
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import os


# --- Helper: Inverse Sigmoid (Logit) ---
def inverse_sigmoid_scaler(mu, mu_min, mu_max, eps=1e-7):
    """
    Converts a value 'mu' from [mu_min, mu_max] back to its
    unbounded logit representation.
    """
    mu_scaled = (mu - mu_min) / (mu_max - mu_min)
    mu_scaled_clamped = torch.clamp(mu_scaled, eps, 1.0 - eps)
    logit = torch.log(mu_scaled_clamped / (1.0 - mu_scaled_clamped))
    return logit


# -------------------------------------------

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


# Load configuration from the JSON file
config = load_config("config.json")
beta = config['beta']
gamma_l = config['gamma_l']

# --- MODIFIED: Load bounds from 'penalty_params' ---
penalty_params_config = config.get('penalty_params', {})
b_min_config = penalty_params_config.get('b_min', -0.5)
b_max_config = penalty_params_config.get('b_max', 3.5)
tau_min_config = penalty_params_config.get('tau_min', 0.0)
tau_max_config = penalty_params_config.get('tau_max', 1.0)
l_eps_config = penalty_params_config.get('l_eps', 0.01)
tau_eps_config = penalty_params_config.get('tau_eps', 0.01)
b_eps_config = penalty_params_config.get('b_eps', 0.01)
# ---

mu_min = config['mu_min']
mu_max = config['mu_max']
v_min = config['v_min']
v_max = config['v_max']
num_samples_value = config['num_samples_value']
zagg_vec = config['zagg_vec']
pi_zagg = config['pi_zagg']
n_v_sim = config['n_v_sim']
model_number = config['model_number_output']
lr_factor = config['lr_factor']
lr_patience = config['lr_patience']

# --- MODIFIED: Load all epoch counts ---
num_epochs_draw = config.get('num_epochs_draw', 5)
num_epochs_p_config = config.get('num_epochs_p', 10)
num_epochs_v_config = config.get('num_epochs_v', 10)
num_epochs_p_stage2_config = config.get('num_epochs_p_stage2', 5)  # Stage 2 hyperparameter
# ---

num_samples_expand_p = config['num_samples_expand_p']
num_samples_expand_v = config['num_samples_expand_v']

# --- NEW: Load two-stage training flag ---
use_two_stage_training_config = config.get('scoring_parameters', {}).get('use_two_stage_training', False)


def contains_nan(tensor):
    return torch.isnan(tensor).any().item()


class UniformSampler:
    """
    Generates uniform samples for (b, mu, g_index).
    """

    def __init__(self, range_x1, range_x2, range_x3, device=None):
        self.range_x1 = range_x1  # b_min, b_max
        self.range_x2 = range_x2  # mu_min, mu_max
        self.range_x3 = range_x3  # [0.0, 1.0] for g_index
        self.device = device
        # --- Store dynamic bounds ---
        self.b_min = range_x1[0]
        self.b_max = range_x1[1]

    def update_b_range(self, b_min, b_max):
        """Method to update the dynamic debt bounds."""
        self.b_min = b_min
        self.b_max = b_max

    def generate_samples(self, num_samples):
        if num_samples <= 0:
            raise ValueError("num_samples must be greater than zero")

        samples_tensor = torch.rand(num_samples, 3, dtype=torch.float32, device=self.device)

        # Sample b (using *current* b_min and b_max)
        samples_tensor[:, 0] = samples_tensor[:, 0] * (self.b_max - self.b_min) + self.b_min
        # Sample mu
        samples_tensor[:, 1] = samples_tensor[:, 1] * (self.range_x2[1] - self.range_x2[0]) + self.range_x2[0]
        # Sample g_index
        samples_tensor[:, 2] = samples_tensor[:, 2] * (self.range_x3[1] - self.range_x3[0]) + self.range_x3[0]

        samples_tensor[samples_tensor[:, 2] <= 0.5, 2] = 0
        samples_tensor[samples_tensor[:, 2] > 0.5, 2] = 1

        return samples_tensor


class define_objective:
    """
    Calculates simulated utility and objective values.
    """

    def __init__(self, x_value_govt, x_lam_govt, x_lam_govt_old, penalty_params, device=None):
        self.device = device
        self.x_value_govt = x_value_govt.to(self.device)
        self.x_lam_govt = x_lam_govt.to(self.device)
        self.x_lam_govt_old = x_lam_govt_old.to(self.device)

        # --- Load penalties and bounds from dictionary ---
        self.l_eps = torch.tensor(penalty_params.get('l_eps', l_eps_config), device=self.device)
        self.tau_eps = torch.tensor(penalty_params.get('tau_eps', tau_eps_config), device=self.device)
        self.b_eps = torch.tensor(penalty_params.get('b_eps', b_eps_config), device=self.device)
        self.tau_min = torch.tensor(penalty_params.get('tau_min', tau_min_config), device=self.device)
        self.tau_max = torch.tensor(penalty_params.get('tau_max', tau_max_config), device=self.device)
        # --- These bounds are now DYNAMIC (passed in via penalty_params) ---
        self.b_min = torch.tensor(penalty_params.get('b_min', b_min_config), device=self.device)
        self.b_max = torch.tensor(penalty_params.get('b_max', b_max_config), device=self.device)
        # ---

    def obj_sim_value(self, x_batch, x_zagg_vec, x_pi_zagg, x_i_ind, x_n_sim, x_print):
        x_zagg_vec = torch.tensor(x_zagg_vec).to(self.device)
        x_pi_zagg = torch.tensor(x_pi_zagg).to(self.device)

        x_b1 = x_batch[:, 0].unsqueeze(1).to(self.device)
        x_mu1 = x_batch[:, 1].unsqueeze(1).to(self.device)
        x_zagg1_indices = x_batch[:, 2].unsqueeze(1).to(self.device)

        x_v0_sim = torch.zeros_like(x_b1)
        x_v0_sim_pure = torch.zeros_like(x_b1)

        x_b_size = x_b1.shape[0]
        state_sequence = torch.zeros((x_b_size, x_n_sim + 1), dtype=torch.float, device=self.device)
        state_sequence[:, 0] = x_zagg1_indices.squeeze()

        # Simulate G states
        for t in range(1, x_n_sim + 1):
            current_states = state_sequence[:, t - 1].long()
            new_states = torch.multinomial(x_pi_zagg[current_states], 1).squeeze()
            state_sequence[:, t] = new_states
        z_state = state_sequence.clone().detach().reshape((x_b_size, x_n_sim + 1)).long()

        x_bar = torch.tensor(0.0, device=self.device)
        x_bar_b_min = self.b_min
        x_bar_b_max = self.b_max
        x_bar_one = torch.tensor(1.0, device=self.device)

        x_c0_list, x_x0_list, x_b1_list = [], [], []

        for i in range(x_n_sim):
            x_b0 = x_b1
            x_mu0 = x_mu1
            x_g0 = x_zagg_vec[z_state[:, i]]

            x_c0 = 1 / x_mu0
            x_x0 = x_c0 + x_g0

            # Labor penalty
            x_l0_orig = 1 - x_x0
            x_l0 = torch.maximum(x_l0_orig, x_bar + self.l_eps)
            x_l0_punish = (1. / self.l_eps) * torch.maximum(x_bar + self.l_eps - x_l0_orig, x_bar)

            # Tau penalty
            x_tau0_orig = 1 - gamma_l * x_c0 / (1 - x_x0 + 1e-8)
            x_tau0_punish_low = (1. / self.tau_eps) * torch.maximum(self.tau_min - x_tau0_orig, x_bar)
            x_tau0_punish_high = (1. / self.tau_eps) * torch.maximum(x_tau0_orig - self.tau_max, x_bar)
            x_tau0_punish = x_tau0_punish_low + x_tau0_punish_high
            x_tau0 = torch.clamp(x_tau0_orig, self.tau_min, self.tau_max)

            # Get policy
            x_x0_govt = torch.cat([x_b0, x_mu0, x_g0], 1)
            x_gg_govt = self.x_lam_govt(x_x0_govt.float())
            x_mu11 = torch.sigmoid(x_gg_govt[:, 0].unsqueeze(1)) * (mu_max - mu_min) + mu_min
            x_mu12 = torch.sigmoid(x_gg_govt[:, 1].unsqueeze(1)) * (mu_max - mu_min) + mu_min

            x_mu1 = x_mu11.clone()
            mask = z_state[:, i] > 0
            x_mu1[mask] = x_mu12[mask]

            indices = z_state[:, i]
            x_e_mu1 = x_pi_zagg[indices, 0].reshape(-1, 1) * x_mu11 \
                      + x_pi_zagg[indices, 1].reshape(-1, 1) * x_mu12

            x_q0 = beta * x_e_mu1 / x_mu0

            # Get next period debt
            x_b1 = (x_b0 + x_g0 - x_tau0 * x_x0) / x_q0

            # Debt (b) penalty
            x_b1_orig = x_b1.clone()
            x_b1_punish_low = torch.maximum(x_bar_b_min - x_b1_orig, x_bar)
            x_b1_punish_high = torch.maximum(x_b1_orig - x_bar_b_max, x_bar)
            x_b1_punish = (1. / self.b_eps) * (x_b1_punish_low + x_b1_punish_high)
            x_b1 = torch.clamp(x_b1_orig, x_bar_b_min, x_bar_b_max)  # Clamp to *dynamic* bounds

            # Utility calculation (with penalty switch)
            x_u0 = (torch.log(x_c0) + gamma_l * torch.log(x_l0)) - (1 - x_i_ind) * (
                    x_l0_punish + x_tau0_punish + x_b1_punish)
            x_u0_pure = (torch.log(x_c0) + gamma_l * torch.log(x_l0))

            x_v0_sim = x_v0_sim + beta ** i * x_u0
            x_v0_sim_pure = x_v0_sim_pure + beta ** i * x_u0_pure

            if i == 0:
                x_c0_chk, x_x0_chk, x_b1_chk, x_tau0_chk = x_c0, x_x0, x_b1, x_tau0_orig

        # Continuation Value
        x_g1 = x_zagg_vec[z_state[:, x_n_sim]]
        x_x0_govt = torch.cat([x_b1, x_mu1, x_g1], 1).to(self.device)
        x_v0_govt = self.x_value_govt(x_x0_govt)
        x_v1_sim = x_v0_sim + beta ** x_n_sim * x_v0_govt[:, 0].unsqueeze(1)
        x_v1_sim_pure = x_v0_sim_pure + beta ** x_n_sim * x_v0_govt[:, 0].unsqueeze(1)

        x_ind_scalar = torch.zeros_like(x_b0, dtype=torch.float32, device=self.device)
        x_value_data = []

        # --- Filtering logic for Value Training (x_i_ind=1) ---
        if x_i_ind > 0:
            condition1 = (x_tau0_chk <= self.tau_min)
            condition3 = (x_b1_chk < x_bar_b_min)
            condition4 = (x_b1_chk > x_bar_b_max)
            condition5 = (x_x0_chk >= x_bar_one)
            condition6 = (x_b1_orig < x_bar_b_min)
            condition7 = (x_b1_orig > x_bar_b_max)
            condition8 = (x_tau0_chk > self.tau_max)

            combined_condition = condition1 | condition3 | condition4 | condition5 | condition6 | condition7 | condition8
            x_ind = torch.any(combined_condition, dim=1, keepdim=True).type(torch.int)

            x_ind_scalar = x_ind.view(-1, 1)
            mask = x_ind_scalar < 1
            mask = mask.squeeze()

            x_batch_filtered = x_batch[mask]
            x_v_sim_filtered = x_v0_sim[mask]

            if x_batch_filtered.shape[0] > 0:  # Avoid error on empty tensor
                x_value_data = torch.cat((x_batch_filtered.detach(), x_v_sim_filtered.detach()), dim=1)
            else:
                x_value_data = torch.empty(0, 4, device=self.device)  # Return empty tensor

            if x_print == 1:
                pass

        # This is used for simulation data history
        x_domain_data = torch.cat((x_batch.detach(), 1.0 - x_ind_scalar.detach()), dim=1)

        return torch.mean(-x_v1_sim), x_domain_data, x_value_data, torch.mean(-x_v1_sim_pure)


class equm_nn(nn.Module):
    def __init__(self, n1, n2, dropout_prob=0.0):
        super(equm_nn, self).__init__()
        self.fc1 = nn.Linear(n1, 64)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.fc3 = nn.Linear(32, n2)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout1(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class equm_trainer:
    def __init__(self, num_epochs_v, num_epochs_p, lr_v, lr_p, batch_size, n_worker, x_lam_govt, x_lam_govt_old,
                 x_value_govt, device=None, i_save=0):
        self.num_epochs_v = num_epochs_v
        self.num_epochs_p = num_epochs_p
        # --- NEW: Store Stage 2 hyperparameter ---
        self.num_epochs_p_stage2 = num_epochs_p_stage2_config
        # ---
        self.lr_v = lr_v
        self.lr_p = lr_p
        self.batch_size = batch_size
        self.n_worker = n_worker
        self.device = device
        self.i_save = i_save
        self.x_value_govt = x_value_govt.to(self.device)
        self.x_lam_govt = x_lam_govt.to(self.device)
        self.x_lam_govt_old = x_lam_govt_old.to(self.device)
        self.x_total_sample_p = int(num_samples_value * num_samples_expand_p)
        self.x_total_sample_v = int(num_samples_value * num_samples_expand_v * num_epochs_draw)

        self.domain_sampler = UniformSampler(range_x1=[b_min_config, b_max_config],
                                             range_x2=[mu_min, mu_max],
                                             range_x3=[0.0, 1.0], device=self.device)

        # --- NEW: Load refinement flag ---
        self.use_two_stage_training = use_two_stage_training_config
        if self.use_two_stage_training:
            print(f"  > Equm trainer: Two-Stage Training ENABLED (Stage 2 epochs: {self.num_epochs_p_stage2})")

    def update_graph(self, losses, title='x_title'):
        plt.clf()
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(title)
        plt.pause(0.01)

    # --- MODIFIED: Added inadmissible_samples ---
    def policy_train(self, x_data, x_zagg_vec, x_pi_zagg, x_i_ind, x_n_sim, x_print,
                     all_losses_list=None, inadmissible_samples=None):

        x_lam_govt = self.x_lam_govt

        # --- STAGE 1: GRADIENT DESCENT (Existing Logic) ---
        print("  Policy Training Stage 1: Gradient Descent...")
        # We need a define_objective. We create one using the *config* bounds
        # The dynamic bounds are passed to define_objective from the dashboard
        equm_updater = define_objective(self.x_value_govt, x_lam_govt, self.x_lam_govt_old,
                                        penalty_params=penalty_params_config,  # Use config bounds
                                        device=self.device)

        # Update sampler to use the *current* bounds from that updater
        self.domain_sampler.update_b_range(equm_updater.b_min.item(), equm_updater.b_max.item())

        optimizer_policy = torch.optim.Adam(x_lam_govt.parameters(), lr=self.lr_p)
        scheduler = ReduceLROnPlateau(optimizer_policy, mode='min', factor=lr_factor, patience=lr_patience)

        # Note: x_data is now passed in from the dashboard
        dataset = TensorDataset(x_data)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=False,
                                 num_workers=self.n_worker)

        with tqdm(total=self.num_epochs_p, desc='P. Stage 1 (Grad)', position=0, leave=False) as pbar_epoch:
            for epoch in range(self.num_epochs_p):
                epoch_loss = 0.0

                if len(data_loader) == 0:
                    print("Warning: Policy Stage 1 data_loader is empty. Skipping epoch.")
                    continue

                for batch_x, in data_loader:
                    v_sim, _, _, v_sim_pure = equm_updater.obj_sim_value(batch_x, x_zagg_vec, x_pi_zagg,
                                                                         x_i_ind, x_n_sim, x_print)
                    loss_value = v_sim
                    optimizer_policy.zero_grad()
                    loss_value.backward()
                    optimizer_policy.step()
                    epoch_loss += v_sim_pure.item()

                    nan_detected = any(torch.isnan(param).any() for param in x_lam_govt.parameters())
                    if nan_detected:
                        for layer in x_lam_govt:
                            if hasattr(layer, 'reset_parameters'):
                                layer.reset_parameters()
                        print(f"NaN detected in parameters at epoch {epoch}. Resetting the neural network.")

                pbar_epoch.update(1)
                avg_epoch_loss = epoch_loss / len(data_loader) if len(data_loader) > 0 else 0
                if all_losses_list is not None:
                    all_losses_list.append(-avg_epoch_loss)
                scheduler.step(avg_epoch_loss)

        # --- STAGE 2: DATA FITTING (New Refinement) ---
        if self.use_two_stage_training and inadmissible_samples is not None:
            print("  Policy Training Stage 2: Data Fitting (Refinement)...")
            x_lam_govt.train()
            loss_fn_mse = nn.MSELoss()

            # 1. Get (Input, Output) pairs for GOOD data from the *updated* network
            with torch.no_grad():
                g_indices = x_data[:, 2].long()
                g_values = torch.tensor(zagg_vec, device=self.device).squeeze()
                x_data_with_g_val = x_data.clone().to(self.device)
                x_data_with_g_val[:, 2] = g_values[g_indices]

                y_good_logits = x_lam_govt(x_data_with_g_val)

            good_dataset = TensorDataset(x_data_with_g_val, y_good_logits.detach())

            # 2. Create (Input, Target) pairs for BAD data
            # Target is [mu_max_logit, mu_max_logit]
            mu_max_logit = inverse_sigmoid_scaler(torch.tensor(mu_max, device=self.device), mu_min, mu_max)
            y_bad_logits = torch.full((inadmissible_samples.shape[0], 2), mu_max_logit, device=self.device)

            g_indices_bad = inadmissible_samples[:, 2].long()
            x_bad_with_g_val = inadmissible_samples.clone().to(self.device)
            x_bad_with_g_val[:, 2] = g_values[g_indices_bad]

            bad_dataset = TensorDataset(x_bad_with_g_val, y_bad_logits)

            # 3. Combine datasets
            combined_dataset = ConcatDataset([good_dataset, bad_dataset])
            data_loader_stage2 = DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=True)

            # 4. Run the refinement training loop
            # --- MODIFIED: Use the config parameter ---
            num_epochs_stage2 = self.num_epochs_p_stage2
            # ---
            with tqdm(total=num_epochs_stage2, desc='P. Stage 2 (MSE)', position=0, leave=False) as pbar_stage2:
                for epoch in range(num_epochs_stage2):
                    for x_batch, y_batch_target in data_loader_stage2:
                        y_pred_logits = x_lam_govt(x_batch)
                        loss = loss_fn_mse(y_pred_logits, y_batch_target)

                        optimizer_policy.zero_grad()
                        loss.backward()
                        optimizer_policy.step()
                    pbar_stage2.update(1)
            print("  Policy refinement complete.")

        # --- End of Stage 2 ---

        x_lam_govt.state_dict()
        self.x_lam_govt_old = copy.deepcopy(x_lam_govt)
        if self.i_save == 1:
            torch.save(x_lam_govt.state_dict(), f'models/trained_policy_nn_{model_number}.pth')

        return x_lam_govt

    def reset_on_nan(x_value_govt, x_value_govt_old):
        x_value_govt.load_state_dict(x_value_govt_old.state_dict())
        print("NaN detected in parameters. Resetting the neural network.")

    # --- MODIFIED: Added x_data, inadmissible_samples and v_threshold ---
    def value_train(self, compute_value, x_data, all_losses_list=None,
                    inadmissible_samples=None, v_threshold=None):

        x_value_govt = self.x_value_govt
        self.domain_sampler.update_b_range(compute_value.b_min.item(), compute_value.b_max.item())
        optimizer_value = torch.optim.Adam(x_value_govt.parameters(), lr=self.lr_v)
        scheduler = ReduceLROnPlateau(optimizer_value, mode='min', factor=lr_factor, patience=10)
        loss_function = nn.MSELoss()
        x_value_govt_old = copy.deepcopy(x_value_govt)
        x_i_ind, x_print = 1, 0

        all_datasets = []  # To store datasets from each draw

        with tqdm(total=self.num_epochs_v, desc='V. Training', position=0, leave=False) as pbar_epoch:
            for epoch in range(self.num_epochs_v):

                # --- MODIFIED: Generate/Combine datasets only on draw epochs ---
                if epoch % num_epochs_draw == 0:
                    all_datasets = []  # Clear dataset list for this draw

                    # Generate "good" data from simulation
                    # x_data is passed in from dashboard
                    _, _, value_data_good, _ = compute_value.obj_sim_value(x_data, zagg_vec, pi_zagg,
                                                                           x_i_ind, n_v_sim, x_print)

                    if value_data_good.shape[0] == 0:
                        print("Warning: No valid *good* value data generated. Skipping draw.")
                        all_datasets.append(TensorDataset(torch.empty(0, 3), torch.empty(0, 1)))  # Add empty
                        continue

                    good_dataset = TensorDataset(value_data_good[:, 0:3], value_data_good[:, 3].unsqueeze(1))

                    # --- NEW: Create "bad" dataset ---
                    if self.use_two_stage_training and inadmissible_samples is not None and v_threshold is not None:
                        # Create (Input, Target) pairs for BAD data
                        y_bad = torch.full((inadmissible_samples.shape[0], 1), v_threshold, device=self.device, dtype=torch.float32)

                        # Convert inadmissible_samples to include g_val
                        g_indices_bad = inadmissible_samples[:, 2].long()
                        g_values = torch.tensor(zagg_vec, device=self.device).squeeze()
                        x_bad_with_g_val = inadmissible_samples.clone().to(self.device)
                        x_bad_with_g_val[:, 2] = g_values[g_indices_bad]

                        bad_dataset = TensorDataset(x_bad_with_g_val, y_bad)

                        # Combine datasets
                        combined_dataset = ConcatDataset([good_dataset, bad_dataset])
                        all_datasets.append(combined_dataset)
                        print(
                            f"  Value training on {len(good_dataset)} good samples and {len(bad_dataset)} bad samples.")
                    else:
                        all_datasets.append(good_dataset)

                # Use the dataset from the current draw cycle
                current_dataset = all_datasets[0]  # Always use the one we just made

                data_loader = DataLoader(current_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=False,
                                         num_workers=self.n_worker)

                if len(data_loader) == 0:
                    print("Warning: DataLoader is empty. Skipping epoch.")
                    pbar_epoch.update(1)
                    continue

                epoch_loss = 0.0
                for idx, (batch_x, batch_y) in enumerate(data_loader):
                    data_x = batch_x.to(self.device)
                    data_y = batch_y.to(self.device)

                    # --- MODIFIED: Data is now pre-formatted with g_val ---
                    # The loader provides (b, mu, g_val)
                    data_fit = x_value_govt(data_x)

                    loss_value = loss_function(data_y, data_fit)
                    optimizer_value.zero_grad()
                    loss_value.backward()
                    optimizer_value.step()
                    epoch_loss += loss_value.item()

                    nan_detected = any(torch.isnan(param).any() for param in x_value_govt.parameters())
                    if nan_detected:
                        x_value_govt.load_state_dict(x_value_govt_old.state_dict())
                        print(f"NaN detected in parameters at epoch {epoch}. Resetting the neural network.")

                pbar_epoch.update(1)
                avg_epoch_loss = epoch_loss / len(data_loader)
                if all_losses_list is not None:
                    all_losses_list.append(avg_epoch_loss)
                scheduler.step(avg_epoch_loss)

        # --- End of value training loop ---

        x_value_govt.state_dict()
        x_value_govt_old = copy.deepcopy(x_value_govt)
        if self.i_save == 1:
            torch.save(x_value_govt.state_dict(), f'models/trained_value_nn_{model_number}.pth')

        return x_value_govt


class policy_equm_funcs:
    def __init__(self, x_policy, x_value, scorer=None, admissibility_threshold=None, device=None):
        self.x_policy = x_policy
        self.x_value = x_value
        # --- NEW: Store the scorer and threshold ---
        self.scorer = scorer
        self.admissibility_threshold = admissibility_threshold
        # ---
        self.device = device
        # --- MODIFIED: Load current bounds from config ---
        self.b_min = penalty_params_config.get('b_min', -0.5)
        self.b_max = penalty_params_config.get('b_max', 3.5)
        self.tau_min = penalty_params_config.get('tau_min', 0.0)
        self.tau_max = penalty_params_config.get('tau_max', 1.0)
        # ---

    def generate_data(self, n):
        g_index = 1
        g_val = zagg_vec[g_index][0]

        x1 = np.linspace(self.b_min, self.b_max, n)
        x2 = np.linspace(mu_min, mu_max, n)
        X1_m, X2_m = np.meshgrid(x1, x2)
        X3_m = np.ones_like(X1_m) * g_val

        inputs = np.column_stack([X1_m.ravel(), X2_m.ravel(), X3_m.ravel()])
        inputs = torch.from_numpy(inputs).to(torch.float32).to(self.device)

        x_b0 = inputs[:, 0].unsqueeze(1)
        x_mu0 = inputs[:, 1].unsqueeze(1)
        x_g0 = inputs[:, 2].unsqueeze(1)

        x_c0 = 1 / x_mu0
        x_x0 = x_c0 + x_g0
        x_tau0_orig = 1 - gamma_l * x_c0 / (1 - x_x0 + 1e-8)

        x_gg_govt = self.x_policy(inputs.float())
        x_mu11 = torch.sigmoid(x_gg_govt[:, 0].unsqueeze(1)) * (mu_max - mu_min) + mu_min
        x_mu12 = torch.sigmoid(x_gg_govt[:, 1].unsqueeze(1)) * (mu_max - mu_min) + mu_min

        pi_zagg_tensor = torch.tensor(pi_zagg, device=self.device)
        probs = pi_zagg_tensor[g_index, :]
        x_e_mu1 = probs[0] * x_mu11 + probs[1] * x_mu12

        x_q0 = beta * x_e_mu1 / x_mu0
        x_tau0 = torch.clamp(x_tau0_orig, self.tau_min, self.tau_max)
        x_b1 = (x_b0 + x_g0 - x_tau0 * x_x0) / x_q0

        Y_value = self.x_value(inputs)[:, 0].reshape(-1, 1)
        Y_policy = x_mu11

        # --- NEW: Filter by Admissibility IF scorer is provided ---
        Y_value_np = Y_value.cpu().detach().numpy().reshape(n, n)
        Y_policy_np = Y_policy.cpu().detach().numpy().reshape(n, n)
        Y_tau_np = x_tau0_orig.cpu().detach().numpy().reshape(n, n)
        Y_b1_np = x_b1.cpu().detach().numpy().reshape(n, n)

        if self.scorer is not None and self.admissibility_threshold is not None:
            print(f"--- Filtering plot data by admissibility (A > {self.admissibility_threshold}) ---")

            g_idx_tensor = torch.tensor(g_index, device=self.device, dtype=torch.float)

            # We must iterate over the grid
            for i in range(n):  # row
                for j in range(n):  # col
                    B_val = X1_m[i, j]
                    lam_val = X2_m[i, j]

                    B_tensor = torch.tensor(B_val, device=self.device)
                    lam_tensor = torch.tensor(lam_val, device=self.device)

                    # Calculate the score for this grid point
                    A = self.scorer.compute_score(B_tensor, lam_tensor, g_idx_tensor)

                    # If score is below threshold, set all Y values to NaN
                    if A <= self.admissibility_threshold:
                        Y_value_np[i, j] = np.nan
                        Y_policy_np[i, j] = np.nan
                        Y_tau_np[i, j] = np.nan
                        Y_b1_np[i, j] = np.nan

            admissible_count = np.sum(~np.isnan(Y_value_np))
            n_total_points = n * n
            print(f"--- Plot filtering complete: {admissible_count}/{n_total_points} points are admissible ---")

        self.X1 = X1_m
        self.X2 = X2_m
        self.X3 = X3_m
        self.Y_policy = Y_policy.cpu().detach().numpy().reshape(n, n)
        self.Y_value = Y_value.cpu().detach().numpy().reshape(n, n)
        self.Y_tau = x_tau0_orig.cpu().detach().numpy().reshape(n, n)
        self.Y_b1 = x_b1.cpu().detach().numpy().reshape(n, n)

    def create_plot(self, save_filename='figures/final_surface_plots.png', title_suffix=''):
        print(f"--- Plotting results for: {title_suffix.strip()} ---")
        print(f"Y_policy min: {self.Y_policy.min()}, max: {self.Y_policy.max()}")
        print(f"Y_value min: {self.Y_value.min()}, max: {self.Y_value.max()}")
        print(f"Y_tau min: {self.Y_tau.min()}, max: {self.Y_tau.max()}")
        print(f"Y_b1 min: {self.Y_b1.min()}, max: {self.Y_b1.max()}")

        fig = plt.figure(figsize=(12, 12))

        os.makedirs(os.path.dirname(save_filename), exist_ok=True)

        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        surf1 = ax1.plot_surface(self.X1, self.X2, self.Y_policy)
        ax1.set_xlabel('B')
        ax1.set_ylabel('Lam')
        ax1.set_zlabel('Policy')
        ax1.set_title('Policy Surface' + title_suffix)

        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        surf2 = ax2.plot_surface(self.X1, self.X2, self.Y_value)
        ax2.set_xlabel('B')
        ax2.set_ylabel('Lam')
        ax2.set_zlabel('Value')
        ax2.set_title('Value Surface' + title_suffix)

        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        surf3 = ax3.plot_surface(self.X1, self.X2, self.Y_tau)
        ax3.set_xlabel('B')
        ax3.set_ylabel('Lam')
        ax3.set_zlabel('Tau')
        ax3.set_title('Implied Tau Surface (High-G State)' + title_suffix)

        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        surf4 = ax4.plot_surface(self.X1, self.X2, self.Y_b1)
        ax4.set_xlabel('B')
        ax4.set_ylabel('Lam')
        ax4.set_zlabel('B_next')
        ax4.set_title('Implied B_next Surface (High-G State)' + title_suffix)

        plt.tight_layout()
        plt.savefig(save_filename)
        plt.close(fig)
        print(f"Plot saved to {save_filename}")