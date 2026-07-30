import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import json
import sys


def load_config(config_file):
    """Loads configuration from a JSON file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{config_file}'.")
        sys.exit(1)


def find_optimal_t0(b0, g0_idx, value_govt, config, device):
    """
    Solves the t=0 Ramsey problem, consistent with C++ logic.
    Finds the optimal [l0, mu_next_g0, mu_next_g1] by searching
    to maximize V = U(c0, l0) + beta * E[V_next(b', mu_next, g')].

    This function does NOT use a policy network,
    it only uses the value_govt continuation value.
    """
    # Load config params
    beta = config['beta']
    gamma_l = config['gamma_l']
    zagg_vec = config['zagg_vec']
    pi_zagg = torch.tensor(config['pi_zagg'], device=device, dtype=torch.float)
    mu_min = config['mu_min']
    mu_max = config['mu_max']
    g_vals = torch.tensor(zagg_vec, device=device, dtype=torch.float).squeeze()

    # --- MODIFIED: Load bounds from 'penalty_params' ---
    penalty_params = config.get('penalty_params', {})
    b_min = penalty_params.get('b_min', -0.5)
    b_max = penalty_params.get('b_max', 3.5)
    # ---

    g0_val = g_vals[g0_idx].item()  # Get g0 as a scalar

    # --- Define search bounds based on C++ logic ---
    l0_min_bound = g0_val + 0.01
    l0_max_bound = 0.999

    # This is the function we want to optimize.
    # We are searching for x = [l0, mu_next_g0, mu_next_g1]
    def objective(x):
        l0, mu_next_g0, mu_next_g1 = x

        # --- 0. Check constraints ---
        if (l0 <= l0_min_bound or l0 >= l0_max_bound or
                mu_next_g0 < mu_min or mu_next_g0 > mu_max or
                mu_next_g1 < mu_min or mu_next_g1 > mu_max):
            return 1e10  # Bad value

        # --- 1. Calculate Current Period Utility U(c0, l0) ---
        c0 = l0 - g0_val
        if c0 <= 0:
            return 1e10

        mu0 = 1.0 / c0
        if mu0 < mu_min or mu0 > mu_max:
            return 1e10  # Bad value

        mu0 = torch.tensor(mu0, device=device, dtype=torch.float)
        l0 = torch.tensor(l0, device=device, dtype=torch.float)
        c0 = torch.tensor(c0, device=device, dtype=torch.float)
        mu_next_g0 = torch.tensor(mu_next_g0, device=device, dtype=torch.float)
        mu_next_g1 = torch.tensor(mu_next_g1, device=device, dtype=torch.float)

        u0 = torch.log(c0) + gamma_l * torch.log(1.0 - l0)  # U(c0, 1-l0)

        # --- 2. Calculate Expected Future Value E[V(b', mu', g')] ---
        e_mu_next = pi_zagg[g0_idx, 0] * mu_next_g0 + pi_zagg[g0_idx, 1] * mu_next_g1
        q0 = beta * e_mu_next / mu0
        if q0.item() == 0:
            return 1e10

        x0 = c0 + g0_val
        tau0_orig = 1.0 - gamma_l * c0 / (1.0 - l0 + 1e-8)
        tau0 = torch.max(tau0_orig, torch.tensor(0.01, device=device))
        b_next = (b0 + g0_val - tau0 * x0) / q0

        input_v_g0 = torch.stack([b_next, mu_next_g0, g_vals[0]], dim=0)
        v_next_g0 = value_govt(input_v_g0)

        input_v_g1 = torch.stack([b_next, mu_next_g1, g_vals[1]], dim=0)
        v_next_g1 = value_govt(input_v_g1)

        e_v_next = pi_zagg[g0_idx, 0] * v_next_g0 + pi_zagg[g0_idx, 1] * v_next_g1

        # --- 3. Total Value (to be maximized) ---
        total_value = u0 + beta * e_v_next

        return -total_value.item()

    # Initial guess
    l0_init = (l0_min_bound + l0_max_bound) / 2.0
    mu_init = (mu_min + mu_max) / 2.0
    x0 = [l0_init, mu_init, mu_init]

    bounds = [(l0_min_bound, l0_max_bound), (mu_min, mu_max), (mu_min, mu_max)]
    opt_result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

    if opt_result.success:
        l0_opt, mu_next_g0_opt, mu_next_g1_opt = opt_result.x
        c0_opt = l0_opt - g0_val
        mu0_opt = 1.0 / c0_opt
        return [mu0_opt, mu_next_g0_opt, mu_next_g1_opt]
    else:
        print(f"Warning: t=0 optimization failed for b0={b0}. Using initial guess.")
        l0_init, mu_init, mu_init = x0
        c0_init = l0_init - g0_val
        mu0_init = 1.0 / c0_init
        return [mu0_init, mu_init, mu_init]


def run_simulation(b_init, g_init_idx, lam_govt, value_govt, config, device, T=100):
    """
    Runs a single simulation path for T periods.
    t=0: Solves Ramsey problem (finds optimal mu0, mu_next_g0, mu_next_g1)
    t>=1: Follows the trained lam_govt policy network
    """
    # Load config params
    beta = config['beta']
    gamma_l = config['gamma_l']
    zagg_vec = config['zagg_vec']
    pi_zagg = torch.tensor(config['pi_zagg'], device=device, dtype=torch.float)
    mu_min = config['mu_min']
    mu_max = config['mu_max']
    g_vals = torch.tensor(zagg_vec, device=device, dtype=torch.float).squeeze()

    # --- MODIFIED: Load bounds from 'penalty_params' ---
    # These will be the *final* dynamic bounds passed from the dashboard
    penalty_params = config.get('penalty_params', {})
    b_min = penalty_params.get('b_min', -0.5)
    b_max = penalty_params.get('b_max', 3.5)
    # ---

    # Data storage
    results = []

    # --- Period 0: Find optimal t=0 plan ---
    print(f"Simulating for b_init={b_init:.2f}... finding optimal t=0 plan...")
    mu0_opt, mu_next_g0_opt, mu_next_g1_opt = find_optimal_t0(
        b_init, g_init_idx, value_govt, config, device
    )

    b_t = b_init
    mu_t = mu0_opt
    g_t_idx = g_init_idx

    # --- Periods t=0 to T ---
    for t in range(T):
        g_t_val = g_vals[g_t_idx].item()

        c_t = 1.0 / mu_t
        x_t = c_t + g_t_val
        l_t = 1.0 - x_t

        if t == 0:
            mu_next_g0 = mu_next_g0_opt
            mu_next_g1 = mu_next_g1_opt
        else:
            input_t = torch.tensor([[b_t, mu_t, g_t_val]], device=device, dtype=torch.float)
            gg_govt = lam_govt(input_t.detach())
            mu_next_g0 = (torch.sigmoid(gg_govt[:, 0]) * (mu_max - mu_min) + mu_min).item()
            mu_next_g1 = (torch.sigmoid(gg_govt[:, 1]) * (mu_max - mu_min) + mu_min).item()

        e_mu_next = pi_zagg[g_t_idx, 0] * mu_next_g0 + pi_zagg[g_t_idx, 1] * mu_next_g1
        q_t = (beta * e_mu_next / mu_t)

        tau_t_orig = 1.0 - gamma_l * c_t / (l_t + 1e-8)
        tau_t = max(tau_t_orig, 0.01)

        b_next = (b_t + g_t_val - tau_t * x_t) / q_t

        # Clamp b_next using the (potentially dynamic) bounds
        b_next_clamped = torch.clamp(b_next, b_min, b_max)

        results.append({
            't': t,
            'b': b_t,
            'mu': mu_t,
            'g_idx': g_t_idx,
            'g_val': g_t_val,
            'c': c_t,
            'l': l_t,
            'tau': tau_t,
            'q': q_t.item(),
            'b_next': b_next_clamped.item()
        })

        g_next_idx = torch.multinomial(pi_zagg[g_t_idx], 1).item()
        mu_next = mu_next_g0 if g_next_idx == 0 else mu_next_g1

        b_t = b_next_clamped.item()
        mu_t = mu_next
        g_t_idx = g_next_idx

    return pd.DataFrame(results)


def plot_simulation_results(sim_dfs, title='Simulation Results'):
    """
    Plots a list of simulation dataframes.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle(title, fontsize=16)

    var_list = [('b', 'Debt (b)'), ('mu', 'Multiplier (mu)'),
                ('g_val', 'Govt. Spending (g)'), ('c', 'Consumption (c)'),
                ('l', 'Labor (l)'), ('tau', 'Tax Rate (tau)')]

    colors = plt.cm.jet(np.linspace(0, 1, len(sim_dfs)))

    for (var, label), ax in zip(var_list, axes.flatten()):
        for i, df in enumerate(sim_dfs):
            b_init = df['b'].iloc[0]
            ax.plot(df['t'], df[var], label=f'b_init = {b_init:.2f}', color=colors[i])
        ax.set_xlabel('Time (t)')
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('figures/simulation_results.png')
    plt.close(fig)
    print(f"\nSimulation plots saved to 'simulation_results.png'")


def plot_simulation_paths_b_mu(sim_dfs, config, title='Simulation State-Space Paths (b vs mu)'):
    """
    Plots the (b, mu) state-space paths from a list of simulation dataframes
    using a scatter plot to show individual time steps.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(title, fontsize=16)

    # --- MODIFIED: Load bounds from 'penalty_params' ---
    penalty_params = config.get('penalty_params', {})
    b_min = penalty_params.get('b_min', -0.5)
    b_max = penalty_params.get('b_max', 3.5)
    # ---

    mu_min = config['mu_min']
    mu_max = config['mu_max']

    colors = plt.cm.jet(np.linspace(0, 1, len(sim_dfs)))

    for i, df in enumerate(sim_dfs):
        b_init = df['b'].iloc[0]
        g_init_idx = df['g_idx'].iloc[0]
        run_label = f'b_init={b_init:.2f}, g_init={g_init_idx}'

        ax.scatter(df['b'], df['mu'],
                   label=run_label, color=colors[i],
                   alpha=0.5, s=10, zorder=2)

        ax.scatter(df['b'].iloc[0], df['mu'].iloc[0],
                   marker='o', s=100, color=colors[i],
                   edgecolors='black', zorder=5, label=f'Start (t=0)')

        ax.scatter(df['b'].iloc[-1], df['mu'].iloc[-1],
                   marker='X', s=120, color=colors[i],
                   edgecolors='black', zorder=5, label=f'End (t={len(df) - 1})')

    ax.set_xlabel('Debt (b)')
    ax.set_ylabel('Multiplier (mu)')
    ax.set_xlim(b_min, b_max)
    ax.set_ylim(mu_min, mu_max)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')

    ax.grid(True, alpha=0.5, linestyle='--')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = 'figures/simulation_paths_b_mu_scatter.png'
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Simulation (b, mu) scatter plot saved to: {save_path}")


def run_and_plot_simulations(lam_govt, value_govt, config, device, T_sim=100):
    """
    Runs simulations using the provided (trained) models and plots the results.
    """
    print("Running simulations with provided models...")

    lam_govt.eval()
    value_govt.eval()

    # --- MODIFIED: Load bounds from 'penalty_params' ---
    penalty_params = config.get('penalty_params', {})
    b_min = penalty_params.get('b_min', -0.5)
    b_max = penalty_params.get('b_max', 3.5)
    # ---

    b_starts = [b_min + 0.01, (b_min + b_max) / 4, (b_min + b_max) / 3]
    g_start_idx = 0

    simulation_data = []
    with torch.no_grad():
        for b0 in b_starts:
            df = run_simulation(b0, g_start_idx, lam_govt, value_govt, config, device, T=T_sim)
            simulation_data.append(df)

    plot_simulation_results(simulation_data, title=f'Simulations (T={T_sim}, g_start={g_start_idx})')
    plot_simulation_paths_b_mu(simulation_data, config,
                               title=f'Simulation Paths (b vs Î¼) (T={T_sim}, g_start={g_start_idx})')

    print("Simulation and plotting complete.")


if __name__ == "__main__":
    print("Running Ramsey_RA_simulation_module.py as main script...")

    CONFIG_FILE = 'config.json'
    config = load_config(CONFIG_FILE)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    n1_p = config['n1_p']
    n2_p = config['n2_p']
    n1_v = config['n1_v']
    n2_v = config['n2_v']

    n_input_p, n_output_p = 3, 2
    test_lam_govt = nn.Sequential(nn.Linear(n_input_p, n1_p),
                                  nn.ReLU(),
                                  nn.Linear(n1_p, n2_p),
                                  nn.ReLU(),
                                  nn.Linear(n2_p, n_output_p)).to(device)

    n_input_v, n_output_v = 3, 1
    test_value_govt = nn.Sequential(nn.Linear(n_input_v, n1_v),
                                    nn.ReLU(),
                                    nn.Linear(n1_v, n2_v),
                                    nn.ReLU(),
                                    nn.Linear(n2_v, n_output_v)).to(device)

    model_number = config['model_number_output']
    POLICY_MODEL_PATH = f'models/trained_policy_nn_{model_number}.pth'
    VALUE_MODEL_PATH = f'models/trained_value_nn_{model_number}.pth'

    try:
        test_lam_govt.load_state_dict(torch.load(POLICY_MODEL_PATH, map_location=device))
        test_value_govt.load_state_dict(torch.load(VALUE_MODEL_PATH, map_location=device))
        print(f"Successfully loaded models for standalone test:\n  {POLICY_MODEL_PATH}\n  {VALUE_MODEL_PATH}")
    except FileNotFoundError as e:
        print(f"Error: Model file not found. {e}")
        print("Please run the main dashboard_v10_value_gpu.py script first to train and save the models.")
        sys.exit(1)

    run_and_plot_simulations(test_lam_govt, test_value_govt, config, device, T_sim=100)