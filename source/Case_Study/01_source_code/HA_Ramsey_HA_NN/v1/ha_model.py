"""
Heterogeneous Agent Model Module.
Revised:
- Implements General CRRA utility (handles sigma=1 and sigma!=1).
- Admissibility check for Bond Price Q constrained to [beta, 1].
"""

import torch
import torch.nn as nn
import numpy as np

class HANetworkFactory:
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def create_actor(config, input_dim=5, output_dim=3):
        h_dim = config['network_architecture']['hidden_dim']
        net = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, output_dim)
        )
        net.apply(HANetworkFactory.init_weights)
        nn.init.xavier_uniform_(net[-1].weight, gain=0.01)
        return net

    @staticmethod
    def create_critic(config, input_dim=5):
        h_dim = config['network_architecture']['hidden_dim']
        net = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1)
        )
        net.apply(HANetworkFactory.init_weights)
        return net

class HAModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        econ = config['economic_parameters']
        self.beta = econ['beta']
        self.alpha = econ['alpha']
        self.sigma = econ['sigma']  # CRRA parameter from JSON
        self.gamma = econ['gamma']
        self.delta = econ['delta']
        self.pi_e = econ['pi_e']
        self.pi_u = econ['pi_u']

        # Store full transition matrix
        self.pi_mat = torch.tensor(econ['pi_matrix'], device=device, dtype=torch.float32)
        self.pi_ee = self.pi_mat[0, 0]
        self.pi_eu = self.pi_mat[0, 1]
        self.pi_ue = self.pi_mat[1, 0]
        self.pi_uu = self.pi_mat[1, 1]

        cb = config['control_bounds']
        self.n_min = cb['n_min']
        self.n_max = cb['n_max']
        self.c_scale = cb['c_scale']

        sb = config['state_bounds']
        self.K_max = sb['K_max']
        self.K_min = sb['K_min']
        self.a_min = sb['a_min']
        self.a_max = sb['a_max']

        self.fb_eps = config['fischer_burmeister']['epsilon']

        self.power_barrier_delta = config['admissibility'].get('power_barrier_delta', 0.1)
        self.power_barrier_power = config['admissibility'].get('power_barrier_power', 2.0)

        self.actor = HANetworkFactory.create_actor(config).to(device)
        self.critic = HANetworkFactory.create_critic(config).to(device)

    def fischer_burmeister(self, a, b):
        return a + b - torch.sqrt(a**2 + b**2 + self.fb_eps**2)

    def forward_physics(self, state):
        """
        Complete forward pass implementing explicit transition dynamics.
        """
        # 1. Unpack State: s = (K, a^e, a^u, c^e, c^u)
        K = state[:, 0:1]
        a_e = state[:, 1:2]
        a_u = state[:, 2:3]
        c_e = state[:, 3:4]
        c_u = state[:, 4:5]

        # 2. Get Policy from Actor
        raw_out = self.actor(state)

        if torch.isnan(raw_out).any():
            return None

        n_e = torch.sigmoid(raw_out[:, 0:1]) * (self.n_max - self.n_min) + self.n_min
        c_prime_e = torch.exp(raw_out[:, 1:2]) * self.c_scale
        c_prime_u = torch.exp(raw_out[:, 2:3]) * self.c_scale

        # --- SAFETY CLAMPS ---
        c_min_safe = 0.01
        c_e = torch.clamp(c_e, min=c_min_safe, max=5.0)
        c_u = torch.clamp(c_u, min=c_min_safe, max=5.0)
        n_e = torch.clamp(n_e, min=0.01, max=0.99)
        c_prime_e = torch.clamp(c_prime_e, min=c_min_safe, max=5.0)
        c_prime_u = torch.clamp(c_prime_u, min=c_min_safe, max=5.0)

        # 3. Explicit Reductions
        Y = (K ** self.alpha) * ((self.pi_e * n_e) ** (1 - self.alpha))
        I = Y + (1 - self.delta) * K - (self.pi_e * c_e) - (self.pi_u * c_u)
        K_prime = torch.clamp(I, min=0.1, max=self.K_max * 1.5)

        w_hat = (n_e ** self.gamma) * (c_e ** self.sigma)

        # Bond Price Q
        term_e = (c_prime_e ** (-self.sigma)) * self.pi_ee
        term_u = (c_prime_u ** (-self.sigma)) * self.pi_eu
        Q = self.beta * (c_e ** self.sigma) * (term_e + term_u)
        Q_safe = torch.clamp(Q, min=0.01, max=20.0)

        # Asset transitions
        wealth_transfer_e = (a_e * self.pi_e * self.pi_ee + a_u * self.pi_u * self.pi_eu) / self.pi_e
        a_prime_e = (1.0 / Q_safe) * (wealth_transfer_e + w_hat * n_e - c_e)

        wealth_transfer_u = (a_e * self.pi_e * self.pi_ue + a_u * self.pi_u * self.pi_uu) / self.pi_u
        a_prime_u = (1.0 / Q_safe) * (wealth_transfer_u - c_u)

        a_prime_e = torch.clamp(a_prime_e, min=-10.0, max=20.0)
        a_prime_u = torch.clamp(a_prime_u, min=-10.0, max=20.0)

        # 4. Euler Discrepancies
        # Note: term_e and term_u used in Q already contain the correct probability weighting
        rhs_e = self.beta * ((c_prime_e ** -self.sigma) * self.pi_ee +
                            (c_prime_u ** -self.sigma) * self.pi_eu)
        phi_e = Q_safe * (c_e ** -self.sigma) - rhs_e

        rhs_u = self.beta * ((c_prime_e ** -self.sigma) * self.pi_ue +
                            (c_prime_u ** -self.sigma) * self.pi_uu)
        phi_u = Q_safe * (c_u ** -self.sigma) - rhs_u

        # 5. FB Residuals
        fb_e = self.fischer_burmeister(phi_e, a_prime_e)
        fb_u = self.fischer_burmeister(phi_u, a_prime_u)

        # 6. Welfare (General CRRA)
        # Handles both Log (sigma=1) and CRRA (sigma!=1)
        if abs(self.sigma - 1.0) < 1e-4:
            u_e = torch.log(c_e) - (n_e ** (1 + self.gamma)) / (1 + self.gamma)
            u_u = torch.log(c_u)
        else:
            u_e = (c_e ** (1 - self.sigma)) / (1 - self.sigma) - (n_e ** (1 + self.gamma)) / (1 + self.gamma)
            u_u = (c_u ** (1 - self.sigma)) / (1 - self.sigma)

        welfare = self.pi_e * u_e + self.pi_u * u_u

        if torch.isnan(welfare).any():
            return None

        next_state = torch.cat([K_prime, a_prime_e, a_prime_u, c_prime_e, c_prime_u], dim=1)

        return {
            'next_state': next_state,
            'welfare': welfare,
            'fb_residuals': (fb_e, fb_u),
            'physics': {
                'Q': Q_safe,
                'K_prime': K_prime,
                'a_prime_e': a_prime_e,
                'a_prime_u': a_prime_u,
                'next_state': next_state
            }
        }

    def power_barrier_score(self, val, v_min, v_max, delta=None, power=None):
        if delta is None: delta = self.power_barrier_delta
        if power is None: power = self.power_barrier_power
        delta = max(delta, 1e-6)

        score = torch.ones_like(val)

        left_mask = (val >= v_min - delta) & (val < v_min)
        score = torch.where(left_mask, ((val - (v_min - delta)) / delta) ** power, score)

        right_mask = (val > v_max) & (val <= v_max + delta)
        score = torch.where(right_mask, ((v_max + delta - val) / delta) ** power, score)

        outside_mask = (val < v_min - delta) | (val > v_max + delta)
        score = torch.where(outside_mask, torch.zeros_like(val), score)
        return score

    def compute_admissibility(self, physics_out, boundary=None):
        """
        Compute Admissibility Score.
        """
        if physics_out is None:
            return torch.zeros(1, device=self.device)

        # 1. Bond Price Feasibility (A_Q)
        # REVISED: Q must be in [beta, 1.0]
        Q = physics_out['Q']
        A_Q = self.power_barrier_score(Q, self.beta, 1.0)

        # 2. Geometric Feasibility (A_geo)
        next_state = physics_out['next_state']

        if boundary is not None and boundary.delaunay is not None:
            # Check if next_state is inside the current alpha-shape
            is_inside = boundary.is_admissible(next_state).float()
            A_geo = is_inside
        else:
            # Fallback (Prior): Box bounds on Capital only
            K_p = physics_out['K_prime']
            A_geo = self.power_barrier_score(K_p, self.K_min, self.K_max)

        # Weighted Sum
        weights = self.config.get('admissibility', {})
        w_geo = weights.get('w_geo', 0.5)
        w_Q = weights.get('w_Q', 0.5)

        return w_geo * A_geo + w_Q * A_Q