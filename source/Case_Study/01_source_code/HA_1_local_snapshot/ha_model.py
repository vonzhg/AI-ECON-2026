"""
Heterogeneous Agent Model Module.
Corrected version with proper transition matrix usage for log utility (σ=1).
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
        # Initialize final layer with smaller weights
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
        self.sigma = econ['sigma']
        self.gamma = econ['gamma']
        self.delta = econ['delta']
        self.pi_e = econ['pi_e']
        self.pi_u = econ['pi_u']

        # Store full transition matrix
        self.pi_mat = torch.tensor(econ['pi_matrix'], device=device, dtype=torch.float32)
        # Extract individual transition probabilities
        self.pi_ee = self.pi_mat[0, 0]  # P(e -> e)
        self.pi_eu = self.pi_mat[0, 1]  # P(e -> u)
        self.pi_ue = self.pi_mat[1, 0]  # P(u -> e)
        self.pi_uu = self.pi_mat[1, 1]  # P(u -> u)

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
        
        # Power barrier parameters for admissibility scoring
        self.power_barrier_delta = config['admissibility'].get('power_barrier_delta', 0.1)
        self.power_barrier_power = config['admissibility'].get('power_barrier_power', 2.0)

        self.actor = HANetworkFactory.create_actor(config).to(device)
        self.critic = HANetworkFactory.create_critic(config).to(device)

    def fischer_burmeister(self, a, b):
        """
        Smoothed Fischer-Burmeister function.
        Φ_ε(a, b) = a + b - sqrt(a² + b² + ε²)
        At (a≥0, b≥0, ab=0) as ε→0.
        """
        return a + b - torch.sqrt(a**2 + b**2 + self.fb_eps**2)

    def forward_physics(self, state):
        """
        Complete forward pass implementing explicit transition dynamics.
        For log utility (σ=1), uses proper transition matrix elements.
        """
        # 1. Unpack State: s = (K, a^e, a^u, c^e, c^u)
        K = state[:, 0:1]
        a_e = state[:, 1:2]
        a_u = state[:, 2:3]
        c_e = state[:, 3:4]
        c_u = state[:, 4:5]

        # 2. Get Policy from Actor: (n^e, c'^e, c'^u) = π_θ(s)
        raw_out = self.actor(state)

        if torch.isnan(raw_out).any():
            return None

        # Transform outputs to valid ranges
        n_e = torch.sigmoid(raw_out[:, 0:1]) * (self.n_max - self.n_min) + self.n_min
        c_prime_e = torch.exp(raw_out[:, 1:2]) * self.c_scale
        c_prime_u = torch.exp(raw_out[:, 2:3]) * self.c_scale

        # --- SAFETY CLAMPS ---
        # With Log utility, we must ensure c > 0 strictly.
        c_min_safe = 0.01
        c_e = torch.clamp(c_e, min=c_min_safe, max=5.0)
        c_u = torch.clamp(c_u, min=c_min_safe, max=5.0)
        n_e = torch.clamp(n_e, min=0.01, max=0.99)
        c_prime_e = torch.clamp(c_prime_e, min=c_min_safe, max=5.0)
        c_prime_u = torch.clamp(c_prime_u, min=c_min_safe, max=5.0)

        # 3. Explicit Reductions (No iteration required)
        
        # Resource Constraint: K' = F(K, n^e π^e) + (1-δ)K - π^e c^e - π^u c^u
        Y = (K ** self.alpha) * ((self.pi_e * n_e) ** (1 - self.alpha))
        I = Y + (1 - self.delta) * K - (self.pi_e * c_e) - (self.pi_u * c_u)
        K_prime = torch.clamp(I, min=0.1, max=self.K_max * 1.5)

        # After-tax wage: ŵ = (n^e)^γ (c^e)^σ
        w_hat = (n_e ** self.gamma) * (c_e ** self.sigma)

        # Bond Price Q (for log utility σ=1):
        # Q = β (c^e)^σ [ (c'^e)^{-σ} π^{ee} + (c'^u)^{-σ} π^{ue} ]
        # For σ=1: Q = β c^e [ (1/c'^e) π^{ee} + (1/c'^u) π^{ue} ]
        term_e = (c_prime_e ** (-self.sigma)) * self.pi_ee
        term_u = (c_prime_u ** (-self.sigma)) * self.pi_ue  # FIXED: use π^{ue}, not (1-π^{ee})
        Q = self.beta * (c_e ** self.sigma) * (term_e + term_u)
        Q_safe = torch.clamp(Q, min=0.01, max=20.0)

        # Asset transitions from budget constraints:
        # a'^e = (1/Q) [ (a^e π^e π^{ee} + a^u π^u π^{eu}) / π^e + ŵ n^e - c^e ]
        wealth_transfer_e = (a_e * self.pi_e * self.pi_ee + a_u * self.pi_u * self.pi_eu) / self.pi_e
        a_prime_e = (1.0 / Q_safe) * (wealth_transfer_e + w_hat * n_e - c_e)

        # a'^u = (1/Q) [ (a^e π^e π^{ue} + a^u π^u π^{uu}) / π^u - c^u ]
        wealth_transfer_u = (a_e * self.pi_e * self.pi_ue + a_u * self.pi_u * self.pi_uu) / self.pi_u
        a_prime_u = (1.0 / Q_safe) * (wealth_transfer_u - c_u)

        a_prime_e = torch.clamp(a_prime_e, min=-10.0, max=20.0)
        a_prime_u = torch.clamp(a_prime_u, min=-10.0, max=20.0)

        # 4. Euler Discrepancies (for complementarity conditions)
        # φ^e = Q (c^e)^{-σ} - β [ (c'^e)^{-σ} π^{ee} + (c'^u)^{-σ} π^{eu} ]
        rhs_e = self.beta * ((c_prime_e ** -self.sigma) * self.pi_ee + 
                            (c_prime_u ** -self.sigma) * self.pi_eu)  # FIXED: π^{eu}
        phi_e = Q_safe * (c_e ** -self.sigma) - rhs_e

        # φ^u = Q (c^u)^{-σ} - β [ (c'^e)^{-σ} π^{ue} + (c'^u)^{-σ} π^{uu} ]
        rhs_u = self.beta * ((c_prime_e ** -self.sigma) * self.pi_ue + 
                            (c_prime_u ** -self.sigma) * self.pi_uu)  # FIXED: proper transitions
        phi_u = Q_safe * (c_u ** -self.sigma) - rhs_u

        # 5. Fischer-Burmeister Residuals for complementarity: (a' ≥ 0, φ ≥ 0, φ·a' = 0)
        fb_e = self.fischer_burmeister(phi_e, a_prime_e)
        fb_u = self.fischer_burmeister(phi_u, a_prime_u)

        # 6. Current Period Welfare (Utility)
        # U(s, n^e) = π^e [u(c^e, n^e)] + π^u [u(c^u, 0)]
        if abs(self.sigma - 1.0) < 1e-4:
            # Log Utility Case (σ = 1): u(c,n) = log(c) - n^{1+γ}/(1+γ)
            u_e = torch.log(c_e) - (n_e ** (1 + self.gamma)) / (1 + self.gamma)
            u_u = torch.log(c_u)
        else:
            # CRRA Case: u(c,n) = c^{1-σ}/(1-σ) - n^{1+γ}/(1+γ)
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
                'a_prime_u': a_prime_u  # Added for complete admissibility check
            }
        }

    def power_barrier_score(self, val, v_min, v_max, delta=None, power=None):
        """
        Power barrier function S(x; [a,b], δ, p) providing smooth scores in [0,1].
        
        S(x; [a,b], δ, p) = 
            0                                if x < a - δ or x > b + δ
            ((x - (a-δ)) / δ)^p              if a - δ ≤ x < a  (smooth ramp up)
            1                                if a ≤ x ≤ b (interior)
            (((b+δ) - x) / δ)^p              if b < x ≤ b + δ (smooth ramp down)
        
        Args:
            val: Values to score
            v_min: Lower bound (a)
            v_max: Upper bound (b)
            delta: Transition width (δ), default from config
            power: Power factor (p), default from config
        
        This provides continuous values based on proximity to constraint boundaries.
        Higher power values create sharper transitions near the boundaries.
        """
        if delta is None:
            delta = self.power_barrier_delta
        if power is None:
            power = self.power_barrier_power
        
        # Ensure delta > 0
        delta = max(delta, 1e-6)
        
        # Initialize scores
        score = torch.ones_like(val)
        
        # Left boundary region: [v_min - δ, v_min)
        left_mask = (val >= v_min - delta) & (val < v_min)
        score = torch.where(left_mask, 
                           ((val - (v_min - delta)) / delta) ** power, 
                           score)
        
        # Right boundary region: (v_max, v_max + δ]
        right_mask = (val > v_max) & (val <= v_max + delta)
        score = torch.where(right_mask, 
                           ((v_max + delta - val) / delta) ** power, 
                           score)
        
        # Outside regions: score = 0
        outside_mask = (val < v_min - delta) | (val > v_max + delta)
        score = torch.where(outside_mask, torch.zeros_like(val), score)
        
        return score

    def compute_admissibility(self, physics_out):
        """
        Compute admissibility score A(s) with three components using power barrier.
        
        A(s) = w_K * A_K + w_a * A_a + w_Q * A_Q
        
        Components:
        - A_K: Capital feasibility (K' in [K_min, K_max])
        - A_a: Asset feasibility (both a'^e and a'^u in [0, K'])  
        - A_Q: Bond price feasibility (Q in [0, β])
        """
        if physics_out is None:
            return torch.zeros(1, device=self.device)

        K_p = physics_out['K_prime']
        a_pe = physics_out['a_prime_e']
        a_pu = physics_out['a_prime_u']  # Now using both asset transitions
        Q = physics_out['Q']

        # Get weights from config (default to equal weights)
        weights = self.config.get('admissibility', {})
        w_K = weights.get('w_K', 1.0 / 3.0)
        w_a = weights.get('w_a', 1.0 / 3.0)
        w_Q = weights.get('w_Q', 1.0 / 3.0)

        # A_K: Capital Feasibility
        A_K = self.power_barrier_score(K_p, self.K_min, self.K_max)

        # A_a: Asset Feasibility - check BOTH a'^e and a'^u
        # Assets should be in [0, K'] (borrowing constraint and cannot exceed capital)
        A_a_e = self.power_barrier_score(a_pe, 0.0, K_p)
        A_a_u = self.power_barrier_score(a_pu, 0.0, K_p)
        A_a = torch.minimum(A_a_e, A_a_u)  # FIXED: check both assets

        # A_Q: Bond Price Feasibility (Q in [0, β])
        A_Q = self.power_barrier_score(Q, 0.0, self.beta)

        # Weighted sum (as per document specification)
        A_total = w_K * A_K + w_a * A_a + w_Q * A_Q
        
        return A_total
