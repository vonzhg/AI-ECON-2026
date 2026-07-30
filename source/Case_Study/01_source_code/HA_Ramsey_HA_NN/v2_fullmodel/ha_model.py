"""
Heterogeneous Agent Model Module.

This module implements the economic model and neural network architecture
for the Ramsey optimal taxation problem with heterogeneous agents.

State Space: s = (K, a^e, a^u, c^e, c^u) ∈ R^5
    - K: Aggregate capital stock
    - a^e, a^u: Asset holdings for employed/unemployed agents
    - c^e, c^u: Consumption levels (co-state variables encoding past commitments)

Control Variables: y = (n^e, c'^e, c'^u) ∈ R^3
    - n^e: Labor supply of employed agents
    - c'^e, c'^u: Next-period consumption choices

Key Features:
    - Actor-Critic architecture for policy/value approximation
    - Fischer-Burmeister complementarity for borrowing constraints
    - Explicit (non-iterative) transition dynamics
    - Target network for stable critic training
"""

import torch
import torch.nn as nn
import numpy as np


class HANetworkFactory:
    """Factory class for creating neural networks with consistent initialization."""
    
    @staticmethod
    def init_weights(m):
        """Kaiming initialization for ReLU networks."""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def create_actor(config, input_dim=5, output_dim=3):
        """
        Create the policy network (actor).
        
        Architecture: MLP with ReLU activations
        Input: State s ∈ R^5
        Output: Raw logits for (n^e, c'^e, c'^u) - transformed via sigmoid in forward_physics
        
        NOTE: Output layer has small initialization to start near center of sigmoid,
        which gives moderate initial policy values.
        """
        h_dim = config['network_architecture']['hidden_dim']
        n_layers = config['network_architecture'].get('layers', 3)
        
        layers = [nn.Linear(input_dim, h_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(h_dim, h_dim), nn.ReLU()])
        layers.append(nn.Linear(h_dim, output_dim))
        
        net = nn.Sequential(*layers)
        net.apply(HANetworkFactory.init_weights)
        
        # Small output weights → sigmoid outputs near 0.5 → mid-range controls
        nn.init.xavier_uniform_(net[-1].weight, gain=0.01)
        return net

    @staticmethod
    def create_critic(config, input_dim=5):
        """
        Create the value network (critic).
        
        Architecture: MLP with ReLU activations
        Input: State s ∈ R^5
        Output: Scalar value V(s)
        """
        h_dim = config['network_architecture']['hidden_dim']
        n_layers = config['network_architecture'].get('layers', 3)
        
        layers = [nn.Linear(input_dim, h_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(h_dim, h_dim), nn.ReLU()])
        layers.append(nn.Linear(h_dim, 1))
        
        net = nn.Sequential(*layers)
        net.apply(HANetworkFactory.init_weights)
        return net


class HAModel(nn.Module):
    """
    Heterogeneous Agent Ramsey Model.
    
    Implements the economic model from the document:
    - Explicit transition dynamics (Eqs. 7-11)
    - Euler discrepancies (Eqs. 12-13)
    - Fischer-Burmeister residuals (Eqs. 14-15)
    - Actor-Critic networks for optimization
    
    The forward_physics method computes one step of the economic transition,
    including all equilibrium conditions needed for training.
    """
    
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        # ==================== Economic Parameters ====================
        econ = config['economic_parameters']
        self.beta = econ['beta']        # Discount factor
        self.alpha = econ['alpha']      # Capital share in production
        self.sigma = econ['sigma']      # CRRA coefficient (risk aversion)
        self.gamma = econ['gamma']      # Inverse Frisch elasticity
        self.delta = econ['delta']      # Depreciation rate
        self.pi_e = econ['pi_e']        # Stationary prob of employed
        self.pi_u = econ['pi_u']        # Stationary prob of unemployed

        # Transition matrix Π
        self.pi_mat = torch.tensor(econ['pi_matrix'], device=device, dtype=torch.float32)
        self.pi_ee = self.pi_mat[0, 0]  # Prob: E → E
        self.pi_eu = self.pi_mat[0, 1]  # Prob: E → U
        self.pi_ue = self.pi_mat[1, 0]  # Prob: U → E
        self.pi_uu = self.pi_mat[1, 1]  # Prob: U → U

        # ==================== Bounds ====================
        # Control bounds
        cb = config['control_bounds']
        self.n_min = cb['n_min']
        self.n_max = cb['n_max']

        # State bounds
        sb = config['state_bounds']
        self.K_max = sb['K_max']
        self.K_min = sb['K_min']
        self.c_min = sb['c_min']
        self.c_max = sb['c_max']
        self.a_min = sb['a_min']
        self.a_max = sb['a_max']

        # Q bounds from admissibility config
        # Economic interpretation: Q ∈ [β, 1) since Q = β × E[MRS]
        adm = config.get('admissibility', {})
        self.Q_min = adm.get('Q_min', self.beta)
        self.Q_max = adm.get('Q_max', 0.995)

        # Fischer-Burmeister smoothing parameter
        self.fb_eps = config['fischer_burmeister']['epsilon']

        # ==================== Neural Networks ====================
        self.actor = HANetworkFactory.create_actor(config).to(device)
        self.critic = HANetworkFactory.create_critic(config).to(device)
        
        # Target critic for stable TD learning
        # Updated via soft update: θ_target = τ*θ + (1-τ)*θ_target
        self.critic_target = HANetworkFactory.create_critic(config).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False  # Frozen - updated manually

    def soft_update_critic_target(self, tau=0.005):
        """
        Soft update target network weights.
        
        This provides stable bootstrap targets for TD learning.
        Standard practice in DDPG, SAC, etc.
        
        Args:
            tau: Interpolation coefficient (small = slow update)
        """
        for param, target_param in zip(self.critic.parameters(), 
                                        self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def hard_update_critic_target(self):
        """Hard copy critic weights to target."""
        self.critic_target.load_state_dict(self.critic.state_dict())

    def fischer_burmeister(self, a, b):
        """
        Smoothed Fischer-Burmeister function for complementarity conditions.
        
        Φ_ε(a,b) = a + b - √(a² + b² + ε²)
        
        Property: Φ_ε(a,b) = 0 ⟺ a ≥ 0, b ≥ 0, a·b = 0 (as ε → 0)
        
        Used to encode the KKT conditions for borrowing constraints:
        - a' ≥ 0 (non-negative assets)
        - φ ≥ 0 (Euler discrepancy)  
        - a' · φ = 0 (complementary slackness)
        
        Args:
            a: First variable (asset a')
            b: Second variable (Euler discrepancy φ)
            
        Returns:
            FB residual (should be 0 at optimum)
        """
        return a + b - torch.sqrt(a**2 + b**2 + self.fb_eps**2)

    def forward_physics(self, state):
        """
        Complete forward pass implementing explicit transition dynamics.
        
        This is the CORE METHOD that implements the economic model.
        Given current state s and policy π_θ(s), compute:
        1. Control variables (n^e, c'^e, c'^u)
        2. Equilibrium prices (Q, ŵ)
        3. Next-period state s'
        4. Euler discrepancies and FB residuals
        5. Period welfare
        
        Input:
            state: Tensor (Batch, 5) = (K, a^e, a^u, c^e, c^u)
        
        Output:
            Dictionary with all computed quantities, or None if invalid
            
        IMPORTANT: This method does NOT check admissibility of s'.
        The caller (training loop) is responsible for:
        - Checking if s' ∈ S_α
        - Projecting s' back if needed
        - Adding appropriate penalties
        """
        # ==================== Unpack State ====================
        K = state[:, 0:1]       # Aggregate capital
        a_e = state[:, 1:2]     # Assets (employed)
        a_u = state[:, 2:3]     # Assets (unemployed)
        c_e = state[:, 3:4]     # Consumption (employed) - STATE variable
        c_u = state[:, 4:5]     # Consumption (unemployed) - STATE variable

        # ==================== Actor Policy ====================
        raw_out = self.actor(state)
        if torch.isnan(raw_out).any():
            return None

        # Transform raw logits to bounded controls via sigmoid
        # This GUARANTEES controls are within bounds
        
        # Labor supply: n^e ∈ [n_min, n_max]
        n_e = torch.sigmoid(raw_out[:, 0:1]) * (self.n_max - self.n_min) + self.n_min

        # Future consumption: c'^e ∈ [c_min, c_max]
        c_prime_e = torch.sigmoid(raw_out[:, 1:2]) * (self.c_max - self.c_min) + self.c_min

        # Future consumption: c'^u ∈ [c_min, c_max]
        c_prime_u = torch.sigmoid(raw_out[:, 2:3]) * (self.c_max - self.c_min) + self.c_min

        # ==================== 1. Explicit Reductions (Document Eqs. 7-11) ====================
        
        # --- Capital Transition (Resource Constraint, Eq. 7) ---
        # K' = F(K, N) + (1-δ)K - π^e·c^e - π^u·c^u
        # where N = π^e · n^e (aggregate labor)
        Y = (K ** self.alpha) * ((self.pi_e * n_e) ** (1 - self.alpha))
        I = Y + (1 - self.delta) * K - (self.pi_e * c_e) - (self.pi_u * c_u)
        K_prime = torch.clamp(I, min=self.K_min * 0.5, max=self.K_max * 1.5)

        # --- After-Tax Wage (Intratemporal FOC, Eq. 8) ---
        # ŵ = (n^e)^γ · (c^e)^σ
        w_hat = (n_e ** self.gamma) * (c_e ** self.sigma)

        # --- Bond Price (Euler Equation, Eq. 9) ---
        # Q = β · (c^e)^σ · [π^{ee}/(c'^e)^σ + π^{eu}/(c'^u)^σ]
        # Note: This is derived from the employed agent's Euler equation
        term_e = (c_prime_e ** (-self.sigma)) * self.pi_ee
        term_u = (c_prime_u ** (-self.sigma)) * self.pi_eu
        Q = self.beta * (c_e ** self.sigma) * (term_e + term_u)
        
        # Clamp Q to economic bounds
        Q_safe = torch.clamp(Q, min=self.Q_min, max=self.Q_max)

        # --- Asset Transitions (Budget Constraints, Eqs. 10-11) ---
        # a'^e = (1/Q) · [wealth_transfer_e + ŵ·n^e - c^e]
        # wealth_transfer_e: expected assets for agents who are EMPLOYED next period
        #   = aᵉ × Pr(E→E) + aᵘ × Pr(U→E) = aᵉ·π^{ee} + aᵘ·π^{ue}
        wealth_transfer_e = (a_e * self.pi_e * self.pi_ee + a_u * self.pi_u * self.pi_ue) / self.pi_e
        a_prime_e_raw = (1.0 / Q_safe) * (wealth_transfer_e + w_hat * n_e - c_e)

        # a'^u = (1/Q) · [wealth_transfer_u - c^u]
        # wealth_transfer_u: expected assets for agents who are UNEMPLOYED next period
        #   = aᵉ × Pr(E→U) + aᵘ × Pr(U→U) = aᵉ·π^{eu} + aᵘ·π^{uu}
        wealth_transfer_u = (a_e * self.pi_e * self.pi_eu + a_u * self.pi_u * self.pi_uu) / self.pi_u
        a_prime_u_raw = (1.0 / Q_safe) * (wealth_transfer_u - c_u)

        # Clamp assets for numerical stability in next iteration
        # NOTE: We return BOTH raw and clamped values:
        # - Raw: for penalty calculation (true violation magnitude)
        # - Clamped: for state transition (prevents NaN explosion)
        a_prime_e = torch.clamp(a_prime_e_raw, min=-5.0, max=self.a_max * 2)
        a_prime_u = torch.clamp(a_prime_u_raw, min=-5.0, max=self.a_max * 2)

        # ==================== 2. Euler Discrepancies (Document Eqs. 12-13) ====================
        # These measure violation of the Euler equation, NORMALIZED by marginal utility
        # At optimum with non-binding constraint: φ = 0
        # With binding constraint (a' = 0): φ > 0 (marginal utility of saving < cost)

        # Original (unnormalized):
        #   φ^e_raw = Q·(c^e)^{-σ} - β·[π^{ee}·(c'^e)^{-σ} + π^{eu}·(c'^u)^{-σ}]
        # Normalized by (c^e)^{-σ}:
        #   φ^e = Q - β·(c^e)^σ·[π^{ee}·(c'^e)^{-σ} + π^{eu}·(c'^u)^{-σ}]
        #       = Q - β·[π^{ee}·(c^e/c'^e)^σ + π^{eu}·(c^e/c'^u)^σ]

        # Employed agent's normalized Euler discrepancy
        rhs_e = self.beta * ((c_e / c_prime_e) ** self.sigma * self.pi_ee +
                            (c_e / c_prime_u) ** self.sigma * self.pi_eu)
        phi_e = Q_safe - rhs_e

        # Original (unnormalized):
        #   φ^u_raw = Q·(c^u)^{-σ} - β·[π^{ue}·(c'^e)^{-σ} + π^{uu}·(c'^u)^{-σ}]
        # Normalized by (c^u)^{-σ}:
        #   φ^u = Q - β·(c^u)^σ·[π^{ue}·(c'^e)^{-σ} + π^{uu}·(c'^u)^{-σ}]
        #       = Q - β·[π^{ue}·(c^u/c'^e)^σ + π^{uu}·(c^u/c'^u)^σ]

        # Unemployed agent's normalized Euler discrepancy
        rhs_u = self.beta * ((c_u / c_prime_e) ** self.sigma * self.pi_ue +
                            (c_u / c_prime_u) ** self.sigma * self.pi_uu)
        phi_u = Q_safe - rhs_u

        # ==================== 3. Fischer-Burmeister Residuals ====================
        # Φ^i = FB(φ^i, a'^i) should be 0 at equilibrium
        fb_e = self.fischer_burmeister(phi_e, a_prime_e)
        fb_u = self.fischer_burmeister(phi_u, a_prime_u)

        # ==================== 4. Period Welfare ====================
        # u(c,n) = c^{1-σ}/(1-σ) - n^{1+γ}/(1+γ)
        # Aggregate: U = π^e · u(c^e, n^e) + π^u · u(c^u, 0)
        u_e = (c_e ** (1 - self.sigma)) / (1 - self.sigma) - \
              (n_e ** (1 + self.gamma)) / (1 + self.gamma)
        u_u = (c_u ** (1 - self.sigma)) / (1 - self.sigma)
        welfare = self.pi_e * u_e + self.pi_u * u_u

        if torch.isnan(welfare).any():
            return None

        # ==================== 5. Next State ====================
        # s' = (K', a'^e, a'^u, c'^e, c'^u)
        # NOTE: c' becomes the next period's state c
        next_state = torch.cat([K_prime, a_prime_e, a_prime_u, c_prime_e, c_prime_u], dim=1)

        return {
            'next_state': next_state,
            'welfare': welfare,
            'fb_residuals': (fb_e, fb_u),
            'controls': {
                'n_e': n_e,
                'c_prime_e': c_prime_e,
                'c_prime_u': c_prime_u
            },
            'physics': {
                'Q': Q_safe,
                'Q_raw': Q,  # Pre-clamped for diagnostics
                'w_hat': w_hat,
                'K_prime': K_prime,
                'next_state': next_state,
                # Raw assets for penalty calculation
                'a_prime_e_raw': a_prime_e_raw,
                'a_prime_u_raw': a_prime_u_raw,
                # Euler discrepancies for debugging
                'phi_e': phi_e,
                'phi_u': phi_u
            }
        }

    def barrier_score(self, val, v_min, v_max, delta):
        """
        Smooth barrier function B(x) ∈ [0,1].

        Returns:
        - 1.0 if val ∈ [v_min, v_max] (interior)
        - Smooth decay to 0 in buffer zones [v_min-δ, v_min] and [v_max, v_max+δ]
        - 0.0 if val outside buffer zones

        Used for admissibility scoring.

        Args:
            val: Values to score
            v_min, v_max: Valid range
            delta: Buffer zone width
        """
        score = torch.ones_like(val)

        # Left barrier
        left_mask = (val >= v_min - delta) & (val < v_min)
        score = torch.where(left_mask, ((val - (v_min - delta)) / delta) ** 2, score)

        # Right barrier
        right_mask = (val > v_max) & (val <= v_max + delta)
        score = torch.where(right_mask, ((v_max + delta - val) / delta) ** 2, score)

        # Outside
        outside_mask = (val < v_min - delta) | (val > v_max + delta)
        score = torch.where(outside_mask, torch.zeros_like(val), score)

        return score

    def compute_admissibility(self, physics_out, boundary=None):
        """
        Compute global admissibility score A(s) ∈ [0, 1].

        A(s) = w_geo · A_geo(s') + w_Q · A_Q(Q)

        Components:
        - A_geo: Is next state s' inside the learned α-shape?
        - A_Q: Is bond price Q within economic bounds?

        Args:
            physics_out: Output dict from forward_physics
            boundary: AlphaBoundary object (optional)

        Returns:
            Tensor (Batch, 1) of scores in [0, 1]
        """
        if physics_out is None:
            return torch.zeros(1, device=self.device)

        adm_conf = self.config.get('admissibility', {})
        w_geo = adm_conf.get('w_geo', 0.5)
        w_Q = adm_conf.get('w_Q', 0.5)
        delta_Q = adm_conf.get('barrier_delta', 0.05)

        # 1. Bond Price Feasibility (A_Q)
        Q = physics_out['Q']
        A_Q = self.barrier_score(Q, self.Q_min, self.Q_max, delta_Q)

        # 2. Geometric Feasibility (A_geo)
        next_state = physics_out['next_state']

        if boundary is not None and boundary.delaunay is not None:
            is_inside = boundary.is_admissible(next_state).float()
            A_geo = is_inside
        else:
            # Fallback: simple box constraint on K'
            K_p = physics_out['K_prime']
            A_geo = self.barrier_score(K_p, self.K_min, self.K_max, 0.1)

        return w_geo * A_geo + w_Q * A_Q