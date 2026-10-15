"""V4 smoke test — capital adjustment cost reduces to V3 at psi_K = 0."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from _helpers import use_version


class TestV4Smoke(unittest.TestCase):
    def setUp(self) -> None:
        use_version("v4")
        torch.manual_seed(0)
        np.random.seed(0)

    def test_psi_K_zero_drops_adjustment_cost(self) -> None:
        """delta_k still computed; adjustment cost zero when psi_K=0."""
        import model, network
        net = network.PolicyNet(hidden=64).to(model.device)
        z_idx, k, b = model.init_cloud(32)
        Z = model.Z_VALS[z_idx]
        s_K, b_next, p_b = net(Z, k, b)
        out = model.cohort_decisions(Z, k, b, s_K, b_next, p_b, psi_K=0.0)
        self.assertTrue(torch.allclose(out["adj_cost"], torch.zeros_like(out["adj_cost"]), atol=1e-12))

    def test_short_training_runs(self) -> None:
        import train, simulate
        net, losses = train.run(hp_overrides={"n_steps": 400, "pretrain_steps": 100, "log_every": 10000}, verbose=False)
        sim = simulate.run(net, T=400, burn=50)
        gate = simulate.validation_gate(sim, losses)
        self.assertTrue(gate["bond_market_clears"])
        self.assertIn("K_volatility", gate)

    def test_marginal_adjustment_term_in_residual(self) -> None:
        """At psi_K = 0 the V4 capital Euler residual equals V3's structurally."""
        import model, network, train
        net = network.PolicyNet(hidden=64).to(model.device)
        z_idx, k, b = model.init_cloud(16)
        R_K_default, _, _ = train.euler_residuals(z_idx, k, b, net)
        R_K_zero, _, _ = train.euler_residuals(z_idx, k, b, net, psi_K=0.0)
        # With ψ=0 the marginal-cost term equals one, so R_K should differ
        # from the default ψ=0.5 case.
        self.assertFalse(torch.allclose(R_K_default, R_K_zero, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
