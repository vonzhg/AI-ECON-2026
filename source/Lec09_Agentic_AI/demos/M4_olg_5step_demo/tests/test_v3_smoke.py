"""V3 smoke test — bonds, market clearing, Fischer–Burmeister."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from _helpers import use_version


class TestV3Smoke(unittest.TestCase):
    def setUp(self) -> None:
        use_version("v3")
        torch.manual_seed(0)
        np.random.seed(0)

    def test_market_clearing_layer(self) -> None:
        """Bond holdings produced by the network must sum to zero."""
        import model, network
        net = network.PolicyNet(hidden=64).to(model.device)
        z_idx, k, b = model.init_cloud(64)
        Z = model.Z_VALS[z_idx]
        s_K, b_next, p_b = net(Z, k, b)
        s = b_next.sum(dim=-1).abs().max().item()
        self.assertLess(s, 1e-5)

    def test_fischer_burmeister_definition(self) -> None:
        """FB(x, y) = x + y - sqrt(x² + y²); zero iff x≥0, y≥0, x·y=0."""
        import model
        x = torch.tensor([0.0, 0.0, 1.0, 2.0, -1.0])
        y = torch.tensor([1.0, 0.0, 0.0, 3.0, 0.0])
        fb = model.fb_residual(x, y).cpu().numpy()
        # First three: complementarity satisfied → FB ≈ 0.
        self.assertLess(np.abs(fb[0]), 1e-4)
        self.assertLess(np.abs(fb[1]), 1e-4)
        self.assertLess(np.abs(fb[2]), 1e-4)
        # Both positive: FB > 0 (a+b > sqrt(a²+b²)).
        self.assertGreater(fb[3], 1e-3)
        # x = -1, y = 0: x is negative ⇒ FB < 0.
        self.assertLess(fb[4], -1e-3)

    def test_short_training_runs(self) -> None:
        import train, simulate
        net, losses = train.run(hp_overrides={"n_steps": 400, "pretrain_steps": 100, "log_every": 10000}, verbose=False)
        sim = simulate.run(net, T=400, burn=50)
        gate = simulate.validation_gate(sim, losses)
        self.assertTrue(gate["bond_market_clears"])

    def test_bonds_off_lever(self) -> None:
        """With bonds_off=True, the bond budget term is zero."""
        import model, network, train
        net = network.PolicyNet(hidden=64).to(model.device)
        z_idx, k, b = model.init_cloud(32)
        # Forward with bonds_off=True
        Z = model.Z_VALS[z_idx]
        s_K, b_next, p_b = net(Z, k, b)
        out = model.cohort_decisions(Z, k, b, s_K, b_next, p_b, bonds_off=True)
        self.assertTrue(torch.equal(out["b_next"], torch.zeros_like(out["b_next"])))
        self.assertTrue(torch.equal(out["bond_cost"], torch.zeros_like(out["bond_cost"])))


if __name__ == "__main__":
    unittest.main()
