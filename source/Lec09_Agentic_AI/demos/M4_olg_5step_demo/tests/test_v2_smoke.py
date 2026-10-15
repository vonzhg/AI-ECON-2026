"""V2 smoke test — Rouwenhorst sanity + 4-state TFP loop."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from _helpers import use_version


class TestV2Smoke(unittest.TestCase):
    def setUp(self) -> None:
        use_version("v2")
        torch.manual_seed(0)
        np.random.seed(0)

    def test_rouwenhorst_well_formed(self) -> None:
        import model
        Z, P_t = model.make_tfp(n_tfp=4)
        self.assertEqual(Z.shape[0], 4)
        self.assertEqual(P_t.shape, (4, 4))
        # Row sums equal one.
        sums = P_t.sum(dim=-1).cpu().numpy()
        self.assertTrue(np.allclose(sums, 1.0, atol=1e-6))
        # Symmetric grid in log space.
        log_z = np.log(Z.cpu().numpy())
        self.assertTrue(np.allclose(log_z, -log_z[::-1], atol=1e-6))

    def test_two_state_tfp_grid_works(self) -> None:
        """Reduce-to-V1 lever: making n_tfp=2 should give a valid 2-state TFP."""
        import model
        Z, P_t = model.make_tfp(n_tfp=2)
        self.assertEqual(Z.shape[0], 2)
        self.assertTrue(np.allclose(P_t.sum(dim=-1).cpu().numpy(), 1.0, atol=1e-6))

    def test_short_training_runs(self) -> None:
        import train, simulate
        net, losses = train.run(hp_overrides={"n_steps": 400, "pretrain_steps": 100, "log_every": 10000}, verbose=False)
        sim = simulate.run(net, T=400, burn=50)
        self.assertEqual(sim["c"].shape[1], 7)
        gate = simulate.validation_gate(sim, losses)
        self.assertIn("K_spread_across_TFP_states", gate)


if __name__ == "__main__":
    unittest.main()
