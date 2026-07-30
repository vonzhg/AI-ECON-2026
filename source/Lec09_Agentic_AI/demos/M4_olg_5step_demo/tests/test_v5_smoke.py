"""V5 smoke test — homotopy schedule produces finite per-phase snapshots."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from _helpers import use_version


class TestV5Smoke(unittest.TestCase):
    def setUp(self) -> None:
        use_version("v5")
        torch.manual_seed(0)
        np.random.seed(0)

    def test_phases_run_in_order_and_finite(self) -> None:
        import train
        # Tiny per-phase budgets.
        hp = dict(phase1_steps=200, phase2_steps=150, phase3_steps=150, phase4_steps=150,
                  pretrain_steps=100, log_every=10000)
        net, history = train.run(hp_overrides=hp, verbose=False)
        names = [p["name"] for p in history["phases"]]
        self.assertEqual(names,
                         ["capital_only", "bond_pretraining", "bond_homotopy", "fine_tuning"])
        self.assertTrue(all(np.isfinite(row).all() for row in history["all"]))

    def test_validation_gate_returns_expected_keys(self) -> None:
        import train, simulate
        hp = dict(phase1_steps=200, phase2_steps=150, phase3_steps=150, phase4_steps=150,
                  pretrain_steps=100, log_every=10000)
        net, history = train.run(hp_overrides=hp, verbose=False)
        sim = simulate.run(net, T=500, burn=50)
        gate = simulate.validation_gate(sim, history)
        for key in ["training_progressed", "all_residual_snapshots_finite",
                    "rms_total_loss_<_8pct", "bond_market_clears",
                    "rk_phase4_end_<_half_phase1_start"]:
            self.assertIn(key, gate)
        self.assertTrue(gate["all_residual_snapshots_finite"])
        self.assertTrue(gate["bond_market_clears"])


if __name__ == "__main__":
    unittest.main()
