"""V1 smoke test — minimal training run + key gate checks."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from _helpers import use_version


class TestV1Smoke(unittest.TestCase):
    def setUp(self) -> None:
        use_version("v1")
        torch.manual_seed(0)
        np.random.seed(0)

    def test_imports_and_short_train(self) -> None:
        import train, simulate, model
        self.assertEqual(model.N, 7)
        net, losses = train.run(hp_overrides={"n_steps": 600, "pretrain_steps": 200, "log_every": 10000}, verbose=False)
        self.assertGreater(len(losses), 0)

    def test_lifecycle_outputs_correct_shape(self) -> None:
        import train, simulate, model
        net, losses = train.run(hp_overrides={"n_steps": 400, "pretrain_steps": 100, "log_every": 10000}, verbose=False)
        sim = simulate.run(net, T=400, burn=50)
        self.assertEqual(sim["c"].shape[1], model.N)            # consumption per cohort
        self.assertEqual(sim["s"].shape[1], model.N - 1)        # savings rates
        self.assertEqual(sim["a"].shape[1], model.N - 1)        # wealth state

    def test_consumption_grows_with_age(self) -> None:
        import train, simulate
        net, losses = train.run(hp_overrides={"n_steps": 1500, "pretrain_steps": 400, "log_every": 10000}, verbose=False)
        sim = simulate.run(net, T=800, burn=100)
        gate = simulate.validation_gate(sim, losses)
        self.assertTrue(gate["consumption_grows_with_age"])


if __name__ == "__main__":
    unittest.main()
