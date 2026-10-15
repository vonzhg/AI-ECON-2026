#!/usr/bin/env python3
"""Thin CLI wrapper that runs the V0 baseline.

Usage:
    python3 run.py            # train V0, simulate, print summary
    python3 run.py --quick    # smaller training budget for smoke tests

For the full multi-version walkthrough, open demo.ipynb instead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V0 baseline.")
    parser.add_argument("--quick", action="store_true",
                        help="Use a smaller training budget for smoke tests.")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here / "versions" / "v0"))
    try:
        import train as v0_train
        import simulate as v0_simulate
    except ImportError as exc:
        print(f"Could not import V0 modules: {exc}", file=sys.stderr)
        print("Run `pip install -e .` from this folder first.", file=sys.stderr)
        return 1

    hp_overrides = {}
    if args.quick:
        hp_overrides.update(n_steps=400, pretrain_steps=100, log_every=100)

    net, losses = v0_train.run(hp_overrides=hp_overrides)
    sim = v0_simulate.run(net, T=1500 if args.quick else 5000, burn=300 if args.quick else 500)
    v0_simulate.print_summary(sim)
    return 0


if __name__ == "__main__":
    sys.exit(main())
