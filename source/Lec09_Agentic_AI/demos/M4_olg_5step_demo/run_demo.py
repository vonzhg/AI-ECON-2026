#!/usr/bin/env python3
"""Terminal walkthrough that mirrors Section 1 of demo.ipynb.

Trains V0, simulates the trained policy, prints a compact summary, and
writes build/terminal_report_v0.json with the reproducibility record.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here / "versions" / "v0"))
    try:
        import model as v0_model
        import train as v0_train
        import simulate as v0_simulate
    except ImportError as exc:
        print(f"Could not import V0 modules: {exc}", file=sys.stderr)
        print("Run `pip install -e .` from this folder first.", file=sys.stderr)
        return 1

    print("=" * 60)
    print("V0 — three-period OLG with neural-network policy")
    print("=" * 60)
    print()
    print("Calibration:")
    for k, v in v0_model.P.items():
        print(f"  {k:>12} = {v}")
    print()

    t0 = time.perf_counter()
    net, losses = v0_train.run()
    train_secs = time.perf_counter() - t0
    print(f"Training finished in {train_secs:.1f}s   final MSE = {losses[-1]:.3e}")
    print()

    sim = v0_simulate.run(net, T=5000, burn=500)
    v0_simulate.print_summary(sim)

    gate = v0_simulate.validation_gate(sim, losses)
    print()
    print("Validation gate:")
    for k, ok in gate.items():
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {k}")
    overall = all(gate.values())
    print()
    print(f"VALIDATION: {'PASS' if overall else 'FAIL'}")

    build_dir = here / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "v0",
        "params": v0_model.P,
        "training": {
            "n_steps": len(losses),
            "final_mse": float(losses[-1]),
            "min_mse": float(min(losses)),
            "wall_time_s": train_secs,
        },
        "ergodic": {
            "E_K": float(sim["K"].mean()),
            "E_K_lo": float(sim["K"][sim["zi"] == 0].mean()),
            "E_K_hi": float(sim["K"][sim["zi"] == 1].mean()),
            "E_r": float(sim["r"].mean()),
        },
        "validation_gate": gate,
        "validation_pass": overall,
    }
    out_path = build_dir / "terminal_report_v0.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_path.relative_to(here)}")
    return 0 if overall else 2


if __name__ == "__main__":
    sys.exit(main())
