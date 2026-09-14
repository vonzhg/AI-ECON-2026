"""Lecture 9 lab, Step 0 for Part B: which terminal agent can you use today?

    python3 scripts/setup_check.py          # what is installed (no network)
    python3 scripts/setup_check.py --live   # also one tiny call per harness to test your login

Run it from labs/Lec09_Agent_Lab.  Bring the output to class: teams are formed
so that each has at least one harness that passes --live.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agent_lab  # noqa: E402

ROUTES = {
    "claude": "Part B as written (TERMINAL_WALKTHROUGH.md Steps 1-8), then the Design Lab",
    "gemini": "rung-3 route: scripts/detection_matrix.py --harness gemini, then the Design Lab",
    "codex": "rung-3 route: scripts/detection_matrix.py --harness codex, then the Design Lab",
}


def version(binary: str) -> str:
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30,
                             stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return "?"
    lines = [l for l in (out.stdout + out.stderr).splitlines() if l.strip() and "arg0 temp" not in l]
    return lines[0].strip() if lines else "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="make one tiny headless call per installed harness")
    args = ap.parse_args()
    usable = []
    for name, h in agent_lab.HARNESSES.items():
        if not agent_lab.harness_available(name):
            print(f"{name:7s} not installed")
            continue
        line = f"{name:7s} {version(h.binary)}"
        if args.live:
            ok, msg = agent_lab.harness_auth_check(name)
            line += "  login OK" if ok else f"  LOGIN FAILED -- {msg.splitlines()[-1]}"
            if ok:
                usable.append(name)
        print(line)
        if name == "gemini":
            print("        note: Gemini CLI ignores project agents, hooks and workspace skills until you trust"
                  " the folder -- start `gemini` here once and accept the prompt.")
    if not args.live:
        print("\nRe-run with --live to test your logins (one short call each).")
        return 0
    print()
    if not usable:
        print("No working harness yet: fix one login above, or pair with a team that has one. "
              "Part A (the notebook) and the design canvas need no account.")
        return 1
    for name in usable:
        print(f"{name}: {ROUTES[name]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
