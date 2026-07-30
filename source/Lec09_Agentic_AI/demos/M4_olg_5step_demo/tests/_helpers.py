"""Test helpers — version path manipulation."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def use_version(name: str) -> None:
    """Insert versions/{name} at the front of sys.path and clear cached version modules.

    Each test should call use_version("vN") before importing model/network/train/simulate
    so the imports resolve to the right version's files.
    """
    target = str(ROOT / "versions" / name)
    # Drop any other version directory currently on the path.
    sys.path[:] = [p for p in sys.path
                   if not (p.endswith("versions/v0") or p.endswith("versions/v1")
                           or p.endswith("versions/v2") or p.endswith("versions/v3")
                           or p.endswith("versions/v4") or p.endswith("versions/v5"))]
    sys.path.insert(0, target)
    # Drop cached version-shared module names.
    for mod in ("model", "network", "train", "simulate", "plotting"):
        sys.modules.pop(mod, None)
