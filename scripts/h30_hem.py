#!/usr/bin/env python3
"""H30: size-doubling heavy-edge matching, then stop (no residual X0 CC)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _kruskal_probe import probe  # noqa: E402


def main():
    ok, _, rounds, _ = probe("h30", 8, 1, 0.0, 0.0, "h30")
    raise SystemExit(0 if ok and rounds <= 30 else 1)


if __name__ == "__main__":
    main()
