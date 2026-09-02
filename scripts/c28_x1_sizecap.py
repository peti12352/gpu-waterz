#!/usr/bin/env python3
"""C28: X1 union-all-in-band + merge-time size cap."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _kruskal_probe import probe  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smax", type=float, default=1024)
    p.add_argument("--bins", type=int, default=16)
    args = p.parse_args()
    ok, kill, rounds, _ = probe("c28", 5, args.bins, args.smax, 0.0, f"c28_b{args.bins}")
    if kill or not ok or rounds > 30:
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
