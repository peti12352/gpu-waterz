#!/usr/bin/env python3
"""R36: relative-contact γ=0.05 α=0.67 (P0t candidate, never graded)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _kruskal_probe import probe_and_batch  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gamma", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.67)
    args = p.parse_args()
    ok = probe_and_batch(
        "r36", 9, args.gamma, args.alpha,
        f"r36_g{args.gamma}_a{args.alpha}",
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
