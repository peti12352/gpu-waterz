#!/usr/bin/env python3
"""S26: Soille strong-connection Kruskal, alpha=1-T, omega sweep."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _kruskal_probe import probe_and_batch  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--omega", type=float, default=0.2)
    args = p.parse_args()
    ok = probe_and_batch("s26", 6, args.omega, 0.0, f"s26_w{args.omega}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
