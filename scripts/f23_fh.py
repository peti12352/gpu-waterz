#!/usr/bin/env python3
"""F23: Felzenszwalb–Huttenlocher MInt + hard cut w<=1-T."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _kruskal_probe import probe_and_batch  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=float, default=200)
    args = p.parse_args()
    ok = probe_and_batch("f23", 3, args.k, 0.0, f"f23_k{int(args.k)}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
