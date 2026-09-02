#!/usr/bin/env python3
"""A27: UF mean>T AND area>=a."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _kruskal_probe import probe_and_batch  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--area", type=int, default=4)
    args = p.parse_args()
    ok = probe_and_batch("a27", 0, float(args.area), 0.0, f"a27_a{args.area}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
