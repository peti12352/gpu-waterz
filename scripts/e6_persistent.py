#!/usr/bin/env python3
"""E6: persistent GPU ContractLayer only if locked inners <= 400."""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAMP = ROOT / "data/cache/e12_inners.txt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inners", type=int, default=0)
    args = p.parse_args()
    inners = args.inners
    if inners <= 0 and STAMP.is_file():
        inners = int(STAMP.read_text().strip().split()[0])
    if inners <= 0:
        # E5b lock: 1031+197+225+325 = 1778
        inners = 1778
    print(f"E6 locked_inners={inners}")
    if inners > 400:
        print(
            "E6 SKIP cannot hit 10 ms: "
            f"{inners} inners > 400 (1778 x 20us = 36ms even fused; "
            "launch-per-inner is worse). Persistent GPU ContractLayer "
            "is not a G6 path on the locked mean-ε."
        )
        return
    print("E6 inners<=400 — not implemented in this cycle (would be next)")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
