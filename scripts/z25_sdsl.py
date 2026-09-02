#!/usr/bin/env python3
"""Z25: size-dependent single linkage (step or smooth) on contact-mean."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _kruskal_probe import probe, probe_and_batch  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--s0", type=float, default=1024)
    p.add_argument("--beta", type=float, default=0.0)
    args = p.parse_args()
    pred = 2 if args.beta > 0 else 1
    ok = probe_and_batch("z25", pred, args.s0, args.beta, f"z25_s{int(args.s0)}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
