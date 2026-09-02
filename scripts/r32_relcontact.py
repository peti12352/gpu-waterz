#!/usr/bin/env python3
"""R32: relative-contact Kruskal. Stamp LOCK only if batched PASS and B<=32."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _kruskal_probe import probe_and_batch  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gamma", type=float, default=0.02)
    p.add_argument("--alpha", type=float, default=0.67)
    args = p.parse_args()
    ok = probe_and_batch(
        "r32", 9, args.gamma, args.alpha,
        f"r32_g{args.gamma}_a{args.alpha}",
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
