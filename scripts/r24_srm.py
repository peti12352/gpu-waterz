#!/usr/bin/env python3
"""R24: Nock-Nielsen SRM on RAG node max-incident-mean + mean>T."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _kruskal_probe import probe_and_batch  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Q", type=float, default=64)
    args = p.parse_args()
    ok = probe_and_batch("r24", 4, args.Q, 0.0, f"r24_q{int(args.Q)}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
