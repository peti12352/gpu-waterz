#!/usr/bin/env python3
"""W31: Beucher/Marcotegui waterfall on the RAG (lowest-pass union)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _kruskal_probe import probe  # noqa: E402


def main():
    ok, _, rounds, _ = probe("w31", 7, 1, 0.0, 0.0, "w31")
    raise SystemExit(0 if ok and rounds <= 30 else 1)


if __name__ == "__main__":
    main()
