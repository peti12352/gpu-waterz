#!/usr/bin/env python3
"""G5: two segment() runs must be byte-identical."""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from segment import segment  # noqa: E402

AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
THRS = [0.3]


def main():
    with h5py.File(AFF, "r") as f:
        aff = f["affinity"][:]
    a = segment(aff, THRS)
    b = segment(aff, THRS)
    ok = all(np.array_equal(x, y) for x, y in zip(a, b))
    print("G5", "PASS" if ok else "FAIL", "nseg", int((np.unique(a[0]) != 0).sum()))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
