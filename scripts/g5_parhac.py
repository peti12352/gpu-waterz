#!/usr/bin/env python3
"""G5r: two ParHAC runs @ 0.3 array_equal."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from segment import _parhac  # noqa: E402
from ref_cpu import extract_parent  # noqa: E402

CACHE = ROOT / "data/cache"


def load_rag():
    z = np.load(CACHE / "rag.npz")
    u = np.ascontiguousarray(z["keys"][:, 0], dtype=np.uint32)
    v = np.ascontiguousarray(z["keys"][:, 1], dtype=np.uint32)
    sm = np.ascontiguousarray(z["stats"][:, 0], dtype=np.float64)
    ct = np.ascontiguousarray(np.rint(z["stats"][:, 1]), dtype=np.int64)
    return u, v, sm, ct


def main():
    u, v, sm, ct = load_rag()
    fr = np.load(CACHE / "wz_fragments.npy")
    max_id = int(max(int(u.max()), int(v.max()), int(fr.max())))
    a = _parhac(u, v, sm, ct, [0.3], max_id)[0.3]
    b = _parhac(u, v, sm, ct, [0.3], max_id)[0.3]
    la = extract_parent(fr, a)
    lb = extract_parent(fr, b)
    eq = bool(np.array_equal(la, lb))
    nseg = int((np.unique(la) != 0).sum())
    print(f"G5r array_equal={eq} nseg={nseg}")
    print("G5r", "PASS" if eq else "FAIL")
    if not eq:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
