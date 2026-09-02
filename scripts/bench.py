#!/usr/bin/env python3
"""G6: stage times on val @ aff 0.3. Wall clock; CUDA events inside .so where present."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from segment import _as_u8, _heap, _rag, _watershed  # noqa: E402
from ref_cpu import extract_parent  # noqa: E402

AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
THR = 0.3


def once(aff_u8):
    t = {}
    t0 = time.perf_counter()
    fr = _watershed(aff_u8, 1e-4, 0.9999)
    t["ws"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    u, v, sm, ct = _rag(aff_u8, fr)
    t["rag"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    snaps = _heap(u, v, sm, ct, [THR], max_id=int(fr.max()))
    t["heap"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    extract_parent(fr, snaps[THR])
    t["extract"] = time.perf_counter() - t0
    t["total"] = t["ws"] + t["rag"] + t["heap"] + t["extract"]
    return t


def main():
    with h5py.File(AFF, "r") as f:
        aff_u8 = _as_u8(f["affinity"][:])
    nvox = int(np.prod(aff_u8.shape[1:]))
    print("warmup", flush=True)
    once(aff_u8)
    times = []
    for i in range(5):
        t = once(aff_u8)
        times.append(t)
        print(f"run{i}", {k: round(v, 4) for k, v in t.items()}, flush=True)
    totals = sorted(t["total"] for t in times)
    med = totals[2]
    gvox = nvox / med / 1e9
    print(f"nvox={nvox} min={totals[0]:.4f} median={med:.4f} max={totals[-1]:.4f} Gvox_s={gvox:.3f}")
    print(f"G6_local {'PASS' if med < 0.050 else 'FAIL'} (need median<0.050s)")
    for k in ("ws", "rag", "heap", "extract"):
        vals = sorted(t[k] for t in times)
        print(f"  {k}_median={vals[2]:.4f}")


if __name__ == "__main__":
    main()
