#!/usr/bin/env python3
"""E3cpu: RAG on waterz fragments. CPU only."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import waterz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ref_cpu import region_graph  # noqa: E402

AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
CACHE = ROOT / "data/cache"
TASK_EDGES = 7_505_458


def waterz_fragments(aff):
    for seg in waterz.agglomerate(aff, [0.0]):
        return seg.astype(np.uint32, copy=False)
    raise RuntimeError("no seg")


def main():
    CACHE.mkdir(exist_ok=True)
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:].astype(np.float32) / 255.0)
    t0 = time.time()
    fr = waterz_fragments(aff)
    print(f"fragments sec={time.time()-t0:.1f} n={(np.unique(fr)!=0).sum()}")
    np.save(CACHE / "wz_fragments.npy", fr)
    t1 = time.time()
    rg = region_graph(aff, fr)
    print(f"rag sec={time.time()-t1:.1f} edges={len(rg)}")
    n = len(rg)
    rel = abs(n - TASK_EDGES) / TASK_EDGES
    bad0 = sum(1 for u, v in rg if u == 0 or v == 0)
    # S3: recompute a sample of means from stored sum/count
    ok_mean = True
    for i, ((u, v), (s, c, m)) in enumerate(rg.items()):
        if i >= 1000:
            break
        if abs(m - s / c) > 1e-6:
            ok_mean = False
            print("mean fail", u, v, s, c, m)
            break
    print(f"rel_edge_err={rel:.6f} bg_endpoints={bad0} mean_s3={ok_mean}")
    ok = rel <= 0.001 and bad0 == 0 and ok_mean
    print("E3cpu", "PASS" if ok else "FAIL")
    # persist edges for E4
    keys = np.array(list(rg.keys()), dtype=np.uint32)
    stats = np.array([[s, c, m] for s, c, m in rg.values()], dtype=np.float64)
    np.savez(CACHE / "rag.npz", keys=keys, stats=stats)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
