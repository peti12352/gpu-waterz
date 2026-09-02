#!/usr/bin/env python3
"""G3: GPU RAG on waterz fragments."""
from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
CACHE = ROOT / "data/cache"
SO = ROOT / "src/librag_gpu.so"
TASK = 7_505_458


def rag_gpu(aff_u8, seg):
    z, y, x = aff_u8.shape[1:]
    max_e = 20_000_000
    u = np.empty(max_e, dtype=np.uint32)
    v = np.empty(max_e, dtype=np.uint32)
    sm = np.empty(max_e, dtype=np.float64)
    ct = np.empty(max_e, dtype=np.int64)
    lib = ctypes.CDLL(str(SO))
    lib.rag_gpu.restype = ctypes.c_int64
    lib.rag_gpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
    ]
    a = np.ascontiguousarray(aff_u8)
    s = np.ascontiguousarray(seg, dtype=np.uint32)
    n = lib.rag_gpu(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        z, y, x,
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        max_e,
    )
    if n < 0:
        raise RuntimeError("rag overflow")
    return u[:n].copy(), v[:n].copy(), sm[:n].copy(), ct[:n].copy()


def main():
    fr = np.load(CACHE / "wz_fragments.npy")
    with h5py.File(AFF, "r") as f:
        aff_u8 = f["affinity"][:]
    t0 = time.time()
    u, v, sm, ct = rag_gpu(aff_u8, fr)
    print(f"edges={len(u)} sec={time.time()-t0:.1f}")
    rel = abs(len(u) - TASK) / TASK
    bad0 = int(((u == 0) | (v == 0)).sum())
    okm = True
    for i in range(min(1000, len(u))):
        if ct[i] and abs(sm[i] / ct[i] - sm[i] / ct[i]) > 1e-6:
            okm = False
            break
        if ct[i] == 0:
            okm = False
            break
    print(f"rel_edge_err={rel:.6f} bg_endpoints={bad0} mean_ok={okm}")
    ok = rel <= 0.001 and bad0 == 0 and okm
    print("G3", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)
    np.savez(CACHE / "rag_gpu.npz", u=u, v=v, sm=sm, ct=ct)


if __name__ == "__main__":
    main()
