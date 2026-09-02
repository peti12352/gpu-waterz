#!/usr/bin/env python3
"""G2: GPU watershed vs waterz no-merge on val."""
from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import waterz

ROOT = Path(__file__).resolve().parents[1]
AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
SO = ROOT / "src/libws_gpu.so"
TASK = 2_175_400


def gpu_ws(aff_u8, low=1e-4, high=0.9999):
    z, y, x = aff_u8.shape[1:]
    out = np.zeros((z, y, x), dtype=np.uint32)
    lib = ctypes.CDLL(str(SO))
    lib.watershed_gpu.restype = ctypes.c_int
    lib.watershed_gpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    a = np.ascontiguousarray(aff_u8)
    n = lib.watershed_gpu(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        z, y, x, low, high,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
    )
    return out, n


def main():
    with h5py.File(AFF, "r") as f:
        aff_u8 = f["affinity"][:]
    cache = ROOT / "data/cache/wz_fragments.npy"
    t0 = time.time()
    if cache.is_file():
        wz = np.load(cache)
    else:
        aff = np.ascontiguousarray(aff_u8.astype(np.float32) / 255.0)
        wz = None
        for seg in waterz.agglomerate(aff, [0.0]):
            wz = seg.astype(np.uint32, copy=False)
            break
    wz_n = int((np.unique(wz) != 0).sum())
    wz_bg = int((wz == 0).sum())
    print(f"waterz n={wz_n} bg={wz_bg} sec={time.time()-t0:.1f}")
    t1 = time.time()
    g1, _ = gpu_ws(aff_u8)
    t2 = time.time()
    g2, _ = gpu_ws(aff_u8)
    gn = int((np.unique(g1) != 0).sum())
    gbg = int((g1 == 0).sum())
    rel = abs(gn - wz_n) / max(wz_n, 1)
    eq = np.array_equal(g1, g2)
    print(f"gpu n={gn} bg={gbg} sec={t2-t1:.1f} rel={rel:.6f} det={eq}")
    ok = abs(wz_n - TASK) < 1000 and rel <= 0.01 and eq and gbg == wz_bg
    print("G2", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)
    np.save(ROOT / "data/cache/gpu_fragments.npy", g1)


if __name__ == "__main__":
    main()
