#!/usr/bin/env python3
"""H0: time heap_s4 wrapper vs waterz merge on cached val RAG."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ref_cpu import heap_from_arrays  # noqa: E402

CACHE = ROOT / "data/cache"
SO = ROOT / "src/libheap_cpu.so"
THRS = [0.2, 0.3, 0.4, 0.5]


def compile_so():
    src = ROOT / "src/heap_s4.cpp"
    subprocess.check_call(
        [
            "g++",
            "-O3",
            "-DNDEBUG",
            "-shared",
            "-fPIC",
            "-I",
            str(ROOT / "src/waterz-upstream/src/waterz"),
            "-o",
            str(SO),
            str(src),
        ]
    )


def load_rag():
    z = np.load(CACHE / "rag.npz")
    u = np.ascontiguousarray(z["keys"][:, 0], dtype=np.uint32)
    v = np.ascontiguousarray(z["keys"][:, 1], dtype=np.uint32)
    sm = np.ascontiguousarray(z["stats"][:, 0], dtype=np.float64)
    ct = np.ascontiguousarray(np.rint(z["stats"][:, 1]), dtype=np.int64)
    return u, v, sm, ct


def main():
    compile_so()
    u, v, sm, ct = load_rag()
    fr = np.load(CACHE / "wz_fragments.npy")
    max_id = int(max(int(u.max()), int(v.max()), int(fr.max())))
    print(f"edges={len(u)} max_id={max_id}", flush=True)
    t0 = time.time()
    snaps = heap_from_arrays(u, v, sm, ct, THRS, max_id=max_id)
    heap_s = time.time() - t0
    print(f"heap_sec={heap_s:.3f}", flush=True)
    for t in THRS:
        n = int((np.unique(snaps[t][1:]) != snaps[t][1:]).sum()) if False else int(
            (snaps[t][1:] == np.arange(1, max_id + 1, dtype=np.uint32)).sum()
        )
        nroot = int((snaps[t] == np.arange(max_id + 1, dtype=np.uint32)).sum())
        print(f"  T={t} nroot={nroot}", flush=True)
    # Fast exact S4 (same mean/combine, heap + incremental adj)
    so2 = ROOT / "src/librac_agg.so"
    src2 = ROOT / "src/rac_agg.cpp"
    subprocess.check_call(
        ["g++", "-O3", "-DNDEBUG", "-shared", "-fPIC", "-o", str(so2), str(src2)]
    )
    import ctypes
    thrs = np.asarray(THRS, dtype=np.float64)
    parents = np.empty((len(THRS), max_id + 1), dtype=np.uint32)
    lib = ctypes.CDLL(str(so2))
    lib.s4_fast_cpu.restype = ctypes.c_int
    lib.s4_fast_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
    ]
    t1 = time.time()
    rc = lib.s4_fast_cpu(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(THRS)),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
    )
    fast_s = time.time() - t1
    print(f"s4_fast_sec={fast_s:.3f} rc={rc}", flush=True)
    for i, t in enumerate(THRS):
        nroot = int((parents[i] == np.arange(max_id + 1, dtype=np.uint32)).sum())
        print(f"  s4_fast T={t} nroot={nroot}", flush=True)
    ok = heap_s <= 4.0 or fast_s <= 4.0
    print(f"H0 heap_sec={heap_s:.3f} s4_fast_sec={fast_s:.3f} {'PASS' if ok else 'FAIL'} (need <=4s)")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
