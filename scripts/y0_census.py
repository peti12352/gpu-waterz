#!/usr/bin/env python3
"""Y0: RAC census on cached RAG at T=0.3, 50-round trajectory."""
from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
SO = ROOT / "src/librac_agg.so"
T = 0.3
NROUNDS = 50


def compile_so():
    src = ROOT / "src/rac_agg.cpp"
    if SO.exists() and SO.stat().st_mtime >= src.stat().st_mtime:
        return
    subprocess.check_call(
        ["g++", "-O3", "-DNDEBUG", "-shared", "-fPIC", "-o", str(SO), str(src)]
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
    n_rnn = np.zeros(NROUNDS + 1, dtype=np.int64)
    n_one = np.zeros(NROUNDS + 1, dtype=np.int64)
    smin = np.zeros(NROUNDS + 1, dtype=np.float64)
    n_ed = np.zeros(NROUNDS + 1, dtype=np.int64)
    lib = ctypes.CDLL(str(SO))
    lib.rac_census_cpu.restype = ctypes.c_int
    lib.rac_census_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
    ]
    rc = lib.rac_census_cpu(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        ctypes.c_double(T),
        ctypes.c_int(NROUNDS),
        ctypes.c_uint32(max_id),
        n_rnn.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        n_one.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        smin.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_ed.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    if rc != 1:
        raise RuntimeError("rac_census_cpu failed")
    print(f"edges={len(u)} max_id={max_id} T={T}", flush=True)
    for r in range(NROUNDS + 1):
        if r > 0 and n_rnn[r] == 0 and n_ed[r] == 0 and n_rnn[r - 1] == 0:
            break
        print(
            f"r={r:02d} n_rnn={n_rnn[r]} n_onesided={n_one[r]} "
            f"smin={smin[r]:.6f} n_edges={n_ed[r]}",
            flush=True,
        )
    live = [int(x) for x in n_rnn.tolist() if True]
    # count how many of first 50 rounds actually ran (n_edges still changing or r=0)
    r50_sum = int(n_rnn[1:].sum()) if NROUNDS >= 1 else 0
    n_rnn_r0 = int(n_rnn[0])
    print(f"Y0 n_rnn_r0={n_rnn_r0} rounds50_rnn_sum={r50_sum}")
    high = n_rnn_r0 > 50000 and sum(1 for x in n_rnn[1:11] if x > 10000) >= 10
    low = n_rnn_r0 < 1000
    if high:
        print("Y0 verdict: RAC is the product (high n_rnn)")
    elif low:
        print("Y0 verdict: RAC-only ~serial; Y2 is the speed path")
    else:
        print("Y0 verdict: mixed; run Y1 for VOI+rounds")


if __name__ == "__main__":
    main()
