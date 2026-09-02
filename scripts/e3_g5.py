#!/usr/bin/env python3
"""G5 on locked paper ParHAC ε=0.033333: two runs, parent arrays equal."""
from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
SO = ROOT / "src/librac_agg.so"
THRS = [0.2, 0.3, 0.4, 0.5]


def load_rag():
    z = np.load(CACHE / "rag.npz")
    u = np.ascontiguousarray(z["keys"][:, 0], dtype=np.uint32)
    v = np.ascontiguousarray(z["keys"][:, 1], dtype=np.uint32)
    sm = np.ascontiguousarray(z["stats"][:, 0], dtype=np.float64)
    ct = np.ascontiguousarray(np.rint(z["stats"][:, 1]), dtype=np.int64)
    return u, v, sm, ct


def run(u, v, sm, ct, max_id, eps):
    thrs = np.asarray(THRS, dtype=np.float64)
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(thrs), 3), dtype=np.int64)
    lib = ctypes.CDLL(str(SO))
    lib.parhac_paper_cpu.restype = ctypes.c_int
    lib.parhac_paper_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    rc = lib.parhac_paper_cpu(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(thrs)),
        ctypes.c_double(eps),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    if rc != 1:
        raise RuntimeError("parhac_paper_cpu failed")
    return parents


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--eps", type=float, default=0.033333)
    args = p.parse_args()
    eps = float(args.eps)
    subprocess.check_call(
        ["g++", "-O3", "-DNDEBUG", "-shared", "-fPIC", "-o", str(SO), str(ROOT / "src/rac_agg.cpp")]
    )
    u, v, sm, ct = load_rag()
    max_id = int(max(int(u.max()), int(v.max())))
    a = run(u, v, sm, ct, max_id, eps)
    b = run(u, v, sm, ct, max_id, eps)
    ok = bool(np.array_equal(a, b))
    print(f"G5 paper-ε {eps} array_equal={ok} shape={a.shape}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
