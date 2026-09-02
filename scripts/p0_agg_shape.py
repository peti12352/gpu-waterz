#!/usr/bin/env python3
"""P0e-j: measure RAG shape. No algorithm lock. Writes data/cache/p0_agg_shape.json."""
from __future__ import annotations

import ctypes
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
SO = ROOT / "src/librac_agg.so"


def main():
    subprocess.check_call(
        ["g++", "-O3", "-DNDEBUG", "-shared", "-fPIC", "-o", str(SO), str(ROOT / "src/rac_agg.cpp")]
    )
    z = np.load(CACHE / "rag.npz")
    u = np.ascontiguousarray(z["keys"][:, 0], dtype=np.uint32)
    v = np.ascontiguousarray(z["keys"][:, 1], dtype=np.uint32)
    sm = np.ascontiguousarray(z["stats"][:, 0], dtype=np.float64)
    ct = np.ascontiguousarray(np.rint(z["stats"][:, 1]), dtype=np.int64)
    max_id = int(max(int(u.max()), int(v.max())))
    print(f"P0 edges={len(u)} max_id={max_id}", flush=True)
    lib = ctypes.CDLL(str(SO))
    lib.p0_agg_shape_cpu.restype = ctypes.c_int
    lib.p0_agg_shape_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.c_uint32,
    ]
    log_path = ROOT / "data/logs/p0_agg_shape.cerr"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # C++ prints to stderr; capture via a wrapper file from the caller.
    rc = lib.p0_agg_shape_cpu(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        ctypes.c_uint32(max_id),
    )
    print(f"P0 rc={rc}", flush=True)
    if rc != 1:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
