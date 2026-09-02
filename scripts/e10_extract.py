#!/usr/bin/env python3
"""E10: GPU labels[i] = parent[seg[i]]. Bit-equal to host remap."""
from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SO = ROOT / "src/libws_gpu.so"
NVCC = "/usr/local/cuda-12.8/bin/nvcc"
CACHE = ROOT / "data/cache"


def main():
    subprocess.check_call([
        NVCC, "-O3", "-arch=sm_120", "--shared", "-Xcompiler", "-fPIC",
        "-I/usr/local/cuda-12.8/targets/x86_64-linux/include",
        "-o", str(SO), str(ROOT / "csrc/ws.cu"),
    ])
    fr = np.load(CACHE / "wz_fragments.npy")
    if fr.dtype != np.uint32:
        fr = fr.astype(np.uint32, copy=False)
    fr = np.ascontiguousarray(fr.ravel())
    max_id = int(fr.max())
    parent = np.arange(max_id + 1, dtype=np.uint32)
    # identity remap: still exercises the kernel; plus a real remap
    parent[1::3] = np.maximum(parent[1::3] - 1, 0)
    host = parent[fr]
    out = np.empty_like(fr)
    ms = ctypes.c_float(0)
    lib = ctypes.CDLL(str(SO))
    lib.extract_gpu.restype = ctypes.c_int
    lib.extract_gpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int64,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_float),
    ]
    rc = lib.extract_gpu(
        fr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        parent.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_int64(fr.size),
        ctypes.c_uint32(max_id),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.byref(ms),
    )
    eq = bool(np.array_equal(host, out))
    print(f"E10 rc={rc} array_equal={eq} device_ms={ms.value:.3f} nvox={fr.size}")
    if not eq:
        raise SystemExit(1)
    print("E10 PASS")


if __name__ == "__main__":
    main()
