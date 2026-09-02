#!/usr/bin/env python3
"""W1: GPU compact-frontier plateau BFS vs host first-BFS bits."""
from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
SO = ROOT / "src/libws_gpu.so"


def compile_so():
    cmd = [
        "/usr/local/cuda-12.8/bin/nvcc",
        "-O3",
        "-arch=sm_120",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        "-I/usr/local/cuda-12.8/targets/x86_64-linux/include",
        "-o",
        str(SO),
        str(ROOT / "csrc/ws.cu"),
    ]
    subprocess.check_call(cmd)


def main():
    compile_so()
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    z, y, x = aff.shape[1:]
    host = np.empty(z * y * x, dtype=np.uint8)
    gpu = np.empty(z * y * x, dtype=np.uint8)
    lib = ctypes.CDLL(str(SO))
    lib.w1_plateau_bfs.restype = ctypes.c_int
    lib.w1_plateau_bfs.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    rc = lib.w1_plateau_bfs(
        aff.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        z, y, x, 1e-4, 0.9999,
        host.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        gpu.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    eq = bool(np.array_equal(host, gpu))
    mism = int((host != gpu).sum())
    print(f"W1 array_equal={eq} mismatch={mism} rc={rc}")
    print("W1", "PASS" if eq and rc == 1 else "FAIL")
    if not eq:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
