#!/usr/bin/env python3
"""E9b: independent-plateau GPU divideplateaus vs host plateau_bfs_only."""
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
NVCC = "/usr/local/cuda-12.8/bin/nvcc"
sys.path.insert(0, str(ROOT / "src"))
from task_gate import AFF_HIGH, AFF_LOW, print_contract  # noqa: E402


def compile_ws():
    subprocess.check_call([
        NVCC, "-O3", "-arch=sm_120", "--shared", "-Xcompiler", "-fPIC",
        "-I/usr/local/cuda-12.8/targets/x86_64-linux/include",
        "-o", str(SO), str(ROOT / "csrc/ws.cu"),
    ])


def main():
    print_contract()
    compile_ws()
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    z, y, x = aff.shape[1:]
    n = z * y * x
    host = np.zeros(n, dtype=np.uint8)
    gpu = np.zeros(n, dtype=np.uint8)
    lib = ctypes.CDLL(str(SO))
    lib.e9b_divide.restype = ctypes.c_int
    lib.e9b_divide.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    rc = lib.e9b_divide(
        aff.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        z, y, x, ctypes.c_float(AFF_LOW), ctypes.c_float(AFF_HIGH),
        host.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        gpu.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    mism = int((host != gpu).sum())
    print(f"E9b rc={rc} mismatch={mism} array_equal={bool(np.array_equal(host, gpu))}")
    if rc != 1 or mism != 0:
        print("E9b FAIL bit mismatch: do not almost")
        raise SystemExit(1)
    print("E9b PASS bit-equal")


if __name__ == "__main__":
    main()
