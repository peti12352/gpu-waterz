#!/usr/bin/env python3
"""W2: GPU basins + parallel BFS. G2 nfrag/bg/det, device <80ms."""
from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
SO = ROOT / "src/libws_gpu.so"
TASK_N = 2_175_400
TASK_BG = 506_568


def compile_so():
    subprocess.check_call([
        "/usr/local/cuda-12.8/bin/nvcc",
        "-O3", "-arch=sm_120", "--shared", "-Xcompiler", "-fPIC",
        "-I/usr/local/cuda-12.8/targets/x86_64-linux/include",
        "-o", str(SO), str(ROOT / "csrc/ws.cu"),
    ])


def main():
    compile_so()
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    z, y, x = aff.shape[1:]
    aff_t = torch.from_numpy(aff).to("cuda")
    torch.cuda.synchronize()
    lib = ctypes.CDLL(str(SO))
    lib.watershed_gpu_d.restype = ctypes.c_int
    lib.watershed_gpu_d.argtypes = [
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float, ctypes.c_void_p,
    ]
    out1 = torch.zeros((z, y, x), dtype=torch.int32, device="cuda")
    out2 = torch.zeros((z, y, x), dtype=torch.int32, device="cuda")

    def run(out):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        n = lib.watershed_gpu_d(
            aff_t.data_ptr(), z, y, x, 1e-4, 0.9999, out.data_ptr())
        end.record()
        torch.cuda.synchronize()
        return n, start.elapsed_time(end)

    n1, _ = run(out1)
    n1, t1 = run(out1)
    n2, t2 = run(out2)
    g1 = out1.cpu().numpy().astype(np.uint32)
    g2 = out2.cpu().numpy().astype(np.uint32)
    gn = int((np.unique(g1) != 0).sum())
    gbg = int((g1 == 0).sum())
    eq = bool(np.array_equal(g1, g2))
    rel = abs(gn - TASK_N) / TASK_N
    ms = min(t1, t2)
    print(f"gpu n={gn} (lib={n1}) bg={gbg} rel={rel:.6f} det={eq} device_ms={t1:.2f},{t2:.2f}")
    ok = rel <= 0.01 and gbg == TASK_BG and eq and ms < 80.0
    print("W2", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
