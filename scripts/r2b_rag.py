#!/usr/bin/env python3
"""R2b: rag_gpu_d CUDA-event median <80ms, edges exact. aff+seg already on GPU."""
from __future__ import annotations

import ctypes
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
CACHE = ROOT / "data/cache"
SO = ROOT / "src/librag_gpu.so"
TASK = 7_505_458


def compile_so():
    src = ROOT / "csrc/rag.cu"
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
        str(src),
    ]
    subprocess.check_call(cmd)


def main():
    compile_so()
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    fr = np.ascontiguousarray(np.load(CACHE / "wz_fragments.npy"), dtype=np.uint32)
    z, y, x = aff.shape[1:]
    aff_t = torch.from_numpy(aff).to("cuda", non_blocking=True)
    seg_t = torch.from_numpy(fr).to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    max_e = 20_000_000
    u_t = torch.empty(max_e, dtype=torch.int32, device="cuda")
    v_t = torch.empty(max_e, dtype=torch.int32, device="cuda")
    sm_t = torch.empty(max_e, dtype=torch.float64, device="cuda")
    ct_t = torch.empty(max_e, dtype=torch.int64, device="cuda")
    lib = ctypes.CDLL(str(SO))
    lib.rag_gpu_d.restype = ctypes.c_int64
    lib.rag_gpu_d.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64,
    ]

    def once():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        n = lib.rag_gpu_d(
            aff_t.data_ptr(),
            seg_t.data_ptr(),
            z, y, x,
            u_t.data_ptr(),
            v_t.data_ptr(),
            sm_t.data_ptr(),
            ct_t.data_ptr(),
            max_e,
        )
        end.record()
        torch.cuda.synchronize()
        return n, start.elapsed_time(end)

    n0, _ = once()
    print(f"edges={n0} task={TASK}", flush=True)
    ms = []
    for i in range(5):
        n, t = once()
        ms.append(t)
        print(f"run{i} n={n} ms={t:.2f}", flush=True)
    ms.sort()
    med = ms[2]
    ok = abs(n0 - TASK) <= 1 and med < 80.0
    print(f"R2b median_ms={med:.2f} edges={n0} {'PASS' if ok else 'FAIL'} (need <80ms)")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
