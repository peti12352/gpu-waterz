#!/usr/bin/env python3
"""R14: device rag_gpu_d + extract_gpu_d. Bit-equal, CUDA-event ms. aff already on GPU."""
from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
CACHE = ROOT / "data/cache"
WS = ROOT / "src/libws_gpu.so"
RAG = ROOT / "src/librag_gpu.so"
NVCC = "/usr/local/cuda-12.8/bin/nvcc"
TASK_E = 7_505_458


def nvcc(so, src):
    subprocess.check_call([
        NVCC, "-O3", "-arch=sm_120", "--shared", "-Xcompiler", "-fPIC",
        "-I/usr/local/cuda-12.8/targets/x86_64-linux/include",
        "-o", str(so), str(src),
    ])


def main():
    nvcc(WS, ROOT / "csrc/ws.cu")
    nvcc(RAG, ROOT / "csrc/rag.cu")
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    fr = np.ascontiguousarray(np.load(CACHE / "wz_fragments.npy"), dtype=np.uint32)
    z, y, x = aff.shape[1:]
    n = fr.size
    max_id = int(fr.max())
    parent = np.arange(max_id + 1, dtype=np.uint32)
    parent[1::3] = np.maximum(parent[1::3] - 1, 0)
    host = parent[fr.ravel()]
    aff_t = torch.from_numpy(aff).to("cuda", non_blocking=True)
    seg_t = torch.from_numpy(fr).to("cuda", non_blocking=True)
    par_t = torch.from_numpy(parent).to("cuda", non_blocking=True)
    out_t = torch.empty(n, dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()
    max_e = 20_000_000
    u_t = torch.empty(max_e, dtype=torch.int32, device="cuda")
    v_t = torch.empty(max_e, dtype=torch.int32, device="cuda")
    sm_t = torch.empty(max_e, dtype=torch.float64, device="cuda")
    ct_t = torch.empty(max_e, dtype=torch.int64, device="cuda")
    libr = ctypes.CDLL(str(RAG))
    libr.rag_gpu_d.restype = ctypes.c_int64
    libr.rag_gpu_d.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64,
    ]
    libw = ctypes.CDLL(str(WS))
    libw.extract_gpu_d.restype = ctypes.c_int
    libw.extract_gpu_d.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
    ]
    s0 = torch.cuda.Event(enable_timing=True)
    s1 = torch.cuda.Event(enable_timing=True)
    s0.record()
    nedge = libr.rag_gpu_d(
        aff_t.data_ptr(), seg_t.data_ptr(), z, y, x,
        u_t.data_ptr(), v_t.data_ptr(), sm_t.data_ptr(), ct_t.data_ptr(), max_e,
    )
    s1.record()
    torch.cuda.synchronize()
    rag_ms = s0.elapsed_time(s1)
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    rc = libw.extract_gpu_d(seg_t.data_ptr(), par_t.data_ptr(), n, out_t.data_ptr())
    e1.record()
    torch.cuda.synchronize()
    ext_ms = e0.elapsed_time(e1)
    got = out_t.cpu().numpy().astype(np.uint32, copy=False)
    eq = bool(np.array_equal(host, got))
    print(f"R14 rag_n={nedge} task={TASK_E} rag_ms={rag_ms:.3f} "
          f"extract_rc={rc} extract_ms={ext_ms:.3f} array_equal={eq}")
    if nedge != TASK_E or not eq:
        raise SystemExit(1)
    print("R14 PASS")


if __name__ == "__main__":
    main()
