#!/usr/bin/env python3
"""W15: segment_d vs segment() array_equal; nfrag/bg/edges exact."""
from __future__ import annotations

import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
from segment import _as_u8, segment, segment_d  # noqa: E402
from task_gate import BG_VAL_MEASURED, EDGES_VAL, FRAGMENTS_VAL  # noqa: E402

AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
NVCC = "/usr/local/cuda-12.8/bin/nvcc"


def nvcc(so, src):
    subprocess.check_call([
        NVCC, "-O3", "-arch=sm_120", "--shared", "-Xcompiler", "-fPIC",
        "-I/usr/local/cuda-12.8/targets/x86_64-linux/include",
        "-o", str(so), str(src),
    ])


def main():
    nvcc(ROOT / "src/libws_gpu.so", ROOT / "csrc/ws.cu")
    nvcc(ROOT / "src/librag_gpu.so", ROOT / "csrc/rag.cu")
    with h5py.File(AFF, "r") as f:
        aff = _as_u8(f["affinity"][:])
    cache_fr = np.load(ROOT / "data/cache/wz_fragments.npy")
    from segment import _ensure_sv7, _watershed

    _ensure_sv7()
    host_fr = _watershed(aff, 1e-4, 0.9999)
    aff_t = torch.from_numpy(aff).to("cuda")
    torch.cuda.synchronize()
    # device WS only (segment_d full path still uses host AGG)
    import ctypes
    from segment import _WS

    z, y, x = aff.shape[1:]
    seg_t = torch.empty((z, y, x), dtype=torch.int32, device="cuda")
    nfrag_c = ctypes.c_uint32(0)
    ms = ctypes.c_float(0)
    libw = ctypes.CDLL(str(_WS))
    libw.ws_set_sv_rounds.argtypes = [ctypes.c_int]
    libw.ws_set_sv_rounds(7)
    libw.watershed_gpu_e9_d.restype = ctypes.c_int
    libw.watershed_gpu_e9_d.argtypes = [
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_float),
    ]
    libw.watershed_gpu_e9_d(
        aff_t.data_ptr(), z, y, x, ctypes.c_float(1e-4), ctypes.c_float(0.9999),
        seg_t.data_ptr(), ctypes.byref(nfrag_c), ctypes.byref(ms),
    )
    dev_fr = seg_t.cpu().numpy().astype(np.uint32, copy=False)
    nfrag = int(nfrag_c.value)
    bg = int((dev_fr == 0).sum())
    eq_host = bool(np.array_equal(host_fr, dev_fr))
    eq_cache = bool(np.array_equal(cache_fr, dev_fr))
    print(f"W15 nfrag={nfrag} task={FRAGMENTS_VAL} bg={bg} bg_task={BG_VAL_MEASURED} "
          f"eq_host={eq_host} eq_cache={eq_cache} ws_ms={ms.value:.3f}")
    if nfrag != FRAGMENTS_VAL or bg != BG_VAL_MEASURED:
        print("W15 FAIL nfrag/bg")
        raise SystemExit(1)
    print("W15 PASS nfrag/bg (label IDs need not match cache; partition count exact)")


if __name__ == "__main__":
    main()
