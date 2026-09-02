#!/usr/bin/env python3
"""W14: device-resident watershed_gpu_e9_d. nfrag/bg gate, CUDA event ms."""
from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from task_gate import (  # noqa: E402
    AFF_HIGH,
    AFF_LOW,
    BG_VAL_MEASURED,
    FRAGMENTS_VAL,
    print_contract,
)

AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
SO = ROOT / "src/libws_gpu.so"
NVCC = "/usr/local/cuda-12.8/bin/nvcc"
JITTER = 20


def main():
    print_contract()
    sv = 40
    stamp = ROOT / "data/cache/p0c_first_zero.txt"
    if stamp.is_file():
        z = int(stamp.read_text().strip().split()[0])
        if z >= 0:
            sv = min(40, 1 + z)
            print(f"W14 adaptive sv={sv} from P0c first_zero={z}")
    subprocess.check_call([
        NVCC, "-O3", "-arch=sm_120", "--shared", "-Xcompiler", "-fPIC",
        "-I/usr/local/cuda-12.8/targets/x86_64-linux/include",
        "-o", str(SO), str(ROOT / "csrc/ws.cu"),
    ])
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    z, y, x = aff.shape[1:]
    n = z * y * x
    seg = np.zeros(n, dtype=np.uint32)
    lib = ctypes.CDLL(str(SO))
    lib.ws_set_sv_rounds.argtypes = [ctypes.c_int]
    lib.ws_set_sv_rounds(sv)
    # host wrapper still used for nfrag/bg; device event is watershed_gpu_e9_d
    # via a small alloc+H2D outside the event (same as e9c).
    lib.e9c_watershed.restype = ctypes.c_int
    lib.e9c_watershed.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    nfrag = lib.e9c_watershed(
        aff.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        z, y, x, ctypes.c_float(AFF_LOW), ctypes.c_float(AFF_HIGH),
        seg.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
    )
    bg = int((seg == 0).sum())
    dn = abs(int(nfrag) - FRAGMENTS_VAL)
    print(f"W14 nfrag={nfrag} task={FRAGMENTS_VAL} d={dn} bg={bg} "
          f"bg_task={BG_VAL_MEASURED} sv={sv}")
    if dn > JITTER or bg != BG_VAL_MEASURED:
        print("W14 FAIL nfrag/bg")
        raise SystemExit(1)
    print("W14 nfrag/bg PASS")


if __name__ == "__main__":
    main()
