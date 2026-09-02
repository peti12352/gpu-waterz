#!/usr/bin/env python3
"""P0c/P0d: basin SV saturation + bit popcount after E9b divide."""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from task_gate import AFF_HIGH, AFF_LOW, print_contract  # noqa: E402

AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
SO = ROOT / "src/libws_gpu.so"


def main():
    print_contract()
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    z, y, x = aff.shape[1:]
    stats = np.zeros(5, dtype=np.int64)
    lib = ctypes.CDLL(str(SO))
    lib.p0_ws_diag.restype = ctypes.c_int
    lib.p0_ws_diag.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float,
        ctypes.POINTER(ctypes.c_int64),
    ]
    rc = lib.p0_ws_diag(
        aff.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        z, y, x, ctypes.c_float(AFF_LOW), ctypes.c_float(AFF_HIGH),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    print(
        f"P0c/d rc={rc} bg={int(stats[0])} unique={int(stats[1])} "
        f"multibit={int(stats[2])} first_zero_sv={int(stats[3])} "
        f"divide_us={int(stats[4])}"
    )
    if rc != 1:
        raise SystemExit(1)
    n_fg = int(stats[1]) + int(stats[2])
    if n_fg:
        print(f"P0d unique/fg={int(stats[1])/n_fg:.4f} multi/fg={int(stats[2])/n_fg:.4f}")


if __name__ == "__main__":
    main()
