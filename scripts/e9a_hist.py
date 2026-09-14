#!/usr/bin/env python3
"""E9a: plateau size histogram after host flow matching k_flow."""
from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
SO = ROOT / "src/libe9a_hist.so"
sys.path.insert(0, str(ROOT / "src"))
from task_gate import AFF_HIGH, AFF_LOW, BG_VAL_MEASURED, print_contract  # noqa: E402


def main():
    print_contract()
    subprocess.check_call([
        "g++", "-O3", "-DNDEBUG", "-shared", "-fPIC",
        "-o", str(SO), str(ROOT / "src/e9a_hist.cpp"),
    ])
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    z, y, x = aff.shape[1:]
    stats = np.zeros(9, dtype=np.int64)
    lib = ctypes.CDLL(str(SO))
    lib.e9a_plateau_hist.restype = ctypes.c_int
    lib.e9a_plateau_hist.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float,
        ctypes.POINTER(ctypes.c_int64),
    ]
    rc = lib.e9a_plateau_hist(
        aff.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        z, y, x, ctypes.c_float(AFF_LOW), ctypes.c_float(AFF_HIGH),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    names = [
        "n_fg", "n_bg", "n_corner", "n_plat_cc", "n_closed_cc",
        "n_closed_vox", "max", "p50", "p99",
    ]
    print("E9a rc", rc)
    for n, v in zip(names, stats):
        print(f"  {n}={int(v)}")
    n_fg, n_bg, max_sz = int(stats[0]), int(stats[1]), int(stats[6])
    print(f"  bg_vs_task={n_bg} (measured {BG_VAL_MEASURED})")
    frac = (max_sz / n_fg) if n_fg else 0.0
    giant = frac > 0.2
    print(f"  max/fg={frac:.6f} giant={giant}")
    if giant:
        print("E9a GIANT plateau >20% fg: E9b cannot save that component")
    else:
        print("E9a no giant plateau: E9b independent FIFO is the speed path")


if __name__ == "__main__":
    main()
