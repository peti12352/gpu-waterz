#!/usr/bin/env python3
"""V35: voxel hysteresis CC high=0.9, low=T. One VOI, then stop."""
from __future__ import annotations

import ctypes
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import AFF_THRESHOLDS, OUT, PY, SO, compile_so, stamp  # noqa: E402
from task_gate import print_contract  # noqa: E402

AFF = OUT / "cremiA_val/affinity.h5"


def main():
    print_contract()
    compile_so()
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    zdim, ydim, xdim = aff.shape[1:]
    print(f"V35 aff={aff.shape} high=0.9", flush=True)
    lib = ctypes.CDLL(str(SO))
    lib.hysteresis_cc_cpu.restype = ctypes.c_int
    lib.hysteresis_cc_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    dest_dir = OUT / "v35_hyst"
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    t0 = time.time()
    for thr in AFF_THRESHOLDS:
        lab = np.empty((zdim, ydim, xdim), dtype=np.uint32)
        rc = lib.hysteresis_cc_cpu(
            aff.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int(zdim), ctypes.c_int(ydim), ctypes.c_int(xdim),
            ctypes.c_double(0.9), ctypes.c_double(float(thr)),
            lab.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        )
        nseg = int((np.unique(lab) != 0).sum())
        dest = dest_dir / f"mine_thr{thr}.h5"
        with h5py.File(dest, "w") as h:
            h.create_dataset("labels", data=lab)
        print(f"  T={thr} rc={rc} nseg={nseg}", flush=True)
        paths.append(str(dest))
    wall = time.time() - t0
    cmd = [str(PY), str(OUT / "baseline/run_baseline.py"), "--candidate", *paths]
    r = subprocess.run(cmd, cwd=str(OUT), capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.stderr:
        sys.stderr.write(r.stderr)
    ok = "ACCURACY GATE: PASS" in r.stdout
    stamp("v35", ok, f"hysteresis high=0.9 wall={wall:.3f}")
    print(f"V35 {'PASS' if ok else 'FAIL'} (one VOI, stop)", flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
