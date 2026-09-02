#!/usr/bin/env python3
"""S21: GASP mean + cannot-link on mean<0.5. Not AbsMax / not w-=1-mean."""
from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import AFF_THRESHOLDS, SO, compile_so, grade_parents, load_rag  # noqa: E402
from task_gate import print_contract  # noqa: E402


def main():
    print_contract()
    compile_so()
    u, v, sm, ct, fr, max_id = load_rag()
    print(f"S21 edges={len(u)}", flush=True)
    thrs = np.asarray(AFF_THRESHOLDS, dtype=np.float64)
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(thrs), 3), dtype=np.int64)
    lib = ctypes.CDLL(str(SO))
    lib.gasp_mean_signed_cpu.restype = ctypes.c_int
    lib.gasp_mean_signed_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.POINTER(ctypes.c_int64),
    ]
    t0 = time.time()
    rc = lib.gasp_mean_signed_cpu(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(thrs)),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    wall = time.time() - t0
    print(f"S21 rc={rc} wall={wall:.3f}", flush=True)
    if rc != 1:
        raise SystemExit(2)
    ok = grade_parents(parents, fr, "S21", "s21_gasp_mean")
    (ROOT / "data/cache/s21_pass.txt").write_text(f"{'PASS' if ok else 'FAIL'} wall={wall:.3f}\n")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
