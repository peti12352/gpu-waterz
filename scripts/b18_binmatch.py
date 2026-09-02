#!/usr/bin/env python3
"""B18: matching-until-empty in BinQueue live-mean bands. Not X1/X1b."""
from __future__ import annotations

import argparse
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
    p = argparse.ArgumentParser()
    p.add_argument("--bins", type=int, default=256)
    args = p.parse_args()
    print_contract()
    compile_so()
    u, v, sm, ct, fr, max_id = load_rag()
    print(f"B18 edges={len(u)} bins={args.bins}", flush=True)
    thrs = np.asarray(AFF_THRESHOLDS, dtype=np.float64)
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(thrs), 3), dtype=np.int64)
    lib = ctypes.CDLL(str(SO))
    lib.binmatch_mean_cpu.restype = ctypes.c_int
    lib.binmatch_mean_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.POINTER(ctypes.c_int64),
    ]
    t0 = time.time()
    rc = lib.binmatch_mean_cpu(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(thrs)),
        ctypes.c_int(args.bins),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    wall = time.time() - t0
    rounds = int(stats[:, 0].sum())
    print(f"B18 rc={rc} wall={wall:.3f} sum_rounds={rounds}", flush=True)
    if rc != 1:
        raise SystemExit(2)
    ok = grade_parents(parents, fr, "B18", f"b18_bins{args.bins}")
    (ROOT / "data/cache/b18_pass.txt").write_text(
        f"{'PASS' if ok else 'FAIL'} rounds={rounds} bins={args.bins} wall={wall:.3f}\n"
    )
    if not ok or (rounds > 80 and wall >= 0.010):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
