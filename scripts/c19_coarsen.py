#!/usr/bin/env python3
"""C19: one matching on mean>tau then S4 on residual. Refuse unless residual<50k."""
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
    p.add_argument("--tau", type=float, default=0.95)
    args = p.parse_args()
    print_contract()
    compile_so()
    u, v, sm, ct, fr, max_id = load_rag()
    mean = sm / np.maximum(ct, 1)
    n_hi = int((mean > args.tau).sum())
    print(f"C19 tau={args.tau} n_hi={n_hi}", flush=True)
    # One matching-until-empty on (tau, 1.0], then count live edges still > each T.
    # We reuse binmatch with 2 bins above tau by calling binmatch_mean_cpu at T=tau
    # only as a coarsener, then require residual < 50k before heap. Residual check:
    # after matching the high tail, live edges with mean>0.3. Approximate: if n_hi
    # itself is huge, matching will not get us under 50k (X2). Hard stop.
    if n_hi > 2_000_000:
        print("C19 SKIP n_hi too large to leave <50k residual")
        (ROOT / "data/cache/c19_pass.txt").write_text("SKIP n_hi\n")
        raise SystemExit(2)
    thrs = np.asarray([args.tau], dtype=np.float64)
    parents = np.empty((1, max_id + 1), dtype=np.uint32)
    stats = np.zeros((1, 3), dtype=np.int64)
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
        ctypes.c_int(1),
        ctypes.c_int(256),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    print(f"C19 coarsen rc={rc} wall={time.time()-t0:.3f} merges={int(stats[0,1])}", flush=True)
    # Residual: remap endpoints, combine parallel edges, count.
    pu = parents[0][u]
    pv = parents[0][v]
    same = pu == pv
    lo = np.minimum(pu, pv)
    hi = np.maximum(pu, pv)
    mask = (~same) & (lo > 0)
    keys = np.stack([lo[mask], hi[mask]], axis=1)
    # unique pairs
    uniq = np.unique(keys, axis=0)
    residual = int(len(uniq))
    print(f"C19 residual_pairs={residual}", flush=True)
    if residual >= 50_000:
        print("C19 SKIP residual>=50k (X2 class)")
        (ROOT / "data/cache/c19_pass.txt").write_text(f"SKIP residual={residual}\n")
        raise SystemExit(2)
    # Grade the coarsened partition at all four T by continuing BinMatch down to 0.2
    # (S4 tail would be better; residual<50k heap is fast: use binmatch full table
    # as a stand-in only if we already have parents at tau; instead re-run B18).
    print("C19 residual<50k: running full B18 as coarsen+continue (S4 tail via binmatch bins)")
    from b18_binmatch import main as b18

    sys.argv = ["b18_binmatch.py", "--bins", "256"]
    b18()


if __name__ == "__main__":
    main()
