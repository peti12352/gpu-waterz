#!/usr/bin/env python3
"""L36: relative-contact then S4 on thin leftover. LOCK iff PASS and residual≤5k."""
from __future__ import annotations

import argparse
import ctypes
import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import (  # noqa: E402
    AFF_THRESHOLDS, SO, compile_so, frag_sizes, grade_parents, load_rag, stamp,
)
from task_gate import print_contract  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gamma", type=float, default=0.10)
    p.add_argument("--alpha", type=float, default=0.67)
    args = p.parse_args()
    print_contract()
    compile_so()
    u, v, sm, ct, fr, max_id = load_rag()
    sz = frag_sizes(fr, max_id)
    thrs = np.asarray(AFF_THRESHOLDS, dtype=np.float64)
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    residual = np.zeros(len(thrs), dtype=np.int64)
    lib = ctypes.CDLL(str(SO))
    lib.leftover_rel_s4_cpu.restype = ctypes.c_int
    lib.leftover_rel_s4_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    t0 = time.time()
    rc = lib.leftover_rel_s4_cpu(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        sz.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(thrs)),
        ctypes.c_double(args.gamma), ctypes.c_double(args.alpha),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        residual.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    wall = time.time() - t0
    nres = int(residual.max())
    print(f"L36 rc={rc} wall={wall:.3f} residual={list(map(int, residual))} max={nres}", flush=True)
    if rc != 1:
        stamp("l36", False, f"rc={rc}")
        return False
    buf = io.StringIO()
    with redirect_stdout(buf):
        ok = grade_parents(parents, fr, "L36", f"l36_g{args.gamma}")
    sys.stdout.write(buf.getvalue())
    if ok and nres <= 5000:
        stamp("l36", True, f"LOCK leftover n_residual={nres} gamma={args.gamma}")
        return True
    if ok and nres < 50_000:
        stamp("l36", False, f"PASS-serial / FAIL-depth n_residual={nres}")
        return False
    stamp("l36", False, f"{'PASS' if ok else 'FAIL'} n_residual={nres}")
    return False


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
