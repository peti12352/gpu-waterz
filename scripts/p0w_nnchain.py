#!/usr/bin/env python3
"""P0w: NN-chain S3 round count. N36 only if rounds≤30 and maxchain≤5k."""
from __future__ import annotations

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
    AFF_THRESHOLDS, SO, compile_so, grade_parents, load_rag, stamp,
)
from task_gate import print_contract  # noqa: E402


def main():
    print_contract()
    compile_so()
    u, v, sm, ct, fr, max_id = load_rag()
    thrs = np.asarray(AFF_THRESHOLDS, dtype=np.float64)
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(thrs), 3), dtype=np.int64)
    lib = ctypes.CDLL(str(SO))
    lib.nn_chain_s3_cpu.restype = ctypes.c_int
    lib.nn_chain_s3_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    t0 = time.time()
    rc = lib.nn_chain_s3_cpu(
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
    rounds = int(stats[:, 0].max())
    chain = int(stats[:, 2].max())
    print(
        f"P0w rc={rc} wall={wall:.3f} rounds={list(map(int, stats[:, 0]))} "
        f"maxchain={list(map(int, stats[:, 2]))}",
        flush=True,
    )
    if rc != 1 or rounds > 30 or chain > 5000:
        stamp("n36", False, f"KILL rounds={rounds} maxchain={chain}")
        print("N36 SKIP (P0w kill)", flush=True)
        return False
    buf = io.StringIO()
    with redirect_stdout(buf):
        ok = grade_parents(parents, fr, "N36", "n36_nnchain")
    sys.stdout.write(buf.getvalue())
    if ok:
        stamp("n36", True, f"LOCK NN-chain rounds={rounds} maxchain={chain}")
        return True
    stamp("n36", False, f"FAIL rounds={rounds}")
    return False


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
