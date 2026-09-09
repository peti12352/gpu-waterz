#!/usr/bin/env python3
"""E6r: device paper-ParHAC ε=0.08. LOCK iff VOI PASS and val AGG≤50ms."""
from __future__ import annotations

import ctypes
import io
import os
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import (  # noqa: E402
    AFF_THRESHOLDS, CACHE, SO, compile_so, grade_parents, load_rag, stamp,
)
from b_dev_aff import nvcc_arch_flags  # noqa: E402
from task_gate import print_contract  # noqa: E402

NVCC = "/usr/local/cuda-12.8/bin/nvcc"
DSO = ROOT / "src/libparhac_d.so"


def compile_d():
    src = ROOT / "csrc/parhac_d.cu"
    if os.environ.get("WATERZ_SKIP_BUILD") == "1" and DSO.is_file():
        return
    if DSO.exists() and DSO.stat().st_mtime >= src.stat().st_mtime:
        return
    subprocess.check_call([
        NVCC, "-O3", *nvcc_arch_flags(), "--shared", "-Xcompiler", "-fPIC",
        "-o", str(DSO), str(src),
    ])


def main():
    print_contract()
    compile_so()
    compile_d()
    u, v, sm, ct, fr, max_id = load_rag()
    thrs = np.asarray(AFF_THRESHOLDS, dtype=np.float64)
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(thrs), 3), dtype=np.int64)
    lib = ctypes.CDLL(str(DSO))
    lib.parhac_paper_d.restype = ctypes.c_int
    lib.parhac_paper_d.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    print("E6r device paper-ε 0.08", flush=True)
    t0 = time.perf_counter()
    rc = lib.parhac_paper_d(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(thrs)),
        ctypes.c_double(0.08),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    tagg = time.perf_counter() - t0
    print(
        f"E6r rc={rc} agg_s={tagg:.4f} outer={list(map(int, stats[:, 0]))} "
        f"inner={list(map(int, stats[:, 2]))} merges={list(map(int, stats[:, 1]))}",
        flush=True,
    )
    if rc != 1:
        stamp("e6r", False, f"rc={rc}")
        return False
    if os.environ.get("WATERZ_E6R_NOGRADE"):
        print("E6r skip grade (WATERZ_E6R_NOGRADE)", flush=True)
        stamp("e6r", False, f"NOGRADE agg_ms={tagg * 1000.0:.2f}")
        return False
    buf = io.StringIO()
    with redirect_stdout(buf):
        ok = grade_parents(parents, fr, "E6r", "e6r_parhac")
    sys.stdout.write(buf.getvalue())
    ms = tagg * 1000.0
    if ok and ms <= 50.0:
        stamp("e6r", True, f"LOCK device ParHAC ε=0.08 agg_ms={ms:.2f}")
        print("E6r PASS LOCK", flush=True)
        return True
    extra = f"{'PASS' if ok else 'FAIL'} agg_ms={ms:.2f} budget=50"
    stamp("e6r", False, extra)
    print(f"E6r {extra}", flush=True)
    return False


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
