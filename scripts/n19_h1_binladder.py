#!/usr/bin/env python3
"""N19 H1: affinity bin-ladder PQ (Luengo-style), work-efficient drain.

NOT N18 parallel BinQueue. Host abort 30s. Kill if wall>800ms or VOI FAIL.
Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import load_rag  # noqa: E402
from b_dev_aff import nvcc_arch_flags  # noqa: E402
from n19_dead import refuse_or_ok, stamp  # noqa: E402
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402
from t3_memsafe import grade_t3, voi_parent_mmap  # noqa: E402

NVCC = "/usr/local/cuda-12.8/bin/nvcc"
DSO = ROOT / "src/libbinladder_d.so"
NOTE = ROOT / "notes/N19_H1.md"
OUT = CACHE / "N19_H1.json"
ABORT_SEC = 30
CLAIM = "not a 2 Gvox/s number; not 3090 Ti"


def compile_so():
    src = ROOT / "csrc/binladder_d.cu"
    if DSO.exists() and DSO.stat().st_mtime >= src.stat().st_mtime:
        return
    subprocess.check_call([
        NVCC, "-O3", *nvcc_arch_flags(), "--shared", "-Xcompiler", "-fPIC",
        "-o", str(DSO), str(src),
    ])


def main():
    print(f"N19 H1 bin-ladder. {CLAIM}.", flush=True)
    msg = refuse_or_ok("N19_H1", force="--force" in sys.argv)
    if msg:
        print(msg, flush=True)
        return 3
    # also refuse reopening N18 parallel
    if refuse_or_ok("N18_B1") is None:
        pass
    busy = card_busy()
    if busy:
        print(f"N19 H1 REFUSE: {busy}", flush=True)
        return 2
    compile_so()
    u, v, sm, ct, _fr, max_id = load_rag()
    lib = ctypes.CDLL(str(DSO))
    lib.binladder_mean_d.restype = ctypes.c_int
    lib.binladder_mean_d.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64), ctypes.c_int,
    ]
    thrs = np.asarray([0.3], dtype=np.float64)
    parents = np.empty((1, max_id + 1), dtype=np.uint32)
    stats = np.zeros((1, 3), dtype=np.int64)
    t0 = time.perf_counter()
    rc = lib.binladder_mean_d(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(1),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int(ABORT_SEC * 1000),
    )
    ms = (time.perf_counter() - t0) * 1000.0
    split, merge, nseg = voi_parent_mmap(parents[0])
    ok, _, _ = grade_t3(split, merge)
    kill = ms > 800 or not ok or rc == 0
    reason = (
        "hang/abort" if rc == 0 else
        f"wall {ms:.0f}>800" if ms > 800 else
        "VOI FAIL" if not ok else "ok"
    )
    doc = {
        "claim": CLAIM, "rc": int(rc), "ms": ms, "split": split, "merge": merge,
        "nseg": nseg, "voi_ok": bool(ok), "kill": kill, "reason": reason,
        "stats": stats[0].tolist(),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        f"# N19 H1 bin-ladder PQ\n\n{CLAIM}.\n\n"
        f"- ms={ms:.1f} voi_ok={ok} kill={kill} reason={reason}\n"
        f"- Luengo ladder; NOT N18 parallel BinQueue\n"
    )
    if kill:
        stamp("N19_H1", reason, {"ms": ms}, "notes/N19_H1.md")
    print(json.dumps(doc), flush=True)
    return 0 if not kill else 1


if __name__ == "__main__":
    raise SystemExit(main())
