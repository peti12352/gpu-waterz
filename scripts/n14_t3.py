#!/usr/bin/env python3
"""N14 T3: GPU FIFO-inside-bin MEAN. Not 1-thread find+union probe.

Val T=0.3 VOI then wall vs E6s. Kill if VOI FAIL or slower than ParHAC.
Not a 2 Gvox/s claim. Not 3090 Ti. No default on FAIL.
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
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402
from t3_memsafe import grade_t3, voi_parent_mmap  # noqa: E402

NVCC = "/usr/local/cuda-12.8/bin/nvcc"
DSO = ROOT / "src/libbinqueue_d.so"
NOTE = ROOT / "notes/N14_T3.md"
OUT = CACHE / "n14_t3.json"
PARHAC_VAL_MS = 202.0
PARHAC_216_MS = 2234.0


def compile_bq():
    src = ROOT / "csrc/binqueue_d.cu"
    if DSO.exists() and DSO.stat().st_mtime >= src.stat().st_mtime:
        return
    subprocess.check_call([
        NVCC, "-O3", *nvcc_arch_flags(), "--shared", "-Xcompiler", "-fPIC",
        "-o", str(DSO), str(src),
    ])


def main():
    print("N14 T3 GPU FIFO-BinQueue. Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N14 T3 REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)
    compile_bq()
    u, v, sm, ct, _fr, max_id = load_rag()
    lib = ctypes.CDLL(str(DSO))
    lib.binqueue_mean_d.restype = ctypes.c_int
    lib.binqueue_mean_d.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_int, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    thrs = np.asarray([0.3], dtype=np.float64)
    parents = np.empty((1, max_id + 1), dtype=np.uint32)
    stats = np.zeros((1, 3), dtype=np.int64)
    t0 = time.perf_counter()
    rc = lib.binqueue_mean_d(
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
    ms = (time.perf_counter() - t0) * 1000.0
    split, merge, nseg = voi_parent_mmap(parents[0])
    ok, sl, ml = grade_t3(split, merge)
    kill_voi = not ok
    kill_slow = ms >= PARHAC_VAL_MS
    keep = bool(ok and ms <= PARHAC_216_MS / 1.2)
    # val already slower than ParHAC val => cannot beat 2234/1.2 on 2.16
    if kill_slow:
        keep = False
    doc = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "gpu": gpu_state(),
        "rc": int(rc),
        "ms": ms,
        "split": split,
        "merge": merge,
        "nseg": nseg,
        "voi_ok": bool(ok),
        "pops": int(stats[0, 2]),
        "merges": int(stats[0, 1]),
        "kill_voi": kill_voi,
        "kill_slow": kill_slow,
        "keep_default": keep,
        "run216": None,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        "# N14 T3 GPU FIFO-BinQueue\n\n"
        "Not a 2 Gvox/s claim. Not 3090 Ti. N=256 MEAN. FIFO inside bin. "
        "Not the N11 1-thread find+union probe. No 2.16 if val already >= ParHAC val.\n\n"
        f"- rc={rc} ms={ms:.1f} pops={int(stats[0, 2])} merges={int(stats[0, 1])}\n"
        f"- T=0.3 split={split} merge={merge} nseg={nseg} voi_ok={ok}\n"
        f"- kill_voi={kill_voi} kill_slow={kill_slow} (val vs {PARHAC_VAL_MS} ms)\n"
        f"- keep_default={keep}. no default on FAIL. no 2.16.\n"
    )
    print(
        f"N14 T3 voi_ok={ok} ms={ms:.1f} keep_default={keep} -> {OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
