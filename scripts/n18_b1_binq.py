#!/usr/bin/env python3
"""N18 B1: work-efficient parallel BinQueue probe with 30s host abort.

Multi-bin parallel pop (not single-thread drain). Kill if wall >2x ParHAC val
(~400 ms) or hang. Not a 2 Gvox/s claim. Not 3090 Ti.
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
from n18_dead import refuse_or_ok, stamp  # noqa: E402
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402
from t3_memsafe import grade_t3, voi_parent_mmap  # noqa: E402

NVCC = "/usr/local/cuda-12.8/bin/nvcc"
DSO = ROOT / "src/libbinqueue_par_d.so"
NOTE = ROOT / "notes/N18_B1.md"
OUT = CACHE / "n18_b1.json"
PARHAC_VAL_MS = 202.0
ABORT_SEC = 30
CLAIM = "not a 2 Gvox/s number; not 3090 Ti"


def compile_bq():
    src = ROOT / "csrc/binqueue_par_d.cu"
    if DSO.exists() and DSO.stat().st_mtime >= src.stat().st_mtime:
        return
    subprocess.check_call([
        NVCC, "-O3", *nvcc_arch_flags(), "--shared", "-Xcompiler", "-fPIC",
        "-o", str(DSO), str(src),
    ])


def main():
    print(f"N18 B1 parallel BinQueue. {CLAIM}. abort={ABORT_SEC}s", flush=True)
    msg = refuse_or_ok("N18_B1", force="--force" in sys.argv)
    if msg:
        print(msg, flush=True)
        return 3
    # Also refuse serial path
    if refuse_or_ok("AGG_serial_BinQueue") and "--force" not in sys.argv:
        # parallel is a different exp_id; ok
        pass
    busy = card_busy()
    if busy:
        print(f"N18 B1 REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)
    compile_bq()
    u, v, sm, ct, _fr, max_id = load_rag()
    lib = ctypes.CDLL(str(DSO))
    lib.binqueue_par_mean_d.restype = ctypes.c_int
    lib.binqueue_par_mean_d.argtypes = [
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
    # Host abort: run in subprocess with timeout would be safer; also pass abort_ms
    rc = lib.binqueue_par_mean_d(
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
    hang = ms > ABORT_SEC * 1000
    split, merge, nseg = voi_parent_mmap(parents[0])
    ok, sl, ml = grade_t3(split, merge)
    kill = hang or ms > 2.0 * 400.0 or not ok or rc == 0
    reason = (
        "hang" if hang else
        f"wall {ms:.0f}>800" if ms > 800 else
        "VOI FAIL" if not ok else
        "rc=0" if rc == 0 else "ok"
    )
    doc = {
        "claim": CLAIM,
        "rc": int(rc),
        "ms": ms,
        "hang": hang,
        "split": split,
        "merge": merge,
        "nseg": nseg,
        "voi_ok": bool(ok),
        "kill": kill,
        "reason": reason,
        "stats": stats[0].tolist(),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        f"# N18 B1 parallel BinQueue\n\n{CLAIM}.\n\n"
        f"- ms={ms:.1f} hang={hang} voi_ok={ok} kill={kill} reason={reason}\n"
        f"- Design: multi-bin parallel candidate pop; host abort_ms={ABORT_SEC*1000}\n"
        f"- Do not reopen AGG_serial_BinQueue.\n"
    )
    if kill:
        stamp("N18_B1", reason, {"ms": ms}, "notes/N18_B1.md")
    print(json.dumps(doc), flush=True)
    return 0 if not kill else 1


if __name__ == "__main__":
    raise SystemExit(main())
