#!/usr/bin/env python3
"""A2: is the device agglomeration deterministic? Test E6s and E6t separately.

TASK.md line 118: "Deterministic: same input -> byte-identical labels, run to
run." A1 showed E6t drifting across runs (merges 27822/27781/27801 at T=0.2),
and because `load_rag()` reads a cached rag.npz the input was byte-identical,
so the drift originates inside the agglomeration itself.

This probe isolates it: run each path twice at T=0.3 only, on the same cached
RAG, and compare the returned parent arrays byte-for-byte. T=0.3-only keeps it
cheap (E6s ~1.3 s, E6t ~10 s per run) and no grading is performed, so it is
safe to run repeatedly on a shared GPU.

Writes no *_pass.txt stamp.
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from _agg_common import CACHE, load_rag  # noqa: E402
from e6r_parhac import DSO, compile_d  # noqa: E402

T = 0.3
NRUN = 2


def bind(lib):
    lib.parhac_paper_d_timed.restype = ctypes.c_int
    lib.parhac_paper_d_timed.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_double),
    ]


def one(lib, u, v, sm, ct, max_id):
    thrs = np.asarray([T], dtype=np.float64)
    parents = np.empty((1, max_id + 1), dtype=np.uint32)
    stats = np.zeros((1, 3), dtype=np.int64)
    device_ms = ctypes.c_double(0.0)
    t0 = time.perf_counter()
    rc = lib.parhac_paper_d_timed(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(1),
        ctypes.c_double(0.08),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.byref(device_ms),
    )
    wall = (time.perf_counter() - t0) * 1000.0
    return {
        "rc": int(rc),
        "parents": parents[0].copy(),
        "outer": int(stats[0, 0]),
        "merges": int(stats[0, 1]),
        "inner": int(stats[0, 2]),
        "device_ms": float(device_ms.value),
        "wall_ms": wall,
    }


def gpu_state():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def main():
    compile_d()
    u, v, sm, ct, _fr, max_id = load_rag()
    lib = ctypes.CDLL(str(DSO))
    bind(lib)
    print(f"A2 determinism probe @T={T}, {NRUN} runs per path. "
          f"GPU: {gpu_state()}", flush=True)
    print("A2 input is cached rag.npz, so the input is byte-identical by "
          "construction.", flush=True)

    out = {"threshold": T, "nrun": NRUN, "gpu": gpu_state(), "paths": {}}
    all_ok = True
    for path in ("E6s", "E6t"):
        if path == "E6t":
            os.environ["WATERZ_PAPER_E6T"] = "1"
        else:
            os.environ.pop("WATERZ_PAPER_E6T", None)
        runs = [one(lib, u, v, sm, ct, max_id) for _ in range(NRUN)]
        base = runs[0]["parents"]
        ident = all(np.array_equal(base, r["parents"]) for r in runs[1:])
        ndiff = [int((base != r["parents"]).sum()) for r in runs[1:]]
        merges = [r["merges"] for r in runs]
        inner = [r["inner"] for r in runs]
        ms = [round(r["device_ms"], 1) for r in runs]
        print(
            f"A2 {path}: byte_identical={ident} ndiff={ndiff} "
            f"merges={merges} inner={inner} device_ms={ms} (provisional)",
            flush=True,
        )
        out["paths"][path] = {
            "byte_identical": bool(ident),
            "ndiff_vs_run0": ndiff,
            "merges": merges,
            "inner": inner,
            "device_ms_provisional": ms,
            "rc": [r["rc"] for r in runs],
        }
        all_ok = all_ok and ident

    out["all_deterministic"] = bool(all_ok)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "a2_determinism.json").write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"A2 {'PASS' if all_ok else 'FAIL'} — TASK requires byte-identical "
        "labels run to run",
        flush=True,
    )
    return all_ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
