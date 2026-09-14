#!/usr/bin/env python3
"""Device paper-ParHAC gate. LOCK iff VOI PASS and T=0.3 CUDA-event AGG≤50ms.

G9 stays BLOCKED: no 3090 Ti. A 5090 number is not 2 Gvox/s.
"""
from __future__ import annotations

import ctypes
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import (  # noqa: E402
    AFF_THRESHOLDS, CACHE, compile_so, grade_parents, load_rag, stamp,
)
from e6r_parhac import DSO, compile_d  # noqa: E402
from task_gate import print_contract  # noqa: E402

E6S_T03_MS = 1597.0


def _apply_p0ab_cap():
    path = CACHE / "p0ab_outer_cap.json"
    if not path.exists():
        return 64
    try:
        chosen = int(json.loads(path.read_text()).get("chosen_cap", 64))
    except (json.JSONDecodeError, TypeError, ValueError):
        return 64
    os.environ["WATERZ_MAX_OUTER"] = str(chosen)
    return chosen


def _bind_timed(lib):
    lib.parhac_paper_d_timed.restype = ctypes.c_int
    lib.parhac_paper_d_timed.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_double),
    ]


def _run_timed(lib, u, v, sm, ct, thrs, max_id):
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(thrs), 3), dtype=np.int64)
    device_ms = ctypes.c_double(0.0)
    t0 = time.perf_counter()
    rc = lib.parhac_paper_d_timed(
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
        ctypes.byref(device_ms),
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return rc, parents, stats, float(device_ms.value), wall_ms


def main():
    print_contract()
    compile_so()
    if DSO.exists():
        DSO.unlink()
    compile_d()
    cap = _apply_p0ab_cap()
    print(
        f"G9 BLOCKED: no 3090 Ti. WATERZ_MAX_OUTER={cap}. "
        "5090 CUDA-event is not a 2 Gvox/s claim.",
        flush=True,
    )
    u, v, sm, ct, fr, max_id = load_rag()
    lib = ctypes.CDLL(str(DSO))
    _bind_timed(lib)
    t03 = np.asarray([0.3], dtype=np.float64)
    print("E6uvw T=0.3 CUDA-event (H2D excluded)", flush=True)
    rc03, _, stats03, ms03, wall03 = _run_timed(lib, u, v, sm, ct, t03, max_id)
    print(
        f"E6uvw T=0.3 rc={rc03} device_ms={ms03:.2f} wall_ms={wall03:.2f} "
        f"outer={list(map(int, stats03[:, 0]))} inner={list(map(int, stats03[:, 2]))} "
        f"merges={list(map(int, stats03[:, 1]))} vs_e6s={ms03 / E6S_T03_MS:.3f}",
        flush=True,
    )
    if ms03 > E6S_T03_MS and not os.environ.get("WATERZ_PAPER_E6S"):
        print(
            f"E6v implementation slower than E6s {E6S_T03_MS:.0f} ms; "
            "retry paper path as E6s (WATERZ_PAPER_E6S=1)",
            flush=True,
        )
        os.environ["WATERZ_PAPER_E6S"] = "1"
        rc03, _, stats03, ms03, wall03 = _run_timed(lib, u, v, sm, ct, t03, max_id)
        print(
            f"E6s-fallback T=0.3 rc={rc03} device_ms={ms03:.2f} wall_ms={wall03:.2f} "
            f"merges={list(map(int, stats03[:, 1]))}",
            flush=True,
        )
    thrs = np.asarray(AFF_THRESHOLDS, dtype=np.float64)
    print("E6uvw four-T VOI", flush=True)
    rc, parents, stats, ms4, wall4 = _run_timed(lib, u, v, sm, ct, thrs, max_id)
    print(
        f"E6uvw four-T rc={rc} device_ms={ms4:.2f} wall_ms={wall4:.2f} "
        f"inner={list(map(int, stats[:, 2]))} merges={list(map(int, stats[:, 1]))}",
        flush=True,
    )
    if rc != 1:
        stamp("e6r", False, f"e6uvw rc={rc}")
        return False
    if os.environ.get("WATERZ_E6S_NOGRADE"):
        stamp("e6r", False, f"NOGRADE T03_ms={ms03:.2f}")
        return False
    buf = io.StringIO()
    with redirect_stdout(buf):
        ok = grade_parents(parents, fr, "E6uvw", "e6uvw_parhac")
    sys.stdout.write(buf.getvalue())
    path = "E6s" if os.environ.get("WATERZ_PAPER_E6S") else "E6uvw"
    if ok and ms03 <= 50.0:
        stamp("e6r", True, f"LOCK device {path} ε=0.08 T03_ms={ms03:.2f}")
        print(
            f"{path} PASS LOCK T03_ms={ms03:.2f}. "
            "G9 BLOCKED: no 3090 Ti. Do not write 2 Gvox/s.",
            flush=True,
        )
        return True
    extra = f"{'PASS' if ok else 'FAIL'} T03_ms={ms03:.2f} budget=50"
    stamp("e6r", False, extra)
    print(
        f"{path} {extra}. G9 BLOCKED: no 3090 Ti. Planning target ~10 ms.",
        flush=True,
    )
    return False


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
