#!/usr/bin/env python3
"""P0ab: raise E6s-a outer cap. T=0.3 nmerge/n_roots first; four-T VOI if no drift."""
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
from _agg_common import CACHE, compile_so, grade_parents, load_rag  # noqa: E402
from e6r_parhac import DSO, compile_d  # noqa: E402
from task_gate import AFF_THRESHOLDS  # noqa: E402

E6S_T03_MERGE = 1853410
E6S_NSEG = (294162, 321994, 345065, 379093)


def _bind(lib):
    lib.parhac_e6s_cap.restype = ctypes.c_int
    lib.parhac_e6s_cap.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64), ctypes.c_int,
    ]


def _run(lib, u, v, sm, ct, thrs, max_id, cap):
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(thrs), 3), dtype=np.int64)
    t0 = time.perf_counter()
    rc = lib.parhac_e6s_cap(
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
        ctypes.c_int(cap),
    )
    ms = (time.perf_counter() - t0) * 1000.0
    return rc, parents, stats, ms


def main():
    compile_so()
    compile_d()
    u, v, sm, ct, fr, max_id = load_rag()
    lib = ctypes.CDLL(str(DSO))
    _bind(lib)
    caps = [256, 512]
    env = os.environ.get("WATERZ_P0AB_CAPS")
    if env:
        caps = [int(x) for x in env.split(",") if x.strip()]
    t03 = np.asarray([0.3], dtype=np.float64)
    rows = []
    keep_cap = None
    for cap in caps:
        print(f"P0ab cap={cap} T=0.3", flush=True)
        rc, parents, stats, ms = _run(lib, u, v, sm, ct, t03, max_id, cap)
        nmerge = int(stats[0, 1])
        ninner = int(stats[0, 2])
        nouter = int(stats[0, 0])
        roots = int(np.unique(parents[0][1:]).size)
        drift = abs(nmerge - E6S_T03_MERGE)
        n_layer_est = nouter // cap if cap and nouter % cap == 0 else None
        row = {
            "cap": cap, "rc": int(rc), "ms": ms,
            "nmerge": nmerge, "ninner": ninner, "nouter": nouter,
            "n_layer_est": n_layer_est,
            "n_roots": roots, "merge_drift": drift,
        }
        print(
            f"P0ab cap={cap} ms={ms:.2f} nmerge={nmerge} drift={drift} "
            f"ninner={ninner} nouter={nouter} roots={roots}",
            flush=True,
        )
        rows.append(row)
        if drift == 0:
            keep_cap = cap
    chosen = keep_cap
    voi = None
    if chosen is None:
        # no exact merge match — still try smallest-drift cap only if drift==0 was required
        print("P0ab no cap matched E6s nmerge; revert to 64", flush=True)
        chosen = 64
    elif os.environ.get("WATERZ_P0AB_NOGRADE"):
        print("P0ab skip four-T", flush=True)
    else:
        print(f"P0ab four-T VOI cap={chosen}", flush=True)
        thrs = np.asarray(AFF_THRESHOLDS, dtype=np.float64)
        rc, parents, stats, ms4 = _run(lib, u, v, sm, ct, thrs, max_id, chosen)
        buf = io.StringIO()
        with redirect_stdout(buf):
            ok = grade_parents(parents, fr, f"P0ab_c{chosen}", f"p0ab_c{chosen}")
        sys.stdout.write(buf.getvalue())
        voi = {"ok": ok, "ms4": ms4, "nmerge": [int(x) for x in stats[:, 1]]}
        if not ok:
            print("P0ab VOI FAIL; revert cap 64", flush=True)
            chosen = 64
    out = {
        "rows": rows,
        "chosen_cap": chosen,
        "voi": voi,
        "e6s_t03_merge": E6S_T03_MERGE,
        "e6s_nseg": list(E6S_NSEG),
    }
    path = CACHE / "p0ab_outer_cap.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"P0ab chosen_cap={chosen} wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
