#!/usr/bin/env python3
"""N15 agg gate: two-run ParHAC T=0.3 VOI vs GT, parent array_equal.

Env must be set before DSO load. No 2.16. Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import ctypes
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from e6r_parhac import DSO, compile_d  # noqa: E402
from _agg_common import load_rag  # noqa: E402
from t3_memsafe import grade_t3, voi_parent_mmap  # noqa: E402

AGG_BASE = 2234.0503642335534
AGG_GATE = AGG_BASE / 1.2


def val_t03_two():
    compile_d()
    lib = ctypes.CDLL(str(DSO))
    lib.parhac_paper_d.restype = ctypes.c_int
    lib.parhac_paper_d.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    u, v, sm, ct, _fr, max_id = load_rag()
    thrs = np.asarray([0.3], dtype=np.float64)
    eps = float(os.environ.get("WATERZ_AGG_EPS") or "0.40")

    def once():
        parents = np.empty((1, max_id + 1), dtype=np.uint32)
        stats = np.zeros((1, 3), dtype=np.int64)
        rc = lib.parhac_paper_d(
            u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_int64(len(u)),
            thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(1),
            ctypes.c_double(eps),
            parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(max_id),
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        )
        split, merge, nseg = voi_parent_mmap(parents[0])
        ok, sl, ml = grade_t3(split, merge)
        return parents[0].copy(), {
            "rc": int(rc),
            "split": split,
            "merge": merge,
            "nseg": nseg,
            "ok": bool(ok),
            "limit_split": sl,
            "limit_merge": ml,
            "inner": int(stats[0, 2]),
            "merges": int(stats[0, 1]),
        }

    p1, m1 = once()
    p2, m2 = once()
    return {
        "ok": bool(m1["ok"] and m2["ok"]),
        "run2_array_equal": bool(np.array_equal(p1, p2)),
        "m1": m1,
        "m2": m2,
        "env": {
            "WATERZ_FUSE_DIRTY": os.environ.get("WATERZ_FUSE_DIRTY"),
            "WATERZ_STICKY_SZ0": os.environ.get("WATERZ_STICKY_SZ0"),
            "WATERZ_MAX_OUTER": os.environ.get("WATERZ_MAX_OUTER"),
            "WATERZ_AGG_LEVERS": os.environ.get("WATERZ_AGG_LEVERS"),
            "WATERZ_AGG_EPS": os.environ.get("WATERZ_AGG_EPS"),
        },
    }


if __name__ == "__main__":
    print(json.dumps(val_t03_two()), flush=True)
