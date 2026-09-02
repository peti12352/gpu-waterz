#!/usr/bin/env python3
"""P0z: E6s-a StarMerge work (n_gc, n_active, n_dirty) at T=0.3. No merge change."""
from __future__ import annotations

import ctypes
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import CACHE, compile_so, load_rag  # noqa: E402
from e6r_parhac import DSO, compile_d  # noqa: E402

HIST_CAP = 8192


def _bind(lib):
    lib.parhac_e6s_profile.restype = ctypes.c_int
    lib.parhac_e6s_profile.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]


def main():
    compile_so()
    if DSO.exists():
        DSO.unlink()
    compile_d()
    u, v, sm, ct, fr, max_id = load_rag()
    lib = ctypes.CDLL(str(DSO))
    _bind(lib)
    thrs = np.asarray([0.3], dtype=np.float64)
    parents = np.empty((1, max_id + 1), dtype=np.uint32)
    stats = np.zeros((1, 3), dtype=np.int64)
    hist_nlive = np.zeros(HIST_CAP, dtype=np.int64)
    hist_nprop = np.zeros(HIST_CAP, dtype=np.int64)
    hist_nmerge = np.zeros(HIST_CAP, dtype=np.int64)
    hist_ngc = np.zeros(HIST_CAP, dtype=np.int64)
    hist_nact = np.zeros(HIST_CAP, dtype=np.int64)
    hist_ndirty = np.zeros(HIST_CAP, dtype=np.int64)
    hist_nstar = np.zeros(HIST_CAP, dtype=np.int64)
    hist_n = ctypes.c_int(0)
    n_layer = ctypes.c_int(0)
    print("P0z T=0.3-only E6s-a instrument (no merge change)", flush=True)
    t0 = time.perf_counter()
    rc = lib.parhac_e6s_profile(
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
        hist_nlive.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        hist_nprop.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        hist_nmerge.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        hist_ngc.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        hist_nact.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        hist_ndirty.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        hist_nstar.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int(HIST_CAP),
        ctypes.byref(hist_n),
        ctypes.byref(n_layer),
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    n = int(hist_n.value)
    ngc = hist_ngc[:n].astype(np.float64)
    nact = hist_nact[:n].astype(np.float64)
    ndirty = hist_ndirty[:n].astype(np.float64)
    nstar = hist_nstar[:n].astype(np.float64)
    nlive = hist_nlive[:n].astype(np.float64)
    nprop = hist_nprop[:n].astype(np.float64)
    nmerge = hist_nmerge[:n].astype(np.float64)
    total_dirty = int(ndirty.sum())
    total_star = int(nstar.sum())
    mean_gc = float(ngc.mean()) if n else 0.0
    go = mean_gc <= 80000.0 and total_dirty <= 50_000_000
    stop = mean_gc > 200000.0 or total_dirty > 150_000_000
    if go:
        branch = "E6t"
    elif stop:
        branch = "STOP Type C"
    else:
        branch = "E6t-marginal"
    summary = {
        "rc": int(rc),
        "wall_ms": wall_ms,
        "n_layer": int(n_layer.value),
        "ninner": int(stats[0, 2]),
        "nmerge": int(stats[0, 1]),
        "nouter": int(stats[0, 0]),
        "hist_n": n,
        "ngc_mean": mean_gc,
        "ngc_p50": float(np.median(ngc)) if n else 0.0,
        "ngc_max": int(ngc.max()) if n else 0,
        "nact_mean": float(nact.mean()) if n else 0.0,
        "ndirty_mean": float(ndirty.mean()) if n else 0.0,
        "ndirty_total": total_dirty,
        "nstar_mean": float(nstar.mean()) if n else 0.0,
        "nstar_total": total_star,
        "nlive_mean": float(nlive.mean()) if n else 0.0,
        "nprop_mean": float(nprop.mean()) if n else 0.0,
        "nmerge_mean": float(nmerge.mean()) if n else 0.0,
        "n_empty_prop": int((nprop == 0).sum()) if n else 0,
        "go": go,
        "stop": stop,
        "branch": branch,
        "budget_ref_ms": 1597.0,
        "wall_vs_ref": wall_ms / 1597.0 if wall_ms else 0.0,
    }
    out = CACHE / "p0z_starmarge.json"
    out.write_text(json.dumps(summary, indent=2))
    print(
        f"P0z wall_ms={wall_ms:.2f} ninner={summary['ninner']} n_layer={summary['n_layer']} "
        f"ngc_mean={mean_gc:.1f} ndirty_total={total_dirty} nstar_total={total_star} "
        f"branch={branch}",
        flush=True,
    )
    print(f"P0z wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
