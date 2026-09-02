#!/usr/bin/env python3
"""P0y: E6r CUDA-event breakdown + T=0.3-only on-device time + inner histogram."""
from __future__ import annotations

import ctypes
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys_path_insert = ROOT / "scripts"
import sys

sys.path.insert(0, str(sys_path_insert))
from _agg_common import CACHE, compile_so, load_rag  # noqa: E402
from e6r_parhac import DSO, compile_d, NVCC  # noqa: E402

HIST_CAP = 8192


def _bind(lib):
    lib.parhac_paper_d_profile.restype = ctypes.c_int
    lib.parhac_paper_d_profile.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64),
    ]


def _run(lib, u, v, sm, ct, thrs, max_id, skip_debug):
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(thrs), 3), dtype=np.int64)
    phase = np.zeros(6, dtype=np.float64)
    hist_nlive = np.zeros(HIST_CAP, dtype=np.int64)
    hist_nprop = np.zeros(HIST_CAP, dtype=np.int64)
    hist_nmerge = np.zeros(HIST_CAP, dtype=np.int64)
    hist_n = ctypes.c_int(0)
    t0 = time.perf_counter()
    rc = lib.parhac_paper_d_profile(
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
        phase.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        hist_nlive.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        hist_nprop.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        hist_nmerge.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int(HIST_CAP),
        ctypes.byref(hist_n),
        ctypes.c_int(1 if skip_debug else 0),
        None,  # hist_nact: not needed here, and it costs an extra sweep
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    n = int(hist_n.value)
    return {
        "rc": int(rc),
        "wall_ms": wall_ms,
        "phase_ms": {
            "compact": float(phase[0]),
            "propose": float(phase[1]),
            "accept": float(phase[2]),
            "d2h": float(phase[3]),
            "memset": float(phase[4]),
            "debug": float(phase[5]),
        },
        "stats_outer": [int(x) for x in stats[:, 0]],
        "stats_merges": [int(x) for x in stats[:, 1]],
        "stats_inner": [int(x) for x in stats[:, 2]],
        "n_hist": n,
        "hist_nlive": [int(x) for x in hist_nlive[:n]],
        "hist_nprop": [int(x) for x in hist_nprop[:n]],
        "hist_nmerge": [int(x) for x in hist_nmerge[:n]],
    }


def main():
    compile_so()
    compile_d()
    u, v, sm, ct, fr, max_id = load_rag()
    lib = ctypes.CDLL(str(DSO))
    _bind(lib)
    t03 = np.asarray([0.3], dtype=np.float64)
    print("P0y T=0.3-only skip_debug=1 (RAG H2D then device AGG)", flush=True)
    r03 = _run(lib, u, v, sm, ct, t03, max_id, skip_debug=True)
    print(
        f"P0y T=0.3 wall_ms={r03['wall_ms']:.2f} "
        f"compact={r03['phase_ms']['compact']:.2f} "
        f"propose={r03['phase_ms']['propose']:.2f} "
        f"accept={r03['phase_ms']['accept']:.2f} "
        f"d2h={r03['phase_ms']['d2h']:.2f} "
        f"memset={r03['phase_ms']['memset']:.2f} "
        f"ninner={r03['stats_inner']} n_hist={r03['n_hist']}",
        flush=True,
    )
    compact_d2h_accept = (
        r03["phase_ms"]["compact"]
        + r03["phase_ms"]["d2h"]
        + r03["phase_ms"]["accept"]
    )
    branch = "E6s-c-only" if compact_d2h_accept <= 50.0 else "E6s-a-then-b"
    print(
        f"P0y compact+d2h+accept={compact_d2h_accept:.2f} ms branch={branch}",
        flush=True,
    )
    nlive = np.asarray(r03["hist_nlive"], dtype=np.float64)
    nprop = np.asarray(r03["hist_nprop"], dtype=np.float64)
    nmerge = np.asarray(r03["hist_nmerge"], dtype=np.float64)
    summary = {
        "t03_wall_ms": r03["wall_ms"],
        "t03_phase_ms": r03["phase_ms"],
        "t03_compact_d2h_accept_ms": compact_d2h_accept,
        "branch": branch,
        "t03_ninner": r03["stats_inner"],
        "t03_nmerge": r03["stats_merges"],
        "hist_n": r03["n_hist"],
        "nlive_mean": float(nlive.mean()) if nlive.size else 0.0,
        "nlive_max": int(nlive.max()) if nlive.size else 0,
        "nlive_p50": float(np.median(nlive)) if nlive.size else 0.0,
        "nprop_mean": float(nprop.mean()) if nprop.size else 0.0,
        "nmerge_mean": float(nmerge.mean()) if nmerge.size else 0.0,
        "nmerge_max": int(nmerge.max()) if nmerge.size else 0,
    }
    out = CACHE / "p0y_e6r.json"
    out.write_text(json.dumps({**summary, "t03": r03}, indent=2))
    print(f"P0y wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
