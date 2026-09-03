#!/usr/bin/env python3
"""P0aa: E6s-a CUDA-event phase split + per-layer outer counts. T=0.3 only."""
from __future__ import annotations

import ctypes
import json
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import CACHE, compile_so, load_rag  # noqa: E402
from e6r_parhac import DSO, compile_d  # noqa: E402

PHASES = (
    "compact", "propose", "pack", "sort", "accept",
    "compress", "freeze", "color", "memset", "d2h", "host",
)


def main():
    compile_so()
    if DSO.exists():
        DSO.unlink()
    compile_d()
    u, v, sm, ct, fr, max_id = load_rag()
    lib = ctypes.CDLL(str(DSO))
    lib.parhac_e6s_p0aa.restype = ctypes.c_int
    lib.parhac_e6s_p0aa.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int64),
    ]
    thrs = np.asarray([0.3], dtype=np.float64)
    parents = np.empty((1, max_id + 1), dtype=np.uint32)
    stats = np.zeros((1, 3), dtype=np.int64)
    phase = np.zeros(11, dtype=np.float64)
    n_layer = ctypes.c_int(0)
    layer_outers = np.zeros(64, dtype=np.int32)
    layer_merges = np.zeros(64, dtype=np.int32)
    layer_first_zero = np.full(64, -1, dtype=np.int32)
    work = np.zeros(2, dtype=np.int64)
    print("P0aa T=0.3 E6s-a phase split", flush=True)
    t0 = time.perf_counter()
    rc = lib.parhac_e6s_p0aa(
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
        phase.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(n_layer),
        layer_outers.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        layer_merges.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        layer_first_zero.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        work.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    nl = int(n_layer.value)
    phases = {k: float(phase[i]) for i, k in enumerate(PHASES)}
    summary = {
        "rc": int(rc),
        "wall_ms": wall_ms,
        "ninner": int(stats[0, 2]),
        "nmerge": int(stats[0, 1]),
        "nouter": int(stats[0, 0]),
        "n_layer": nl,
        "phases": phases,
        "phase_sum_ms": float(sum(phases.values())),
        "layer_outers": [int(x) for x in layer_outers[:nl]],
        "layer_merges": [int(x) for x in layer_merges[:nl]],
        "hit_64": all(int(x) == 64 for x in layer_outers[:nl]) if nl else False,
        "budget_ref_ms": 1597.0,
        "wall_vs_ref": wall_ms / 1597.0 if wall_ms else 0.0,
    }
    # A2. first_zero[i] is the outer index at which layer i ran out of edges
    # above TL. An exit there is a no-op for the result, so the outer count an
    # A1-style break would produce is first_zero+1 where it fired and the full
    # cap where it never did.
    fz = [int(x) for x in layer_first_zero[:nl]]
    obs = [int(x) for x in layer_outers[:nl]]
    proj = [(f + 1) if f >= 0 else o for f, o in zip(fz, obs)]
    summary["layer_first_zero"] = fz
    summary["outers_now"] = sum(obs)
    summary["outers_if_exit"] = sum(proj)
    summary["outer_reduction"] = (
        sum(obs) / sum(proj) if sum(proj) else 0.0
    )
    summary["sum_nlive"] = int(work[0])
    summary["sum_above"] = int(work[1])
    summary["above_frac"] = (
        float(work[1]) / float(work[0]) if work[0] else 0.0
    )
    out = CACHE / "p0aa_e6s.json"
    out.write_text(json.dumps(summary, indent=2))
    print(
        f"P0aa wall_ms={wall_ms:.2f} n_layer={nl} ninner={summary['ninner']} "
        f"nmerge={summary['nmerge']} hit_64={summary['hit_64']}",
        flush=True,
    )
    for k, v in phases.items():
        print(f"  {k:10s} {v:8.2f} ms", flush=True)
    print(
        f"A2 outers {summary['outers_now']} -> {summary['outers_if_exit']} "
        f"({summary['outer_reduction']:.2f}x)",
        flush=True,
    )
    print(f"A2 first_zero per layer: {summary['layer_first_zero']}", flush=True)
    print(
        f"A2 sum_nlive={summary['sum_nlive']} sum_above={summary['sum_above']} "
        f"above_frac={summary['above_frac']:.6f}",
        flush=True,
    )
    print(f"P0aa wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
