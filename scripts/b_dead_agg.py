#!/usr/bin/env python3
"""B: delete unused cu/cv/csm/cct in parhac_e6s_dev. Identity + agg_mem_peak.

Runs paper ParHAC on rag.npz (val). Compares parents to a pre-delete snapshot
if present, otherwise writes the snapshot. Rebuilds libparhac_d.so.
Idle-5090. Not a 2 Gvox/s claim.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import AFF_THRESHOLDS, CACHE, load_rag  # noqa: E402
from p1_make_big_indep import card_busy, gpu_state  # noqa: E402

NVCC = "/usr/local/cuda-12.8/bin/nvcc"
DSO = ROOT / "src/libparhac_d.so"
SRC = ROOT / "csrc/parhac_d.cu"
SNAP = CACHE / "b_dead_agg_parents.npy"
OUT = CACHE / "b_dead_agg.json"
GIB = 1024.0 ** 3
EPS = 0.08


def compile_d(force=False):
    if not force and DSO.exists() and DSO.stat().st_mtime >= SRC.stat().st_mtime:
        return
    subprocess.check_call([
        NVCC, "-O3", "-arch=sm_120", "--shared", "-Xcompiler", "-fPIC",
        "-o", str(DSO), str(SRC),
    ])


def run_parhac(u, v, sm, ct, max_id):
    thrs = np.asarray(AFF_THRESHOLDS, dtype=np.float64)
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(thrs), 3), dtype=np.int64)
    lib = ctypes.CDLL(str(DSO))
    lib.agg_mem_reset.restype = None
    lib.agg_mem_peak.restype = ctypes.c_size_t
    lib.agg_mem_reset()
    lib.parhac_paper_d.restype = ctypes.c_int
    lib.parhac_paper_d.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    rc = lib.parhac_paper_d(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(thrs)),
        ctypes.c_double(EPS),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    if rc != 1:
        raise RuntimeError(f"parhac_paper_d rc={rc}")
    peak = int(lib.agg_mem_peak())
    return parents, peak, stats


def fp_parents(parents):
    return hashlib.sha256(np.ascontiguousarray(parents).tobytes()).hexdigest()[:32]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "after"
    print(f"B dead-agg mode={mode}. Not a 2 Gvox/s claim.", flush=True)
    busy = card_busy()
    if busy:
        print(f"B REFUSE card busy: {busy}", flush=True)
        raise SystemExit(2)
    print(f"GPU idle: {gpu_state()}", flush=True)
    if mode == "after":
        compile_d(force=True)
    u, v, sm, ct, _fr, max_id = load_rag()
    parents, peak, stats = run_parhac(u, v, sm, ct, max_id)
    fp = fp_parents(parents)
    identical = None
    if mode == "before":
        np.save(SNAP, parents)
        identical = True
    elif SNAP.exists():
        ref = np.load(SNAP)
        identical = bool(ref.shape == parents.shape and np.array_equal(ref, parents))
    nedge = int(len(u))
    dead_bytes = nedge * (4 + 4 + 8 + 8)
    doc = {
        "claim": "not a 2 Gvox/s number",
        "mode": mode,
        "nedge": nedge,
        "max_id": int(max_id),
        "peak_bytes": peak,
        "peak_gib": peak / GIB,
        "dead_24b_per_edge_bytes": dead_bytes,
        "parents_fp": fp,
        "identical_to_before": identical,
        "stats": stats.tolist(),
        "gpu": gpu_state(),
        "pass": bool(identical is not False),
    }
    dest = CACHE / f"b_dead_agg_{mode}.json"
    dest.write_text(json.dumps(doc, indent=2) + "\n")
    if mode == "after":
        OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(
        f"B dead-agg {mode} peak={peak / GIB:.3f} GiB fp={fp} "
        f"ident={identical} {'PASS' if doc['pass'] else 'FAIL'} -> {dest}",
        flush=True,
    )
    return 0 if doc["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
