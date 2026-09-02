#!/usr/bin/env python3
"""E2: existing Y2 matching at paper ε values only.

ε ∈ {0.1, 1.0, 0.0} on cached host-S1 RAG.
ε=0.01 already TASK-PASS in LOG (Y2); not re-run (501s).
No additive bands.
PASS = shipped run_baseline.py prints ACCURACY GATE: PASS.
"""
from __future__ import annotations

import ctypes
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ref_cpu import extract_parent  # noqa: E402
from task_gate import AFF_THRESHOLDS, print_contract  # noqa: E402

CACHE = ROOT / "data/cache"
OUT = ROOT / "data/ws_bounty"
SO = ROOT / "src/librac_agg.so"
PY = ROOT / ".venv/bin/python"
# Paper Figure 1: ParHac-0.1, 1.0, 0.0 (ParHAC-E: ε=0, TL=Wmax).
EPS_LIST = [0.10, 1.0, 0.0]


def compile_so():
    subprocess.check_call(
        ["g++", "-O3", "-DNDEBUG", "-shared", "-fPIC", "-o", str(SO), str(ROOT / "src/rac_agg.cpp")]
    )


def load_rag():
    z = np.load(CACHE / "rag.npz")
    u = np.ascontiguousarray(z["keys"][:, 0], dtype=np.uint32)
    v = np.ascontiguousarray(z["keys"][:, 1], dtype=np.uint32)
    sm = np.ascontiguousarray(z["stats"][:, 0], dtype=np.float64)
    ct = np.ascontiguousarray(np.rint(z["stats"][:, 1]), dtype=np.int64)
    return u, v, sm, ct


def run_one(u, v, sm, ct, fr, max_id, eps):
    thrs = np.asarray(AFF_THRESHOLDS, dtype=np.float64)
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(thrs), 3), dtype=np.int64)
    lib = ctypes.CDLL(str(SO))
    lib.parhac_agg_cpu.restype = ctypes.c_int
    lib.parhac_agg_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    t0 = time.time()
    rc = lib.parhac_agg_cpu(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(thrs)),
        ctypes.c_double(eps),
        ctypes.c_int(0),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    wall = time.time() - t0
    if rc != 1:
        raise RuntimeError("parhac_agg_cpu failed")
    dest_dir = OUT / f"e2_eps{eps}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, thr in enumerate(AFF_THRESHOLDS):
        lab = extract_parent(fr, parents[i]).astype(np.uint32, copy=False)
        dest = dest_dir / f"mine_thr{thr}.h5"
        with h5py.File(dest, "w") as h:
            h.create_dataset("labels", data=lab)
        nseg = int((np.unique(lab) != 0).sum())
        print(
            f"  T={thr} nseg={nseg} rounds={int(stats[i,0])} "
            f"merges={int(stats[i,1])}",
            flush=True,
        )
        paths.append(str(dest))
    print(f"  wall={wall:.3f}", flush=True)
    cmd = [
        str(PY),
        str(OUT / "baseline/run_baseline.py"),
        "--candidate",
        *paths,
    ]
    r = subprocess.run(cmd, cwd=str(OUT), capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.stderr:
        sys.stderr.write(r.stderr)
    return "ACCURACY GATE: PASS" in r.stdout, wall


def main():
    print_contract()
    print("E2 skip ε=0.01: LOG Y2 ACCURACY GATE PASS (0.2 merge 0.3485 vs 0.3525)")
    compile_so()
    u, v, sm, ct = load_rag()
    fr = np.load(CACHE / "wz_fragments.npy")
    max_id = int(max(int(u.max()), int(v.max()), int(fr.max())))
    print(f"edges={len(u)} max_id={max_id}", flush=True)
    any_pass = False
    for eps in EPS_LIST:
        print(f"E2 eps={eps}", flush=True)
        ok, wall = run_one(u, v, sm, ct, fr, max_id, eps)
        print(f"E2 eps={eps} {'PASS' if ok else 'FAIL'} wall={wall:.3f}", flush=True)
        any_pass = any_pass or ok
    if not any_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
