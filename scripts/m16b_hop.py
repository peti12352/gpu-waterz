#!/usr/bin/env python3
"""M16b: M16 + hop-k repulsive fragment pairs from 3-channel aff."""
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
AFF = OUT / "cremiA_val/affinity.h5"
SO = ROOT / "src/librac_agg.so"
PY = ROOT / ".venv/bin/python"


def main():
    print_contract()
    subprocess.check_call(
        ["g++", "-O3", "-DNDEBUG", "-shared", "-fPIC", "-o", str(SO), str(ROOT / "src/rac_agg.cpp")]
    )
    z = np.load(CACHE / "rag.npz")
    u = np.ascontiguousarray(z["keys"][:, 0], dtype=np.uint32)
    v = np.ascontiguousarray(z["keys"][:, 1], dtype=np.uint32)
    sm = np.ascontiguousarray(z["stats"][:, 0], dtype=np.float64)
    ct = np.ascontiguousarray(np.rint(z["stats"][:, 1]), dtype=np.int64)
    fr = np.ascontiguousarray(np.load(CACHE / "wz_fragments.npy"), dtype=np.uint32)
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    Z, Y, X = aff.shape[1:]
    max_id = int(max(int(u.max()), int(v.max()), int(fr.max())))
    hops = np.asarray([2, 4, 8], dtype=np.int32)
    print(f"M16b edges={len(u)} hops={list(hops)} max_id={max_id}", flush=True)
    thrs = np.asarray(AFF_THRESHOLDS, dtype=np.float64)
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(thrs), 3), dtype=np.int64)
    lib = ctypes.CDLL(str(SO))
    lib.mutex_rag_hop_cpu.restype = ctypes.c_int
    lib.mutex_rag_hop_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    t0 = time.time()
    rc = lib.mutex_rag_hop_cpu(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        aff.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        fr.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        Z, Y, X,
        hops.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(len(hops)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(thrs)),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    wall = time.time() - t0
    print(f"M16b rc={rc} wall={wall:.3f}", flush=True)
    if rc != 1:
        raise SystemExit(2)
    dest_dir = OUT / "m16b_hop"
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, thr in enumerate(AFF_THRESHOLDS):
        lab = extract_parent(fr, parents[i]).astype(np.uint32, copy=False)
        dest = dest_dir / f"mine_thr{thr}.h5"
        with h5py.File(dest, "w") as h:
            h.create_dataset("labels", data=lab)
        print(f"  T={thr} nseg={int((np.unique(lab)!=0).sum())} "
              f"merges={int(stats[i,1])} mutex={int(stats[i,2])}", flush=True)
        paths.append(str(dest))
    cmd = [str(PY), str(OUT / "baseline/run_baseline.py"), "--candidate", *paths]
    r = subprocess.run(cmd, cwd=str(OUT), capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    ok = "ACCURACY GATE: PASS" in r.stdout
    print(f"M16b {'PASS' if ok else 'FAIL'} wall={wall:.3f}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
