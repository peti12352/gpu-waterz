#!/usr/bin/env python3
"""E13: official UPGMA ParHAC. Stop when UPGMA Wmax <= T. No hybrid cut."""
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


def main():
    print_contract()
    subprocess.check_call([
        "g++", "-O3", "-DNDEBUG", "-shared", "-fPIC",
        "-o", str(SO), str(ROOT / "src/rac_agg.cpp"),
    ])
    z = np.load(CACHE / "rag.npz")
    u = np.ascontiguousarray(z["keys"][:, 0], dtype=np.uint32)
    v = np.ascontiguousarray(z["keys"][:, 1], dtype=np.uint32)
    sm = np.ascontiguousarray(z["stats"][:, 0], dtype=np.float64)
    ct = np.ascontiguousarray(np.rint(z["stats"][:, 1]), dtype=np.int64)
    fr = np.load(CACHE / "wz_fragments.npy")
    max_id = int(max(int(u.max()), int(v.max()), int(fr.max())))
    eps = 0.1
    print(f"E13 edges={len(u)} max_id={max_id} eps={eps} UPGMA", flush=True)
    thrs = np.asarray(AFF_THRESHOLDS, dtype=np.float64)
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(thrs), 3), dtype=np.int64)
    lib = ctypes.CDLL(str(SO))
    lib.parhac_upgma_cpu.restype = ctypes.c_int
    lib.parhac_upgma_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    t0 = time.time()
    rc = lib.parhac_upgma_cpu(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(thrs)),
        ctypes.c_double(eps),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    wall = time.time() - t0
    print(f"E13 rc={rc} wall={wall:.3f}", flush=True)
    if rc != 1:
        raise SystemExit(2)
    dest_dir = OUT / "e13_upgma"
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, thr in enumerate(AFF_THRESHOLDS):
        lab = extract_parent(fr, parents[i]).astype(np.uint32, copy=False)
        dest = dest_dir / f"mine_thr{thr}.h5"
        with h5py.File(dest, "w") as h:
            h.create_dataset("labels", data=lab)
        nseg = int((np.unique(lab) != 0).sum())
        print(
            f"  T={thr} nseg={nseg} layers={int(stats[i,0])} "
            f"merges={int(stats[i,1])} inner={int(stats[i,2])}",
            flush=True,
        )
        paths.append(str(dest))
    cmd = [str(PY), str(OUT / "baseline/run_baseline.py"), "--candidate", *paths]
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(OUT), capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.stderr:
        sys.stderr.write(r.stderr)
    ok = "ACCURACY GATE: PASS" in r.stdout
    print(f"E13 UPGMA {'PASS' if ok else 'FAIL'} wall={wall:.3f}")
    if not ok:
        print("E13 killed — do not invent a contact-mean cut of the UPGMA dendrogram")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
