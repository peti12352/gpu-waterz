#!/usr/bin/env python3
"""Y2: ParHAC matching (1+eps) under waterz means. Same eps all T."""
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

CACHE = ROOT / "data/cache"
OUT = ROOT / "data/ws_bounty"
SO = ROOT / "src/librac_agg.so"
THRS = [0.2, 0.3, 0.4, 0.5]
EPS_LIST = [0.01, 0.02, 0.05, 0.10]
DELTA_LIST = [1.0 / 1024.0, 1.0 / 4096.0]


def compile_so():
    src = ROOT / "src/rac_agg.cpp"
    subprocess.check_call(
        ["g++", "-O3", "-DNDEBUG", "-shared", "-fPIC", "-o", str(SO), str(src)]
    )


def load_rag():
    z = np.load(CACHE / "rag.npz")
    u = np.ascontiguousarray(z["keys"][:, 0], dtype=np.uint32)
    v = np.ascontiguousarray(z["keys"][:, 1], dtype=np.uint32)
    sm = np.ascontiguousarray(z["stats"][:, 0], dtype=np.float64)
    ct = np.ascontiguousarray(np.rint(z["stats"][:, 1]), dtype=np.int64)
    return u, v, sm, ct


def nseg(lab):
    return int((np.unique(lab) != 0).sum())


def max_comp_frac(fr, parent):
    lab = extract_parent(fr, parent)
    vals, cnt = np.unique(lab, return_counts=True)
    if len(vals) and vals[0] == 0:
        cnt = cnt[1:]
    tot = int((fr != 0).sum())
    if tot == 0 or len(cnt) == 0:
        return 0.0
    return float(cnt.max()) / float(tot)


def run_one(u, v, sm, ct, fr, max_id, eps, additive):
    thrs = np.asarray(THRS, dtype=np.float64)
    parents = np.empty((len(THRS), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(THRS), 3), dtype=np.int64)
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
        ctypes.c_int(len(THRS)),
        ctypes.c_double(eps),
        ctypes.c_int(1 if additive else 0),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    wall = time.time() - t0
    if rc != 1:
        raise RuntimeError("parhac_agg_cpu failed")
    tag = f"d{eps}" if additive else f"eps{eps}"
    dest_dir = OUT / f"y2_{tag}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, thr in enumerate(THRS):
        lab = extract_parent(fr, parents[i]).astype(np.uint32, copy=False)
        dest = dest_dir / f"mine_thr{thr}.h5"
        with h5py.File(dest, "w") as h:
            h.create_dataset("labels", data=lab)
        mcf = max_comp_frac(fr, parents[i])
        print(
            f"  T={thr} nseg={nseg(lab)} rounds={int(stats[i,0])} "
            f"merges={int(stats[i,1])} max_comp={mcf:.4f}",
            flush=True,
        )
        paths.append(str(dest))
    print(f"  wall={wall:.3f}", flush=True)
    cmd = [
        str(ROOT / ".venv/bin/python"),
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
    compile_so()
    u, v, sm, ct = load_rag()
    fr = np.load(CACHE / "wz_fragments.npy")
    max_id = int(max(int(u.max()), int(v.max()), int(fr.max())))
    print(f"edges={len(u)} max_id={max_id}", flush=True)
    for eps in EPS_LIST:
        print(f"Y2 eps={eps}", flush=True)
        ok, wall = run_one(u, v, sm, ct, fr, max_id, eps, False)
        print(f"Y2 eps={eps} {'PASS' if ok else 'FAIL'} wall={wall:.3f}", flush=True)
        if ok:
            print(f"AGG=parhac-ε {eps}")
            raise SystemExit(0)
    print("Y2 multiplicative FAIL; trying Y2b additive band", flush=True)
    for d in DELTA_LIST:
        print(f"Y2b delta={d}", flush=True)
        ok, wall = run_one(u, v, sm, ct, fr, max_id, d, True)
        print(f"Y2b d={d} {'PASS' if ok else 'FAIL'} wall={wall:.3f}", flush=True)
        if ok:
            print(f"AGG=parhac-δ {d}")
            raise SystemExit(0)
    print("Y2 FAIL")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
