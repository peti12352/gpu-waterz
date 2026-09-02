#!/usr/bin/env python3
"""X1: bucketed live-mean on cached RAG. Grade VOI. No CUDA."""
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
SO = ROOT / "src/libbucket_agg.so"
THRS = [0.2, 0.3, 0.4, 0.5]
BUCKETS = [16, 64, 256, 1024]


def compile_so():
    src = ROOT / "src/bucket_agg.cpp"
    if SO.exists() and SO.stat().st_mtime >= src.stat().st_mtime:
        return
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


def bucket_parents(u, v, sm, ct, max_id, n_buckets):
    thrs = np.asarray(THRS, dtype=np.float64)
    parents = np.empty((len(THRS), max_id + 1), dtype=np.uint32)
    lib = ctypes.CDLL(str(SO))
    lib.bucket_agg_cpu.restype = ctypes.c_int
    lib.bucket_agg_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
    ]
    lib.bucket_agg_cpu(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(THRS)),
        ctypes.c_int(n_buckets),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
    )
    return parents


def nseg(lab):
    return int((np.unique(lab) != 0).sum())


def write_and_grade(fr, parents, tag):
    dest_dir = OUT / f"x1_{tag}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, thr in enumerate(THRS):
        lab = extract_parent(fr, parents[i]).astype(np.uint32, copy=False)
        dest = dest_dir / f"mine_thr{thr}.h5"
        with h5py.File(dest, "w") as h:
            h.create_dataset("labels", data=lab)
        print(f"  wrote {dest} nseg={nseg(lab)}", flush=True)
        paths.append(str(dest))
    cmd = [
        str(ROOT / ".venv/bin/python"),
        str(OUT / "baseline/run_baseline.py"),
        "--candidate",
        *paths,
    ]
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(OUT), capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.stderr:
        sys.stderr.write(r.stderr)
    return "ACCURACY GATE: PASS" in r.stdout


def main():
    compile_so()
    u, v, sm, ct = load_rag()
    fr = np.load(CACHE / "wz_fragments.npy")
    max_id = int(max(int(u.max()), int(v.max()), int(fr.max())))
    print(f"edges={len(u)} max_id={max_id}", flush=True)
    for B in BUCKETS:
        t0 = time.time()
        parents = bucket_parents(u, v, sm, ct, max_id, B)
        print(f"X1 B={B} agg sec={time.time()-t0:.2f}", flush=True)
        ok = write_and_grade(fr, parents, f"B{B}")
        print(f"X1 B={B} {'PASS' if ok else 'FAIL'}", flush=True)
        if ok:
            print(f"AGG=bucket-{B}", flush=True)
            raise SystemExit(0)
    print("X1 FAIL all B", flush=True)
    print("AGG=unset", flush=True)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
