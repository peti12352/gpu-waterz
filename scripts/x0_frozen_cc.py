#!/usr/bin/env python3
"""X0 / X0b: frozen CC on cached RAG. Grade VOI. No CUDA."""
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
SO = ROOT / "src/libfrozen_cc.so"
THRS = [0.2, 0.3, 0.4, 0.5]


def compile_so():
    src = ROOT / "src/frozen_cc.cpp"
    if SO.exists() and SO.stat().st_mtime >= src.stat().st_mtime:
        return
    subprocess.check_call(
        ["g++", "-O3", "-DNDEBUG", "-shared", "-fPIC", "-o", str(SO), str(src)]
    )


def load_rag():
    z = np.load(CACHE / "rag.npz")
    u = np.ascontiguousarray(z["keys"][:, 0], dtype=np.uint32)
    v = np.ascontiguousarray(z["keys"][:, 1], dtype=np.uint32)
    mean = np.ascontiguousarray(z["stats"][:, 2], dtype=np.float64)
    return u, v, mean


def frozen_parent(u, v, mean, thr, max_id):
    parent = np.empty(max_id + 1, dtype=np.uint32)
    lib = ctypes.CDLL(str(SO))
    lib.frozen_cc_cpu.restype = ctypes.c_int
    lib.frozen_cc_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
    ]
    lib.frozen_cc_cpu(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        mean.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(len(u)),
        ctypes.c_double(thr),
        parent.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
    )
    return parent


def apply_minsize(lab, min_size):
    if min_size <= 0:
        return lab
    flat = lab.reshape(-1)
    mx = int(flat.max()) + 1
    cnt = np.bincount(flat, minlength=mx)
    cnt[0] = min_size
    keep = cnt >= min_size
    remap = np.arange(mx, dtype=np.uint32)
    remap[~keep] = 0
    return remap[flat].reshape(lab.shape)


def nseg(lab):
    return int((np.unique(lab) != 0).sum())


def write_and_grade(fr, parents, min_size, tag):
    dest_dir = OUT / f"x0_{tag}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for thr, p in zip(THRS, parents):
        lab = extract_parent(fr, p).astype(np.uint32, copy=False)
        lab = apply_minsize(lab, min_size)
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
    ok = "ACCURACY GATE: PASS" in r.stdout
    return ok


def main():
    compile_so()
    u, v, mean = load_rag()
    fr = np.load(CACHE / "wz_fragments.npy")
    max_id = int(max(int(u.max()), int(v.max()), int(fr.max())))
    print(f"edges={len(u)} max_id={max_id} fr_n={nseg(fr)}", flush=True)

    t0 = time.time()
    parents = [frozen_parent(u, v, mean, t, max_id) for t in THRS]
    print(f"X0 unions sec={time.time()-t0:.2f}", flush=True)
    ok = write_and_grade(fr, parents, 0, "d0_m0")
    print("X0", "PASS" if ok else "FAIL", flush=True)
    if ok:
        print("AGG=cc", flush=True)
        raise SystemExit(0)

    deltas = [0.02, 0.05, 0.08, 0.10]
    minsizes = [0, 16, 64]
    winners = []
    for d in deltas:
        t0 = time.time()
        parents = [frozen_parent(u, v, mean, t + d, max_id) for t in THRS]
        print(f"X0b delta={d} unions sec={time.time()-t0:.2f}", flush=True)
        for ms in minsizes:
            tag = f"d{d}_m{ms}".replace(".", "p")
            ok = write_and_grade(fr, parents, ms, tag)
            print(f"X0b delta={d} min_size={ms} {'PASS' if ok else 'FAIL'}", flush=True)
            if ok:
                winners.append((d, ms))
    if winners:
        print("X0b WINNERS", winners, flush=True)
        print(f"AGG=cc+delta first={winners[0]}", flush=True)
        raise SystemExit(0)
    print("X0b FAIL no (delta,min_size) passes all four T", flush=True)
    print("AGG=unset go X1", flush=True)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
