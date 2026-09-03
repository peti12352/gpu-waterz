#!/usr/bin/env python3
"""C1: time the region-graph build, A/B against another rag.cu revision.

Timing on a shared card is only meaningful as a matched comparison, so this
builds two revisions of csrc/rag.cu and interleaves their runs rather than
timing one and trusting a number from an earlier session. Pass a baseline
source to compare against:

    c1_rag_ab.py                     # time the working tree only
    c1_rag_ab.py --base /tmp/rag_head.cu

The reported figure is device time around the RAG kernels via CUDA events, and
the edge count is printed so a build that got faster by dropping edges is
obvious rather than flattering.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import CACHE  # noqa: E402

NVCC = "/usr/local/cuda-12.8/bin/nvcc"
AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
EDGES_VAL = 7505458


def build(src: Path, out: Path):
    subprocess.check_call([
        NVCC, "-O3", "-arch=sm_120", "--shared", "-Xcompiler", "-fPIC",
        "-I/usr/local/cuda-12.8/targets/x86_64-linux/include",
        "-o", str(out), str(src),
    ], stderr=subprocess.DEVNULL)


def run(so: Path, aff, fr, reps: int):
    """Return (edge count, list of device ms).

    rag_gpu prints `RAG device_ms=...` on stderr from CUDA events placed around
    the kernels, so the timing excludes the 540 MB affinity upload this host
    entry point does and the graded path does not.
    """
    lib = ctypes.CDLL(str(so))
    lib.rag_last_ms.restype = ctypes.c_float
    lib.rag_gpu.restype = ctypes.c_int64
    lib.rag_gpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
    ]
    z, y, x = fr.shape
    cap = EDGES_VAL * 2
    u = np.empty(cap, dtype=np.uint32)
    v = np.empty(cap, dtype=np.uint32)
    sm = np.empty(cap, dtype=np.float64)
    ct = np.empty(cap, dtype=np.int64)
    n = 0
    out = []
    for _ in range(reps):
        n = lib.rag_gpu(
            aff.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            fr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_int64(z), ctypes.c_int64(y), ctypes.c_int64(x),
            u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_int64(cap),
        )
        out.append(float(lib.rag_last_ms()))
    return int(n), out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=None)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    fr = np.ascontiguousarray(np.load(CACHE / "wz_fragments.npy"), dtype=np.uint32)

    builds = [("work", ROOT / "csrc/rag.cu", Path("/tmp/c1_rag_work.so"))]
    if args.base:
        builds.append(("base", args.base, Path("/tmp/c1_rag_base.so")))
    for _, src, so in builds:
        build(src, so)

    # Interleave so a drift in load hits both arms equally.
    res = {name: [] for name, _, _ in builds}
    nedge = {}
    for _ in range(args.reps):
        for name, _, so in builds:
            n, ms = run(so, aff, fr, 1)
            res[name].extend(ms)
            nedge[name] = n

    out = {}
    for name in res:
        v = sorted(res[name])
        out[name] = {
            "nedge": nedge[name],
            "median_ms": statistics.median(v),
            "min_ms": v[0],
            "max_ms": v[-1],
            "edges_ok": nedge[name] == EDGES_VAL,
        }
        r = out[name]
        print(f"C1 {name:5s} median={r['median_ms']:8.2f} min={r['min_ms']:8.2f} "
              f"max={r['max_ms']:8.2f} nedge={r['nedge']} "
              f"edges_ok={r['edges_ok']}", flush=True)
    if "base" in out:
        sp = out["base"]["median_ms"] / out["work"]["median_ms"]
        out["speedup"] = sp
        print(f"C1 speedup {sp:.2f}x (base/work, medians)", flush=True)
    p = CACHE / "c1_rag_ab.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"C1 wrote {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
