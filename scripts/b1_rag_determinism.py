#!/usr/bin/env python3
"""B1: is the device RAG builder deterministic?

csrc/rag.cu accumulates contact sums with `atomicAdd(&tab[s].sum, a)` on a
float32. Float addition is not associative, so if two runs interleave their
atomics differently the resulting sums differ in the low bits. That would make
the edge weights, and therefore every downstream merge decision, vary run to
run, a direct violation of TASK.md's "same input -> byte-identical labels".

This probe calls rag_gpu (the host-pointer entry point, which allocates and
frees its own device memory, so no torch is needed) twice on the same input and
compares u/v/sm/ct bit-for-bit. It also reports the edge count against the
TASK figure so a regression in the builder itself cannot hide behind a
determinism pass.

Peak device use is ~2.8 GB, safe to run alongside another job.
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from task_gate import EDGES_VAL  # noqa: E402

AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
CACHE = ROOT / "data/cache"
RAG = ROOT / "src/librag_gpu.so"
NVCC = "/usr/local/cuda-12.8/bin/nvcc"
MAX_E = 20_000_000
NRUN = 2


def build():
    subprocess.check_call([
        NVCC, "-O3", "-arch=sm_120", "--shared", "-Xcompiler", "-fPIC",
        "-I/usr/local/cuda-12.8/targets/x86_64-linux/include",
        "-o", str(RAG), str(ROOT / "csrc/rag.cu"),
    ])


def gpu_state():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def main():
    build()
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    fr = np.ascontiguousarray(np.load(CACHE / "wz_fragments.npy"), dtype=np.uint32)
    z, y, x = aff.shape[1:]
    lib = ctypes.CDLL(str(RAG))
    lib.rag_gpu.restype = ctypes.c_int64
    lib.rag_gpu.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64,
    ]
    print(f"B1 RAG determinism, {NRUN} runs, {z}x{y}x{x}. GPU: {gpu_state()}",
          flush=True)

    runs = []
    for r in range(NRUN):
        u = np.zeros(MAX_E, dtype=np.uint32)
        v = np.zeros(MAX_E, dtype=np.uint32)
        sm = np.zeros(MAX_E, dtype=np.float64)
        ct = np.zeros(MAX_E, dtype=np.int64)
        n = lib.rag_gpu(
            aff.ctypes.data_as(ctypes.c_void_p),
            fr.ctypes.data_as(ctypes.c_void_p),
            z, y, x,
            u.ctypes.data_as(ctypes.c_void_p), v.ctypes.data_as(ctypes.c_void_p),
            sm.ctypes.data_as(ctypes.c_void_p), ct.ctypes.data_as(ctypes.c_void_p),
            MAX_E,
        )
        n = int(n)
        print(f"B1 run{r} nedge={n}", flush=True)
        # The hash table emits edges in slot order, which itself depends on
        # insertion race, so compare as an order-independent keyed mapping
        # rather than as raw arrays.
        key = (u[:n].astype(np.uint64) << np.uint64(32)) | v[:n].astype(np.uint64)
        o = np.argsort(key, kind="stable")
        runs.append({
            "n": n,
            "key": key[o],
            "sm": sm[:n][o],
            "ct": ct[:n][o],
        })

    base = runs[0]
    out = {"nedge": [r["n"] for r in runs], "edges_expected": EDGES_VAL,
           "gpu": gpu_state(), "runs": NRUN}
    same_n = all(r["n"] == base["n"] for r in runs)
    key_eq = same_n and all(np.array_equal(base["key"], r["key"]) for r in runs[1:])
    ct_eq = key_eq and all(np.array_equal(base["ct"], r["ct"]) for r in runs[1:])
    sm_eq = key_eq and all(np.array_equal(base["sm"], r["sm"]) for r in runs[1:])
    sm_ulp = 0.0
    if key_eq:
        for r in runs[1:]:
            d = np.abs(base["sm"] - r["sm"])
            sm_ulp = max(sm_ulp, float(d.max()))
    ndiff_sm = 0
    if key_eq:
        ndiff_sm = int(sum(int((base["sm"] != r["sm"]).sum()) for r in runs[1:]))

    out.update({
        "same_edge_count": bool(same_n),
        "same_edge_set": bool(key_eq),
        "count_bit_identical": bool(ct_eq),
        "sum_bit_identical": bool(sm_eq),
        "sum_max_abs_drift": sm_ulp,
        "sum_n_edges_differing": ndiff_sm,
        "edge_count_matches_task": bool(base["n"] == EDGES_VAL),
    })

    # Determinism alone is satisfiable by a wrong-but-stable builder, so also
    # check against the CPU reference RAG in rag.npz (built by e3cpu.py via
    # region_graph, independent of this code path). Counts are integers and
    # must agree exactly; sums are compared with tolerance because the oracle
    # accumulates float affinities while the GPU now sums bytes exactly.
    oracle = CACHE / "rag.npz"
    if oracle.exists():
        z = np.load(oracle)
        ok_u = z["keys"][:, 0].astype(np.uint64)
        ok_v = z["keys"][:, 1].astype(np.uint64)
        okey = (ok_u << np.uint64(32)) | ok_v
        oo = np.argsort(okey, kind="stable")
        okey = okey[oo]
        osm = z["stats"][:, 0][oo]
        oct_ = np.rint(z["stats"][:, 1]).astype(np.int64)[oo]
        set_eq = bool(base["key"].size == okey.size
                      and np.array_equal(base["key"], okey))
        ct_exact = bool(set_eq and np.array_equal(base["ct"], oct_))
        sm_max = float(np.abs(base["sm"] - osm).max()) if set_eq else float("inf")
        out.update({
            "oracle_edge_set_equal": set_eq,
            "oracle_count_exact": ct_exact,
            "oracle_sum_max_abs_diff": sm_max,
        })
        print(
            f"B1 vs CPU oracle: edge_set_equal={set_eq} count_exact={ct_exact} "
            f"sum_max_abs_diff={sm_max:.3e}",
            flush=True,
        )
        oracle_ok = set_eq and ct_exact and sm_max < 1e-3
        out["oracle_ok"] = bool(oracle_ok)
    else:
        print("B1 oracle rag.npz absent, skipping reference comparison",
              flush=True)
        oracle_ok = True
        out["oracle_ok"] = None
    print(
        f"B1 same_edge_count={same_n} same_edge_set={key_eq} "
        f"ct_bit_identical={ct_eq} sm_bit_identical={sm_eq}\n"
        f"   sm edges differing={ndiff_sm} max_abs_drift={sm_ulp:.3e}\n"
        f"   nedge={base['n']} expected={EDGES_VAL} "
        f"match={base['n'] == EDGES_VAL}",
        flush=True,
    )
    ok = bool(ct_eq and sm_eq and base["n"] == EDGES_VAL and oracle_ok)
    out["pass"] = ok
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "b1_rag_determinism.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"B1 {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
