#!/usr/bin/env python3
"""C2: the watershed regression instrument, used to gate every ws.cu change.

Track C has to cut the watershed's measured 68.53 B/vox (scripts/c1_memory_budget.py)
without changing what it computes. Checking only nfrag, as the earlier gates
did, is far too weak: a change can preserve the fragment count while moving
voxels between fragments. So this records, on val:

  determinism     two runs, fragment arrays byte-identical
  nfrag / bg      against TASK.md's 2175400 and 506568
  fingerprint     sha256 of the sorted region-size histogram, which is
                  invariant to label renumbering but changes if a single voxel
                  moves between fragments
  oracle          array_equal against the CPU reference wz_fragments.npy
  peak bytes      from the in-library allocation counter

Run it before and after a change and compare the JSON. The fingerprint is the
gate that matters: equal fingerprint plus equal nfrag means the segmentation is
identical up to relabeling.

Uses watershed_gpu_e9, the host-pointer entry point, so no torch is needed.
Peak device use is the watershed's own ~11.5 GiB at val, which is why this must
not be run when the GPU is short of memory.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from task_gate import BG_VAL_MEASURED, FRAGMENTS_VAL  # noqa: E402

AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
CACHE = ROOT / "data/cache"
WS = ROOT / "src/libws_gpu.so"
NVCC = "/usr/local/cuda-12.8/bin/nvcc"
GIB = 1024.0 ** 3
NRUN = 2


def build():
    subprocess.check_call([
        NVCC, "-O3", "-arch=sm_120", "--shared", "-Xcompiler", "-fPIC",
        "-I/usr/local/cuda-12.8/targets/x86_64-linux/include",
        "-o", str(WS), str(ROOT / "csrc/ws.cu"),
    ])


def fingerprint(seg):
    """sha256 of the sorted nonzero region-size histogram.

    Invariant to relabeling, sensitive to any voxel changing fragment.
    """
    counts = np.bincount(seg.ravel())
    sizes = np.sort(counts[1:][counts[1:] > 0]).astype(np.int64)
    h = hashlib.sha256()
    h.update(sizes.tobytes())
    return h.hexdigest()[:32], int(sizes.size), int(sizes.sum())


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
    z, y, x = aff.shape[1:]
    lib = ctypes.CDLL(str(WS))
    lib.watershed_gpu_e9.restype = ctypes.c_int
    lib.watershed_gpu_e9.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_uint32),
    ]
    for name in ("reset", "peak", "cur"):
        fn = getattr(lib, f"ws_mem_{name}")
        fn.restype = None if name == "reset" else ctypes.c_size_t
        fn.argtypes = []

    print(f"C2 watershed invariants, {z}x{y}x{x}. GPU: {gpu_state()}", flush=True)
    runs = []
    peak = 0
    for r in range(NRUN):
        seg = np.zeros((z, y, x), dtype=np.uint32)
        lib.ws_mem_reset()
        lib.watershed_gpu_e9(
            aff.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            z, y, x, ctypes.c_float(1e-4), ctypes.c_float(0.9999),
            seg.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        )
        peak = max(peak, int(lib.ws_mem_peak()))
        runs.append(seg)
        print(f"C2 run{r} done", flush=True)

    base = runs[0]
    det = all(np.array_equal(base, s) for s in runs[1:])
    ndiff = [int((base != s).sum()) for s in runs[1:]]
    nfrag = int((np.unique(base) != 0).sum())
    bg = int((base == 0).sum())
    fp, nreg, covered = fingerprint(base)
    leaked = int(lib.ws_mem_cur())

    oracle_eq = None
    op = CACHE / "wz_fragments.npy"
    if op.exists():
        ref = np.ascontiguousarray(np.load(op), dtype=np.uint32)
        oracle_eq = bool(ref.shape == base.shape and np.array_equal(ref, base))
        ref_fp, _, _ = fingerprint(ref)
        oracle_fp_eq = bool(ref_fp == fp)
    else:
        oracle_fp_eq = None

    print(
        f"C2 deterministic={det} ndiff={ndiff}\n"
        f"   nfrag={nfrag} (TASK {FRAGMENTS_VAL}) "
        f"bg={bg} (TASK {BG_VAL_MEASURED})\n"
        f"   fingerprint={fp} nregions={nreg} voxels_covered={covered}\n"
        f"   peak={peak / GIB:.3f} GiB ({peak / base.size:.2f} B/vox) "
        f"leaked={leaked}\n"
        f"   vs CPU oracle: array_equal={oracle_eq} fingerprint_equal={oracle_fp_eq}",
        flush=True,
    )

    ok = bool(det and nfrag == FRAGMENTS_VAL and bg == BG_VAL_MEASURED
              and leaked == 0)
    out = {
        "shape": [z, y, x],
        "deterministic": bool(det),
        "ndiff_vs_run0": ndiff,
        "nfrag": nfrag,
        "nfrag_expected": FRAGMENTS_VAL,
        "bg": bg,
        "bg_expected": BG_VAL_MEASURED,
        "size_histogram_sha256": fp,
        "nregions": nreg,
        "voxels_covered": covered,
        "peak_bytes": peak,
        "peak_bytes_per_vox": peak / base.size,
        "leaked_bytes": leaked,
        "oracle_array_equal": oracle_eq,
        "oracle_fingerprint_equal": oracle_fp_eq,
        "gpu": gpu_state(),
        "pass": ok,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / "c2_ws_invariants.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"C2 {'PASS' if ok else 'FAIL'} -> {dest.name}", flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
