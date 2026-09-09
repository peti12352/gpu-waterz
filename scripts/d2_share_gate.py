#!/usr/bin/env python3
"""D2 gate: the watershed's parent array can be the caller's label buffer.

`parent` was a fourth per-voxel uint32 array on top of bits, flag and vcount.
It does not need to be: the stage ends in k_write_labels, which is
`seg[i] = psum[parent[i]] + 1`, where thread i reads slot i of parent and writes
slot i of seg while psum is a separate array. No thread reads a slot another
thread writes, so parent and seg may be the same storage. e9b's parent has the
same property and is dead before e9c starts, so both stages borrow the buffer.

Gated by building both from one source, the reference with
-DWS_NO_BUFFER_SHARE, and comparing fragments. There is no CPU oracle at crop
scale and the full-volume one cannot be run while the card is shared, so a
second library is the only honest reference; comparing the shared build against
segment() would compare it against itself.

Reports the peak too, since saving memory is the entire point.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
NVCC = "/usr/local/cuda/bin/nvcc"
GIB = 1024.0 ** 3


def build(out, extra):
    subprocess.check_call(
        [NVCC, "-O3", "-arch=sm_120", "--shared", "-Xcompiler", "-fPIC",
         *extra, "-o", str(out), str(ROOT / "csrc/ws.cu")])


def fingerprint(seg):
    """sha256 of the sorted region-size histogram: invariant to renumbering,
    sensitive to one voxel changing fragment."""
    counts = np.bincount(seg.ravel())
    sizes = np.sort(counts[1:][counts[1:] > 0]).astype(np.int64)
    return hashlib.sha256(sizes.tobytes()).hexdigest()[:32]


def run(so, aff, z, y, x):
    lib = ctypes.CDLL(str(so))
    lib.watershed_gpu_e9.restype = ctypes.c_int
    lib.watershed_gpu_e9.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.ws_mem_peak.restype = ctypes.c_size_t
    lib.ws_mem_reset.restype = None
    seg = np.zeros((z, y, x), dtype=np.uint32)
    lib.ws_mem_reset()
    lib.watershed_gpu_e9(
        aff.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        z, y, x, ctypes.c_float(1e-4), ctypes.c_float(0.9999),
        seg.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
    )
    return seg, int(lib.ws_mem_peak())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aff", default=str(AFF))
    ap.add_argument("--crop", nargs=3, type=int, default=[48, 768, 768],
                    metavar=("Z", "Y", "X"))
    args = ap.parse_args()

    ref_so = ROOT / "src/libws_gpu_noshare.so"
    cur_so = ROOT / "src/libws_gpu.so"
    print("D2g building both widths", flush=True)
    build(cur_so, [])
    build(ref_so, ["-DWS_NO_BUFFER_SHARE"])

    cz, cy, cx = args.crop
    with h5py.File(args.aff, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:, :cz, :cy, :cx], dtype=np.uint8)
    z, y, x = aff.shape[1:]
    nvox = z * y * x
    print(f"D2g {z}x{y}x{x} = {nvox / 1e6:.1f} Mvox", flush=True)

    ref, ref_peak = run(ref_so, aff, z, y, x)
    cur, cur_peak = run(cur_so, aff, z, y, x)

    same = bool(np.array_equal(ref, cur))
    fp_same = fingerprint(ref) == fingerprint(cur)
    nfrag = int((np.unique(cur) != 0).sum())
    bg = int((cur == 0).sum())
    print(f"D2g reference (own parent)  peak {ref_peak / GIB:.4f} GiB "
          f"({ref_peak / nvox:.2f} B/vox)\n"
          f"D2g shared    (borrows seg) peak {cur_peak / GIB:.4f} GiB "
          f"({cur_peak / nvox:.2f} B/vox)\n"
          f"D2g identical={same} fingerprint_equal={fp_same} "
          f"nfrag={nfrag} bg={bg}", flush=True)

    saved = ref_peak - cur_peak
    print(f"D2g saved {saved / 1e6:.1f} MB = {saved / nvox:.2f} B/vox = "
          f"{saved / ref_peak * 100:.1f}% of the watershed peak", flush=True)
    for name, target in (("2.16 Gvox", 2_160_000_000),
                         ("1.44 Gvox", 1_440_000_000)):
        print(f"D2g   at {name}: {saved / nvox * target / GIB:.2f} GiB saved, "
              f"ws scratch {cur_peak / nvox * target / GIB:.2f} GiB",
              flush=True)

    ok = same and fp_same and saved > 0
    print(f"D2g {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
