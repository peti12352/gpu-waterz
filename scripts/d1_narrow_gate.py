#!/usr/bin/env python3
"""D1 gate: narrowing the dedup hash slot changes memory, not labels.

The dedup table is the agglomeration's largest allocation. Its slot was
{uint64 key, double sm, uint64 ct} = 24 B and is now {uint64, uint32, uint32}
= 16 B. Both build from one source, the wide one with -DHSLOT_WIDE, so this
compares the two side by side in one process on the same input.

Comparing against segment() would prove nothing, since both paths link the same
library and would move together. Two libraries is the only honest reference.

The claim being gated is that narrowing is exact rather than approximate: sm is
already whole affinity bytes by the time it reaches the table and ct is a face
count, so both adds are integer sums in either width, and the only difference
is range. So labels must be byte-identical and no overflow may be reported.
"""
from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
GIB = 1024 ** 3


def run(S, so, aff_u8, thresholds):
    S._PARHAC_D = so
    lib = ctypes.CDLL(str(so))
    lib.agg_mem_peak.restype = ctypes.c_size_t
    lib.agg_mem_reset()
    aff_d = S.DevBuf.from_host(aff_u8)
    labs = S.segment_d(aff_d, thresholds)
    aff_d.free()
    return labs, int(lib.agg_mem_peak())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aff", default=str(AFF))
    ap.add_argument("--crop", nargs=3, type=int, default=[48, 768, 768],
                    metavar=("Z", "Y", "X"))
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.2, 0.3, 0.4, 0.5])
    args = ap.parse_args()

    import segment as S

    narrow_so = ROOT / "src/libparhac_d.so"
    wide_so = ROOT / "src/libparhac_d_wide.so"
    if not wide_so.is_file():
        print(f"D1g FAIL build the reference first:\n"
              f"  nvcc -O3 -arch=sm_120 -DHSLOT_WIDE -Xcompiler -fPIC -shared "
              f"-o {wide_so} csrc/parhac_d.cu")
        return False

    cz, cy, cx = args.crop
    with h5py.File(args.aff, "r") as f:
        aff = f["affinity"][:, :cz, :cy, :cx]
    aff_u8 = np.ascontiguousarray(S._as_u8(aff))
    nvox = int(aff_u8[0].size)
    print(f"D1g {aff_u8.shape[1:]} = {nvox / 1e6:.1f} Mvox, "
          f"T={args.thresholds}", flush=True)

    wide_labs, wide_peak = run(S, wide_so, aff_u8, args.thresholds)
    print(f"D1g wide   HSlot 24 B  agg peak {wide_peak / GIB:.4f} GiB "
          f"({wide_peak / nvox:.2f} B/vox)", flush=True)
    narrow_labs, narrow_peak = run(S, narrow_so, aff_u8, args.thresholds)
    print(f"D1g narrow HSlot 16 B  agg peak {narrow_peak / GIB:.4f} GiB "
          f"({narrow_peak / nvox:.2f} B/vox)", flush=True)

    ok = True
    for t, a, b in zip(args.thresholds, wide_labs, narrow_labs):
        same = bool(np.array_equal(a, b))
        ok = ok and same
        nseg = int((np.unique(b) != 0).sum())
        print(f"D1g T={t} identical={same} nseg={nseg}", flush=True)

    saved = wide_peak - narrow_peak
    print(f"D1g saved {saved / 1e6:.1f} MB = {saved / wide_peak * 100:.1f}% "
          f"of the agglomeration peak, {saved / nvox:.2f} B/vox", flush=True)
    for name, target in (("2.16 Gvox", 2_160_000_000), ("1.44 Gvox", 1_440_000_000)):
        print(f"D1g   at {name}: {saved / nvox * target / GIB:.2f} GiB saved",
              flush=True)

    if saved <= 0:
        ok = False
        print("D1g FAIL narrowing did not reduce the peak", flush=True)
    print(f"D1g {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
