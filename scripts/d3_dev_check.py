#!/usr/bin/env python3
"""D3 gate: the device-resident path returns exactly what the host path returns.

segment_d used to need torch as nothing but a device allocator, which is why it
had never run on this machine. It now calls cudaMalloc through ctypes and reads
its input through __cuda_array_interface__, so it accepts a torch tensor, a
cupy array or its own DevBuf without importing any of them.

Three input shapes are checked against segment(), which is the trusted path:
a host array, a uint8 array already in VRAM, and a float32 array already in
VRAM quantised on the device by aff_f32_to_u8_d. All three must be
byte-identical to the host result, and the labels must be readable while still
in VRAM, since that is what TASK grades.

Runs on a crop by default: the card is shared and a full-volume run would risk
an OOM in a co-tenant, which is not a cost worth paying for a correctness gate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aff", default=str(AFF))
    ap.add_argument("--crop", nargs=3, type=int, default=[32, 256, 256],
                    metavar=("Z", "Y", "X"))
    ap.add_argument("--threshold", type=float, default=0.3)
    args = ap.parse_args()

    import segment as S

    z, y, x = args.crop
    with h5py.File(args.aff, "r") as f:
        aff = f["affinity"][:, :z, :y, :x]
    aff_u8 = np.ascontiguousarray(S._as_u8(aff))
    thr = [args.threshold]
    print(f"D3 crop {aff_u8.shape[1:]} = {aff_u8[0].size / 1e6:.1f} Mvox "
          f"T={args.threshold}", flush=True)

    ref = S.segment(aff_u8, thr)[0]
    print(f"D3 host segment nseg={int((np.unique(ref) != 0).sum())} "
          f"backend={S.AGG_BACKEND}", flush=True)

    ok = True

    def check(name, labs):
        nonlocal ok
        same = bool(np.array_equal(ref, labs))
        ok = ok and same
        print(f"D3 {name}: identical={same} dtype={labs.dtype} "
              f"shape={labs.shape}", flush=True)

    check("host input, host out", S.segment_d(aff_u8, thr)[0])
    print(f"D3 device agglomeration backend={S.AGG_BACKEND}", flush=True)
    if S.AGG_BACKEND != "gpu_dev":
        ok = False
        print("D3 FAIL the RAG left the device", flush=True)

    aff_d = S.DevBuf.from_host(aff_u8)
    out = S.segment_d(aff_d, thr, return_device=True)
    assert len(out) == 1 and out[0].ptr, "labels must come back as a device buffer"
    check("u8 in VRAM, labels in VRAM", out[0].to_host())
    for b in out:
        b.free()

    aff_f32 = (aff_u8.astype(np.float32) / 255.0)
    aff_f = S.DevBuf.from_host(aff_f32)
    out = S.segment_d(aff_f, thr, return_device=True)
    check("f32 in VRAM, quantised on device", out[0].to_host())
    for b in out:
        b.free()
    aff_f.free()

    # Determinism of the device path itself, same input twice.
    a = S.segment_d(aff_d, thr, return_device=True)
    b = S.segment_d(aff_d, thr, return_device=True)
    det = bool(np.array_equal(a[0].to_host(), b[0].to_host()))
    ok = ok and det
    print(f"D3 device path deterministic={det}", flush=True)
    for buf in (*a, *b):
        buf.free()
    aff_d.free()

    # Four thresholds exercise the branch where only the last threshold writes
    # its labels over the fragment buffer and the earlier ones cannot.
    many = [0.2, 0.3, 0.4, 0.5]
    ref_many = S.segment(aff_u8, many)
    aff_d = S.DevBuf.from_host(aff_u8)
    got = S.segment_d(aff_d, many)
    for t, a, b in zip(many, ref_many, got):
        same = bool(np.array_equal(a, b))
        ok = ok and same
        print(f"D3 four thresholds T={t}: identical={same}", flush=True)
    aff_d.free()

    _, ms = S.cuda_event_time(lambda: None)
    print(f"D3 cuda event timer works ({ms:.3f} ms for an empty region)",
          flush=True)

    print(f"D3 {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
