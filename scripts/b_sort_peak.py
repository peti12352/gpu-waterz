#!/usr/bin/env python3
"""Measure e9b SortPairs tmp_bytes: in/out vs dry DoubleBuffer query.

Parked val only. Does not change sort behavior. Idle-5090.
Writes data/cache/b_sort_peak.json. Not a 2 Gvox/s claim. No 2.16 allocation.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from b_dev_aff import AFF, CACHE, GIB, WS, bind, build, run_split  # noqa: E402
from p1_make_big_indep import card_busy, fingerprint, gpu_state, nfrag_bg  # noqa: E402
from task_gate import FRAGMENTS_VAL  # noqa: E402

import h5py
import numpy as np

OUT = CACHE / "b_sort_peak.json"
NVOX_216 = 2_160_000_000
USABLE = 23.0


def peak_lines(lib, cap=256):
    lib.ws_mem_peak_lines.restype = ctypes.c_int
    lib.ws_mem_peak_lines.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int,
    ]
    lines = (ctypes.c_int * cap)()
    byts = (ctypes.c_size_t * cap)()
    n = min(cap, lib.ws_mem_peak_lines(lines, byts, cap))
    rows = sorted(
        ([int(lines[i]), int(byts[i])] for i in range(n)),
        key=lambda r: r[1],
        reverse=True,
    )
    return rows


def main():
    print("B sort-peak tmp measure. Not a 2 Gvox/s claim.", flush=True)
    busy = card_busy()
    if busy:
        print(f"B REFUSE card busy: {busy}", flush=True)
        raise SystemExit(2)
    print(f"GPU idle: {gpu_state()}", flush=True)
    os.environ["WATERZ_WS_MEMLOG"] = "1"
    build()
    lib = ctypes.CDLL(str(WS))
    bind(lib)
    lib.ws_sort_tmp_inout.restype = ctypes.c_size_t
    lib.ws_sort_tmp_dbl.restype = ctypes.c_size_t
    lib.ws_sort_nC.restype = ctypes.c_int
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    seg, meta = run_split(lib, aff, park=True)
    nfrag, bg = nfrag_bg(seg)
    fp, _, _ = fingerprint(seg)
    oracle = None
    op = CACHE / "wz_fragments.npy"
    if op.exists():
        ref = np.ascontiguousarray(np.load(op), dtype=np.uint32)
        oracle = bool(ref.shape == seg.shape and np.array_equal(ref, seg))
    nvox = int(seg.size)
    nC = int(lib.ws_sort_nC())
    tmp_in = int(lib.ws_sort_tmp_inout())
    tmp_dbl = int(lib.ws_sort_tmp_dbl())
    peak = int(meta["peak_bytes"])
    by_line = peak_lines(lib)
    nc_b = nC * 4
    extra_n = max(0, tmp_in - tmp_dbl)
    pred_ws_now = (peak / nvox) * NVOX_216 / GIB
    # If DoubleBuffer only drops CUB extra-N and user arrays stay 6*nC,
    # tracked peak falls by extra_n (the in/out third pair inside tmp).
    pred_ws_dbl = ((peak - extra_n) / nvox) * NVOX_216 / GIB
    fused_now = pred_ws_now + 5.0  # seg 4 B/vox + bits 1 B/vox at 2.16 = 5*2.16e9/2^30
    # Use the same formula as b_fit.py: (tracked + seg + bits) * scale
    peak_ws_val = peak + 4 * nvox + nvox
    pred_fused_now = peak_ws_val * (NVOX_216 / nvox) / GIB
    pred_fused_dbl = (peak_ws_val - extra_n) * (NVOX_216 / nvox) / GIB
    ok = bool(nfrag == FRAGMENTS_VAL and oracle is not False
              and meta["leaked_bytes"] == 0 and nC > 0 and tmp_in > 0)
    doc = {
        "claim": "not a 2 Gvox/s number",
        "measured_on": "val 180 Mvox, idle 5090",
        "no_2p16_allocation": True,
        "nfrag": nfrag,
        "bg": bg,
        "fingerprint": fp,
        "oracle_array_equal": oracle,
        "nvox": nvox,
        "nC": nC,
        "nC_bytes": nc_b,
        "nC_gib": nc_b / GIB,
        "tmp_inout_bytes": tmp_in,
        "tmp_inout_gib": tmp_in / GIB,
        "tmp_dbl_bytes": tmp_dbl,
        "tmp_dbl_gib": tmp_dbl / GIB,
        "tmp_extra_n_bytes": extra_n,
        "tmp_extra_n_gib": extra_n / GIB,
        "ws_tracked_bytes": peak,
        "ws_tracked_gib": peak / GIB,
        "ws_peak_by_line": by_line,
        "pred_2p16_tracked_gib": pred_ws_now,
        "pred_2p16_tracked_dbl_only_gib": pred_ws_dbl,
        "pred_2p16_fused_gib": pred_fused_now,
        "pred_2p16_fused_dbl_only_gib": pred_fused_dbl,
        "usable_gib": USABLE,
        "dbl_only_fits": bool(pred_fused_dbl <= USABLE),
        "leaked": meta["leaked_bytes"],
        "meta": meta,
        "gpu": gpu_state(),
        "pass": ok,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(
        f"B sort-peak nC={nC} nfrag={nfrag} oracle={oracle} "
        f"tmp_inout={tmp_in / GIB:.3f} tmp_dbl={tmp_dbl / GIB:.3f} "
        f"extra_n={extra_n / GIB:.3f} peak={peak / GIB:.3f} GiB "
        f"fused_now={pred_fused_now:.2f} fused_dbl={pred_fused_dbl:.2f} "
        f"{'PASS' if ok else 'FAIL'} -> {OUT}",
        flush=True,
    )
    print("B peak_by_line:", by_line[:8], flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
