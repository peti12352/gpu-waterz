#!/usr/bin/env python3
"""Gate DoubleBuffer sort + one-sort gather. Parked val. Idle-5090.

Writes data/cache/b_dbl_buf.json. Not a 2 Gvox/s claim. No 2.16 allocation.
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
from b_sort_peak import peak_lines  # noqa: E402
from p1_make_big_indep import card_busy, fingerprint, gpu_state, nfrag_bg  # noqa: E402
from task_gate import FRAGMENTS_VAL  # noqa: E402

import h5py
import numpy as np

OUT = CACHE / "b_dbl_buf.json"
NVOX_216 = 2_160_000_000
USABLE = 23.0
BASE_PEAK = 2172426888  # b_sort_peak.json before this lever


def main():
    print("B DoubleBuffer + gather. Not a 2 Gvox/s claim.", flush=True)
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
    peak_ws_val = peak + 4 * nvox + nvox
    pred_fused = peak_ws_val * (NVOX_216 / nvox) / GIB
    pred_tracked = (peak / nvox) * NVOX_216 / GIB
    ok = bool(nfrag == FRAGMENTS_VAL and oracle is not False
              and meta["leaked_bytes"] == 0)
    doc = {
        "claim": "not a 2 Gvox/s number",
        "lever": "DoubleBuffer keys+idx; gather corners+vc; delay vcount",
        "nfrag": nfrag,
        "bg": bg,
        "fingerprint": fp,
        "oracle_array_equal": oracle,
        "nvox": nvox,
        "nC": nC,
        "tmp_inout_bytes": tmp_in,
        "tmp_dbl_bytes": tmp_dbl,
        "tmp_inout_gib": tmp_in / GIB,
        "tmp_dbl_gib": tmp_dbl / GIB,
        "peak_bytes": peak,
        "peak_gib": peak / GIB,
        "base_peak_bytes": BASE_PEAK,
        "saved_bytes": BASE_PEAK - peak,
        "saved_gib": (BASE_PEAK - peak) / GIB,
        "ws_peak_by_line": by_line,
        "pred_2p16_tracked_gib": pred_tracked,
        "pred_2p16_fused_gib": pred_fused,
        "usable_gib": USABLE,
        "fits": bool(pred_fused <= USABLE),
        "leaked": meta["leaked_bytes"],
        "meta": meta,
        "gpu": gpu_state(),
        "pass": ok,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(
        f"B dbl nfrag={nfrag} oracle={oracle} peak={peak / GIB:.3f} "
        f"(was {BASE_PEAK / GIB:.3f}) tmp_dbl={tmp_dbl / GIB:.3f} "
        f"fused_2.16={pred_fused:.2f} "
        f"{'FITS' if doc['fits'] else 'OVER'} {USABLE} "
        f"{'PASS' if ok else 'FAIL'} -> {OUT}",
        flush=True,
    )
    print("B peak_by_line:", by_line[:8], flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
