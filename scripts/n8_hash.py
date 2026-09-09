#!/usr/bin/env python3
"""N8: SM/warp + CUB RBK dirty-hash vs G15 parent. Gate hash ≤40 ms.

Idle-5090. Not a 2 Gvox/s claim.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import CACHE, load_rag  # noqa: E402
from e6r_parhac import DSO, compile_d  # noqa: E402
from g_levers import bind, run_four, run_p0aa  # noqa: E402
from p1_make_big_indep import card_busy, gpu_state  # noqa: E402

OUT = CACHE / "n8_hash.json"
HASH_GATE_MS = 40.0


def iou(lib):
    lib.parhac_compact_iou.restype = None
    lib.parhac_compact_iou.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    scan = ctypes.c_double(0)
    hsh = ctypes.c_double(0)
    rdx = ctypes.c_double(0)
    lib.parhac_compact_iou(
        ctypes.byref(scan), ctypes.byref(hsh), ctypes.byref(rdx))
    return float(scan.value), float(hsh.value), float(rdx.value)


def main():
    print("N8 dirty-hash dedup. Not a 2 Gvox/s claim.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N8 REFUSE card busy: {busy}", flush=True)
        raise SystemExit(2)
    print(f"GPU idle: {gpu_state()}", flush=True)
    if DSO.exists():
        DSO.unlink()
    compile_d()
    u, v, sm, ct, _fr, max_id = load_rag()
    lib = ctypes.CDLL(str(DSO))
    bind(lib)
    os.environ["WATERZ_AGG_LEVERS"] = "15"

    os.environ["WATERZ_HASH_DEDUP"] = "0"
    ref4 = run_four(lib, u, v, sm, ct, max_id, 0.08)
    ref_aa = run_p0aa(lib, u, v, sm, ct, max_id, 0.08)
    ref_scan, ref_hash, ref_rdx = iou(lib)

    os.environ["WATERZ_HASH_DEDUP"] = "1"
    got4 = run_four(lib, u, v, sm, ct, max_id, 0.08)
    got_aa = run_p0aa(lib, u, v, sm, ct, max_id, 0.08)
    got_scan, got_hash, got_rdx = iou(lib)

    parent_ok = bool(np.array_equal(ref4["parents"], got4["parents"]))
    hash_ok = got_hash <= HASH_GATE_MS
    ok = parent_ok and hash_ok
    doc = {
        "claim": "not a 2 Gvox/s number",
        "gpu": gpu_state(),
        "parent_array_equal": parent_ok,
        "nseg_ref": ref4["nseg_unique_parents"],
        "nseg_got": got4["nseg_unique_parents"],
        "g15_hash_ms": ref_hash,
        "dedup_hash_ms": got_hash,
        "hash_gate_ms": HASH_GATE_MS,
        "hash_under_gate": hash_ok,
        "g15_scan_ms": ref_scan,
        "dedup_scan_ms": got_scan,
        "g15_radix_ms": ref_rdx,
        "dedup_radix_ms": got_rdx,
        "g15_compact_ms": ref_aa["phases"]["compact"],
        "dedup_compact_ms": got_aa["phases"]["compact"],
        "pass": ok,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(
        f"N8 hash parent={parent_ok} hash={got_hash:.2f} "
        f"(g15 {ref_hash:.2f}) {'<=' if hash_ok else '>'}{HASH_GATE_MS} "
        f"{'PASS' if ok else 'FAIL'} -> {OUT}",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
