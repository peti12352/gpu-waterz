#!/usr/bin/env python3
"""N7: pin whether 1150 ms is an E2 IOU.

G15 compact split: k_rewrite_dirty scan vs hash emit vs layer compact_radix.
If scan << hash, CSR splice cannot deliver the credited 3.86x. Stop CUDA.
Idle-5090. Not a 2 Gvox/s claim.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import CACHE, load_rag  # noqa: E402
from e6r_parhac import DSO, compile_d  # noqa: E402
from g_levers import bind, run_p0aa  # noqa: E402
from p1_make_big_indep import card_busy, gpu_state  # noqa: E402

OUT = CACHE / "n7_compact_iou.json"
G15_COMPACT_MS = 218.83
E2_FACTOR = 0.2588850437504452
GATE = G15_COMPACT_MS * 0.35  # splice must beat this to continue


def main():
    print("N7 compact IOU. Not a 2 Gvox/s claim.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N7 REFUSE card busy: {busy}", flush=True)
        raise SystemExit(2)
    print(f"GPU idle: {gpu_state()}", flush=True)
    compile_d()
    u, v, sm, ct, _fr, max_id = load_rag()
    lib = ctypes.CDLL(str(DSO))
    bind(lib)
    lib.parhac_compact_iou.restype = None
    lib.parhac_compact_iou.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    os.environ["WATERZ_AGG_LEVERS"] = "15"
    aa = run_p0aa(lib, u, v, sm, ct, max_id, 0.08)
    scan = ctypes.c_double(0)
    hsh = ctypes.c_double(0)
    rdx = ctypes.c_double(0)
    lib.parhac_compact_iou(
        ctypes.byref(scan), ctypes.byref(hsh), ctypes.byref(rdx))
    compact = float(aa["phases"]["compact"])
    scan_ms = float(scan.value)
    hash_ms = float(hsh.value)
    radix_ms = float(rdx.value)
    other = compact - scan_ms - hash_ms - radix_ms
    # Decision: scan dominates iff it is the largest of {scan, hash, radix}
    # and at least 40% of compact. Otherwise hash/radix own the wall.
    parts = {"rewrite_scan": scan_ms, "hash": hash_ms, "compact_radix": radix_ms}
    owner = max(parts, key=parts.get)
    scan_frac = scan_ms / compact if compact else 0.0
    scan_dominates = owner == "rewrite_scan" and scan_frac >= 0.40
    credited = compact * E2_FACTOR
    doc = {
        "claim": "not a 2 Gvox/s number",
        "measured_on": "val RAG, idle 5090, G15 p0aa",
        "gpu": gpu_state(),
        "p0aa": {k: v for k, v in aa.items() if k != "phases"},
        "phases": aa["phases"],
        "compact_ms": compact,
        "rewrite_scan_ms": scan_ms,
        "hash_ms": hash_ms,
        "compact_radix_ms": radix_ms,
        "other_compact_ms": other,
        "scan_frac": scan_frac,
        "owner": owner,
        "scan_dominates": scan_dominates,
        "e2_credited_compact_ms": credited,
        "splice_gate_ms": GATE,
        "g15_compact_ref_ms": G15_COMPACT_MS,
        "verdict": (
            "SCAN_DOMINATES implement CSR splice"
            if scan_dominates
            else "SCAN_NOT_DOMINANT stop CUDA; 1150 is an E2 IOU"
        ),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(
        f"N7 compact={compact:.2f} scan={scan_ms:.2f} hash={hash_ms:.2f} "
        f"radix={radix_ms:.2f} other={other:.2f}\n"
        f"N7 owner={owner} scan_frac={scan_frac:.3f} "
        f"dominates={scan_dominates} -> {doc['verdict']}\n"
        f"N7 wrote {OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
