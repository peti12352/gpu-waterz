#!/usr/bin/env python3
"""Measure one W2 lever on the parked device path. Val only.

Usage: b_w2_lever.py NAME
Rebuilds libws_gpu.so, runs ws_flow_d + park + ws_label_d, gates nfrag/oracle.
Writes data/cache/b_w2_<name>.json. Not a 2 Gvox/s claim.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from b_dev_aff import AFF, CACHE, GIB, NVCC, WS, bind, build, run_split  # noqa: E402
from p1_make_big_indep import card_busy, fingerprint, gpu_state, nfrag_bg  # noqa: E402
from task_gate import FRAGMENTS_VAL  # noqa: E402

import h5py
import numpy as np


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "anon"
    print(f"B W2 lever={name}. Not a 2 Gvox/s claim.", flush=True)
    busy = card_busy()
    if busy:
        print(f"B REFUSE card busy: {busy}", flush=True)
        raise SystemExit(2)
    print(f"GPU idle: {gpu_state()}", flush=True)
    build()
    import ctypes
    lib = ctypes.CDLL(str(WS))
    bind(lib)
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
    peak = int(meta["peak_bytes"])
    ok = bool(nfrag == FRAGMENTS_VAL and oracle is not False
              and meta["leaked_bytes"] == 0)
    doc = {
        "claim": "not a 2 Gvox/s number",
        "lever": name,
        "nfrag": nfrag,
        "bg": bg,
        "fingerprint": fp,
        "oracle_array_equal": oracle,
        "peak_bytes": peak,
        "peak_gib": peak / GIB,
        "pred_2p16_tracked_gib": (peak / nvox) * 2.16e9 / GIB,
        "leaked": meta["leaked_bytes"],
        "meta": meta,
        "gpu": gpu_state(),
        "pass": ok,
    }
    dest = CACHE / f"b_w2_{name}.json"
    dest.write_text(json.dumps(doc, indent=2) + "\n")
    print(
        f"B {name} nfrag={nfrag} oracle={oracle} peak={peak / GIB:.3f} GiB "
        f"{'PASS' if ok else 'FAIL'} -> {dest}",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
