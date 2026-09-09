#!/usr/bin/env python3
"""N19 val_ws: timed val watershed + identity diagnostic. No agg/four/2.16.

Abort via parent timeout. Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import ctypes
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from b_dev_aff import bind, run_split  # noqa: E402
from p1_make_big_indep import AFF, CACHE, nfrag_bg  # noqa: E402
from task_gate import FRAGMENTS_VAL  # noqa: E402

BG_VAL = 506_568
GOLD = CACHE / "wz_fragments.npy"


def main():
    lib = ctypes.CDLL(str(ROOT / "src/libws_gpu.so"))
    bind(lib)
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    t0 = time.perf_counter()
    seg, meta = run_split(lib, aff, park=False)
    ms = (time.perf_counter() - t0) * 1000.0
    nfrag, bg = nfrag_bg(seg)
    gold = np.load(GOLD) if GOLD.is_file() else None
    eq = bool(gold is not None and np.array_equal(seg, gold))
    doc = {
        "ok": True,
        "ws_ms": ms,
        "nfrag": nfrag,
        "bg": bg,
        "identity": bool(eq and nfrag == FRAGMENTS_VAL and bg == BG_VAL),
        "array_equal": eq,
        "meta": meta,
        "env": {
            "WATERZ_FOLD_FLATTEN": os.environ.get("WATERZ_FOLD_FLATTEN"),
            "WATERZ_SHARE_OFF": os.environ.get("WATERZ_SHARE_OFF"),
            "WATERZ_HOOK_ROOT": os.environ.get("WATERZ_HOOK_ROOT"),
            "WATERZ_PLAYNE_HOOK": os.environ.get("WATERZ_PLAYNE_HOOK"),
            "WATERZ_VCOUNT_PRIV": os.environ.get("WATERZ_VCOUNT_PRIV"),
            "WATERZ_BLOCK_FLAG": os.environ.get("WATERZ_BLOCK_FLAG"),
        },
    }
    print(json.dumps(doc), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
