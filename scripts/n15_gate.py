#!/usr/bin/env python3
"""N15 WS identity gate: two-run byte-ident vs wz_fragments.npy.

Env (UF, parks, fold/share/jump) must be set in the process before import
of the DSO. Parent scripts spawn this as a subprocess. No 2.16 here.
Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import ctypes
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from p1_make_big_indep import (  # noqa: E402
    AFF, CACHE, fingerprint, nfrag_bg, same_partition,
)
from b_dev_aff import bind, run_split  # noqa: E402
from task_gate import FRAGMENTS_VAL  # noqa: E402

BG_VAL = 506_568
GOLD = CACHE / "wz_fragments.npy"
WS_BASE = 2579.0453841909766
WS_GATE = WS_BASE / 1.2


def ident_two_run():
    lib = ctypes.CDLL(str(ROOT / "src/libws_gpu.so"))
    bind(lib)
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    gold = np.load(GOLD)
    seg1, meta1 = run_split(lib, aff, park=True)
    seg2, meta2 = run_split(lib, aff, park=True)
    nfrag, bg = nfrag_bg(seg1)
    fp1, nsz1, _ = fingerprint(seg1)
    fpg, nszg, _ = fingerprint(gold)
    ident = bool(np.array_equal(seg1, gold) and nfrag == FRAGMENTS_VAL
                 and bg == BG_VAL)
    det = bool(np.array_equal(seg1, seg2))
    return {
        "identity": ident,
        "array_equal": bool(np.array_equal(seg1, gold)),
        "run2_array_equal": det,
        "nfrag": nfrag,
        "bg": bg,
        "ndiff_raw": int(np.count_nonzero(seg1 != gold)),
        "same_partition": bool(same_partition(seg1, gold)),
        "fp_eq": bool(fp1 == fpg),
        "fp_pred": fp1,
        "fp_gold": fpg,
        "n_nonzero_sizes_pred": nsz1,
        "n_nonzero_sizes_gold": nszg,
        "peak_bytes": meta1.get("peak_bytes"),
        "meta1": meta1,
        "meta2": meta2,
        "env": {
            "WATERZ_FOLD_FLATTEN": os.environ.get("WATERZ_FOLD_FLATTEN"),
            "WATERZ_SHARE_OFF": os.environ.get("WATERZ_SHARE_OFF"),
            "WATERZ_JUMP_FLATTEN": os.environ.get("WATERZ_JUMP_FLATTEN"),
            "WATERZ_E9B_FOLD_ONLY": os.environ.get("WATERZ_E9B_FOLD_ONLY"),
            "WATERZ_HOOK_ROOT": os.environ.get("WATERZ_HOOK_ROOT"),
            "WATERZ_UF_ALGO": os.environ.get("WATERZ_UF_ALGO"),
        },
    }


if __name__ == "__main__":
    print(json.dumps(ident_two_run()), flush=True)
