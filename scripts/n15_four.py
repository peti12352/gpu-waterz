#!/usr/bin/env python3
"""N15 four-T ε=0.08 gate. Env from caller. Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import ctypes
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import AFF_THRESHOLDS, grade_parents, load_rag  # noqa: E402
from e6r_parhac import DSO, compile_d  # noqa: E402
from n12_compact import bind, run_parhac  # noqa: E402


def main():
    compile_d()
    lib = bind(ctypes.CDLL(str(DSO)))
    u, v, sm, ct, fr, max_id = load_rag()
    four = np.asarray(AFF_THRESHOLDS, dtype=np.float64)
    rc4, p4, st4, ms4 = run_parhac(lib, u, v, sm, ct, max_id, four, 0.08)
    four_ok = False
    tag = os.environ.get("WATERZ_FOUR_TAG", "n15_four")
    if rc4 == 1:
        four_ok = grade_parents(p4, fr, tag, tag)
    doc = {
        "four_pass": bool(four_ok),
        "rc": int(rc4),
        "ms": ms4,
        "thresholds": list(AFF_THRESHOLDS),
        "env": {
            "WATERZ_FUSE_DIRTY": os.environ.get("WATERZ_FUSE_DIRTY"),
            "WATERZ_STICKY_SZ0": os.environ.get("WATERZ_STICKY_SZ0"),
            "WATERZ_MAX_OUTER": os.environ.get("WATERZ_MAX_OUTER"),
        },
    }
    print(json.dumps(doc), flush=True)
    return 0 if four_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
