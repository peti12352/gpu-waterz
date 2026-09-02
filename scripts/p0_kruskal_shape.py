#!/usr/bin/env python3
"""P0m/n/r: constrained CC giant / degree / residual. Writes p0_kruskal_shape.json."""
from __future__ import annotations

import ctypes
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import CACHE, SO, compile_so, frag_sizes, load_rag  # noqa: E402

E4 = {0.2: 294518, 0.3: 322167, 0.4: 345376, 0.5: 379876}


def parse(text: str) -> dict:
    cells = []
    for m in re.finditer(
        r"P0m tau=([0-9.]+) a=(\d+) keep=(\d+) ncc=(\d+) max_vox=(\d+) giant=([0-9.]+)",
        text,
    ):
        cells.append({
            "tau": float(m.group(1)),
            "a": int(m.group(2)),
            "keep": int(m.group(3)),
            "ncc": int(m.group(4)),
            "max_vox": int(m.group(5)),
            "giant": float(m.group(6)),
        })
    degs = {}
    for m in re.finditer(
        r"P0n (\S+) ne=(\d+) nact=(\d+) deg_max=(\d+) p50=(\d+) p99=(\d+) giant_vfrac=([0-9.]+)",
        text,
    ):
        degs[m.group(1)] = {
            "ne": int(m.group(2)), "nact": int(m.group(3)),
            "deg_max": int(m.group(4)), "p50": int(m.group(5)),
            "p99": int(m.group(6)), "giant_vfrac": float(m.group(7)),
        }
    res = []
    for m in re.finditer(
        r"P0r tau=([0-9.]+) a=(\d+) nsuper=(\d+) residual_T=([0-9.]+) nedge=(\d+) giant=([0-9.]+)",
        text,
    ):
        res.append({
            "tau": float(m.group(1)), "a": int(m.group(2)),
            "nsuper": int(m.group(3)), "T": float(m.group(4)),
            "nedge": int(m.group(5)), "giant": float(m.group(6)),
        })
    a27 = []
    for c in cells:
        if abs(c["tau"] - 0.2) > 1e-9 and abs(c["tau"] - 0.3) > 1e-9 \
                and abs(c["tau"] - 0.4) > 1e-9 and abs(c["tau"] - 0.5) > 1e-9:
            continue
        if c["a"] < 2:
            continue
        if c["giant"] > 0.05:
            continue
        e4 = E4.get(round(c["tau"], 1))
        if e4 is None:
            continue
        if 0.85 * e4 <= c["ncc"] <= 1.15 * e4:
            a27.append(c)
    branch = ["A27"] if a27 else ["Z25", "F23", "R24", "C28", "S26"]
    return {"p0m": cells, "p0n": degs, "p0r": res, "a27_hits": a27, "branch": branch}


def main():
    compile_so()
    u, v, sm, ct, fr, max_id = load_rag()
    sz = frag_sizes(fr, max_id)
    total = int(sz[1:].sum())
    print(f"P0k edges={len(u)} max_id={max_id} total_vox={total}", flush=True)
    lib = ctypes.CDLL(str(SO))
    lib.p0_kruskal_shape_cpu.restype = ctypes.c_int
    lib.p0_kruskal_shape_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_uint32,
        ctypes.c_int64,
    ]
    rc = lib.p0_kruskal_shape_cpu(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        sz.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        ctypes.c_uint32(max_id),
        ctypes.c_int64(total),
    )
    print(f"P0k rc={rc}", flush=True)
    if rc != 1:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
