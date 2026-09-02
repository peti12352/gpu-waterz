#!/usr/bin/env python3
"""P0v: leftover after relative-contact. Writes p0_r32_leftover.json."""
from __future__ import annotations

import ctypes
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import CACHE, SO, compile_so, frag_sizes, load_rag  # noqa: E402


def parse(text: str) -> dict:
    rows = []
    for m in re.finditer(
        r"P0v gamma=([0-9.]+) alpha=([0-9.]+) T=([0-9.]+) n_residual=(\d+) n_high=([0-9.]+) giant_if_union=([0-9.]+)",
        text,
    ):
        rows.append({
            "gamma": float(m.group(1)),
            "alpha": float(m.group(2)),
            "T": float(m.group(3)),
            "n_residual": int(m.group(4)),
            "n_high": int(float(m.group(5))),
            "giant_if_union": float(m.group(6)),
        })
    l36 = any(r["n_residual"] < 50_000 for r in rows)
    return {"p0v": rows, "l36": l36, "branch": ["L36"] if l36 else ["R36"]}


def main():
    compile_so()
    u, v, sm, ct, fr, max_id = load_rag()
    sz = frag_sizes(fr, max_id)
    total = int(sz[1:].sum())
    lib = ctypes.CDLL(str(SO))
    lib.p0_rel_leftover_cpu.restype = ctypes.c_int
    lib.p0_rel_leftover_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
        ctypes.c_uint32, ctypes.c_int64,
    ]
    rc = lib.p0_rel_leftover_cpu(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        sz.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        ctypes.c_uint32(max_id),
        ctypes.c_int64(total),
    )
    print(f"P0v rc={rc}", flush=True)
    return 0 if rc == 1 else 1
    # JSON is written by run_e6r_tree from the captured P0v lines.


if __name__ == "__main__":
    raise SystemExit(main())
