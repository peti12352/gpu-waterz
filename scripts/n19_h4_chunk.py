#!/usr/bin/env python3
"""N19 H4: Lu–Zlateski chunked MEAN correctness probe on val RAG.

Split nodes into 2 chunks by id; delay boundary edges; compare parents to ParHAC.
Stamp if mismatch. No speed claim. Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import ctypes
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import load_rag  # noqa: E402
from e6r_parhac import DSO, compile_d  # noqa: E402
from n17_gate import DEEP  # noqa: E402
from n19_dead import refuse_or_ok, stamp  # noqa: E402
from p1_make_big_indep import CACHE, card_busy  # noqa: E402
import os

CLAIM = "not a 2 Gvox/s number; not 3090 Ti"
NOTE = ROOT / "notes/N19_H4.md"
OUT = CACHE / "N19_H4.json"


def parhac(u, v, sm, ct, max_id, thr=0.3, eps=0.40):
    compile_d()
    lib = ctypes.CDLL(str(DSO))
    lib.parhac_paper_d.restype = ctypes.c_int
    lib.parhac_paper_d.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    thrs = np.asarray([thr], dtype=np.float64)
    parents = np.empty((1, max_id + 1), dtype=np.uint32)
    stats = np.zeros((1, 3), dtype=np.int64)
    rc = lib.parhac_paper_d(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(1),
        ctypes.c_double(eps),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    return parents[0], int(rc)


def chunked_mean_probe(u, v, sm, ct, max_id, thr=0.3):
    """Simplified boundary-freeze: interior edges merge first, then all."""
    mid = max_id // 2
    interior = []
    boundary = []
    for i in range(len(u)):
        a, b = int(u[i]), int(v[i])
        a_left, b_left = a <= mid, b <= mid
        if a_left == b_left:
            interior.append(i)
        else:
            boundary.append(i)
    # Phase 1: ParHAC on interior-only graph is not identical to freeze semantics;
    # approximate: run full ParHAC as reference; chunked = interior then full.
    # Correctness claim for this simplified probe: final parents == full ParHAC
    # when we only *delay* boundary by running two-pass mean greedy on CPU.
    parent = np.arange(max_id + 1, dtype=np.uint32)
    mean = np.where(ct > 0, sm / ct.astype(np.float64), 0.0)

    def find(x):
        while parent[x] != x:
            x = int(parent[x])
        return x

    def unite(a, b):
        a, b = find(a), find(b)
        if a == b:
            return
        if a > b:
            a, b = b, a
        parent[b] = a

    def merge_list(idxs):
        order = sorted(idxs, key=lambda i: -mean[i])
        for i in order:
            if mean[i] < thr:
                break
            unite(int(u[i]), int(v[i]))

    merge_list(interior)
    merge_list(boundary)
    for i in range(max_id + 1):
        parent[i] = find(i)
    return parent, len(interior), len(boundary)


def main():
    print(f"N19 H4 Lu chunk probe. {CLAIM}.", flush=True)
    msg = refuse_or_ok("N19_H4", force="--force" in sys.argv)
    if msg:
        print(msg, flush=True)
        return 3
    busy = card_busy()
    if busy:
        print(f"N19 H4 REFUSE: {busy}", flush=True)
        return 2
    os.environ.update({**DEEP, "WATERZ_EMIT_HOLES": "1", "WATERZ_AGG_LEVERS": "15"})
    u, v, sm, ct, _fr, max_id = load_rag()
    pref, rc = parhac(u, v, sm, ct, max_id)
    pchunk, n_int, n_bnd = chunked_mean_probe(u, v, sm, ct, max_id)
    # Compare partitions via parent roots equality after remap is hard;
    # compare root labels after compressing both.
    eq = bool(np.array_equal(pref, pchunk))
    # Also relative: same components?
    def comps(p):
        return tuple(sorted(set(int(x) for x in p)))
    # weaker: number of roots
    nref = len(set(pref.tolist()))
    nch = len(set(pchunk.tolist()))
    kill = not eq
    reason = (
        "parents mismatch vs ParHAC (simplified freeze ≠ paper; stamp)"
        if kill else "parents match ParHAC"
    )
    doc = {
        "claim": CLAIM, "array_equal": eq, "n_roots_ref": nref, "n_roots_chunk": nch,
        "n_interior": n_int, "n_boundary": n_bnd, "kill": kill, "reason": reason,
        "speed_claim": False, "ran_216": False,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        f"# N19 H4 Lu chunk probe\n\n{CLAIM}.\n\n"
        f"- eq={eq} roots {nref} vs {nch} int/bnd={n_int}/{n_bnd}\n"
        f"- kill={kill} {reason}\n- no speed claim; no 2.16\n"
    )
    stamp("N19_H4", reason, doc, "notes/N19_H4.md")
    print(json.dumps(doc), flush=True)
    return 0 if not kill else 1


if __name__ == "__main__":
    raise SystemExit(main())
