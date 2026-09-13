#!/usr/bin/env python3
"""N20_D2: dendrogram height h of exact S3 heap on the full val RAG.

Abboud arXiv:2404.14730: NN-chain average-linkage is O(m * h * log n).
P0i height=9 was a 50k-edge sample. This is the full 7.5M-edge RAG.
"""
from __future__ import annotations

import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import load_rag  # noqa: E402
from n19_dead import stamp  # noqa: E402
from n20_lib import load_n20  # noqa: E402
from task_gate import AFF_THRESHOLDS  # noqa: E402

CACHE = ROOT / "data/cache"
CLAIM = "N20 diagnostic; not a throughput claim"
U32P = ctypes.POINTER(ctypes.c_uint32)
F64P = ctypes.POINTER(ctypes.c_double)
I64P = ctypes.POINTER(ctypes.c_int64)


def main():
    import socket
    print(f"N20_D2 host={socket.gethostname()} loading rag", flush=True)
    u, v, sm, ct, _fr, max_id = load_rag()
    print(f"N20_D2 nedge={len(u)} max_id={max_id}", flush=True)
    lib = load_n20()
    thrs = np.asarray(AFF_THRESHOLDS, dtype=np.float64)
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(thrs), 3), dtype=np.int64)
    t0 = time.perf_counter()
    rc = lib.n20_lw_heap_cpu(
        u.ctypes.data_as(U32P), v.ctypes.data_as(U32P),
        sm.ctypes.data_as(F64P), ct.ctypes.data_as(I64P),
        ctypes.c_int64(len(u)), thrs.ctypes.data_as(F64P),
        ctypes.c_int(len(thrs)), ctypes.c_int(0),
        parents.ctypes.data_as(U32P), ctypes.c_uint32(max_id),
        stats.ctypes.data_as(I64P),
    )
    ms = (time.perf_counter() - t0) * 1000.0
    heights = [int(stats[i, 1]) for i in range(len(thrs))]
    hmax = max(heights)
    # Exact chain work vs 1680 ms: if h is large, O(m h log n) is not a closer.
    from n20_res import attach  # noqa: E402
    doc = {
        "claim": CLAIM, "rc": int(rc), "wall_ms": ms,
        "heights_by_T": dict(zip(AFF_THRESHOLDS, heights)),
        "h_max": hmax,
        "merges_by_T": [int(stats[i, 0]) for i in range(len(thrs))],
        "p0i_sample_h": 9,
        "chain_closer": False if hmax >= 30 else "maybe",
        "reason": (
            f"full-RAG S3 heap height={hmax} (P0i sample was 9); "
            "Abboud O(m h log n) is not a 1680ms closer" if hmax >= 30
            else f"height={hmax} small; still serial heap"
        ),
        "best_agg_ms": 1679.9,
        "ran_216": False,
    }
    from n20_res import attach  # noqa: E402
    doc = attach(doc, "N20_D2")
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / "N20_D2_parents.npy", parents)
    (CACHE / "N20_D2.json").write_text(json.dumps(doc, indent=2) + "\n")
    (ROOT / "notes/N20_D2.md").write_text(
        f"# N20 D2 height\n\n{CLAIM}.\n\n- h_max={hmax} wall_ms={ms:.1f}\n"
        f"- {doc['reason']}\n"
    )
    stamp("N20_D2", doc["reason"], doc, "notes/N20_D2.md")
    print(json.dumps(doc, indent=2), flush=True)
    return 0 if rc == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
