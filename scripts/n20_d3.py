#!/usr/bin/env python3
"""N20_D3: true S3 RNN rounds with cap 5000. P0w was cap=200 and did not reweight.

Bruynooghe 1977: RNN is exact for reducible average. This uses Graph S3 unite.
GPU NN-chain only if rounds finish well below cap.
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
CAP = 5000
U32P = ctypes.POINTER(ctypes.c_uint32)
F64P = ctypes.POINTER(ctypes.c_double)
I64P = ctypes.POINTER(ctypes.c_int64)


def main():
    import socket
    from n20_res import wait_idle
    print(f"N20_D3 host={socket.gethostname()} loading rag", flush=True)
    print(json.dumps(wait_idle("N20_D3"), indent=2), flush=True)
    u, v, sm, ct, _fr, max_id = load_rag()
    print(f"N20_D3 nedge={len(u)} max_id={max_id} cap={CAP}", flush=True)
    lib = load_n20()
    thrs = np.asarray(AFF_THRESHOLDS, dtype=np.float64)
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(thrs), 3), dtype=np.int64)
    t0 = time.perf_counter()
    rc = lib.n20_rnn_s3_cpu(
        u.ctypes.data_as(U32P), v.ctypes.data_as(U32P),
        sm.ctypes.data_as(F64P), ct.ctypes.data_as(I64P),
        ctypes.c_int64(len(u)), thrs.ctypes.data_as(F64P),
        ctypes.c_int(len(thrs)), ctypes.c_int64(CAP),
        parents.ctypes.data_as(U32P), ctypes.c_uint32(max_id),
        stats.ctypes.data_as(I64P),
    )
    ms = (time.perf_counter() - t0) * 1000.0
    rounds = [int(stats[i, 0]) for i in range(len(thrs))]
    caps = [int(stats[i, 2]) for i in range(len(thrs))]
    hit = any(caps)
    go_gpu = (not hit) and max(rounds) <= 30
    doc = {
        "claim": CLAIM, "rc": int(rc), "wall_ms": ms, "cap": CAP,
        "rounds": dict(zip(AFF_THRESHOLDS, rounds)),
        "merges": [int(stats[i, 1]) for i in range(len(thrs))],
        "hit_cap": caps,
        "p0w_cap": 200,
        "go_gpu_rnn": go_gpu,
        "reason": (
            "still >200 S3-RNN rounds; no GPU NN-chain"
            if hit or max(rounds) > 200
            else "finished under cap"
        ),
        "ran_216": False,
        "best_agg_ms": 1679.9,
    }
    from n20_res import attach  # noqa: E402
    doc = attach(doc, "N20_D3")
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / "N20_D3_parents.npy", parents)
    (CACHE / "N20_D3.json").write_text(json.dumps(doc, indent=2) + "\n")
    (ROOT / "notes/N20_D3.md").write_text(
        f"# N20 D3 RNN rounds\n\n{CLAIM}.\n\n- rounds={rounds} cap={CAP} "
        f"hit={hit} wall_ms={ms:.1f}\n- go_gpu_rnn={go_gpu}\n"
    )
    stamp("N20_D3", doc["reason"], doc, "notes/N20_D3.md")
    print(json.dumps(doc, indent=2), flush=True)
    return 0 if rc == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
