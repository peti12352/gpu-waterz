#!/usr/bin/env python3
"""N20_RNN: exact S3 RNN-HAC vs E4/s4 heap parents, then four-T, then timing.

Oracle is s4_fast_cpu (contact-mean heap), not ParHAC. RAC-class: VOI-legal
and serial. Kill as closer if not faster than 1679.9 ms even if VOI PASSes.
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
from t3_memsafe import voi_parent_mmap  # noqa: E402
from task_gate import AFF_THRESHOLDS, BASE_MERGE, BASE_SPLIT, SLACK  # noqa: E402

CACHE = ROOT / "data/cache"
CLAIM = "N20 legal S3 schedule; not a throughput claim"
U32P = ctypes.POINTER(ctypes.c_uint32)
F64P = ctypes.POINTER(ctypes.c_double)
I64P = ctypes.POINTER(ctypes.c_int64)
BEST_AGG = 1679.9


def main():
    d3 = json.loads((CACHE / "N20_D3.json").read_text()) if (
        CACHE / "N20_D3.json").is_file() else {}
    u, v, sm, ct, _fr, max_id = load_rag()
    lib = load_n20()
    thrs = np.asarray(AFF_THRESHOLDS, dtype=np.float64)
    parents_h = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    parents_r = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(thrs), 3), dtype=np.int64)
    pin = CACHE / "N20_D2_parents.npy"
    d2j = json.loads((CACHE / "N20_D2.json").read_text()) if (
        CACHE / "N20_D2.json").is_file() else {}
    if pin.is_file():
        parents_h = np.load(pin)
        rc_h = 1
        heap_ms = float(d2j.get("wall_ms", -1.0))
        heap_src = "N20_D2_parents.npy"
    else:
        t0 = time.perf_counter()
        rc_h = lib.s4_fast_cpu(
            u.ctypes.data_as(U32P), v.ctypes.data_as(U32P),
            sm.ctypes.data_as(F64P), ct.ctypes.data_as(I64P),
            ctypes.c_int64(len(u)), thrs.ctypes.data_as(F64P),
            ctypes.c_int(len(thrs)), parents_h.ctypes.data_as(U32P),
            ctypes.c_uint32(max_id),
        )
        heap_ms = (time.perf_counter() - t0) * 1000.0
        heap_src = "s4_fast_cpu"
    pin_d3 = CACHE / "N20_D3_parents.npy"
    if pin_d3.is_file() and d3:
        parents_r = np.load(pin_d3)
        rc_r = 1
        rnn_ms = float(d3.get("wall_ms", -1.0))
        stats[:, 0] = [d3.get("rounds", {}).get(T, -1) for T in AFF_THRESHOLDS]
        # d3 rounds dict keys may be floats serialized as strings
        if isinstance(d3.get("rounds"), dict):
            for i, T in enumerate(AFF_THRESHOLDS):
                key = T if T in d3["rounds"] else str(T)
                stats[i, 0] = int(d3["rounds"].get(key, stats[i, 0]))
                caps = d3.get("hit_cap") or []
                stats[i, 2] = int(caps[i]) if i < len(caps) else 0
        rnn_src = "N20_D3_parents.npy"
    else:
        t0 = time.perf_counter()
        rc_r = lib.n20_rnn_s3_cpu(
            u.ctypes.data_as(U32P), v.ctypes.data_as(U32P),
            sm.ctypes.data_as(F64P), ct.ctypes.data_as(I64P),
            ctypes.c_int64(len(u)), thrs.ctypes.data_as(F64P),
            ctypes.c_int(len(thrs)), ctypes.c_int64(5000),
            parents_r.ctypes.data_as(U32P), ctypes.c_uint32(max_id),
            stats.ctypes.data_as(I64P),
        )
        rnn_ms = (time.perf_counter() - t0) * 1000.0
        rnn_src = "n20_rnn_s3_cpu"
    rows = []
    four_ok = True
    same = []
    for i, T in enumerate(AFF_THRESHOLDS):
        same.append(bool(np.array_equal(parents_h[i], parents_r[i])))
        split, merge, nseg = voi_parent_mmap(parents_r[i])
        sl, ml = BASE_SPLIT[T] + SLACK, BASE_MERGE[T] + SLACK
        ok = split <= sl and merge <= ml
        four_ok = four_ok and ok
        rows.append({
            "T": T, "split": split, "merge": merge, "nseg": nseg,
            "ok": bool(ok), "parents_eq_heap": same[-1],
            "rounds": int(stats[i, 0]), "hit_cap": int(stats[i, 2]),
        })
    faster = rnn_ms < BEST_AGG
    kill_closer = (not four_ok) or (not faster)
    reason = (
        "correct_but_serial" if four_ok and not faster else
        "four-T FAIL or cap" if not four_ok else "faster than ParHAC"
    )
    doc = {
        "claim": CLAIM, "rc_heap": int(rc_h), "rc_rnn": int(rc_r),
        "heap_src": heap_src, "rnn_src": rnn_src,
        "heap_ms": heap_ms, "rnn_ms": rnn_ms, "best_agg_ms": BEST_AGG,
        "parents_eq_heap": same, "four_ok": four_ok, "rows": rows,
        "d3": {k: d3.get(k) for k in ("rounds", "hit_cap", "go_gpu_rnn")},
        "kill_as_closer": kill_closer, "reason": reason,
        "go_gpu": bool(d3.get("go_gpu_rnn")), "ran_216": False,
    }
    from n20_res import attach  # noqa: E402
    doc = attach(doc, "N20_RNN")
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "N20_RNN.json").write_text(json.dumps(doc, indent=2) + "\n")
    (ROOT / "notes/N20_RNN.md").write_text(
        f"# N20 RNN\n\n{CLAIM}.\n\n- heap_ms={heap_ms:.1f} rnn_ms={rnn_ms:.1f} "
        f"vs ParHAC {BEST_AGG}\n- four_ok={four_ok} parents_eq={same}\n"
        f"- {reason}\n- GPU port: {doc['go_gpu']}\n"
    )
    stamp("N20_RNN", reason, doc, "notes/N20_RNN.md")
    print(json.dumps(doc, indent=2), flush=True)
    return 0 if rc_r == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
