#!/usr/bin/env python3
"""N20_LU2: paper Algorithm 2 with spatial B^C vs s4 heap.

Lu, Zlateski, Seung arXiv:2106.10795 Alg 2: freeze nodes that touch a fake
chunk boundary; S3 heap on the rest; residual heap with B empty.
H4 used node-id split. This uses fragment bbox vs z=midplane.
Oracle is the heap, not ParHAC.
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
CLAIM = "N20 paper Alg 2; not a throughput claim"
U32P = ctypes.POINTER(ctypes.c_uint32)
U8P = ctypes.POINTER(ctypes.c_uint8)
F64P = ctypes.POINTER(ctypes.c_double)
I64P = ctypes.POINTER(ctypes.c_int64)
BEST_AGG = 1679.9
CHUNK = 1 << 20


def fragment_z_bbox(fr_path, max_id):
    fr = np.load(fr_path, mmap_mode="r")
    zdim = int(fr.shape[0])
    mn = np.full(max_id + 1, zdim, dtype=np.int32)
    mx = np.full(max_id + 1, -1, dtype=np.int32)
    for z in range(zdim):
        sl = np.asarray(fr[z]).reshape(-1)
        for i in range(0, sl.size, CHUNK):
            ids = sl[i:i + CHUNK]
            if ids.size == 0:
                continue
            # unique ids in this z-slice
            u = np.unique(ids)
            u = u[u != 0]
            if u.size == 0:
                continue
            mn[u] = np.minimum(mn[u], z)
            mx[u] = np.maximum(mx[u], z)
        print(f"  z={z}/{zdim}", flush=True)
    return mn, mx, zdim


def main():
    u, v, sm, ct, _fr, max_id = load_rag()
    fr_path = CACHE / "wz_fragments.npy"
    print("N20_LU2 bbox...", flush=True)
    from n20_res import wait_idle
    print(json.dumps(wait_idle("N20_LU2"), indent=2), flush=True)
    t0 = time.perf_counter()
    mn, mx, zdim = fragment_z_bbox(fr_path, max_id)
    plane = zdim // 2
    boundary = ((mn <= plane) & (mx >= plane)).astype(np.uint8)
    boundary[0] = 0
    n_b = int(boundary.sum())
    bbox_ms = (time.perf_counter() - t0) * 1000.0
    lib = load_n20()
    thrs = np.asarray(AFF_THRESHOLDS, dtype=np.float64)
    parents_h = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    parents_l = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
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
    t0 = time.perf_counter()
    rc_l = lib.n20_lu_alg2_cpu(
        u.ctypes.data_as(U32P), v.ctypes.data_as(U32P),
        sm.ctypes.data_as(F64P), ct.ctypes.data_as(I64P),
        ctypes.c_int64(len(u)), boundary.ctypes.data_as(U8P),
        thrs.ctypes.data_as(F64P), ctypes.c_int(len(thrs)),
        parents_l.ctypes.data_as(U32P), ctypes.c_uint32(max_id),
        stats.ctypes.data_as(I64P),
    )
    lu_ms = (time.perf_counter() - t0) * 1000.0
    rows = []
    four_ok = True
    same = []
    for i, T in enumerate(AFF_THRESHOLDS):
        same.append(bool(np.array_equal(parents_h[i], parents_l[i])))
        split, merge, nseg = voi_parent_mmap(parents_l[i])
        sl, ml = BASE_SPLIT[T] + SLACK, BASE_MERGE[T] + SLACK
        ok = split <= sl and merge <= ml
        four_ok = four_ok and ok
        rows.append({
            "T": T, "split": split, "merge": merge, "nseg": nseg, "ok": bool(ok),
            "parents_eq_heap": same[-1],
            "interior_merges": int(stats[i, 0]),
            "frozen_pops": int(stats[i, 1]),
            "nres": int(stats[i, 2]),
        })
    nres0 = rows[1]["nres"] if len(rows) > 1 else 0
    no_speed = nres0 > 0.2 * len(u)
    reason = (
        "no_speed_path residual large" if four_ok and no_speed else
        "correct_but_serial" if four_ok and lu_ms >= BEST_AGG else
        "four-T FAIL" if not four_ok else "ok"
    )
    doc = {
        "claim": CLAIM, "plane": plane, "zdim": zdim, "n_boundary": n_b,
        "bbox_ms": bbox_ms, "rc_heap": int(rc_h), "heap_src": heap_src,
        "rc_lu": int(rc_l),
        "heap_ms": heap_ms, "lu_ms": lu_ms, "best_agg_ms": BEST_AGG,
        "parents_eq_heap": same, "four_ok": four_ok, "rows": rows,
        "no_speed_path": bool(no_speed), "reason": reason, "ran_216": False,
    }
    from n20_res import attach  # noqa: E402
    doc = attach(doc, "N20_LU2")
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "N20_LU2.json").write_text(json.dumps(doc, indent=2) + "\n")
    (ROOT / "notes/N20_LU2.md").write_text(
        f"# N20 LU2\n\n{CLAIM}.\n\n- plane z={plane}/{zdim} n_boundary={n_b}\n"
        f"- heap_ms={heap_ms:.1f} lu_ms={lu_ms:.1f} vs {BEST_AGG}\n"
        f"- four_ok={four_ok} parents_eq={same}\n- {reason}\n"
    )
    stamp("N20_LU2", reason, doc, "notes/N20_LU2.md")
    print(json.dumps(doc, indent=2), flush=True)
    return 0 if rc_l == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
