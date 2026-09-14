#!/usr/bin/env python3
"""N19 H2: NNG / reciprocal-NN filter before ParHAC (Haris 1998).

Keep edge if each endpoint is among the other's top-1 by mean aff.
Then T=0.3 VOI + four-T. No 2.16 on FAIL.
Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import json
import os
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
from n13_baseline import parks_env  # noqa: E402
from n17_gate import DEEP  # noqa: E402
from n19_dead import refuse_or_ok, stamp  # noqa: E402
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402
from t3_memsafe import grade_t3, voi_parent_mmap  # noqa: E402
from n15_four import main as four_main  # may not work; run subprocess

CLAIM = "not a 2 Gvox/s number; not 3090 Ti"
NOTE = ROOT / "notes/N19_H2.md"
OUT = CACHE / "N19_H2.json"


def nng_filter(u, v, sm, ct, max_id):
    """Keep reciprocal nearest-neighbor edges by mean affinity."""
    mean = np.where(ct > 0, sm / ct.astype(np.float64), 0.0)
    best = np.full(max_id + 1, -1, dtype=np.int64)
    best_m = np.full(max_id + 1, -1.0, dtype=np.float64)
    for i in range(len(u)):
        a, b, m = int(u[i]), int(v[i]), float(mean[i])
        if m > best_m[a]:
            best_m[a] = m
            best[a] = i
        if m > best_m[b]:
            best_m[b] = m
            best[b] = i
    keep = np.zeros(len(u), dtype=bool)
    for i in range(len(u)):
        a, b = int(u[i]), int(v[i])
        if best[a] < 0 or best[b] < 0:
            continue
        # reciprocal: a's best edge endpoint is b and b's is a
        ea, eb = int(best[a]), int(best[b])
        oa = int(v[ea]) if int(u[ea]) == a else int(u[ea])
        ob = int(v[eb]) if int(u[eb]) == b else int(u[eb])
        if oa == b and ob == a:
            keep[i] = True
            keep[ea] = True
            keep[eb] = True
    # also keep the best edge itself for each node
    for a in range(max_id + 1):
        if best[a] >= 0:
            keep[best[a]] = True
    idx = np.nonzero(keep)[0]
    return u[idx].copy(), v[idx].copy(), sm[idx].copy(), ct[idx].copy(), int(idx.size)


def run_parhac(u, v, sm, ct, max_id, thr, eps):
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
    t0 = time.perf_counter()
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
    ms = (time.perf_counter() - t0) * 1000.0
    return parents[0], ms, int(rc), stats[0]


def main():
    print(f"N19 H2 NNG filter. {CLAIM}.", flush=True)
    msg = refuse_or_ok("N19_H2", force="--force" in sys.argv)
    if msg:
        print(msg, flush=True)
        return 3
    busy = card_busy()
    if busy:
        print(f"N19 H2 REFUSE: {busy}", flush=True)
        return 2
    os.environ.update({**DEEP, "WATERZ_EMIT_HOLES": "1", "WATERZ_AGG_LEVERS": "15"})
    u, v, sm, ct, fr, max_id = load_rag()
    u2, v2, sm2, ct2, nkeep = nng_filter(u, v, sm, ct, max_id)
    print(f"N19 H2 edges {len(u)} -> {nkeep}", flush=True)
    p1, ms1, rc1, st1 = run_parhac(u2, v2, sm2, ct2, max_id, 0.3, 0.40)
    split, merge, nseg = voi_parent_mmap(p1)
    ok, _, _ = grade_t3(split, merge)
    # four-T on filtered graph via temporary: use grade at 0.08 multi
    from _agg_common import AFF_THRESHOLDS, grade_parents
    from n12_compact import bind, run_parhac as rp
    lib = bind(ctypes.CDLL(str(DSO)))
    four = np.asarray(AFF_THRESHOLDS, dtype=np.float64)
    rc4, p4, st4, ms4 = rp(lib, u2, v2, sm2, ct2, max_id, four, 0.08)
    four_ok = False
    if rc4 == 1:
        four_ok = grade_parents(p4, fr, "n19_h2", "n19_h2")
    kill = not ok or not four_ok
    reason = "VOI FAIL" if not ok else ("four-T FAIL" if not four_ok else "ok")
    doc = {
        "claim": CLAIM, "n_edges_in": len(u), "n_edges_out": nkeep,
        "ms_t03": ms1, "split": split, "merge": merge, "nseg": nseg,
        "voi_ok": bool(ok), "four_ok": bool(four_ok), "kill": kill,
        "reason": reason, "ms_four": ms4,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        f"# N19 H2 NNG\n\n{CLAIM}.\n\n"
        f"- edges {len(u)}->{nkeep}\n"
        f"- voi_ok={ok} four_ok={four_ok} kill={kill} reason={reason}\n"
        f"- no 2.16 on FAIL\n"
    )
    if kill:
        stamp("N19_H2", reason, doc, "notes/N19_H2.md")
    print(json.dumps(doc), flush=True)
    return 0 if not kill else 1


if __name__ == "__main__":
    raise SystemExit(main())
