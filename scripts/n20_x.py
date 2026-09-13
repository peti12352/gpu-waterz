#!/usr/bin/env python3
"""N20_X4 complete-link and N20_X5 WPGMA. Expected four-T FAIL. No 2.16.

SeqHAC arXiv:2106.05610: complete = min similarity on the cut.
Murtagh/Contreras: WPGMA alpha=1/2. Contact-count UPGMA is waterz S3, not this.
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
CLAIM = "N20 atlas hole; not a throughput claim"
U32P = ctypes.POINTER(ctypes.c_uint32)
F64P = ctypes.POINTER(ctypes.c_double)
I64P = ctypes.POINTER(ctypes.c_int64)


def run_one(name, linkage, reason_expect):
    import socket
    from n20_res import wait_idle
    print(f"{name} host={socket.gethostname()} linkage={linkage}. {CLAIM}.", flush=True)
    print(json.dumps(wait_idle(name), indent=2), flush=True)
    u, v, sm, ct, _fr, max_id = load_rag()
    lib = load_n20()
    thrs = np.asarray(AFF_THRESHOLDS, dtype=np.float64)
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(thrs), 3), dtype=np.int64)
    t0 = time.perf_counter()
    rc = lib.n20_lw_heap_cpu(
        u.ctypes.data_as(U32P), v.ctypes.data_as(U32P),
        sm.ctypes.data_as(F64P), ct.ctypes.data_as(I64P),
        ctypes.c_int64(len(u)), thrs.ctypes.data_as(F64P),
        ctypes.c_int(len(thrs)), ctypes.c_int(linkage),
        parents.ctypes.data_as(U32P), ctypes.c_uint32(max_id),
        stats.ctypes.data_as(I64P),
    )
    ms = (time.perf_counter() - t0) * 1000.0
    rows = []
    four_ok = True
    for i, T in enumerate(AFF_THRESHOLDS):
        split, merge, nseg = voi_parent_mmap(parents[i])
        sl, ml = BASE_SPLIT[T] + SLACK, BASE_MERGE[T] + SLACK
        ok = split <= sl and merge <= ml
        four_ok = four_ok and ok
        rows.append({
            "T": T, "split": split, "merge": merge, "nseg": nseg,
            "ok": bool(ok), "limit_s": sl, "limit_m": ml,
            "merges": int(stats[i, 0]), "height": int(stats[i, 1]),
            "heap_ms": int(stats[i, 2]),
        })
    kill = (rc != 1) or (not four_ok)
    reason = reason_expect if not four_ok else (
        "rc!=1" if rc != 1 else "unexpected four-T PASS")
    doc = {
        "claim": CLAIM, "exp_id": name, "linkage": linkage, "rc": int(rc),
        "wall_ms": ms, "rows": rows, "four_ok": four_ok,
        "kill": kill, "reason": reason, "ran_216": False,
    }
    from n20_res import attach  # noqa: E402
    doc = attach(doc, name)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{name}.json").write_text(json.dumps(doc, indent=2) + "\n")
    (ROOT / f"notes/{name}.md").write_text(
        f"# {name}\n\n{CLAIM}.\n\n- wall_ms={ms:.1f} four_ok={four_ok} "
        f"kill={kill} {reason}\n- no 2.16\n"
    )
    stamp(name, reason, {k: doc[k] for k in ("four_ok", "wall_ms", "rows")},
          f"notes/{name}.md")
    print(json.dumps({"exp": name, "four_ok": four_ok, "kill": kill,
                      "rows": rows}, indent=2), flush=True)
    return 0 if not kill else 1


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    rc = 0
    if which in ("x4", "both"):
        rc |= run_one("N20_X4", 1, "complete-link under-merge (expected)")
    if which in ("x5", "both"):
        rc |= run_one("N20_X5", 2, "WPGMA != contact-n UPGMA (expected)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
