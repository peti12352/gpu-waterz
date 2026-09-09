#!/usr/bin/env python3
"""N19 X2: GASP Average on cached RAG — VOI kill expected.

Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import load_rag  # noqa: E402
from n19_dead import refuse_or_ok, stamp  # noqa: E402
from p1_make_big_indep import CACHE, card_busy  # noqa: E402
from t3_memsafe import grade_t3, voi_parent_mmap  # noqa: E402

CLAIM = "not a 2 Gvox/s number; not 3090 Ti"
NOTE = ROOT / "notes/N19_X2.md"
OUT = CACHE / "N19_X2.json"


def gasp_average_cpu(u, v, sm, ct, max_id, thr):
    """Greedy average-linkage like merge while mean >= thr (signed-less)."""
    parent = np.arange(max_id + 1, dtype=np.uint32)
    mean = np.where(ct > 0, sm / ct.astype(np.float64), 0.0)
    order = np.argsort(-mean)
    for i in order:
        if mean[i] < thr:
            break
        a, b = int(u[i]), int(v[i])
        # find
        while parent[a] != a:
            a = int(parent[a])
        while parent[b] != b:
            b = int(parent[b])
        if a == b:
            continue
        if a > b:
            a, b = b, a
        parent[b] = a
    # compress
    for i in range(max_id + 1):
        a = i
        while parent[a] != a:
            a = int(parent[a])
        parent[i] = a
    return parent


def main():
    print(f"N19 X2 GASP Average. {CLAIM}.", flush=True)
    msg = refuse_or_ok("N19_X2", force="--force" in sys.argv)
    if msg:
        print(msg, flush=True)
        return 3
    busy = card_busy()
    if busy:
        print(f"N19 X2 REFUSE: {busy}", flush=True)
        return 2
    u, v, sm, ct, _fr, max_id = load_rag()
    t0 = time.perf_counter()
    parents = gasp_average_cpu(u, v, sm, ct, max_id, 0.3)
    ms = (time.perf_counter() - t0) * 1000.0
    split, merge, nseg = voi_parent_mmap(parents)
    ok, _, _ = grade_t3(split, merge)
    kill = not ok
    reason = "VOI FAIL (expected partition-class)" if kill else "unexpected PASS"
    doc = {
        "claim": CLAIM, "ms": ms, "split": split, "merge": merge, "nseg": nseg,
        "voi_ok": bool(ok), "four_ok": False, "kill": kill, "reason": reason,
        "ran_216": False,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        f"# N19 X2 GASP Average\n\n{CLAIM}.\n\n"
        f"- ms={ms:.1f} voi_ok={ok} kill={kill}\n- no 2.16 on FAIL\n"
    )
    stamp("N19_X2", reason, doc, "notes/N19_X2.md")
    print(json.dumps(doc), flush=True)
    return 0 if not kill else 1


if __name__ == "__main__":
    raise SystemExit(main())
