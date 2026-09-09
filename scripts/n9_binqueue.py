#!/usr/bin/env python3
"""N9 Track D: CPU MEAN + BinQueue T=0.3 VOI. Stock waterz, not hist-q, not X1.

N in {256, 1024, 4096}. Serial FIFO inside the bin. Live S3 mean.
Kill on VOI fail. No CUDA port from this script.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from p1_make_big_indep import CACHE  # noqa: E402
from task_gate import BASE_MERGE, BASE_SPLIT, SLACK  # noqa: E402
from voi_numpy import voi_split_merge  # noqa: E402

T = 0.3
SCORE = 0.7
MEAN = "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>"
NS = (256, 1024, 4096)
OUT = CACHE / "n9_binqueue.json"
NOTE = ROOT / "notes/N9_BINS.md"
AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
GT = ROOT / "data/ws_bounty/cremiA_val/gt.h5"
FRAG = CACHE / "wz_fragments.npy"


def grade(split, merge):
    sl, ml = BASE_SPLIT[T] + SLACK, BASE_MERGE[T] + SLACK
    return bool(split <= sl and merge <= ml), sl, ml


def one(aff, fr, gt, n):
    import waterz
    t0 = time.perf_counter()
    lab = None
    for seg in waterz.agglomerate(
        aff,
        [SCORE],
        fragments=fr,
        scoring_function=MEAN,
        discretize_queue=n,
        aff_threshold_low=1e-4,
        aff_threshold_high=0.9999,
        force_rebuild=True,
    ):
        lab = np.array(seg, dtype=np.uint32, copy=True)
    sec = time.perf_counter() - t0
    split, merge = voi_split_merge(lab, gt)
    ok, sl, ml = grade(split, merge)
    nseg = int((np.unique(lab) != 0).sum())
    return {
        "N": n,
        "sec": sec,
        "voi_split": float(split),
        "voi_merge": float(merge),
        "limit_split": sl,
        "limit_merge": ml,
        "nseg": nseg,
        "pass": ok,
    }


def write_note(doc):
    lines = [
        "# N9 Track D: MEAN + BinQueue T=0.3",
        "",
        "Stock `OneMinus<MeanAffinity<...>>`. Serial FIFO (`BinQueue`). "
        "Not hist-q, not X1. T=0.3 only. Limits: split <= 0.4738, merge <= 0.2611.",
        "",
    ]
    for r in doc["rows"]:
        lines.append(
            f"- N={r['N']}: split={r['voi_split']:.6f} merge={r['voi_merge']:.6f} "
            f"nseg={r['nseg']} sec={r['sec']:.1f} {'PASS' if r['pass'] else 'FAIL'}"
        )
    lines += [
        "",
        f"any_pass={doc['any_pass']}. GPU bucket only if PASS and propose visits "
        "drop >=5x and visits translate to wall (N7: E2 visits != wall).",
        "If FAIL: do not tune N, do not port.",
        "",
    ]
    NOTE.write_text("\n".join(lines))


def main():
    print("N9 MEAN+BinQueue T=0.3. No CUDA. Not a 2 Gvox/s claim.", flush=True)
    if not FRAG.is_file():
        print("N9 BINS FAIL missing wz_fragments.npy", flush=True)
        return 1
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:].astype(np.float32) / 255.0)
    with h5py.File(GT, "r") as f:
        key = "gt" if "gt" in f else "labels"
        gt = np.ascontiguousarray(f[key][:].astype(np.uint32))
    rows = []
    for n in NS:
        # waterz.agglomerate mutates the fragment buffer in place.
        fr = np.load(FRAG).astype(np.uint64, copy=True)
        print(f"N9 BINS N={n} start", flush=True)
        r = one(aff, fr, gt, n)
        del fr
        rows.append(r)
        print(
            f"N9 BINS N={n} split={r['voi_split']:.6f} merge={r['voi_merge']:.6f} "
            f"{'PASS' if r['pass'] else 'FAIL'} sec={r['sec']:.1f}",
            flush=True,
        )
    any_pass = any(r["pass"] for r in rows)
    doc = {
        "claim": "not a 2 Gvox/s number",
        "scoring_function": MEAN,
        "threshold_score": SCORE,
        "aff_threshold": T,
        "rows": rows,
        "any_pass": any_pass,
        "gpu_bucket": False,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    write_note(doc)
    print(f"N9 BINS any_pass={any_pass} -> {OUT} {NOTE}", flush=True)
    return 0 if any_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
