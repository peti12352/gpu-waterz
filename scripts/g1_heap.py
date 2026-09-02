#!/usr/bin/env python3
"""G1: faithful vendored S4 heap on E3 fragments. VOI vs voi.csv, +0.005."""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import waterz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ref_cpu import extract_parent, heap_from_arrays, rag_arrays  # noqa: E402

AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
GT = ROOT / "data/ws_bounty/cremiA_val/gt.h5"
VOI = ROOT / "data/ws_bounty/baseline/voi.csv"
CACHE = ROOT / "data/cache"
THRS = [0.2, 0.3, 0.4, 0.5]
EPS = 0.005


def main():
    base = {}
    with open(VOI) as f:
        for r in csv.DictReader(f):
            base[round(float(r["aff_threshold"]), 4)] = (
                float(r["voi_split"]),
                float(r["voi_merge"]),
            )
    fr = np.load(CACHE / "wz_fragments.npy")
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:].astype(np.float32) / 255.0)
    t0 = time.time()
    u, v, sm, ct = rag_arrays(aff, fr)
    print(f"edges={len(u)} rag_sec={time.time()-t0:.1f}", flush=True)
    with h5py.File(GT, "r") as f:
        gt = f["gt"][:].astype(np.uint64)
    t1 = time.time()
    snaps = heap_from_arrays(u, v, sm, ct, THRS, max_id=int(fr.max()))
    print(f"heap_sec={time.time()-t1:.1f}", flush=True)
    ok = True
    for thr in THRS:
        lab = extract_parent(fr, snaps[thr]).astype(np.uint64)
        nseg = int((np.unique(lab) != 0).sum())
        print(f"aff={thr} nseg={nseg}", flush=True)
        s = waterz.evaluate(lab, gt)
        bs, bm = base[thr]
        ds, dm = s["voi_split"] - bs, s["voi_merge"] - bm
        good = ds <= EPS and dm <= EPS
        ok = ok and good
        print(
            f"aff={thr} split={s['voi_split']:.6f} base={bs:.6f} d={ds:+.6f} "
            f"merge={s['voi_merge']:.6f} base={bm:.6f} d={dm:+.6f} "
            f"{'PASS' if good else 'FAIL'}"
        )
        with h5py.File(CACHE / f"g1_thr{thr}.h5", "w") as h:
            h.create_dataset("labels", data=lab.astype(np.uint32), compression="gzip")
    print("G1", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
