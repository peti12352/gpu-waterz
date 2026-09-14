#!/usr/bin/env python3
"""E4: exact S4 heap (stock waterz) on E3 waterz fragments. VOI vs voi.csv, +0.005."""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import waterz

ROOT = Path(__file__).resolve().parents[1]
AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
GT = ROOT / "data/ws_bounty/cremiA_val/gt.h5"
VOI = ROOT / "data/ws_bounty/baseline/voi.csv"
CACHE = ROOT / "data/cache"
# waterz sorts scores ascending -> aff 0.5, 0.4, 0.3, 0.2
SCORE_THEN_AFF = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]
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
    with h5py.File(GT, "r") as f:
        gt = f["gt"][:].astype(np.uint64)
    seg = fr.astype(np.uint64, copy=True)
    scores = [s for s, _ in SCORE_THEN_AFF]
    t0 = time.time()
    ok = True
    for (score, aff_thr), lab in zip(SCORE_THEN_AFF, waterz.agglomerate(aff, scores, fragments=seg)):
        lab = np.array(lab, dtype=np.uint64, copy=True)
        nseg = int((np.unique(lab) != 0).sum())
        print(f"aff={aff_thr} nseg={nseg} heap_so_far={time.time()-t0:.1f}", flush=True)
        s = waterz.evaluate(lab.astype(np.uint64), gt)
        bs, bm = base[aff_thr]
        ds, dm = s["voi_split"] - bs, s["voi_merge"] - bm
        good = ds <= EPS and dm <= EPS
        ok = ok and good
        print(
            f"aff={aff_thr} split={s['voi_split']:.6f} base={bs:.6f} d={ds:+.6f} "
            f"merge={s['voi_merge']:.6f} base={bm:.6f} d={dm:+.6f} "
            f"{'PASS' if good else 'FAIL'}"
        )
        with h5py.File(CACHE / f"exact_thr{aff_thr}.h5", "w") as h:
            h.create_dataset("labels", data=lab.astype(np.uint32), compression="gzip")
    print(f"total_sec={time.time()-t0:.1f}")
    print("E4", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
