#!/usr/bin/env python3
"""Q20: Funke quantile scoring vs MeanAffinity voi.csv. Stock waterz on E3 fragments."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from task_gate import AFF_THRESHOLDS, print_contract  # noqa: E402

CACHE = ROOT / "data/cache"
OUT = ROOT / "data/ws_bounty"
PY = ROOT / ".venv/bin/python"
SCORE = "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>"


def main():
    print_contract()
    fr = np.load(CACHE / "wz_fragments.npy")
    aff_path = OUT / "cremiA_val/affinity.h5"
    with h5py.File(aff_path, "r") as f:
        aff = f["affinity"][:].astype(np.float32) / 255.0
    import waterz

    scores = sorted(1.0 - t for t in AFF_THRESHOLDS)
    dest_dir = OUT / "q20_quantile"
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    # Copy on each yield: waterz reuses one buffer (list() would alias the last).
    for lab, score in zip(
        waterz.agglomerate(
            aff,
            scores,
            fragments=fr.astype(np.uint64, copy=False),
            scoring_function=SCORE,
            discretize_queue=256,
            aff_threshold_low=1e-4,
            aff_threshold_high=0.9999,
            force_rebuild=True,
        ),
        scores,
    ):
        t = 1.0 - score
        dest = dest_dir / f"mine_thr{t}.h5"
        lab32 = np.array(lab, dtype=np.uint32, copy=True)
        with h5py.File(dest, "w") as h:
            h.create_dataset("labels", data=lab32)
        print(f"Q20 T={t} nseg={int((np.unique(lab32) != 0).sum())}", flush=True)
        paths.append((t, str(dest)))
    paths.sort(key=lambda x: x[0])
    cmd = [str(PY), str(OUT / "baseline/run_baseline.py"), "--candidate", *[p for _, p in paths]]
    r = subprocess.run(cmd, cwd=str(OUT), capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    ok = "ACCURACY GATE: PASS" in r.stdout
    print(f"Q20 {'PASS' if ok else 'FAIL'}", flush=True)
    (ROOT / "data/cache/q20_pass.txt").write_text("PASS\n" if ok else "FAIL\n")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
