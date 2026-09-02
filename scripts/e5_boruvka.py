#!/usr/bin/env python3
"""E5: live-mean Borůvka on waterz fragments; shipped grader."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ref_cpu import agglomerate_boruvka  # noqa: E402

AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
CACHE = ROOT / "data/cache"
OUT = ROOT / "data/ws_bounty"
THRS = [0.2, 0.3, 0.4, 0.5]


def main():
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:].astype(np.float32) / 255.0)
    fr = np.load(CACHE / "wz_fragments.npy")
    t0 = time.time()
    labs = agglomerate_boruvka(aff, THRS, fragments=fr)
    print(f"boruvka_sec={time.time()-t0:.1f}", flush=True)
    paths = []
    for thr in THRS:
        p = OUT / f"mine_thr{thr}.h5"
        lab = labs[thr].astype(np.uint32)
        with h5py.File(p, "w") as h:
            h.create_dataset("labels", data=lab)
        nseg = int((np.unique(lab) != 0).sum())
        paths.append(str(p))
        print(f"wrote {p} nseg={nseg}", flush=True)
    cmd = [
        str(ROOT / ".venv/bin/python"),
        str(OUT / "baseline/run_baseline.py"),
        "--candidate",
        *paths,
    ]
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(OUT))
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
