#!/usr/bin/env python3
"""E5 fallback: dump E4 exact-S4 labels as mine_thr*.h5 and run shipped grader."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
OUT = ROOT / "data/ws_bounty"
THRS = [0.2, 0.3, 0.4, 0.5]


def main():
    paths = []
    for thr in THRS:
        src = CACHE / f"exact_thr{thr}.h5"
        dst = OUT / f"mine_thr{thr}.h5"
        with h5py.File(src, "r") as f:
            lab = f["labels"][:].astype(np.uint32)
        with h5py.File(dst, "w") as h:
            h.create_dataset("labels", data=lab)
        nseg = int((np.unique(lab) != 0).sum())
        paths.append(str(dst))
        print(f"wrote {dst} nseg={nseg}", flush=True)
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
