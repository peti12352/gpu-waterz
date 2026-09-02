#!/usr/bin/env python3
"""E2cpu: ref_cpu watershed vs waterz no-merge fragments on val. CPU only."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import waterz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ref_cpu import watershed  # noqa: E402

AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
TASK_FRAGS = 2_175_400


def main():
    with h5py.File(AFF, "r") as f:
        aff_u8 = f["affinity"][:]
    aff = np.ascontiguousarray(aff_u8.astype(np.float32) / 255.0)
    t0 = time.time()
    wz = None
    for seg in waterz.agglomerate(aff, [0.0]):
        wz = seg.astype(np.uint32, copy=False)
        break
    t1 = time.time()
    wz_n = int((wz > 0).max() and wz.max())
    # waterz IDs may have gaps; count unique
    wz_ids = np.unique(wz)
    wz_count = int(wz_ids[0] != 0) + int(wz_ids.size - (1 if wz_ids[0] == 0 else 0))
    wz_count = int((wz_ids != 0).sum())
    wz_bg = int((wz == 0).sum())
    print(f"waterz fragments={wz_count} bg={wz_bg} sec={t1-t0:.1f}")
    print(f"TASK fragments={TASK_FRAGS} d={wz_count - TASK_FRAGS}")

    t2 = time.time()
    ours = watershed(aff)
    t3 = time.time()
    our_count = int((np.unique(ours) != 0).sum())
    our_bg = int((ours == 0).sum())
    print(f"ref_cpu fragments={our_count} bg={our_bg} sec={t3-t2:.1f}")
    rel = abs(our_count - wz_count) / max(wz_count, 1)
    print(f"rel_count_err={rel:.6f}")
    print(f"bg_match={our_bg == wz_bg} our_bg={our_bg} wz_bg={wz_bg}")
    ok_wz = abs(wz_count - TASK_FRAGS) < 1000
    ok_rel = rel <= 0.01
    print("E2cpu", "PASS" if (ok_wz and ok_rel) else "FAIL")
    if not (ok_wz and ok_rel):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
