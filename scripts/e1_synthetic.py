#!/usr/bin/env python3
"""E1a-f. Assert S1-S4 on tiny volumes. No GPU."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ref_cpu import agglomerate_exact, region_graph, watershed  # noqa: E402

LOW, HIGH = 1e-4, 0.9999


def e1a():
    aff = np.zeros((3, 1, 1, 1), np.float32)
    lab = watershed(aff, LOW, HIGH)
    assert lab.shape == (1, 1, 1)
    assert lab[0, 0, 0] == 0


def e1b():
    # two pairs along x, weak contact in the middle -> 2 fragments, 1 RAG edge
    aff = np.zeros((3, 1, 1, 4), np.float32)
    aff[2, 0, 0, 1] = 0.9  # voxel1-voxel0
    aff[2, 0, 0, 3] = 0.9  # voxel3-voxel2
    aff[2, 0, 0, 2] = 0.25  # contact between the two pairs
    fr = watershed(aff, LOW, HIGH)
    n = len(set(fr.ravel()) - {0})
    assert n == 2, n
    rg = region_graph(aff, fr)
    assert len(rg) == 1
    (u, v), (s, c, mean) = next(iter(rg.items()))
    assert u > 0 and v > 0
    assert abs(mean - 0.25) < 1e-6
    # merge iff mean > aff_thr
    out = agglomerate_exact(aff, [0.20, 0.30], LOW, HIGH, fragments=fr)
    assert len(set(out[0.20].ravel()) - {0}) == 1
    assert len(set(out[0.30].ravel()) - {0}) == 2


def e1c():
    aff = np.ones((3, 2, 2, 2), np.float32)
    lab = watershed(aff, LOW, HIGH)
    ids = set(lab.ravel()) - {0}
    assert len(ids) == 1, ids
    lab2 = watershed(aff, LOW, HIGH)
    assert np.array_equal(lab, lab2)


def e1d():
    aff = np.zeros((3, 3, 3, 3), np.float32)
    aff[:, 1, 1, 1] = 0.0
    # a live pair in the center-x
    aff[2, 1, 1, 1] = 0.8
    fr = watershed(aff, LOW, HIGH)
    assert (fr[0] == 0).all() and (fr[2] == 0).all()
    rg = region_graph(aff, fr)
    for u, v in rg:
        assert u > 0 and v > 0


def e1e():
    # S3 area-weighted mean, then a 3-node RAG with a shared neighbour.
    e1 = [0.4 * 2, 2, 0.4]
    e2 = [0.8 * 3, 3, 0.8]
    merged_mean = (e1[0] + e2[0]) / (e1[1] + e2[1])
    assert abs(merged_mean - 3.2 / 5) < 1e-12
    from ref_cpu import _heap_agglomerate, extract

    # fragments 1-2-3: edges (1,2) mean 0.9 n=1; (1,3) 0.4 n=2; (2,3) 0.8 n=3
    # merge 1-2 first (mean 0.9); (1,3) and (2,3) combine -> mean 3.2/5 = 0.64
    edges = {
        (1, 2): [0.9, 1, 0.9],
        (1, 3): [0.8, 2, 0.4],
        (2, 3): [2.4, 3, 0.8],
    }
    snaps = _heap_agglomerate(edges, [0.85, 0.50])
    # aff 0.85: merge while mean>0.85 -> only 1-2. 1 and 2 same root; 3 separate.
    uf = snaps[0.85]
    assert uf.find(1) == uf.find(2)
    assert uf.find(3) != uf.find(1)
    # aff 0.50: also merge the combined 0.64 edge
    uf2 = snaps[0.50]
    assert uf2.find(1) == uf2.find(2) == uf2.find(3)
    dummy = np.array([0, 1, 2, 3], np.uint32)
    lab = extract(dummy, uf2)
    assert lab[1] == lab[2] == lab[3]


def e1f():
    aff = np.zeros((3, 3, 3, 3), np.float32)
    aff[1] = 0.9
    aff[2] = 0.9
    aff[0] = 0.9
    n_full = len(set(watershed(aff, LOW, HIGH).ravel()) - {0})
    aff0 = aff.copy()
    aff0[0] = 0.0
    n_noz = len(set(watershed(aff0, LOW, HIGH).ravel()) - {0})
    assert n_full != n_noz, (n_full, n_noz)


if __name__ == "__main__":
    for fn in (e1a, e1b, e1c, e1d, e1e, e1f):
        fn()
        print(f"PASS {fn.__name__}")
    print("E1 PASS")
