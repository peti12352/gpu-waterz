#!/usr/bin/env python3
"""E4: stitch union-find over seam representatives, not 0.46 of all voxels.

Phase 1 is tile-local min-index UF (same as W5). Phase 2 unions only the
p1 roots that participate in a cross-tile edge. The min-index fixed point
is unchanged if every intra-tile edge and every cross-tile edge is applied,
the latter after contraction to representatives.

Kill: parent differs from the whole-graph min-index UF.
Pass: identity and stitch domain << 0.46 nvox.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from w5_tile_uf import (
    CACHE, TILES, bfs_tree_edges, exact_partition, rounds_to_converge,
    same_label_edges, tile_arrays, _chase,
)


def rep_stitch(n, ei, ej, tid, p1):
    cross = tid[ei] != tid[ej]
    a = p1[ei[cross]]
    b = p1[ej[cross]]
    keep = a != b
    a, b = a[keep], b[keep]
    reps = np.unique(np.concatenate([a, b])) if a.size else np.empty(
        0, dtype=np.int32)
    parent = p1.copy()
    if a.size == 0:
        return 0, parent, reps
    rounds, parent = rounds_to_converge(
        n, a, b, parent=parent, comp_idx=reps.astype(np.int32))
    _chase(parent, None)
    return rounds, parent, reps


def one(shape, lab, ei, ej, tile, tag, uneven=False):
    n = int(np.prod(shape))
    tid, _face = tile_arrays(shape, tile)
    intra = tid[ei] == tid[ej]
    p1 = exact_partition(n, ei[intra], ej[intra])
    t0 = time.time()
    st_rounds, p2, reps = rep_stitch(n, ei, ej, tid, p1)
    dt = time.time() - t0
    gold = exact_partition(n, ei, ej)
    ok = bool(np.array_equal(gold, p2))
    return {
        "tag": tag, "tile": list(tile), "uneven": uneven,
        "n_cross": int((~intra).sum()),
        "base_rounds": rounds_to_converge(n, ei, ej)[0],
        "stitch_rounds": st_rounds,
        "n_rep": int(reps.size),
        "rep_frac": float(reps.size / n),
        "parent_identical": ok,
        "sec": round(dt, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", nargs=3, type=int, default=[32, 128, 128])
    ap.add_argument("--out", default="e4_rep_uf.json")
    args = ap.parse_args()

    src = CACHE / "gpu_fragments.npy"
    vol = np.load(src, mmap_mode="r")
    Z, Y, X = args.crop
    lab = np.ascontiguousarray(vol[:Z, :Y, :X])
    n = lab.size
    nfrag = int(np.unique(lab).size) - int((lab == 0).any())
    print(f"E4 crop {lab.shape} nvox={n} nfrag={nfrag}", flush=True)

    ei, ej = same_label_edges(lab)
    tei, tej = bfs_tree_edges(n, ei, ej, lab)
    graphs = {"dense": (ei, ej), "tree": (tei, tej)}
    tiles = list(TILES) + [(5, 7, 33), (9, 5, 40), (4, 4, 4)]
    rows = []
    for tag, (a, b) in graphs.items():
        for tile in tiles:
            r = one(lab.shape, lab, a, b, tile, tag,
                    uneven=tile not in TILES)
            rows.append(r)
            print(f"E4 {tag:5s} tile={str(tile):14s} "
                  f"reps={r['n_rep']:7d} frac={r['rep_frac']:.5f} "
                  f"stitch={r['stitch_rounds']:3d} "
                  f"identical={r['parent_identical']}", flush=True)

    bad = [r for r in rows if not r["parent_identical"]]
    max_frac = max(r["rep_frac"] for r in rows)
    ok = (not bad) and max_frac < 0.46
    print(f"\nE4 {len(rows)} configs, {len(bad)} identity fails, "
          f"max rep_frac={max_frac:.5f} vs W5 0.46")
    print(f"E4 {'PASS' if ok else 'FAIL'}")
    dest = CACHE / args.out
    dest.write_text(json.dumps({
        "crop": list(lab.shape), "nvox": n, "nfrag": nfrag,
        "rows": rows, "n_fail": len(bad), "max_rep_frac": max_frac,
        "pass": ok,
    }, indent=2) + "\n")
    print(f"E4 wrote {dest.name}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
