#!/usr/bin/env python3
"""W5: how many stitch rounds a tile-local union-find needs, measured on the
real fragment volume.

Correctness of W5 is settled elsewhere: `w0_ws_ref.py --w5` shows the
two-phase order reaches the identical parent array, because the fixed point of
min-index hooking does not depend on the order edges were applied in
(ws.cu:1221). What is *not* settled by that is the payoff, and the payoff
hinges on one number: how many hook+compress rounds the cross-tile stitch
needs, against how many the current whole-volume loop needs.

That number depends only on component geometry, and `data/cache/gpu_fragments.npy`
is exactly the component geometry the graded pipeline produces, 2175400
fragments over 125x1200x1200. So it can be measured here without a card.

The flow graph itself is not recoverable from the labels, since k_flow needs
affinities and the tarball is not on this machine. Two graphs are therefore
built on each fragment and both are reported, because the true flow graph sits
between them:

  dense   every 6-adjacent same-label pair. The most edges a fragment can
          have, so the fewest rounds, an optimistic bound.
  tree    a BFS spanning tree of each fragment from its minimum-index voxel.
          The fewest edges that still connect it, so the longest chains and
          the most rounds, a pessimistic bound. This is the closer of the
          two in kind, since a flow field is a forest pointing at minima.

Both bounds are measured for the baseline and for W5 on the same graph, so the
ratio is apples to apples even where the absolute counts are not.

Usage:
    w5_tile_uf.py                        # default crop and tile sweep
    w5_tile_uf.py --crop 125 256 256
    w5_tile_uf.py --graph dense          # skip the slower tree build
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"

# Tile shapes worth considering, as (TZ, TY, TX). TX is 32 or more so a warp
# stages contiguous x, which is the same reason vox.cuh orders its grid that
# way. The shared-memory cost is 4 B of uint32 parent plus 1 B of direction
# byte per tile voxel, and the default limit is 48 KB per block, so anything
# past ~9800 voxels needs the opt-in carveout.
TILES = [
    (8, 8, 32),
    (8, 16, 32),
    (16, 16, 32),
    (8, 32, 32),
    (16, 16, 16),
    (12, 16, 32),
    (32, 32, 32),
]


def shared_bytes(tile: tuple[int, int, int]) -> int:
    return int(np.prod(tile)) * 5


# --------------------------------------------------------------------------
# graph construction over a labelled volume
# --------------------------------------------------------------------------

def same_label_edges(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every 6-adjacent pair of voxels carrying the same nonzero label."""
    Z, Y, X = lab.shape
    idx = np.arange(lab.size, dtype=np.int32).reshape(Z, Y, X)
    flat = lab.ravel()
    us, vs = [], []
    for ax in (0, 1, 2):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[ax] = slice(0, -1)
        hi[ax] = slice(1, None)
        a = idx[tuple(lo)].ravel()
        b = idx[tuple(hi)].ravel()
        keep = (flat[a] == flat[b]) & (flat[a] != 0)
        us.append(a[keep])
        vs.append(b[keep])
        del a, b, keep
    return np.concatenate(us), np.concatenate(vs)


def bfs_tree_edges(n: int, ei: np.ndarray, ej: np.ndarray, lab: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """A BFS spanning tree of each fragment, rooted at its lowest voxel index.

    Level-synchronous and deterministic: within a level a newly reached voxel
    takes the lowest-indexed parent that reached it, so the tree does not
    depend on array order.
    """
    flat = lab.ravel()
    dist = np.full(n, -1, dtype=np.int32)
    nz = np.flatnonzero(flat).astype(np.int32)
    # min voxel index per label, which is also the min-index root the
    # union-find would settle on
    order = np.argsort(flat[nz], kind="stable")
    s = nz[order]
    keys = flat[s]
    first = np.ones(s.size, dtype=bool)
    first[1:] = keys[1:] != keys[:-1]
    roots = s[first]
    del nz, order, s, keys, first
    dist[roots] = 0
    tei, tej = [], []
    cur = 0
    while True:
        a_at, b_at = dist[ei] == cur, dist[ej] == cur
        m1 = a_at & (dist[ej] < 0)
        m2 = b_at & (dist[ei] < 0)
        child = np.concatenate([ej[m1], ei[m2]])
        par = np.concatenate([ei[m1], ej[m2]])
        del a_at, b_at, m1, m2
        if child.size == 0:
            break
        # one parent per child, the lowest-indexed one
        o = np.lexsort((par, child))
        child, par = child[o], par[o]
        keep = np.ones(child.size, dtype=bool)
        keep[1:] = child[1:] != child[:-1]
        child, par = child[keep], par[keep]
        dist[child] = cur + 1
        tei.append(par)
        tej.append(child)
        cur += 1
    return np.concatenate(tei), np.concatenate(tej)


# --------------------------------------------------------------------------
# the device round loop, level-synchronous
# --------------------------------------------------------------------------

def _chase(parent: np.ndarray, idx: np.ndarray | None) -> None:
    """k_uf_compress_c over `idx` (or everything when None).

    uf_find walks to the root and the kernel writes only its own slot, so a
    restricted compress does not shorten the chains it walks through. That is
    reproduced literally: only `idx` entries are written.
    """
    if idx is None:
        while True:
            nxt = parent[parent]
            if np.array_equal(nxt, parent):
                return
            parent[:] = nxt
    else:
        r = parent[idx]
        while True:
            nr = parent[r]
            if np.array_equal(nr, r):
                break
            r = nr
        parent[idx] = r


def rounds_to_converge(n: int, ei: np.ndarray, ej: np.ndarray,
                       parent: np.ndarray | None = None,
                       comp_idx: np.ndarray | None = None,
                       cap: int = 400) -> tuple[int, np.ndarray]:
    """Rounds of hook-then-compress until a round changes nothing.

    The hook applies `atomicMin` semantics: every edge reads the same parent
    snapshot and the smallest writer to a slot wins, which is what
    np.minimum.at does. The device is asynchronous and can see a neighbour's
    write within the same round, so it converges in no more rounds than this;
    this is an upper bound that scales the same way.
    """
    if parent is None:
        parent = np.arange(n, dtype=np.int32)
    rounds = 0
    while rounds < cap:
        pa, pb = parent[ei], parent[ej]
        d = pa != pb
        if not d.any():
            break
        before = parent.copy()
        np.minimum.at(parent, np.maximum(pa[d], pb[d]),
                      np.minimum(pa[d], pb[d]))
        del pa, pb, d
        _chase(parent, comp_idx)
        rounds += 1
        if np.array_equal(before, parent):
            break
    return rounds, parent


def exact_partition(n: int, ei: np.ndarray, ej: np.ndarray) -> np.ndarray:
    """Min-index root per voxel, by pointer jumping to the fixed point.

    Same result as the round loop but without counting rounds, so it is the
    cheap way to get phase 1's output.
    """
    parent = np.arange(n, dtype=np.int32)
    while True:
        pa, pb = parent[ei], parent[ej]
        d = pa != pb
        if not d.any():
            break
        np.minimum.at(parent, np.maximum(pa[d], pb[d]),
                      np.minimum(pa[d], pb[d]))
        del pa, pb, d
        while True:
            nxt = parent[parent]
            if np.array_equal(nxt, parent):
                break
            parent[:] = nxt
    return parent


# --------------------------------------------------------------------------
# tiling
# --------------------------------------------------------------------------

def tile_arrays(shape: tuple[int, int, int], tile: tuple[int, int, int]
                ) -> tuple[np.ndarray, np.ndarray]:
    """Per-voxel tile id and tile-face mask, as w0_ws_ref.tile_index but
    built axis by axis to keep the peak allocation down.
    """
    Z, Y, X = shape
    TZ, TY, TX = tile
    ntx, nty = (X + TX - 1) // TX, (Y + TY - 1) // TY
    z = np.arange(Z, dtype=np.int32)
    y = np.arange(Y, dtype=np.int32)
    x = np.arange(X, dtype=np.int32)
    tid = ((((z // TZ)[:, None, None] * nty) + (y // TY)[None, :, None]) * ntx
           + (x // TX)[None, None, :])
    tid = np.ascontiguousarray(np.broadcast_to(tid, (Z, Y, X))).ravel()
    fz = ((z % TZ == 0) | (z % TZ == TZ - 1))[:, None, None]
    fy = ((y % TY == 0) | (y % TY == TY - 1))[None, :, None]
    fx = ((x % TX == 0) | (x % TX == TX - 1))[None, None, :]
    face = np.broadcast_to(fz | fy | fx, (Z, Y, X))
    return tid, np.ascontiguousarray(face).ravel()


def measure(shape, lab, ei, ej, tile, tag) -> dict:
    """Baseline rounds against W5 stitch rounds on one graph and one tiling."""
    n = int(np.prod(shape))
    flat = lab.ravel()
    active = flat != 0

    t0 = time.time()
    base_rounds, base_parent = rounds_to_converge(n, ei, ej)
    t_base = time.time() - t0

    tid, face = tile_arrays(shape, tile)
    intra = tid[ei] == tid[ej]
    n_intra, n_cross = int(intra.sum()), int((~intra).sum())

    # phase 1: intra-tile only, resolved in shared memory. Free in the round
    # accounting because it is one kernel with no global round trip.
    p1 = exact_partition(n, ei[intra], ej[intra])

    # phase 2 domains. The hook covers every edge incident to a face voxel,
    # which is a superset of the cross-tile edges. The compress covers face
    # voxels and every phase-1 root, because a phase-1 root is the only thing
    # a parent entry can point at when phase 2 starts, so leaving them out
    # would let chains grow without bound.
    inc = face[ei] | face[ej]
    is_root = p1 == np.arange(n, dtype=np.int32)
    comp = np.flatnonzero((face | is_root) & active).astype(np.int32)
    t0 = time.time()
    stitch_rounds, p2 = rounds_to_converge(n, ei[inc], ej[inc],
                                           parent=p1.copy(), comp_idx=comp)
    t_stitch = time.time() - t0

    # one full-volume compress at the end, so every voxel holds a true root
    _chase(p2, None)
    ok = bool(np.array_equal(base_parent, p2))

    return {"tag": tag, "tile": list(tile), "shared_bytes": shared_bytes(tile),
            "n_edges": int(ei.size), "n_intra": n_intra, "n_cross": n_cross,
            "base_rounds": base_rounds, "stitch_rounds": stitch_rounds,
            "round_ratio": float(base_rounds / max(stitch_rounds, 1)),
            "face_frac": float(face.mean()),
            "hook_list_frac": float((face & active).mean()),
            "comp_list_frac": float(comp.size / n),
            "parent_identical": ok,
            "sec_base": round(t_base, 1), "sec_stitch": round(t_stitch, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", nargs=3, type=int, default=[125, 256, 256],
                    help="ZYX crop of gpu_fragments.npy to measure on")
    ap.add_argument("--graph", choices=["dense", "tree", "both"],
                    default="both")
    ap.add_argument("--out", default="w5_tile_uf.json")
    args = ap.parse_args()

    src = CACHE / "gpu_fragments.npy"
    vol = np.load(src, mmap_mode="r")
    Z, Y, X = args.crop
    lab = np.ascontiguousarray(vol[:Z, :Y, :X])
    shape = lab.shape
    n = lab.size
    nfrag = int(np.unique(lab).size) - int((lab == 0).any())
    print(f"W5 crop {shape} = {n/1e6:.2f} Mvox from {src.name}, "
          f"{nfrag} fragments, bg {float((lab==0).mean()):.3f}")

    graphs = {}
    ei, ej = same_label_edges(lab)
    print(f"W5 dense graph: {ei.size/1e6:.2f} M edges "
          f"({ei.size/n:.2f} per voxel)")
    if args.graph in ("dense", "both"):
        graphs["dense"] = (ei, ej)
    if args.graph in ("tree", "both"):
        t0 = time.time()
        tei, tej = bfs_tree_edges(n, ei, ej, lab)
        print(f"W5 tree graph:  {tei.size/1e6:.2f} M edges "
              f"({time.time()-t0:.1f}s)")
        graphs["tree"] = (tei, tej)

    rows = []
    for tag, (a, b) in graphs.items():
        for tile in TILES:
            r = measure(shape, lab, a, b, tile, tag)
            rows.append(r)
            print(f"W5 {tag:5s} tile={str(tile):14s} "
                  f"shmem={r['shared_bytes']/1024:5.1f}K "
                  f"base={r['base_rounds']:3d} stitch={r['stitch_rounds']:3d} "
                  f"ratio={r['round_ratio']:5.2f}x "
                  f"hook_list={r['hook_list_frac']:.3f} "
                  f"comp_list={r['comp_list_frac']:.3f} "
                  f"identical={r['parent_identical']}")

    bad = [r for r in rows if not r["parent_identical"]]
    print(f"\nW5 {len(rows)} configurations, "
          f"{len(bad)} with a non-identical parent array")
    for tag in graphs:
        sub = [r for r in rows if r["tag"] == tag]
        best = min(sub, key=lambda r: r["comp_list_frac"] * r["stitch_rounds"])
        print(f"W5 {tag}: lowest stitch work at tile {best['tile']} "
              f"-- {best['stitch_rounds']} rounds over "
              f"{best['comp_list_frac']:.3f} of the volume, against "
              f"{best['base_rounds']} rounds over all of it, i.e. "
              f"{best['base_rounds'] / max(best['stitch_rounds'] * best['comp_list_frac'], 1e-9):.1f}x "
              f"less compress work")

    dest = CACHE / args.out
    dest.write_text(json.dumps(
        {"crop": list(shape), "nvox": n, "nfrag": nfrag,
         "tiles": [list(t) for t in TILES], "rows": rows}, indent=2) + "\n")
    print(f"W5 wrote {dest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
