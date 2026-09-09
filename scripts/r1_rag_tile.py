#!/usr/bin/env python3
"""R1: tiled RAG pre-aggregation, gated on the real fragment volume.

No affinities on this machine, so the check is the face-count half of the
edge record (the `n` field). isum is the same reduction over the same faces
with integer weights, so an exact `n` match is also an exact `isum` match
whenever the affinities come back.

C1 already collapses a warp of 32 consecutive x. R1's payoff is the extra
collapse across the tile's y and z, counted here as global atomic sequences
(one per distinct key that leaves a tile, versus one per distinct key that
leaves a warp).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"
TX, TY, TZ = 32, 4, 2


def faces(lab: np.ndarray) -> dict[tuple[int, int], int]:
    Z, Y, X = lab.shape
    out: dict[tuple[int, int], int] = defaultdict(int)
    for ax, sl in ((0, (slice(1, None), slice(None), slice(None))),
                   (1, (slice(None), slice(1, None), slice(None))),
                   (2, (slice(None), slice(None), slice(1, None)))):
        a = lab[sl]
        bsl = list(sl)
        bsl[ax] = slice(0, -1)
        b = lab[tuple(bsl)]
        keep = (a != b) & (a != 0) & (b != 0)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        keys = np.stack([lo[keep], hi[keep]], axis=1)
        u, c = np.unique(keys, axis=0, return_counts=True)
        for (p, q), n in zip(u.tolist(), c.tolist()):
            out[(int(p), int(q))] += int(n)
    return out


def tile_atomics(lab: np.ndarray) -> tuple[dict[tuple[int, int], int], int, int]:
    """Per-tile unique keys (R1 flushes) and per-warp unique keys (C1)."""
    Z, Y, X = lab.shape
    merged: dict[tuple[int, int], int] = defaultdict(int)
    n_tile = n_warp = 0
    for z0 in range(0, Z, TZ):
        for y0 in range(0, Y, TY):
            for x0 in range(0, X, TX):
                tile = lab[z0:z0 + TZ, y0:y0 + TY, x0:x0 + TX]
                local = faces(tile)
                # Faces on the negative tile boundary, against the halo
                # outside the tile, are part of the global set but not of
                # `faces(tile)`. Add them so the union matches `faces(lab)`.
                if z0 > 0:
                    a = lab[z0, y0:y0 + TY, x0:x0 + TX]
                    b = lab[z0 - 1, y0:y0 + TY, x0:x0 + TX]
                    keep = (a != b) & (a != 0) & (b != 0)
                    lo, hi = np.minimum(a, b), np.maximum(a, b)
                    for p, q in zip(lo[keep].ravel(), hi[keep].ravel()):
                        local[(int(p), int(q))] += 1
                if y0 > 0:
                    a = lab[z0:z0 + TZ, y0, x0:x0 + TX]
                    b = lab[z0:z0 + TZ, y0 - 1, x0:x0 + TX]
                    keep = (a != b) & (a != 0) & (b != 0)
                    lo, hi = np.minimum(a, b), np.maximum(a, b)
                    for p, q in zip(lo[keep].ravel(), hi[keep].ravel()):
                        local[(int(p), int(q))] += 1
                if x0 > 0:
                    a = lab[z0:z0 + TZ, y0:y0 + TY, x0]
                    b = lab[z0:z0 + TZ, y0:y0 + TY, x0 - 1]
                    keep = (a != b) & (a != 0) & (b != 0)
                    lo, hi = np.minimum(a, b), np.maximum(a, b)
                    for p, q in zip(lo[keep].ravel(), hi[keep].ravel()):
                        local[(int(p), int(q))] += 1
                n_tile += len(local)
                for k, n in local.items():
                    merged[k] += n
                # C1: one warp per x-row of the tile
                tz, ty, tx = tile.shape
                for lz in range(tz):
                    for ly in range(ty):
                        row = {}
                        for lx in range(tx):
                            z, y, x = z0 + lz, y0 + ly, x0 + lx
                            id1 = int(lab[z, y, x])
                            for dz, dy, dx in ((-1, 0, 0), (0, -1, 0), (0, 0, -1)):
                                zz, yy, xx = z + dz, y + dy, x + dx
                                if zz < 0 or yy < 0 or xx < 0:
                                    continue
                                id2 = int(lab[zz, yy, xx])
                                if id1 == 0 or id2 == 0 or id1 == id2:
                                    continue
                                lo, hi = (id1, id2) if id1 < id2 else (id2, id1)
                                row[(lo, hi)] = row.get((lo, hi), 0) + 1
                        n_warp += len(row)
    return merged, n_tile, n_warp


def main() -> int:
    src = CACHE / "gpu_fragments.npy"
    vol = np.load(src, mmap_mode="r")
    lab = np.ascontiguousarray(vol[:40, :128, :128])
    print(f"R1 crop {lab.shape} from {src.name}")
    ref = faces(lab)
    got, n_tile, n_warp = tile_atomics(lab)
    same = ref == got
    print(f"R1 edges {len(ref)} tiled {len(got)} identical={same}")
    if not same:
        only_ref = len(set(ref) - set(got))
        only_got = len(set(got) - set(ref))
        ndiff = sum(1 for k in set(ref) | set(got) if ref.get(k) != got.get(k))
        print(f"R1 only_ref={only_ref} only_got={only_got} count_diff={ndiff}")
    print(f"R1 C1 warp atomics={n_warp}  R1 tile flushes={n_tile}  "
          f"ratio={n_warp / max(n_tile, 1):.2f}x")
    dest = CACHE / "r1_rag_tile.json"
    dest.write_text(json.dumps({
        "crop": list(lab.shape), "n_edges": len(ref),
        "identical": bool(same),
        "c1_atomics": n_warp, "r1_flushes": n_tile,
        "ratio": n_warp / max(n_tile, 1),
    }, indent=2) + "\n")
    print(f"R1 wrote {dest.name}")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
