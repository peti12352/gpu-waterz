#!/usr/bin/env python3
"""W0: a CPU replica of the csrc/ws.cu watershed, to serve as the bit-identity
gate for W1-W4 on a machine with no GPU.

Why this exists. The plan gates every watershed change on "bit-identical" or on
`nfrag == 2175400` and `np.array_equal` against the current watershed. Both of
those need a card. `scripts/g0_agg_ref.py` showed the way out for the
agglomeration half: a faithful CPU replica turns a GPU gate into a CPU gate, and
in the process measures the logical work each lever removes. This is the same
move for the watershed.

SCOPE, and a warning. ws.cu contains THREE watershed implementations, and this
file models the two older ones, not the one in production:

  `plateau_basins`   ws.cu:703, reached via watershed_gpu. Commented "Exact S1
                     plateau+basin on host (G2-locked)". Sequential; mutates
                     seg in place as the plateau BFS runs. Only caller in the
                     tree is scripts/g2_ws.py.
  `watershed_device` ws.cu:603, reached via watershed_gpu_d. k_flow, then a
                     compact-frontier plateau BFS, then a union-find over the
                     rewritten direction field, reading an immutable `orig`.
  `watershed_gpu_e9` ws.cu:1796/1965, e9b_divide_d + e9c_basins_d with
                     k_hook_bidir / k_hook_remain. THIS is what src/segment.py
                     calls from both segment() and segment_d(), so this is what
                     produced data/cache/gpu_fragments.npy and nfrag=2175400.

All three are replicated here, and diffing them gives a useful result:
`plateau_basins` is an exact specification of e9 -- identical partitions on
every case tried -- while `watershed_device` differs on every case. So W1's
gate, "array_equal against the current watershed", can be discharged on a CPU
against the sequential host routine in under a second. `watershed_device`
cannot be used as an oracle for anything.

e9 and the host agree despite one being a parallel per-plateau BFS and the
other a single global FIFO, because plateaus are connected components of the
reciprocal subgraph and therefore disjoint, and within a plateau both process
that plateau's corners in ascending voxel order and then expand FIFO. Both also
mutate the direction field in place as they pop. That last part is what
`watershed_device` does differently: it reads reciprocity from an immutable
`orig` copy, so it never sees a neighbour's byte get cleared.

Usage:
    w0_ws_ref.py --shape 24 24 24 --seed 0 --levels 4     # one synthetic case
    w0_ws_ref.py --sweep                                  # differential sweep
    w0_ws_ref.py --sweep --legacy                         # include the dead pair
    w0_ws_ref.py --sweep --blocks                         # W1 block-label gate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"
SENT = np.uint32(0xFFFFFFFF)

# ws.cu:110. Direction d and the bit that the neighbour in direction d would
# have to set to point back at us.
DBIT = (0x01, 0x02, 0x04, 0x08, 0x10, 0x20)
RBIT = (0x08, 0x10, 0x20, 0x01, 0x02, 0x04)
MARK = 0x40           # ws.cu uses bit 6 as the BFS visited mark
HIGH = 0x80000000     # and bit 31 as "basin label assigned"


class Work:
    """Logical work counters, in the same spirit as g0_agg_ref.Work.

    These are what the W1-W4 levers are supposed to reduce, so they are counted
    rather than estimated: union sites for W1, plateau visits for W3, full
    volume passes for W2/W4.
    """

    def __init__(self) -> None:
        self.flow_visits = 0
        self.bfs_pushes = 0
        self.bfs_edge_tests = 0
        self.hook_calls = 0
        self.hook_unions = 0        # hooks that actually changed a parent
        self.find_steps = 0
        self.full_vox_passes = 0
        self.plateau_voxels = 0

    def asdict(self) -> dict:
        return {k: int(v) for k, v in vars(self).items()}


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def offsets(Y: int, X: int) -> tuple[int, ...]:
    """ws.cu:113 neigh_i. Flat index deltas for the six face neighbours."""
    yx = Y * X
    return (-yx, -X, -1, yx, X, 1)


def oob_masks(Z: int, Y: int, X: int) -> list[np.ndarray]:
    """ws.cu:123 oob_d, vectorised: True where direction d leaves the volume."""
    z, y, x = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X),
                          indexing="ij")
    return [(z == 0).ravel(), (y == 0).ravel(), (x == 0).ravel(),
            (z == Z - 1).ravel(), (y == Y - 1).ravel(), (x == X - 1).ravel()]


# --------------------------------------------------------------------------
# k_flow (ws.cu:135)
# --------------------------------------------------------------------------

def flow(aff: np.ndarray, low: float, high: float, W: Work) -> np.ndarray:
    """Replica of k_flow. aff is (3, Z, Y, X) uint8; returns a uint8 direction
    byte per voxel.

    Channel c of aff is the affinity across the face in the negative c
    direction, so `aat(0, z, y, x)` is the -z face of voxel (z,y,x) and
    `aat(0, z+1, y, x)` is its +z face. Out-of-volume faces read as `low`,
    which is what makes the max strictly greater than `low` unreachable for
    them and is why no direction bit is ever set out of bounds.
    """
    _, Z, Y, X = aff.shape
    a = aff.astype(np.float32) / np.float32(255.0)
    W.flow_visits += Z * Y * X

    n = []
    for c, ax in enumerate((0, 1, 2)):
        neg = np.full((Z, Y, X), low, dtype=np.float32)
        pos = np.full((Z, Y, X), low, dtype=np.float32)
        sl_all = [slice(None)] * 3
        sl_lo, sl_hi = list(sl_all), list(sl_all)
        sl_lo[ax] = slice(1, None)
        sl_hi[ax] = slice(0, -1)
        # the -c face of every voxel except the first plane in c
        neg[tuple(sl_lo)] = a[c][tuple(sl_lo)]
        # the +c face is the -c face of the next voxel along c
        pos[tuple(sl_hi)] = a[c][tuple(sl_lo)]
        n.append((neg, pos))

    nz, pz = n[0]
    ny, py = n[1]
    nx, px = n[2]
    m = np.maximum.reduce([nx, ny, nz, px, py, pz])
    bits = np.zeros((Z, Y, X), dtype=np.uint8)
    for val, bit in ((nz, 0x01), (ny, 0x02), (nx, 0x04),
                     (pz, 0x08), (py, 0x10), (px, 0x20)):
        bits |= np.where((val == m) | (val >= high), bit, 0).astype(np.uint8)
    bits[m <= low] = 0
    return bits.ravel()


def assert_no_oob_bits(bits: np.ndarray, Z: int, Y: int, X: int) -> None:
    """The whole of ws.cu's host path dereferences `seg[i + dir[d]]` with no
    bounds check, so it is only correct because k_flow cannot set a bit that
    leaves the volume. Check that rather than trust it.
    """
    for d, oob in enumerate(oob_masks(Z, Y, X)):
        bad = np.flatnonzero((bits & DBIT[d]).astype(bool) & oob)
        assert bad.size == 0, (
            f"k_flow set out-of-bounds direction {d} at {bad[:4]}; the host "
            f"reference would read outside the array")


# --------------------------------------------------------------------------
# HOST reference: plateau_basins (ws.cu:703)
# --------------------------------------------------------------------------

def plateau_basins_host(bits: np.ndarray, Z: int, Y: int, X: int,
                        W: Work) -> tuple[np.ndarray, int]:
    """Faithful port of `plateau_basins`, including its in-place mutation.

    Returns (seg, nfrag). seg is 0 on background and 1..nfrag on fragments.
    """
    seg = bits.astype(np.uint32).copy()
    dirs = offsets(Y, X)

    # -- phase 1a: seed every voxel that has a non-reciprocal outgoing edge.
    # Reads only bits 0-5, and only writes MARK, so this is order-independent.
    bfs: list[int] = []
    for i in range(seg.size):
        for d in range(6):
            if seg[i] & DBIT[d]:
                W.bfs_edge_tests += 1
                if not (seg[i + dirs[d]] & RBIT[d]):
                    seg[i] |= MARK
                    bfs.append(i)
                    break
    W.full_vox_passes += 1
    nseed = len(bfs)

    # -- phase 1b: BFS along reciprocal edges. Each processed voxel is
    # overwritten by `to_set`, the last of its non-reciprocal directions, which
    # is what makes this order-dependent.
    bi = 0
    while bi < len(bfs):
        i = int(bfs[bi])
        to_set = np.uint32(0)
        for d in range(6):
            if seg[i] & DBIT[d]:
                j = i + dirs[d]
                W.bfs_edge_tests += 1
                if seg[j] & RBIT[d]:
                    if not (seg[j] & MARK):
                        seg[j] |= MARK
                        bfs.append(j)
                        W.bfs_pushes += 1
                else:
                    to_set = np.uint32(DBIT[d])
        seg[i] = to_set
        bi += 1
    W.plateau_voxels += len(bfs)

    # -- phase 2: label basins by following the (now single-bit) flow.
    next_id = 1
    order: list[int] = []
    for i in range(seg.size):
        if seg[i] == 0:
            seg[i] |= HIGH
            continue
        if (seg[i] & HIGH) or not seg[i]:
            continue
        order = [i]
        seg[i] |= MARK
        bi = 0
        while bi < len(order):
            me = int(order[bi])
            d = 0
            while d < 6:
                if seg[me] & DBIT[d]:
                    him = me + dirs[d]
                    if seg[him] & HIGH:
                        lab = seg[him]
                        for it in order:
                            seg[it] = lab
                        order = []
                        d = 6
                        continue
                    if not (seg[him] & MARK):
                        seg[him] |= MARK
                        order.append(him)
                d += 1
            bi += 1
        if order:
            lab = np.uint32(HIGH | next_id)
            for it in order:
                seg[it] = lab
            next_id += 1
    W.full_vox_passes += 1
    seg &= np.uint32(0x7FFFFFFF)
    return seg, next_id - 1


# --------------------------------------------------------------------------
# DEVICE path: plateau BFS over an immutable `orig`, then union-find
# --------------------------------------------------------------------------

def plateau_bfs_device(bits: np.ndarray, Z: int, Y: int, X: int,
                       W: Work) -> np.ndarray:
    """Replica of k_corner_flag / k_expand_atomic / k_rewrite_visited.

    Differs from the host in exactly one respect: every reciprocity test reads
    the immutable `orig`, so the answer does not depend on visit order.
    """
    orig = bits
    dirs = offsets(Y, X)
    size = orig.size

    # k_corner_flag: seed set, identical predicate to the host's phase 1a.
    seeds = []
    for i in range(size):
        for d in range(6):
            if orig[i] & DBIT[d]:
                if not (orig[i + dirs[d]] & RBIT[d]):
                    seeds.append(i)
                    break
    W.full_vox_passes += 1

    # k_mark_vis_frontier + k_expand_atomic, level-synchronous.
    vis = np.zeros(size, dtype=bool)
    front = np.array(seeds, dtype=np.int64)
    vis[front] = True
    nit = 0
    while front.size:
        nxt = []
        for i in front.tolist():
            b = orig[i]
            for d in range(6):
                if not (b & DBIT[d]):
                    continue
                j = i + dirs[d]
                W.bfs_edge_tests += 1
                if orig[j] & RBIT[d] and not vis[j]:
                    vis[j] = True
                    nxt.append(j)
                    W.bfs_pushes += 1
        front = np.array(nxt, dtype=np.int64)
        nit += 1
    W.plateau_voxels += int(vis.sum())

    # k_rewrite_visited: visited voxels keep only the last non-reciprocal
    # direction; unvisited voxels keep their original byte.
    out = orig.copy()
    oob = oob_masks(Z, Y, X)
    for i in np.flatnonzero(vis).tolist():
        b = orig[i]
        to_set = 0
        for d in range(6):
            if not (b & DBIT[d]):
                continue
            if oob[d][i]:
                to_set = DBIT[d]
                continue
            j = i + dirs[d]
            if not (orig[j] & RBIT[d]):
                to_set = DBIT[d]
        out[i] = to_set
    W.full_vox_passes += 1
    return out


class UF:
    """ws.cu:373 uf_hook / ws.cu:245 uf_find, sequentially.

    The device versions race on `atomicCAS`, but every hook makes the smaller
    index the parent, so the component partition and its representative (the
    minimum index) are the same for any interleaving. That is what makes a
    sequential replica legitimate here.
    """

    def __init__(self, n: int, W: Work) -> None:
        self.p = np.arange(n, dtype=np.uint32)
        self.W = W

    def find(self, x: int) -> int:
        p = self.p
        r = int(x)
        while p[r] != r:
            r = int(p[r])
            self.W.find_steps += 1
        y = int(x)
        while y != r:
            n = int(p[y])
            p[y] = r
            y = n
        return r

    def hook(self, a: int, b: int) -> None:
        self.W.hook_calls += 1
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        lo, hi = (ra, rb) if ra < rb else (rb, ra)
        self.p[hi] = lo
        self.W.hook_unions += 1

    def compress(self) -> None:
        for i in range(self.p.size):
            self.p[i] = self.find(i)
        self.W.full_vox_passes += 1


def watershed_uf_device(bits0: np.ndarray, bits1: np.ndarray,
                        Z: int, Y: int, X: int,
                        W: Work) -> tuple[np.ndarray, int]:
    """Replica of watershed_device's union-find half (ws.cu:626-676)."""
    size = bits0.size
    dirs = offsets(Y, X)
    oob = oob_masks(Z, Y, X)
    uf = UF(size, W)

    def uf_link() -> None:
        for i in range(size):
            b = bits1[i]
            if not b:
                continue
            for d in range(6):
                if (b & DBIT[d]) and not oob[d][i]:
                    uf.hook(i, i + dirs[d])
        W.full_vox_passes += 1

    def uf_link_zero_orig() -> None:
        for i in range(size):
            if bits1[i] != 0 or bits0[i] == 0:
                continue
            b = bits0[i]
            dest_any = dest_last = -1
            for d in range(6):
                if not (b & DBIT[d]) or oob[d][i]:
                    continue
                j = i + dirs[d]
                dest_any = j
                if bits1[j] != 0 or bits0[j] == 0:
                    dest_last = j
            if dest_last >= 0:
                uf.hook(i, dest_last)
            elif dest_any >= 0:
                uf.hook(i, dest_any)
        W.full_vox_passes += 1

    for _ in range(2):
        uf_link()
        uf_link_zero_orig()
        uf.compress()

    p = uf.p
    mainroot = np.zeros(size, dtype=bool)
    mainroot[p[bits1 != 0]] = True
    W.full_vox_passes += 1

    # k_collect_extra_target: two-hop search for a main-root neighbour.
    target = np.full(size, SENT, dtype=np.uint32)
    for i in range(size):
        if bits0[i] == 0:
            continue
        r = int(p[i])
        if mainroot[r]:
            continue
        best = int(SENT)
        for d in range(6):
            if oob[d][i]:
                continue
            j = i + dirs[d]
            if mainroot[p[j]] and j < best:
                best = j
            for d2 in range(6):
                if oob[d2][j]:
                    continue
                k = j + dirs[d2]
                if mainroot[p[k]] and k < best:
                    best = k
        if best != int(SENT):
            target[r] = min(int(target[r]), best)
    W.full_vox_passes += 1

    for i in range(size):
        if p[i] != i or mainroot[i]:
            continue
        if target[i] != SENT:
            uf.hook(i, int(target[i]))
    uf.compress()

    p = uf.p
    sz = np.bincount(p, minlength=size).astype(np.uint32)
    W.full_vox_passes += 1

    # k_merge_tiny: absorb roots of <= 2 voxels into a neighbouring component.
    for i in range(size):
        r = int(p[i])
        if r != i or bits0[i] == 0 or sz[r] > 2:
            continue
        best = -1
        for d in range(6):
            if oob[d][i]:
                continue
            j = i + dirs[d]
            if bits0[j] == 0 or p[j] == r:
                continue
            if best < 0 or j < best:
                best = j
        if best >= 0:
            uf.hook(r, int(p[best]))
    uf.compress()
    W.full_vox_passes += 1

    p = uf.p
    bgroot = np.zeros(size, dtype=bool)
    bgroot[p[bits0 == 0]] = True
    flag = ((p == np.arange(size, dtype=np.uint32)) & ~bgroot)
    nfrag = int(flag.sum())
    psum = np.concatenate(([0], np.cumsum(flag)[:-1])).astype(np.uint32)
    seg = np.where(bgroot[p], 0, psum[p] + 1).astype(np.uint32)
    W.full_vox_passes += 3
    return seg, nfrag


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# PRODUCTION path: watershed_gpu_e9 = k_flow + e9b_divide_d + e9c_basins_d
# --------------------------------------------------------------------------

def face_pairs(Z: int, Y: int, X: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Every in-bounds (i, j) pair across a face in positive direction d.

    d must be 3, 4 or 5 (+z, +y, +x). Enumerating only the positive directions
    visits each face exactly once, and the reverse bit of d is DBIT[d - 3].
    """
    assert d in (3, 4, 5)
    idx = np.arange(Z * Y * X, dtype=np.int64).reshape(Z, Y, X)
    ax = d - 3
    lo = [slice(None)] * 3
    hi = [slice(None)] * 3
    lo[ax] = slice(0, -1)
    hi[ax] = slice(1, None)
    return idx[tuple(lo)].ravel(), idx[tuple(hi)].ravel()


def reciprocal_edges(bits: np.ndarray, Z: int, Y: int, X: int
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Faces where both voxels point at each other -- k_hook_bidir's edge set."""
    us, vs = [], []
    for d in (3, 4, 5):
        i, j = face_pairs(Z, Y, X, d)
        fwd = (bits[i] & DBIT[d]).astype(bool)
        back = (bits[j] & DBIT[d - 3]).astype(bool)
        keep = fwd & back
        us.append(i[keep])
        vs.append(j[keep])
    return np.concatenate(us), np.concatenate(vs)


def flow_edges(bits: np.ndarray, Z: int, Y: int, X: int
               ) -> tuple[np.ndarray, np.ndarray]:
    """k_hook_remain's edge set: every set direction bit, reciprocal or not.

    Note it does not require bits[j] != 0, so a zero-bit voxel can be pulled
    into a component as a target. Whether that can actually happen is checked
    by `assert_bg_isolated`.
    """
    us, vs = [], []
    for d in (3, 4, 5):
        i, j = face_pairs(Z, Y, X, d)
        f = (bits[i] & DBIT[d]).astype(bool)
        us.append(i[f])
        vs.append(j[f])
        b = (bits[j] & DBIT[d - 3]).astype(bool)
        us.append(j[b])
        vs.append(i[b])
    return np.concatenate(us), np.concatenate(vs)


def cc_min_root(n: int, ei: np.ndarray, ej: np.ndarray, W: Work) -> np.ndarray:
    """Root array after k_hook_* / k_uf_compress_c have run to convergence.

    Every hook in ws.cu makes the smaller index the parent -- `atomicMin` in
    k_hook_bidir and k_hook_remain, `lo` in uf_hook -- so the fixed point is
    independent of the interleaving and the root of a component is its minimum
    member. That is what makes it legitimate to compute the converged partition
    directly instead of simulating the rounds. Round *counts* do depend on the
    interleaving and are not claimed here.
    """
    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        r = int(x)
        while parent[r] != r:
            r = int(parent[r])
            W.find_steps += 1
        y = int(x)
        while y != r:
            nx = int(parent[y])
            parent[y] = r
            y = nx
        return r

    for a, b in zip(ei.tolist(), ej.tolist()):
        W.hook_calls += 1
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb
            W.hook_unions += 1
    for i in range(n):
        find(i)
    return parent.astype(np.uint32)


def tiled_hook_edges(bits: np.ndarray, Z: int, Y: int, X: int,
                     TX: int = 32, TY: int = 4, TZ: int = 4
                     ) -> set[tuple[int, int]]:
    """The edge set k_hook_bidir_tiled issues, using its exact index maths.

    W4 replaces k_hook_bidir's (X/256, Y, Z) launch -- which puts a block's
    +/-z neighbour X*Y*4 bytes away, 23 MB at 2.16 Gvox, and so misses a 6 MB
    L2 on every one of six gathers per voxel per round -- with a 32x4x4 tile
    staged in shared memory along with its one-voxel halo.

    Correctness of that kernel reduces to one claim: it issues hooks for
    exactly the same (i, j) pairs. Staleness of the staged values does not
    matter, because every write is an atomicMin into a root's slot so parent
    entries only decrease, the round loop runs to convergence, and the fixed
    point of min-index hooking is the component minimum for any read order.
    Dropping or inventing an edge would matter, and a halo off-by-one is the
    obvious way to do that. So this replicates the shared-memory indexing
    literally -- ws_sidx, the (gx, gy, gz) bounds test, the inert bits=0 fill
    for out-of-volume slots -- and the caller diffs it against the untiled set.
    """
    SX, SY = TX + 2, TY + 2
    dlz = (-1, 0, 0, 1, 0, 0)
    dly = (0, -1, 0, 0, 1, 0)
    dlx = (0, 0, -1, 0, 0, 1)

    def sidx(lz, ly, lx):
        return ((lz + 1) * SY + (ly + 1)) * SX + (lx + 1)

    out: set[tuple[int, int]] = set()
    for z0 in range(0, Z, TZ):
        for y0 in range(0, Y, TY):
            for x0 in range(0, X, TX):
                # stage tile + halo exactly as the kernel does
                sb: dict[int, int] = {}
                for lz in range(-1, TZ + 1):
                    for ly in range(-1, TY + 1):
                        for lx in range(-1, TX + 1):
                            gx, gy, gz = x0 + lx, y0 + ly, z0 + lz
                            s = sidx(lz, ly, lx)
                            if 0 <= gx < X and 0 <= gy < Y and 0 <= gz < Z:
                                sb[s] = int(bits[(gz * Y + gy) * X + gx])
                            else:
                                sb[s] = 0
                for lz in range(TZ):
                    for ly in range(TY):
                        for lx in range(TX):
                            z, y, x = z0 + lz, y0 + ly, x0 + lx
                            if z >= Z or y >= Y or x >= X:
                                continue
                            b = sb[sidx(lz, ly, lx)]
                            if not b:
                                continue
                            i = (z * Y + y) * X + x
                            for d in range(6):
                                if not (b & DBIT[d]):
                                    continue
                                if (d == 0 and z == 0) or (d == 3 and z == Z - 1) \
                                   or (d == 1 and y == 0) or (d == 4 and y == Y - 1) \
                                   or (d == 2 and x == 0) or (d == 5 and x == X - 1):
                                    continue
                                nb = sidx(lz + dlz[d], ly + dly[d], lx + dlx[d])
                                if not (sb[nb] & RBIT[d]):
                                    continue
                                j = i + offsets(Y, X)[d]
                                out.add((min(i, j), max(i, j)))
    return out


def untiled_hook_edges(bits: np.ndarray, Z: int, Y: int, X: int
                       ) -> set[tuple[int, int]]:
    """The edge set k_hook_bidir issues, straight from its source."""
    dirs = offsets(Y, X)
    oob = oob_masks(Z, Y, X)
    out: set[tuple[int, int]] = set()
    for i in range(bits.size):
        b = int(bits[i])
        if not b:
            continue
        for d in range(6):
            if not (b & DBIT[d]) or oob[d][i]:
                continue
            j = i + dirs[d]
            if not (bits[j] & RBIT[d]):
                continue
            out.add((min(i, j), max(i, j)))
    return out


def uf_rounds(n: int, ei: np.ndarray, ej: np.ndarray) -> int:
    """How many hook+compress rounds the union-find needs to converge.

    Level-synchronous model of the device loop: every edge hooks from the same
    parent snapshot (`atomicMin` into the round's output), then
    k_uf_compress_c flattens, and the caller stops when a round changes
    nothing. The device is asynchronous and can converge in fewer rounds, so
    this is an upper bound on its count -- but it scales the same way, which is
    the question being asked.

    Why it matters: the plan's whole scaling argument is that "iteration counts
    stay constant" while bytes scale 12x. That is measured and true for
    agglomeration, because make_big.py mirror-tiles the volume so the graph is
    12 disjoint copies. It cannot be true for a union-find, whose round count
    grows with the longest chain it has to collapse, and the graded volume is
    3x2x2 tiles of val -- so chains that ran along an axis get up to 3x longer.
    """
    parent = np.arange(n, dtype=np.int64)
    rounds = 0
    while True:
        rounds += 1
        snap = parent.copy()
        pa, pb = snap[ei], snap[ej]
        diff = pa != pb
        if not diff.any():
            return rounds - 1
        lo = np.minimum(pa[diff], pb[diff])
        hi = np.maximum(pa[diff], pb[diff])
        # atomicMin semantics: many writers to the same slot, smallest wins.
        np.minimum.at(parent, hi, lo)
        # k_uf_compress_c: full path compression.
        changed = False
        for i in range(n):
            r = i
            while parent[r] != r:
                r = parent[r]
            if parent[i] != r:
                changed = True
            j = i
            while parent[j] != r:
                parent[j], j = r, parent[j]
        if not changed and not diff.any():
            return rounds
        if rounds > 200:
            return rounds
        # converged when a full round moved nothing
        if np.array_equal(snap, parent):
            return rounds


# --------------------------------------------------------------------------
# W5: tile-local union-find, then a stitch over tile faces
# --------------------------------------------------------------------------
#
# The 317-of-499 ms half of the watershed union-find is k_uf_compress_c, and
# ws.cu:1214 already records why: it moves about 2 GB and runs some 30x off
# bandwidth roofline, so its cost is chains of dependent random loads. W4
# tiled the *hook*, which was the bandwidth half. Nothing so far touches the
# latency half, and it is the larger one.
#
# W5 resolves each tile's union-find entirely in shared memory, where a
# pointer chase costs tens of cycles instead of a 23 MB-stride L2 miss, and
# then stitches across tiles over the voxels that sit on a tile face.
#
# The licence for this is ws.cu:1221, written for k_uf_jump: what the gate
# pins is the fixed point, not the path to it. Both loops run to convergence,
# and the fixed point of min-index hooking is "every entry holds its
# component's minimum index" regardless of the order the edges were applied
# in. So splitting the edge set into intra-tile and cross-tile halves and
# applying them in two phases lands on the same array.
#
# Two things have to be true for that argument to hold in the kernel, and both
# are checked here rather than asserted:
#
#   1. Local index order inside a tile agrees with global index order, or
#      "minimum local index" would not be "minimum global index". It does:
#      global index is lexicographic in (z, y, x), local is lexicographic in
#      (lz, ly, lx), and z = z0 + lz with the tile origin fixed.
#   2. The two phases together cover every edge. Phase 1 covers every
#      intra-tile edge; phase 2 covers every edge incident to a face voxel,
#      and a cross-tile edge has both endpoints on a face. Union is the lot.

W5_TX, W5_TY, W5_TZ = 32, 16, 8


def tile_index(Z: int, Y: int, X: int, TZ: int, TY: int, TX: int
               ) -> tuple[np.ndarray, np.ndarray]:
    """Per-voxel tile id and tile-face mask, for a TZ x TY x TX tiling.

    A voxel is on a face if it sits in the first or last plane of its tile in
    any axis. Tiles at the far edge of the volume can be truncated, so their
    last plane is not at local index T-1; such a voxel goes unmarked, which is
    safe because the neighbour it would have needed marking for is outside the
    volume and no edge exists.
    """
    z, y, x = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X),
                          indexing="ij")
    lz, ly, lx = z % TZ, y % TY, x % TX
    tz, ty, tx = z // TZ, y // TY, x // TX
    ntx, nty = (X + TX - 1) // TX, (Y + TY - 1) // TY
    tid = ((tz * nty + ty) * ntx + tx).ravel().astype(np.int64)
    face = ((lz == 0) | (lz == TZ - 1) | (ly == 0) | (ly == TY - 1)
            | (lx == 0) | (lx == TX - 1)).ravel()
    return tid, face


def w5_partition(n: int, ei: np.ndarray, ej: np.ndarray,
                 tid: np.ndarray, face: np.ndarray) -> np.ndarray:
    """The parent array W5 converges to, computed in its two phases.

    Phase 1 applies only intra-tile edges, which is what k_uf_tile_local does
    in shared memory. Phase 2 applies every edge incident to a face voxel,
    which is a superset of the cross-tile edges and is what the stitch kernel
    covers by iterating its face list. Both phases run to their fixed point,
    so this is exact rather than a round-by-round simulation.
    """
    intra = tid[ei] == tid[ej]
    p = uf_partition(n, ei[intra], ej[intra])
    inc = face[ei] | face[ej]
    a, b = ei[inc], ej[inc]
    # Continue the same union-find rather than starting a fresh one: phase 2
    # inherits phase 1's parent array, exactly as the kernel does.
    def find(x):
        r = int(x)
        while p[r] != r:
            r = int(p[r])
        y = int(x)
        while p[y] != r:
            p[y], y = r, int(p[y])
        return r
    for u, v in zip(a.tolist(), b.tolist()):
        ru, rv = find(u), find(v)
        if ru != rv:
            p[max(ru, rv)] = min(ru, rv)
    for i in range(n):
        find(i)
    return p.astype(np.uint32)


def w5_check(bits: np.ndarray, Z: int, Y: int, X: int,
             edges: tuple[np.ndarray, np.ndarray], tag: str,
             tile: tuple[int, int, int]) -> dict:
    """Does the two-phase order reach the same array as the one-phase order?

    This is the whole correctness question for W5, and it is a partition
    identity rather than a timing claim, so a CPU can settle it.
    """
    ei, ej = edges
    TZ, TY, TX = tile
    tid, face = tile_index(Z, Y, X, TZ, TY, TX)
    ref = cc_min_root(bits.size, ei, ej, Work())
    got = w5_partition(bits.size, ei, ej, tid, face)
    active = bits != 0
    nz_face = int((face & active).sum())
    return {"tag": tag, "tile": [TZ, TY, TX],
            "parent_identical": bool(np.array_equal(ref, got)),
            "ndiff": int((ref != got).sum()),
            "n_edges": int(ei.size),
            "n_intra": int((tid[ei] == tid[ej]).sum()),
            "n_cross": int((tid[ei] != tid[ej]).sum()),
            "face_frac": float(face.mean()),
            "stitch_list_frac": float(nz_face / max(bits.size, 1))}


def nonempty_tile_equiv(bits: np.ndarray, Z: int, Y: int, X: int,
                        tile: tuple[int, int, int]) -> dict:
    """W3: skipping tiles whose bits are all zero must change no edge.

    A tile with every byte zero has no voxel with a direction bit, so
    k_hook_* would return at its `if (!b) return` for every thread in it. The
    check is that the tile-level predicate agrees with that, i.e. no edge has
    an endpoint in a tile we would have skipped.
    """
    TZ, TY, TX = tile
    tid, _ = tile_index(Z, Y, X, TZ, TY, TX)
    ntile = int(tid.max()) + 1
    nz = np.zeros(ntile, dtype=bool)
    np.logical_or.at(nz, tid, bits != 0)
    ei, ej = flow_edges(bits, Z, Y, X)
    er, ea = reciprocal_edges(bits, Z, Y, X)
    bad = int((~nz[tid[ei]]).sum() + (~nz[tid[ej]]).sum()
              + (~nz[tid[er]]).sum() + (~nz[tid[ea]]).sum())
    return {"tile": [TZ, TY, TX], "n_tile": ntile,
            "n_tile_nonempty": int(nz.sum()),
            "tile_keep_frac": float(nz.mean()),
            "edges_into_empty_tile": bad}


def assert_bg_isolated(bits: np.ndarray, Z: int, Y: int, X: int) -> None:
    """No voxel points at a zero-bit voxel.

    e9c's k_root_flag only allocates a label to a root with bits != 0, while
    k_write_labels gives every bits != 0 voxel `psum[parent[i]] + 1`. If a
    component's minimum member had bits == 0 it would be an unflagged root, and
    psum at that index is the count of flagged roots below it -- i.e. some other
    fragment's label. So the labelling is only collision-free because zero-bit
    voxels are isolated in the flow graph.

    For the *original* k_flow output that is forced: bits[j] == 0 means every
    face of j is <= low, but a neighbour i pointing at j does so across a face
    that is i's maximum and > low, and that face is also one of j's. After the
    divide it is no longer forced, which is exactly why it is checked here.
    """
    ei, ej = flow_edges(bits, Z, Y, X)
    bad = ej[bits[ej] == 0]
    assert bad.size == 0, (
        f"{bad.size} flow edges point at a zero-bit voxel (e.g. {bad[:4]}); "
        f"e9c would give the source voxel another fragment's label")


def e9b_divide(bits: np.ndarray, Z: int, Y: int, X: int,
               W: Work) -> tuple[np.ndarray, dict]:
    """Replica of e9b_divide_d: plateau union-find, then the per-plateau BFS
    that rewrites each plateau into single-bit exits.

    Returns the rewritten direction field and a dict of the intermediate
    quantities W1 needs to reason about (plateau count, corner count, the
    queue-sizing identity).
    """
    size = bits.size
    dirs = offsets(Y, X)
    oob = oob_masks(Z, Y, X)

    # -- plateau union-find (k_hook_bidir + k_uf_compress_c to convergence)
    ei, ej = reciprocal_edges(bits, Z, Y, X)
    parent = cc_min_root(size, ei, ej, W)
    W.full_vox_passes += 2

    # -- k_corner_flag: a voxel with at least one non-reciprocal exit
    flag = np.zeros(size, dtype=bool)
    for i in range(size):
        b = bits[i]
        if not b:
            continue
        for d in range(6):
            if not (b & DBIT[d]) or oob[d][i]:
                continue
            if not (bits[i + dirs[d]] & RBIT[d]):
                flag[i] = True
                break
    W.full_vox_passes += 1

    # -- k_count_v2: vcount[root] over voxels that are a corner or have a
    # reciprocal neighbour, which together is every non-zero voxel.
    has_recip = np.zeros(size, dtype=bool)
    has_recip[ei] = True
    has_recip[ej] = True
    in_plat = (bits != 0) & (flag | has_recip)
    vcount = np.bincount(parent[in_plat], minlength=size).astype(np.uint32)
    W.full_vox_passes += 1
    assert np.array_equal(in_plat, bits != 0), (
        "k_count_v2's in_plat predicate is not the same set as bits != 0, so "
        "vcount[root] is not the plateau's voxel count and qsz undersizes")

    # -- corner list, then a stable sort by plateau root. CUB's radix sort is
    # LSD and therefore stable, so within a plateau the corners stay in
    # ascending voxel-index order; that order is what fixes the BFS result.
    corners = np.flatnonzero(flag).astype(np.uint32)
    nC = int(corners.size)
    keys = parent[corners]
    order = np.argsort(keys, kind="stable")
    corners_out = corners[order]
    keys_out = keys[order]

    # -- plateau table (k_run_start / k_scatter_plat / k_plat_meta)
    if nC:
        start = np.ones(nC, dtype=bool)
        start[1:] = keys_out[1:] != keys_out[:-1]
        plat_begin = np.flatnonzero(start)
        plat_root = keys_out[plat_begin]
        plat_nseed = np.diff(np.append(plat_begin, nC))
        qsz = vcount[plat_root]
    else:
        plat_begin = np.array([], dtype=np.int64)
        plat_root = np.array([], dtype=np.uint32)
        plat_nseed = np.array([], dtype=np.int64)
        qsz = np.array([], dtype=np.uint32)
    P = int(plat_begin.size)

    # -- k_or40_u32 then k_indep_bfs, one sequential BFS per plateau.
    seg = bits.copy()
    seg[corners_out] |= MARK
    qmax = 0
    for p in range(P):
        c0 = int(plat_begin[p])
        nseed = int(plat_nseed[p])
        cap = int(qsz[p])
        q = [int(v) for v in corners_out[c0:c0 + nseed]]
        assert len(q) <= cap, f"plateau {p} overflows its queue at seeding"
        bi = 0
        while bi < len(q):
            i = q[bi]
            b = seg[i]
            to_set = 0
            for d in range(6):
                if not (b & DBIT[d]):
                    continue
                j = i + dirs[d]
                W.bfs_edge_tests += 1
                if seg[j] & RBIT[d]:
                    if not (seg[j] & MARK):
                        seg[j] |= MARK
                        q.append(j)
                        W.bfs_pushes += 1
                        assert len(q) <= cap, (
                            f"plateau {p} BFS exceeded qsz={cap}; the "
                            f"tail <= vcount argument in k_plat_meta is wrong")
                else:
                    to_set = DBIT[d]
            seg[i] = to_set
            bi += 1
        qmax = max(qmax, len(q) - cap)
        W.plateau_voxels += len(q)
    W.full_vox_passes += 1

    meta = {"nC": nC, "P": P, "qslack_max": int(qmax),
            "uf_recip_edges": int(ei.size)}
    return seg, meta


def e9c_basins(bits: np.ndarray, Z: int, Y: int, X: int,
               W: Work) -> tuple[np.ndarray, int]:
    """Replica of e9c_basins_d: connected components of the rewritten flow
    field, then a scan over roots to allocate fragment ids.
    """
    size = bits.size
    assert_bg_isolated(bits, Z, Y, X)
    ei, ej = flow_edges(bits, Z, Y, X)
    parent = cc_min_root(size, ei, ej, W)
    W.full_vox_passes += 2

    flag = (bits != 0) & (parent == np.arange(size, dtype=np.uint32))
    nfrag = int(flag.sum())
    psum = np.concatenate(([0], np.cumsum(flag)[:-1])).astype(np.uint32)
    seg = np.where(bits != 0, psum[parent] + 1, 0).astype(np.uint32)
    W.full_vox_passes += 3
    return seg, nfrag


def w3_label_of(mask: np.ndarray, blkscan: np.ndarray, r: int) -> int:
    """Device label_of: exclusive root count below r, plus one."""
    blk = r >> 10
    wi = (r >> 5) & 31
    bit = r & 31
    w = mask[blk * 32: blk * 32 + 32]
    extra = int(sum(int(w[k]).bit_count() for k in range(wi)))
    extra += int(int(w[wi]) & ((1 << bit) - 1 if bit else 0)).bit_count()
    return int(blkscan[blk]) + extra + 1


def w3_labels(bits: np.ndarray, parent: np.ndarray
              ) -> tuple[np.ndarray, int]:
    """The W3 bitmask labelling, bit-identical to the flag+scan path."""
    n = bits.size
    nblk = (n + 1023) // 1024
    nwords = nblk * 32
    mask = np.zeros(nwords, dtype=np.uint32)
    roots = (bits != 0) & (parent == np.arange(n, dtype=np.uint32))
    idx = np.flatnonzero(roots)
    np.bitwise_or.at(mask, idx >> 5, (np.uint32(1) << (idx & 31)))
    pop = np.array([int(x).bit_count() for x in mask], dtype=np.uint32)
    blk = pop.reshape(nblk, 32).sum(axis=1).astype(np.uint32)
    blkscan = np.concatenate(([0], np.cumsum(blk)[:-1])).astype(np.uint32)
    seg = np.zeros(n, dtype=np.uint32)
    nz = np.flatnonzero(bits)
    for i in nz.tolist():
        seg[i] = w3_label_of(mask, blkscan, int(parent[i]))
    return seg, int(blk.sum())


def e9_watershed(bits: np.ndarray, Z: int, Y: int, X: int,
                 W: Work) -> tuple[np.ndarray, int, dict]:
    """The whole production watershed, k_flow output in, labels out."""
    div, meta = e9b_divide(bits, Z, Y, X, W)
    seg, nfrag = e9c_basins(div, Z, Y, X, W)
    return seg, nfrag, meta


# --------------------------------------------------------------------------
# W1: 2x2x2 block-based labelling, and whether the plan's version is sound
# --------------------------------------------------------------------------

def block_ids(Z: int, Y: int, X: int) -> tuple[np.ndarray, int]:
    """Map each voxel to its 2x2x2 block, returning (blk, nblock)."""
    bz, by, bx = (Z + 1) // 2, (Y + 1) // 2, (X + 1) // 2
    z, y, x = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X),
                          indexing="ij")
    blk = ((z // 2) * by + (y // 2)) * bx + (x // 2)
    return blk.ravel().astype(np.int64), bz * by * bx


def uf_partition(n: int, ei: np.ndarray, ej: np.ndarray) -> np.ndarray:
    """Min-index root per element, over an arbitrary edge list."""
    p = np.arange(n, dtype=np.int64)

    def find(x):
        r = x
        while p[r] != r:
            r = p[r]
        while p[x] != r:
            p[x], x = r, p[x]
        return r

    for a, b in zip(ei.tolist(), ej.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            p[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)], dtype=np.int64)


def w1_block_check(bits: np.ndarray, Z: int, Y: int, X: int,
                   edges: tuple[np.ndarray, np.ndarray], tag: str) -> dict:
    """Compare three labellings of the same union-find edge set.

    voxel   one parent slot per voxel. This is what ws.cu does today and is by
            definition the right answer.
    naive   one label per 2x2x2 block, unioning blocks whenever any edge
            crosses between them. This is W1 as the plan words it: "one uint32
            label per 8 voxels", 0.5 B/vox.
    slots   one slot per intra-block connectivity class. Exact by construction,
            because it is the same union-find on the same edges with the parent
            array merely indexed differently. The open question for this one is
            not correctness but how many slots a block needs.

    Komura's BKE justifies `naive` for 8-connected 2D and 26-connected 3D
    labelling, where every pair of voxels inside a 2x2(x2) block is directly
    adjacent, so any two foreground voxels in a block are necessarily in the
    same component and collapsing the block loses nothing. The watershed here
    is 6-connected and its edges are flow directions rather than foreground
    adjacency, so that argument does not carry and `naive` has to be tested.
    """
    ei, ej = edges
    size = bits.size
    blk, nblk = block_ids(Z, Y, X)
    active = bits != 0

    vox_root = uf_partition(size, ei, ej)

    # naive: contract every block to a single node.
    blk_root = uf_partition(nblk, blk[ei], blk[ej])
    naive_lab = blk_root[blk]

    # An honest comparison only looks at active voxels: the naive scheme would
    # also glue background voxels to their block-mates, which is a separate and
    # even more obvious failure.
    def npart(lab):
        return len(set(zip(lab[active].tolist(), )))

    same_blocks = blk[ei] == blk[ej]
    intra = uf_partition(size, ei[same_blocks], ej[same_blocks])
    # slots per block = number of distinct intra-block classes among active
    # voxels of that block
    cls = np.full(size, -1, dtype=np.int64)
    cls[active] = intra[active]
    per_block: dict[int, set] = {}
    for i in np.flatnonzero(active).tolist():
        per_block.setdefault(int(blk[i]), set()).add(int(cls[i]))
    slots = np.array([len(v) for v in per_block.values()]) if per_block \
        else np.array([0])

    # Does naive merge two voxels that the voxel-level union-find keeps apart?
    a = vox_root[active]
    b = naive_lab[active]
    # pairs merged by naive but not by voxel: compare partition refinement
    import collections
    grp = collections.defaultdict(set)
    for x, y in zip(b.tolist(), a.tolist()):
        grp[x].add(y)
    overmerged = sum(1 for v in grp.values() if len(v) > 1)
    nvox_comp = len(set(a.tolist()))
    nblk_comp = len(grp)

    return {"tag": tag,
            "n_voxel_components": nvox_comp,
            "n_naive_components": nblk_comp,
            "naive_overmerged_groups": overmerged,
            "naive_is_exact": bool(overmerged == 0
                                   and nblk_comp == nvox_comp),
            "slots_mean": float(slots.mean()),
            "slots_max": int(slots.max()),
            "slots_hist": {int(k): int(v) for k, v in
                           zip(*np.unique(slots, return_counts=True))}}


def synth(shape: tuple[int, int, int], seed: int, levels: int) -> np.ndarray:
    """Synthetic affinities with a deliberately small number of distinct
    values, so ties -- and therefore plateaus -- are common. Real CREMI
    affinities tie far less often, which makes this the harder input for every
    plateau code path.
    """
    rng = np.random.default_rng(seed)
    Z, Y, X = shape
    q = 255 // max(levels - 1, 1)
    return (rng.integers(0, levels, size=(3, Z, Y, X)) * q).astype(np.uint8)


def relabel_canonical(seg: np.ndarray) -> np.ndarray:
    """Fragment ids are arbitrary up to renumbering; compare partitions by
    first-appearance order so a pure id permutation is not called a difference.
    """
    out = np.zeros_like(seg)
    _, first = np.unique(seg, return_index=True)
    nxt = 1
    for v in seg[np.sort(first)]:
        if v == 0:
            continue
        out[seg == v] = nxt
        nxt += 1
    return out


def one_case(shape, seed, levels, low, high, verbose=True,
             legacy=False, blocks=False, w5=False, tile=None) -> dict:
    Z, Y, X = shape
    W = Work()
    aff = synth(shape, seed, levels)
    bits = flow(aff, low, high, W)
    assert_no_oob_bits(bits, Z, Y, X)

    # The production path, and the host reference that is meant to specify it.
    seg_h, nfrag_h = plateau_basins_host(bits, Z, Y, X, W)
    seg_e9, nfrag_e9, meta = e9_watershed(bits, Z, Y, X, W)

    same_e9 = np.array_equal(relabel_canonical(seg_h),
                             relabel_canonical(seg_e9))
    r = {"shape": list(shape), "seed": seed, "levels": levels,
         "nvox": int(bits.size),
         "nfrag_host": nfrag_h, "nfrag_e9": nfrag_e9,
         "bg_host": int((seg_h == 0).sum()),
         "bg_e9": int((seg_e9 == 0).sum()),
         "e9_matches_host": bool(same_e9), **meta, "work": W.asdict()}

    if blocks:
        # The two union-finds W1 proposes to convert, checked separately: the
        # plateau one runs on the reciprocal subgraph of the raw field, the
        # basin one on every flow edge of the divided field.
        div, _ = e9b_divide(bits, Z, Y, X, Work())
        r["w1"] = [
            w1_block_check(bits, Z, Y, X,
                           reciprocal_edges(bits, Z, Y, X), "plateau"),
            w1_block_check(div, Z, Y, X,
                           flow_edges(div, Z, Y, X), "basin"),
        ]
        if verbose:
            for c in r["w1"]:
                print(f"W1   {c['tag']:8s} voxel_cc={c['n_voxel_components']:6d} "
                      f"naive_cc={c['n_naive_components']:6d} "
                      f"overmerged={c['naive_overmerged_groups']:5d} "
                      f"exact={c['naive_is_exact']!s:5s} "
                      f"slots mean={c['slots_mean']:.2f} max={c['slots_max']}")

    if w5:
        # Both union-finds W5 replaces: the plateau one over the reciprocal
        # subgraph of the raw field, the basin one over every flow edge of the
        # divided field. A tile smaller than the shipped 32x16x8 is used on
        # these small cases so that there is more than one tile to stitch.
        tl = tuple(tile) if tile else (4, 4, 8)
        div, _ = e9b_divide(bits, Z, Y, X, Work())
        r["w5"] = [
            w5_check(bits, Z, Y, X, reciprocal_edges(bits, Z, Y, X),
                     "plateau", tl),
            w5_check(div, Z, Y, X, flow_edges(div, Z, Y, X), "basin", tl),
        ]
        r["w3"] = [nonempty_tile_equiv(bits, Z, Y, X, tl),
                   nonempty_tile_equiv(div, Z, Y, X, tl)]
        # Label-fusion identity: bitmask label_of must reproduce the scan.
        p = cc_min_root(div.size, *flow_edges(div, Z, Y, X), Work())
        flag = (div != 0) & (p == np.arange(div.size, dtype=np.uint32))
        psum = np.concatenate(([0], np.cumsum(flag)[:-1])).astype(np.uint32)
        ref = np.where(div != 0, psum[p] + 1, 0).astype(np.uint32)
        got, nf = w3_labels(div, p)
        r["w3_label"] = {"identical": bool(np.array_equal(ref, got)),
                         "nfrag_scan": int(flag.sum()), "nfrag_mask": nf,
                         "ndiff": int((ref != got).sum())}
        if verbose:
            for c in r["w5"]:
                print(f"W5   {c['tag']:8s} tile={c['tile']} "
                      f"identical={c['parent_identical']!s:5s} "
                      f"ndiff={c['ndiff']:5d} "
                      f"edges={c['n_edges']:6d} "
                      f"cross={c['n_cross']:6d} "
                      f"stitch_list={c['stitch_list_frac']:.3f}")
            for c in r["w3"]:
                print(f"W3   tiles kept {c['n_tile_nonempty']}/{c['n_tile']} "
                      f"({c['tile_keep_frac']:.3f}) "
                      f"edges_into_empty={c['edges_into_empty_tile']}")
            print(f"W3   labels identical={r['w3_label']['identical']} "
                  f"nfrag {r['w3_label']['nfrag_scan']}/"
                  f"{r['w3_label']['nfrag_mask']} "
                  f"ndiff={r['w3_label']['ndiff']}")

    if legacy:
        bits1 = plateau_bfs_device(bits, Z, Y, X, W)
        seg_d, nfrag_d = watershed_uf_device(bits, bits1, Z, Y, X, W)
        r["nfrag_legacy_dev"] = nfrag_d
        r["legacy_dev_matches_host"] = bool(np.array_equal(
            relabel_canonical(seg_h), relabel_canonical(seg_d)))
    if verbose:
        print(f"W0 {str(shape):12s} seed={seed} lv={levels:2d}  "
              f"nfrag host={nfrag_h:6d} e9={nfrag_e9:6d}  "
              f"bg {r['bg_host']:6d}/{r['bg_e9']:6d}  "
              f"P={meta['P']:5d} nC={meta['nC']:6d}  "
              f"e9-vs-host={'SAME' if same_e9 else 'DIFF'}")
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", nargs=3, type=int, default=[16, 16, 16])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--levels", type=int, default=4,
                    help="distinct affinity values; lower = more plateaus")
    ap.add_argument("--low", type=float, default=0.1)
    ap.add_argument("--high", type=float, default=0.8)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--blocks", action="store_true",
                    help="run the W1 block-labelling soundness check")
    ap.add_argument("--w5", action="store_true",
                    help="run the W5 tile-local union-find identity check")
    ap.add_argument("--tile", nargs=3, type=int, default=None,
                    metavar=("TZ", "TY", "TX"),
                    help="tile shape for --w5 (default 4 4 8 on small cases)")
    ap.add_argument("--legacy", action="store_true",
                    help="also run the two dead implementations")
    ap.add_argument("--out", default="w0_ws_ref.json")
    args = ap.parse_args()

    kw = dict(legacy=args.legacy, blocks=args.blocks, w5=args.w5,
              tile=args.tile)
    rows = []
    if args.sweep:
        # Uneven tile boundaries are the failure mode a tiled kernel has, so
        # the sweep runs shapes that do and do not divide the tile.
        shapes = ((12, 12, 12), (13, 9, 17), (5, 4, 8)) if args.w5 \
            else ((12, 12, 12),)
        for shape in shapes:
            for levels in (2, 3, 4, 8, 32):
                for seed in range(4):
                    rows.append(one_case(shape, seed, levels,
                                         args.low, args.high, **kw))
    else:
        rows.append(one_case(tuple(args.shape), args.seed, args.levels,
                             args.low, args.high, **kw))

    ndiff = sum(1 for r in rows if not r["e9_matches_host"])
    print(f"\nW0 {len(rows)} case(s); e9 differs from the host reference in "
          f"{ndiff}")
    if args.w5:
        nw5 = sum(1 for r in rows for c in r["w5"]
                  if not c["parent_identical"])
        nw3 = sum(c["edges_into_empty_tile"] for r in rows for c in r["w3"])
        tot = sum(len(r["w5"]) for r in rows)
        print(f"W5 parent array differs from the one-phase union-find in "
              f"{nw5}/{tot} checks")
        print(f"W3 edges pointing into an all-zero tile: {nw3}")
        nl = sum(1 for r in rows if not r.get("w3_label", {}).get("identical", True))
        print(f"W3 bitmask labels differ from the scan in {nl}/{len(rows)}")
    if args.legacy:
        nl = sum(1 for r in rows if not r["legacy_dev_matches_host"])
        print(f"W0 legacy watershed_device differs from the host reference "
              f"in {nl}")
    dest = CACHE / args.out
    dest.write_text(json.dumps({"rows": rows, "ndiff": ndiff}, indent=2) + "\n")
    print(f"W0 wrote {dest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
