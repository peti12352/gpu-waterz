#!/usr/bin/env python3
"""G0: a CPU replica of parhac_e6s_dev, and the restructured variant beside it.

This exists because the machine has no GPU and the g1-g4 changes touch a hot
path whose whole value is that it is bit-identical. Writing that CUDA blind and
calling it done would be worthless. So instead the algorithm is re-implemented
here in numpy twice, once mirroring the current kernel sequence, once with the
active-list / dirty-set / root-list restructuring, and the two are required to
agree exactly.

Why a sequential replica can be exact. Every racing write in the GPU loop is
order-independent by construction: prop is an atomicMax, sz and the HSlot sums
are integer atomicAdds, and k_scale_sm_bytes rescales the contact sums to whole
affinity bytes precisely so that double addition of them is exact and therefore
commutes (see the comment at parhac_d.cu:129). The proposal priority is hashed
from the canonical root pair rather than the array index (parhac_d.cu:262), so
edge order does not enter any decision either. That is why the GPU path is
deterministic at all, and it is what makes a serial model faithful rather than
merely similar.

Faithfulness is not assumed. `--mode base` reproduces the counters that
p0aa_e6s.py recorded from the real device run on this exact cached RAG at
T=0.3, eps=0.08: n_layer, the 17-element layer_outers and layer_merges vectors,
nouter, ninner, nmerge, sum_nlive and sum_above. Matching two 17-element
vectors and five scalars is a tight enough fingerprint that a model which does
so is running the same algorithm.

The restructuring rests on one lemma, which `--mode both` checks empirically on
every merging iteration rather than trusting the argument:

    A parallel-edge collision can only occur between two dirty edges.

    Proof. A collision needs two alive edges with the same (u,v) after the
    rewrite. An edge is only rewritten if an endpoint's root changed, so at
    least one member of a colliding pair is dirty, and after rewrite it has the
    receiving red r as an endpoint. A non-dirty edge has both endpoints
    untouched, so for it to collide it must already have had r as an endpoint --
    but every edge incident to r is dirty, because r is in the dirty set. So
    both members are dirty. []

That lemma is what licenses compacting only the incidence rows of the merged
blues and their receiving reds instead of the whole live edge array, which is
44% of agglomeration cost by the m1 byte model.

Runtime is a few minutes for base at val scale; it does the same ~700 GB of
logical work the GPU does. Use --max-layer to cut it short while iterating.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
CACHE = ROOT / "data/cache"

U64 = np.uint64
SM_BYTE_SCALE = 255.0

# The counters p0aa_e6s.py read back from the device on this same rag.npz.
FINGERPRINT = "p0aa_e6s.json"


def u64(x):
    return np.uint64(x)


def color_of(i, seed):
    """parhac_d.cu:250. Vectorised over i; seed is scalar."""
    x = np.asarray(i, dtype=U64) * u64(0x9E3779B97F4A7C15)
    x = x ^ (u64(seed) * u64(0xBF58476D1CE4E5B9))
    x = x ^ (x >> u64(30))
    x = x * u64(0x94D049BB133111EB)
    x = x ^ (x >> u64(27))
    return np.where((x & u64(1)).astype(bool), 1, 2).astype(np.uint8)


def prop_pri_bits(seed, a, b, r):
    """parhac_d.cu:278. Position-free: a function of the root pair and the red."""
    a = np.asarray(a, dtype=U64)
    b = np.asarray(b, dtype=U64)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    h = (lo << u64(32)) | hi
    h = h ^ u64(seed)
    h = h ^ (h >> u64(33))
    h = h * u64(0xFF51AFD7ED558CCD)
    h = h ^ (h >> u64(29))
    h = h * u64(0xC4CEB9FE1A85EC53)
    h = h ^ (np.asarray(r, dtype=U64) * u64(0xD1B54A32D192ED03))
    h = h ^ (h >> u64(32))
    return (h >> u64(32)) & u64(0x7FFFFFFF)


def load_rag():
    z = np.load(CACHE / "rag.npz")
    u = np.ascontiguousarray(z["keys"][:, 0], dtype=np.uint32)
    v = np.ascontiguousarray(z["keys"][:, 1], dtype=np.uint32)
    # k_scale_sm_bytes: sm becomes a whole number of affinity bytes, held here
    # as int64 rather than a double, which is the same value exactly.
    sm = np.rint(np.ascontiguousarray(z["stats"][:, 0], dtype=np.float64)
                 * SM_BYTE_SCALE).astype(np.int64)
    ct = np.rint(np.ascontiguousarray(z["stats"][:, 1], dtype=np.float64)).astype(np.int64)
    max_id = int(max(u.max(), v.max()))
    return u, v, sm, ct, max_id


def compress(parent):
    """k_compress. Pointer doubling to the fixpoint is what dfind computes.

    Semantically this is a no-op for every decision the loop makes: it changes
    stored parent values but not root membership, and every read goes through a
    find. It is kept because the base model must mirror the kernel sequence.
    """
    while True:
        nxt = parent[parent]
        if np.array_equal(nxt, parent):
            return parent
        parent = nxt


def find_roots(parent, idx):
    """dfind_nocomp over a subset, by doubling until every entry is a root."""
    a = parent[idx]
    while True:
        nxt = parent[a]
        same = nxt == a
        if same.all():
            return a
        a = nxt


def dedup(u, v, sm, ct):
    """The reduce-by-key half of compact_radix / hash_combine_live.

    Groups by the canonical (u,v) and sums sm and ct. Both are integers, so the
    sum is exact and independent of grouping order, which is why this matches
    whichever of the two device paths ran.
    """
    key = (u.astype(np.uint64) << u64(32)) | v.astype(np.uint64)
    uniq, inv = np.unique(key, return_inverse=True)
    sm2 = np.bincount(inv, weights=sm, minlength=uniq.size).astype(np.int64)
    ct2 = np.bincount(inv, weights=ct, minlength=uniq.size).astype(np.int64)
    u2 = (uniq >> u64(32)).astype(np.uint32)
    v2 = (uniq & u64(0xFFFFFFFF)).astype(np.uint32)
    return u2, v2, sm2, ct2


def rewrite_keep(u, v, parent):
    """k_rewrite: endpoints to roots, canonicalised, self and background out."""
    a = find_roots(parent, u)
    b = find_roots(parent, v)
    keep = (a != b) & (a != 0) & (b != 0)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return lo, hi, keep


def accept_reds(reds, blues, addsz, parent, sz, frozen, eps):
    """k_accept_reds, one group per unique red, in the sorted order.

    Serial over groups, which is what the kernel is too: each red is handled by
    exactly one thread, and a blue is claimed by exactly one red because prop
    holds a single winner per blue, so there is nothing for two threads to race
    over. The take-and-break at the eps size cap is reproduced exactly.
    """
    n = reds.size
    merged = 0
    acc_blues = []
    acc_reds = []
    i = 0
    while i < n:
        r0 = reds[i]
        j = i + 1
        while j < n and reds[j] == r0:
            j += 1
        r = r0
        while parent[r] != r:
            r = parent[r]
        if frozen[r0] or r == 0:
            i = j
            continue
        cap = int(np.rint(float(sz[r]) * eps))
        if eps > 0 and cap == 0:
            cap = 1
        acc = 0
        for k in range(i, j):
            b = blues[k]
            while parent[b] != b:
                b = parent[b]
            if b == r or b == 0:
                continue
            add = int(addsz[k])
            acc += add
            parent[b] = r
            sz[r] += add
            merged += 1
            acc_blues.append(int(b))
            acc_reds.append(int(r))
            if acc > cap:
                break
        i = j
    return merged, np.asarray(acc_blues, dtype=np.uint32), \
        np.asarray(acc_reds, dtype=np.uint32)


def assert_lemma(u, v, sm, ct, mean, alive, parent, TL, amask,
                 layer, outer, inner):
    """The three invariants the restructured compaction has to preserve.

    Checking these directly is what turns "the argument looks right" into
    evidence. Each failure names which invariant broke and where.
    """
    where = f"L{layer} o{outer} i{inner}"
    idx = np.flatnonzero(alive)

    # 1. Every alive edge has root endpoints, canonically ordered. If the dirty
    #    set were incomplete, some edge would still name a merged blue.
    a = find_roots(parent, u[idx])
    b = find_roots(parent, v[idx])
    bad = (a != u[idx]) | (b != v[idx])
    assert not bad.any(), (
        f"{where}: {int(bad.sum())} alive edges have stale endpoints, "
        f"so the dirty set missed them")

    # 2. No two alive edges share a key. This is the collision lemma: if a
    #    non-dirty edge could collide with a dirty one, a duplicate would
    #    survive here and its mean would be too high.
    key = (u[idx].astype(U64) << u64(32)) | v[idx].astype(U64)
    assert np.unique(key).size == key.size, (
        f"{where}: {key.size - np.unique(key).size} duplicate keys survived, "
        f"so the collision lemma is false")

    # 3. The active set is exactly the alive edges at or above TL.
    want = idx[mean[idx] >= TL]
    got = np.flatnonzero(amask & alive)
    assert np.array_equal(got, want), (
        f"{where}: active set drifted from the mean>=TL set by "
        f"{abs(got.size - want.size)} edges")


class Work:
    """Logical work counters, in units the m1 byte model consumes.

    These are the point of the exercise as much as the equality check is: they
    replace the ndirty figure borrowed from p0z_starmarge.json, which was
    measured on the E6t branch with a different iteration count.
    """

    def __init__(self):
        self.propose_edge_visits = 0
        self.pack_node_visits = 0
        self.compact_edge_visits = 0
        self.outer_node_visits = 0
        self.freeze_node_visits = 0
        self.compress_node_visits = 0
        self.ndirty_total = 0
        self.nactive_total = 0
        self.dirty_scan_visits = 0
        self.csr_lookup_visits = 0
        self.csr_rebuild_visits = 0
        self.csr_splice_visits = 0

    def as_dict(self):
        return {k: int(v) for k, v in vars(self).items()}


def _csr_build(u, v, nnode, alive):
    """Incidence lists: each live edge sits in both endpoints' chains."""
    idx = np.flatnonzero(alive)
    nslot = u.size * 2
    head = np.full(nnode, -1, dtype=np.int32)
    nxt = np.full(nslot, -1, dtype=np.int32)
    if idx.size == 0:
        return head, nxt
    nodes = np.concatenate([u[idx], v[idx]]).astype(np.int32)
    slots = np.concatenate([idx * 2, idx * 2 + 1]).astype(np.int32)
    order = np.argsort(nodes, kind="stable")
    nodes, slots = nodes[order], slots[order]
    same = nodes[1:] == nodes[:-1]
    nxt[slots[:-1][same]] = slots[1:][same]
    first = np.ones(nodes.size, dtype=bool)
    first[1:] = ~same
    head[nodes[first]] = slots[first]
    return head, nxt


def _csr_walk(head, nxt, roots, alive):
    """Edge ids on the incidence lists of `roots`, skipping dead slots."""
    seen = []
    nvisit = 0
    for r in np.unique(roots).tolist():
        s = int(head[int(r)])
        while s >= 0:
            nvisit += 1
            e = s // 2
            if alive[e]:
                seen.append(e)
            s = int(nxt[s])
    if not seen:
        return np.empty(0, dtype=np.int64), nvisit
    return np.unique(np.asarray(seen, dtype=np.int64)), nvisit


def _csr_splice(head, nxt, blues, reds):
    """Prepend each blue's chain onto its receiving red. O(nmerge)."""
    nsplice = 0
    for b, r in zip(blues.tolist(), reds.tolist()):
        b, r = int(b), int(r)
        if b == r or head[b] < 0:
            continue
        tail = head[b]
        nsplice += 1
        while nxt[tail] >= 0:
            tail = int(nxt[tail])
            nsplice += 1
        nxt[tail] = head[r]
        head[r] = head[b]
        head[b] = -1
    return nsplice


def run(mode, u0, v0, sm0, ct0, max_id, thr, eps, max_outer, max_layer, trace,
        lemma=False, size_asym=True, stale_active=False, csr=False,
        compact_theta=0.0):
    """One threshold, mirroring the oi/layer/outer/inner nesting exactly."""
    nnode = max_id + 1
    parent = np.arange(nnode, dtype=np.uint32)
    sz = np.ones(nnode, dtype=np.uint32)
    sz[0] = 0
    T = thr * SM_BYTE_SCALE
    fast = mode == "fast"
    W = Work()
    # G4: one root list for the whole run, compacted after each accept.
    # Incremental sz is the value accept_reds already maintains; the
    # rebuilt recount is asserted equal on the live roots every outer.
    roots = np.arange(1, nnode, dtype=np.uint32)
    frozen = np.zeros(nnode, dtype=np.uint8)
    n_sz_ok = 0

    u, v, sm, ct = u0.copy(), v0.copy(), sm0.copy(), ct0.copy()
    nlive = u.size
    alive = None
    ninner = nouter = nmerge = 0
    layer_outers, layer_merges, nlive_trace = [], [], []
    inner_merges = []  # N20_D1: hm per inner, StarMerge EV
    sum_nlive = sum_above = 0

    for layer in range(10000):
        if max_layer and layer >= max_layer:
            break
        parent = compress(parent)
        W.compress_node_visits += nnode

        # In fast mode the edge arrays carry dead slots, so squeeze them out
        # before the layer head slices [:nlive]. A layer boundary is where the
        # device does its one unconditional compact_radix anyway, so paying for
        # a full pass here costs nothing that the base path does not also pay.
        if fast and alive is not None:
            sel = np.flatnonzero(alive)
            u, v, sm, ct = u[sel], v[sel], sm[sel], ct[sel]
            nlive = sel.size

        # k_wmax_live over the live edges.
        a, b = find_roots(parent, u[:nlive]), find_roots(parent, v[:nlive])
        ok = (a != b) & (a != 0) & (b != 0) & (ct[:nlive] >= 1)
        if not ok.any():
            break
        wmax = float(np.max(sm[:nlive][ok] / ct[:nlive][ok]))
        if wmax <= T:
            break
        TL = max(wmax / (1.0 + eps), T)

        # Layer compaction: compact_radix with TL=0.0, so no weight filtering.
        lo, hi, keep = rewrite_keep(u[:nlive], v[:nlive], parent)
        W.compact_edge_visits += nlive
        if not keep.any():
            break
        u, v, sm, ct = dedup(lo[keep], hi[keep], sm[:nlive][keep], ct[:nlive][keep])
        nlive = u.size
        if nlive <= 0:
            break
        mean = sm / ct

        # --- fast-mode state -------------------------------------------------
        # alive[] replaces array compaction; the incidence index is a linked
        # list of row segments per node so that a merge splices the blue's rows
        # onto the red's in O(1) instead of rebuilding.
        if fast:
            alive = np.ones(nlive, dtype=bool)
            csr_head = csr_nxt = None
            if csr:
                csr_head, csr_nxt = _csr_build(u, v, nnode, alive)
                W.csr_rebuild_visits += int(alive.sum()) * 2
            # The active set is held as a mask rather than an index list. Both
            # represent the same set; the mask avoids a sort-based set
            # difference over a list that reaches 3.6 M entries in layer 0,
            # which is a property of this numpy model and not of the device
            # design. nactive_total still counts the list length, which is the
            # figure the cost model wants.
            amask = mean >= TL
            amask_layer0 = amask.copy() if stale_active else None

        layer_m = 0
        layer_o = 0
        done = False

        for outer in range(max_outer):
            cseed = 0xC0FFEE + layer * 10007 + outer
            # dcolor and dfrozen are both memset per outer round, not per
            # layer: a red that froze because it grew past (1+eps) is released
            # when the next colouring is drawn. Holding it frozen for the whole
            # layer caps every cluster at three nodes.
            if fast:
                # G4 CUDA: no full compress/zero/rebuild. Frozen and sz0
                # touch only the maintained root list. Incremental sz is
                # asserted against a recount so a drift cannot hide.
                W.outer_node_visits += roots.size
                frozen[roots] = 0
                rebuilt = np.zeros(nnode, dtype=np.uint32)
                r_all = find_roots(parent, np.arange(1, nnode, dtype=np.uint32))
                np.add.at(rebuilt, r_all, 1)
                if not np.array_equal(sz[roots], rebuilt[roots]):
                    raise AssertionError(
                        f"L{layer} o{outer}: incremental sz drifted from "
                        f"rebuild on {int((sz[roots] != rebuilt[roots]).sum())} "
                        f"roots")
                n_sz_ok += 1
                color = None
                sz0 = sz.copy()
            else:
                frozen = np.zeros(nnode, dtype=np.uint8)
                parent = compress(parent)
                W.compress_node_visits += nnode
                W.outer_node_visits += 6 * nnode
                sz[:] = 0
                r_all = find_roots(parent, np.arange(1, nnode, dtype=np.uint32))
                np.add.at(sz, r_all, 1)
                color = np.zeros(nnode, dtype=np.uint8)
                is_root = parent == np.arange(nnode, dtype=np.uint32)
                is_root[0] = False
                color[is_root] = color_of(np.flatnonzero(is_root), cseed)
                sz0 = sz.copy()

            for inner in range(64):
                seed = 0xA5A5 + inner * 17 + outer
                if fast:
                    idx = np.flatnonzero(amask & alive)
                    W.propose_edge_visits += idx.size
                    W.nactive_total += idx.size
                else:
                    idx = np.arange(nlive)
                    W.propose_edge_visits += nlive

                a = find_roots(parent, u[idx])
                b = find_roots(parent, v[idx])
                elig = (ct[idx] >= 1) & (mean[idx] >= TL) & (a != b) & (a != 0) & (b != 0)
                nabove = int(elig.sum())
                sum_nlive += nlive
                sum_above += nabove
                ninner += 1
                if nabove == 0:
                    done = True

                ea, eb = a[elig], b[elig]
                ca, cb = color_of(ea, cseed), color_of(eb, cseed)
                diff = ca != cb
                ea, eb, ca = ea[diff], eb[diff], ca[diff]
                red = np.where(ca == 1, ea, eb)
                blue = np.where(ca == 1, eb, ea)
                # V2: the sz[r] >= sz[bl] half of k_propose's predicate. It
                # rejects every colour-valid pair in which the red is the
                # smaller side, which is about half of them, and is the
                # suspected reason layers need ~40 outer rounds each.
                sel = frozen[red] == 0
                if size_asym:
                    sel &= sz[red] >= sz[blue]
                red, blue = red[sel], blue[sel]
                ea, eb = ea[sel], eb[sel]

                if red.size:
                    pri = prop_pri_bits(seed, ea, eb, red)
                    pack = (pri << u64(32)) | red.astype(U64)
                    # atomicMax per blue: order-independent, so a stable
                    # lexsort and taking the last of each group is identical.
                    o = np.lexsort((pack, blue))
                    bs, ps = blue[o], pack[o]
                    last = np.append(bs[1:] != bs[:-1], True)
                    pblue, ppack = bs[last], ps[last]
                else:
                    pblue = np.empty(0, dtype=np.uint32)
                    ppack = np.empty(0, dtype=U64)

                if fast:
                    W.pack_node_visits += pblue.size
                else:
                    W.pack_node_visits += nnode

                nprop = pblue.size
                if nprop == 0:
                    break

                # k_pack_prop_fused key/payload, then the ascending sort.
                r_of = (ppack & u64(0xFFFFFFFF)).astype(np.uint32)
                pri_of = (ppack >> u64(32)).astype(U64) & u64(0x7FFFFFFF)
                key = (r_of.astype(U64) << u64(32)) | (u64(0x7FFFFFFF) - pri_of)
                so = np.argsort(key, kind="stable")
                reds, blues = r_of[so], pblue[so]
                addsz = sz[pblue][so]

                hm, mb, mr = accept_reds(reds, blues, addsz, parent, sz,
                                         frozen, eps)
                parent = compress(parent)
                W.compress_node_visits += (mb.size if fast else nnode)
                if fast and hm > 0:
                    roots = roots[parent[roots] == roots]


                # k_freeze / k_freeze_reds.
                if fast:
                    ur = np.unique(reds)
                    ur = ur[ur != 0]
                    W.freeze_node_visits += ur.size
                    rr = find_roots(parent, ur)
                    s0 = np.where(sz0[ur] != 0, sz0[ur], 1).astype(np.float64)
                    frozen[ur[sz[rr] > (1.0 + eps) * s0]] = 1
                else:
                    W.freeze_node_visits += nnode
                    cand = np.flatnonzero(color == 1)
                    rr = find_roots(parent, cand)
                    s0 = np.where(sz0[cand] != 0, sz0[cand], 1).astype(np.float64)
                    frozen[cand[sz[rr] > (1.0 + eps) * s0]] = 1

                nmerge += hm
                layer_m += hm
                inner_merges.append(int(hm))
                if trace > 1:
                    print(f"G0     L{layer} o{outer} i{inner} "
                          f"nlive={nlive} above={nabove} nprop={nprop} "
                          f"hm={hm} szmax={int(sz.max())}", flush=True)
                if hm == 0:
                    break

                # --- compaction ---------------------------------------
                if fast:
                    # The dirty set is the alive edges incident to a merged
                    # blue or to a receiving red. On the device this comes from
                    # a CSR incidence index; here it is a masked scan, which
                    # selects exactly the same set and removes a whole class of
                    # index-maintenance bugs from the equivalence test. The
                    # scan itself is charged separately from the rows it finds,
                    # because the two tiers of g2 differ only in that term.
                    dmask = np.zeros(nnode, dtype=bool)
                    dmask[mb] = True
                    dmask[mr] = True
                    live_idx = np.flatnonzero(alive)
                    if csr:
                        if csr_head is None:
                            csr_head, csr_nxt = _csr_build(u, v, nnode, alive)
                            W.csr_rebuild_visits += int(live_idx.size) * 2
                        dirty_roots = np.concatenate([mb, mr])
                        de, nvis = _csr_walk(csr_head, csr_nxt, dirty_roots,
                                             alive)
                        W.csr_lookup_visits += nvis
                        W.dirty_scan_visits += nvis
                        de_scan = live_idx[dmask[u[live_idx]]
                                           | dmask[v[live_idx]]]
                        if de.size != de_scan.size or not np.array_equal(
                                np.sort(de), np.sort(de_scan)):
                            raise AssertionError(
                                f"L{layer} o{outer} i{inner}: CSR dirty set "
                                f"{de.size} != scan {de_scan.size}")
                        W.csr_splice_visits += _csr_splice(
                            csr_head, csr_nxt, mb, mr)
                    else:
                        W.dirty_scan_visits += live_idx.size
                        de = live_idx[dmask[u[live_idx]] | dmask[v[live_idx]]]
                    W.ndirty_total += de.size
                    W.compact_edge_visits += de.size

                    lo, hi, keep = rewrite_keep(u[de], v[de], parent)
                    u[de], v[de] = lo, hi
                    dead = de[~keep]
                    alive[dead] = False
                    live_de = de[keep]

                    # Collapse duplicates inside the dirty set. The lemma above
                    # is that no other edge can collide with these, and it is
                    # asserted rather than assumed.
                    if live_de.size:
                        k = (u[live_de].astype(U64) << u64(32)) \
                            | v[live_de].astype(U64)
                        o = np.argsort(k, kind="stable")
                        ks, es = k[o], live_de[o]
                        first = np.append(True, ks[1:] != ks[:-1])
                        grp = np.cumsum(first) - 1
                        keeper = es[first]
                        smsum = np.bincount(grp, weights=sm[es],
                                            minlength=keeper.size).astype(np.int64)
                        ctsum = np.bincount(grp, weights=ct[es],
                                            minlength=keeper.size).astype(np.int64)
                        alive[es] = False
                        alive[keeper] = True
                        sm[keeper], ct[keeper] = smsum, ctsum
                        mean[keeper] = smsum / ctsum
                        # Only a dirty edge can cross TL, because only a dirty
                        # edge's mean changes: it either absorbed a duplicate or
                        # it died. So the active set is exact after this delta.
                        #
                        # stale_active models the plan's claim that the list can
                        # be rebuilt "once per layer": it drops dead edges but
                        # never re-tests a survivor's mean. That misses edges
                        # that rise through TL by absorbing a higher-mean
                        # duplicate, and the run diverges.
                        amask[de] = False
                        if not stale_active:
                            amask[keeper] = mean[keeper] >= TL
                        else:
                            amask[keeper] = amask_layer0[keeper]
                    else:
                        amask[de] = False

                    nlive = int(alive.sum())
                    if compact_theta > 0 and alive.size > 0:
                        hole = 1.0 - nlive / alive.size
                        if hole > compact_theta:
                            sel = np.flatnonzero(alive)
                            u, v, sm, ct = u[sel], v[sel], sm[sel], ct[sel]
                            mean = mean[sel]
                            amask = amask[sel]
                            if stale_active and amask_layer0 is not None:
                                amask_layer0 = amask_layer0[sel]
                            alive = np.ones(sel.size, dtype=bool)
                            nlive = sel.size
                            if csr:
                                csr_head, csr_nxt = _csr_build(
                                    u, v, nnode, alive)
                                W.csr_rebuild_visits += nlive * 2
                    if lemma:
                        assert_lemma(u, v, sm, ct, mean, alive, parent, TL,
                                     amask, layer, outer, inner)
                else:
                    lo, hi, keep = rewrite_keep(u[:nlive], v[:nlive], parent)
                    W.compact_edge_visits += nlive
                    if not keep.any():
                        nlive = 0
                        break
                    u, v, sm, ct = dedup(lo[keep], hi[keep],
                                         sm[:nlive][keep], ct[:nlive][keep])
                    nlive = u.size
                    mean = sm / ct

                nlive_trace.append(nlive)
                if nlive <= 0:
                    break

            nouter += 1
            layer_o += 1
            if nlive <= 0 or done:
                break

        layer_outers.append(layer_o)
        layer_merges.append(layer_m)
        if trace:
            print(f"G0   layer {layer:2d} TL={TL:8.3f} outers={layer_o:3d} "
                  f"merges={layer_m:8d} nlive={nlive}", flush=True)
        if layer_m == 0:
            break

    root = find_roots(parent, np.arange(nnode, dtype=np.uint32))
    return {
        "mode": mode,
        "eps": eps,
        "size_asym": bool(size_asym),
        "n_layer": len(layer_outers),
        "nouter": nouter,
        "ninner": ninner,
        "nmerge": nmerge,
        "layer_outers": layer_outers,
        "layer_merges": layer_merges,
        "inner_merges": inner_merges,
        "sum_nlive": sum_nlive,
        "sum_above": sum_above,
        "nlive_final": int(nlive),
        "nseg": int(np.unique(root[1:]).size),
        "root": root,
        "work": W.as_dict(),
        "inc_sz_outers": n_sz_ok,
    }


def compare(base, fast):
    """Every counter the device reports, plus the parent array itself."""
    out = []
    same_root = np.array_equal(base["root"], fast["root"])
    out.append(("parent array bit-identical", same_root))
    for k in ("n_layer", "nouter", "ninner", "nmerge", "sum_nlive",
              "sum_above", "nlive_final", "nseg", "layer_outers",
              "layer_merges"):
        out.append((k, base[k] == fast[k]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=("base", "fast", "both"))
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--eps", type=float, default=0.08)
    ap.add_argument("--max-outer", type=int, default=64)
    ap.add_argument("--max-layer", type=int, default=0,
                    help="stop after N layers; 0 runs to convergence")
    ap.add_argument("-t", "--trace", action="count", default=0,
                    help="-t per layer, -tt per inner iteration")
    ap.add_argument("--lemma", action="store_true",
                    help="check the compaction invariants every iteration")
    ap.add_argument("--no-size-asym", action="store_true",
                    help="V2: drop the sz[red] >= sz[blue] propose predicate")
    ap.add_argument("--stale-active", action="store_true",
                    help="G3 variant: rebuild the active list only at layer "
                         "entry, as the plan proposes")
    ap.add_argument("--csr", action="store_true",
                    help="E2: dirty-combine via incidence lists, not a "
                         "full-array endpoint scan")
    ap.add_argument("--compact-theta", type=float, default=0.0,
                    help="E2b: physically compact when hole fraction exceeds "
                         "this; 0 disables")
    ap.add_argument("--out", default="g0_agg_ref.json")
    ap.add_argument("--sub", type=int, default=0,
                    help="induced subgraph on node ids below N, for fast "
                         "correctness iteration")
    args = ap.parse_args()

    u, v, sm, ct, max_id = load_rag()
    if args.sub:
        # Induced subgraph on the low node ids. The RAG is built in raster
        # order, so this is a spatially contiguous chunk of the volume rather
        # than a random sample, which keeps the degree distribution and the
        # plateau structure realistic. Used to iterate on correctness in
        # seconds; the fingerprint check needs the whole graph.
        keep = (u < args.sub) & (v < args.sub)
        u, v, sm, ct = u[keep], v[keep], sm[keep], ct[keep]
        max_id = int(max(u.max(), v.max()))
    print(f"G0 rag.npz  nedge={u.size}  nnode={max_id + 1}  "
          f"T={args.threshold}  eps={args.eps}", flush=True)

    modes = ["base", "fast"] if args.mode == "both" else [args.mode]
    got = {}
    for m in modes:
        t0 = time.perf_counter()
        got[m] = run(m, u, v, sm, ct, max_id, args.threshold, args.eps,
                     args.max_outer, args.max_layer, args.trace, args.lemma,
                     not args.no_size_asym, args.stale_active,
                     args.csr, args.compact_theta)
        dt = time.perf_counter() - t0
        r = got[m]
        extra = (f" inc_sz_ok={r['inc_sz_outers']}" if m == "fast" else "")
        print(f"G0 {m:4s} {dt:7.1f}s  layers={r['n_layer']} "
              f"outers={r['nouter']} inners={r['ninner']} "
              f"merges={r['nmerge']} nseg={r['nseg']}{extra}", flush=True)

    ok = True
    if (
        "base" in got and not args.max_layer and not args.sub
        and abs(args.eps - 0.08) < 1e-12 and abs(args.threshold - 0.3) < 1e-12
    ):
        ref = json.loads((CACHE / FINGERPRINT).read_text())
        b = got["base"]
        checks = [
            ("n_layer", b["n_layer"], ref["n_layer"]),
            ("nouter", b["nouter"], ref["nouter"]),
            ("ninner", b["ninner"], ref["ninner"]),
            ("nmerge", b["nmerge"], ref["nmerge"]),
            ("sum_nlive", b["sum_nlive"], ref["sum_nlive"]),
            ("sum_above", b["sum_above"], ref["sum_above"]),
            ("layer_outers", b["layer_outers"], ref["layer_outers"]),
            ("layer_merges", b["layer_merges"], ref["layer_merges"]),
        ]
        print(f"\nG0 base vs {FINGERPRINT} (recorded from the device run)")
        for name, mine, theirs in checks:
            good = mine == theirs
            ok &= good
            print(f"G0   {'ok  ' if good else 'DIFF'} {name:14s} "
                  f"model={mine} device={theirs}")

    if len(modes) == 2:
        print("\nG0 base vs fast equivalence")
        for name, good in compare(got["base"], got["fast"]):
            ok &= good
            print(f"G0   {'ok  ' if good else 'DIFF'} {name}")
        print("\nG0 logical work, base -> fast")
        wb, wf = got["base"]["work"], got["fast"]["work"]
        for k in wb:
            r = wb[k] / wf[k] if wf[k] else float("inf")
            print(f"G0   {k:24s} {wb[k]:14d} -> {wf[k]:12d}  {r:8.1f}x")

    report = {m: {k: val for k, val in r.items() if k != "root"}
              for m, r in got.items()}
    report["pass"] = bool(ok)
    # Recorded so m1_cost_model.py can refuse work counters taken from a
    # subgraph, whose visit ratios are not the full graph's.
    report["nedge"] = int(u.size)
    report["nnode"] = int(max_id + 1)
    report["sub"] = int(args.sub)
    report["threshold"] = args.threshold
    report["eps"] = args.eps
    report["max_layer"] = args.max_layer
    dest = CACHE / args.out
    dest.write_text(json.dumps(report, indent=2, default=int) + "\n")
    print(f"\nG0 {'PASS' if ok else 'FAIL'}; wrote {dest.name}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
