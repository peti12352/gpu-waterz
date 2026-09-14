"""CPU oracle for S1-S4. No GPU. Affinity is float32 [3,Z,Y,X] in [0, 1]."""
from __future__ import annotations

import heapq
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

# dir order matches basic_watershed.hpp: -z -y -x +z +y +x
_DZ = np.array([-1, 0, 0, 1, 0, 0], dtype=np.int64)
_DY = np.array([0, -1, 0, 0, 1, 0], dtype=np.int64)
_DX = np.array([0, 0, -1, 0, 0, 1], dtype=np.int64)
_DIRBIT = np.array([0x01, 0x02, 0x04, 0x08, 0x10, 0x20], dtype=np.uint32)
_REVBIT = np.array([0x08, 0x10, 0x20, 0x01, 0x02, 0x04], dtype=np.uint32)


def _as_aff(aff: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(aff)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    else:
        a = a.astype(np.float32, copy=False)
    if a.ndim != 4 or a.shape[0] != 3:
        raise ValueError(f"aff must be [3,Z,Y,X], got {a.shape}")
    return a


def _neighbor_affs(aff: np.ndarray, low: float) -> np.ndarray:
    """Return [6,Z,Y,X] affinities: -z -y -x +z +y +x. OOB = low."""
    _, z, y, x = aff.shape
    out = np.empty((6, z, y, x), dtype=np.float32)
    out[0] = low
    out[0, 1:] = aff[0, 1:]
    out[1] = low
    out[1, :, 1:] = aff[1, :, 1:]
    out[2] = low
    out[2, :, :, 1:] = aff[2, :, :, 1:]
    out[3] = low
    out[3, :-1] = aff[0, 1:]
    out[4] = low
    out[4, :, :-1] = aff[1, :, 1:]
    out[5] = low
    out[5, :, :, :-1] = aff[2, :, :, 1:]
    return out


_WS_SO = Path(__file__).resolve().parent / "libws_cpu.so"


def _watershed_c(aff: np.ndarray, low: float, high: float) -> np.ndarray | None:
    if not _WS_SO.is_file():
        return None
    import ctypes

    aff = np.ascontiguousarray(aff, dtype=np.float32)
    _, z, y, x = aff.shape
    out = np.zeros((z, y, x), dtype=np.uint32)
    lib = ctypes.CDLL(str(_WS_SO))
    lib.watershed_cpu.restype = ctypes.c_uint32
    lib.watershed_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.watershed_cpu(
        aff.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        z,
        y,
        x,
        ctypes.c_float(low),
        ctypes.c_float(high),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
    )
    return out


def watershed(aff: np.ndarray, low: float = 1e-4, high: float = 0.9999) -> np.ndarray:
    """S1. Deterministic plateau: last exit dir in (-z,-y,-x,+z,+y,+x) wins."""
    aff = _as_aff(aff)
    c = _watershed_c(aff, low, high)
    if c is not None:
        return c
    na = _neighbor_affs(aff, low)
    z, y, x = aff.shape[1:]
    size = z * y * x
    m = na.max(axis=0)
    bits = np.zeros((z, y, x), dtype=np.uint32)
    live = m > low
    for d in range(6):
        bits |= np.where(live & ((na[d] == m) | (na[d] >= high)), _DIRBIT[d], 0).astype(
            np.uint32
        )

    raw = bits.reshape(-1).copy()
    stride_z, stride_y, stride_x = y * x, x, 1
    step = np.array(
        [-stride_z, -stride_y, -stride_x, stride_z, stride_y, stride_x], dtype=np.int64
    )

    bfs: deque[int] = deque()
    for idx in range(size):
        b = raw[idx]
        if not b:
            continue
        for d in range(6):
            if b & _DIRBIT[d]:
                him = idx + int(step[d])
                if not (raw[him] & _REVBIT[d]):
                    raw[idx] |= 0x40
                    bfs.append(idx)
                    break

    while bfs:
        idx = bfs.popleft()
        to_set = 0
        b = raw[idx]
        for d in range(6):
            if b & _DIRBIT[d]:
                him = idx + int(step[d])
                if raw[him] & _REVBIT[d]:
                    if not (raw[him] & 0x40):
                        bfs.append(him)
                        raw[him] |= 0x40
                else:
                    to_set = int(_DIRBIT[d])
        raw[idx] = to_set

    HIGH = np.uint32(0x80000000)
    MASK = np.uint32(0x7FFFFFFF)
    next_id = 1
    for idx in range(size):
        if raw[idx] == 0:
            raw[idx] = HIGH
            continue
        if (raw[idx] & HIGH) or raw[idx] == 0:
            continue
        q = [idx]
        raw[idx] |= 0x40
        qi = 0
        inherited = 0
        while qi < len(q):
            me = q[qi]
            qi += 1
            b = raw[me]
            for d in range(6):
                if b & _DIRBIT[d]:
                    him = me + int(step[d])
                    if raw[him] & HIGH:
                        inherited = int(raw[him])
                        for it in q:
                            raw[it] = inherited
                        q.clear()
                        break
                    if not (raw[him] & 0x40):
                        raw[him] |= 0x40
                        q.append(him)
            if not q:
                break
        if q:
            lab = HIGH | np.uint32(next_id)
            for it in q:
                raw[it] = lab
            next_id += 1

    return (raw & MASK).astype(np.uint32).reshape(z, y, x)


_RAG_SO = Path(__file__).resolve().parent / "librag_cpu.so"


def _region_graph_c(aff: np.ndarray, seg: np.ndarray) -> dict[tuple[int, int], list] | None:
    if not _RAG_SO.is_file():
        return None
    import ctypes

    aff = np.ascontiguousarray(aff, dtype=np.float32)
    seg = np.ascontiguousarray(seg, dtype=np.uint32)
    _, z, y, x = aff.shape
    max_e = 20_000_000
    u = np.empty(max_e, dtype=np.uint32)
    v = np.empty(max_e, dtype=np.uint32)
    sm = np.empty(max_e, dtype=np.float64)
    ct = np.empty(max_e, dtype=np.int64)
    lib = ctypes.CDLL(str(_RAG_SO))
    lib.rag_cpu.restype = ctypes.c_int64
    lib.rag_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
    ]
    n = lib.rag_cpu(
        aff.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        seg.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        z,
        y,
        x,
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        max_e,
    )
    if n < 0:
        return None
    acc: dict[tuple[int, int], list] = {}
    for i in range(int(n)):
        c = int(ct[i])
        s = float(sm[i])
        acc[(int(u[i]), int(v[i]))] = [s, c, s / c if c else 0.0]
    return acc


def rag_arrays(aff: np.ndarray, seg: np.ndarray):
    """Return u, v, sum, count from C++ RAG (waterz float incremental)."""
    import ctypes

    aff = np.ascontiguousarray(_as_aff(aff), dtype=np.float32)
    seg = np.ascontiguousarray(seg, dtype=np.uint32)
    _, z, y, x = aff.shape
    max_e = 20_000_000
    u = np.empty(max_e, dtype=np.uint32)
    v = np.empty(max_e, dtype=np.uint32)
    sm = np.empty(max_e, dtype=np.float64)
    ct = np.empty(max_e, dtype=np.int64)
    lib = ctypes.CDLL(str(_RAG_SO))
    lib.rag_cpu.restype = ctypes.c_int64
    lib.rag_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
    ]
    n = lib.rag_cpu(
        aff.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        seg.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        z,
        y,
        x,
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        max_e,
    )
    if n < 0:
        raise RuntimeError("rag_cpu overflow")
    return u[:n].copy(), v[:n].copy(), sm[:n].copy(), ct[:n].copy()


def region_graph(aff: np.ndarray, seg: np.ndarray) -> dict[tuple[int, int], list]:
    """S2+S3. Keys (min_id, max_id), both > 0. Values [sum, count, mean]."""
    aff = _as_aff(aff)
    c = _region_graph_c(aff, seg)
    if c is not None:
        return c
    seg = np.ascontiguousarray(seg)
    keys_all = []
    vals_all = []
    for d, sl_a, sl_b in (
        (0, (slice(1, None), slice(None), slice(None)), (slice(0, -1), slice(None), slice(None))),
        (1, (slice(None), slice(1, None), slice(None)), (slice(None), slice(0, -1), slice(None))),
        (2, (slice(None), slice(None), slice(1, None)), (slice(None), slice(None), slice(0, -1))),
    ):
        id1 = seg[sl_a].ravel()
        id2 = seg[sl_b].ravel()
        av = aff[d][sl_a].ravel()
        mask = id1 != id2
        if not mask.any():
            continue
        a = np.minimum(id1[mask], id2[mask]).astype(np.uint64, copy=False)
        b = np.maximum(id1[mask], id2[mask]).astype(np.uint64, copy=False)
        v = av[mask].astype(np.float64, copy=False)
        keep = a > 0
        a, b, v = a[keep], b[keep], v[keep]
        keys_all.append((a << 32) | b)
        vals_all.append(v)
    if not keys_all:
        return {}
    keys = np.concatenate(keys_all)
    vals = np.concatenate(vals_all)
    if keys.size == 0:
        return {}
    order = np.argsort(keys, kind="stable")
    keys, vals = keys[order], vals[order]
    cuts = np.flatnonzero(np.diff(keys)) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [keys.size]))
    acc: dict[tuple[int, int], list] = {}
    for s, e in zip(starts, ends):
        k = int(keys[s])
        u, v = k >> 32, k & 0xFFFFFFFF
        sm = float(vals[s:e].sum())
        n = int(e - s)
        acc[(u, v)] = [sm, n, sm / n]
    return acc


class _UF:
    def __init__(self):
        self.p: dict[int, int] = {}

    def find(self, x: int) -> int:
        while self.p.get(x, x) != x:
            self.p[x] = self.p.get(self.p[x], self.p[x])
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> int:
        a, b = self.find(a), self.find(b)
        if a == b:
            return a
        self.p[b] = a
        return a


_HEAP_SO = Path(__file__).resolve().parent / "libheap_cpu.so"
_BORUVKA_SO = Path(__file__).resolve().parent / "libboruvka_cpu.so"


def _heap_agglomerate_c(
    edges: dict[tuple[int, int], list],
    aff_thresholds: list[float],
) -> dict[float, _UF] | None:
    if not _HEAP_SO.is_file() or not edges:
        return None
    import ctypes

    n = len(edges)
    u = np.empty(n, dtype=np.uint32)
    v = np.empty(n, dtype=np.uint32)
    sm = np.empty(n, dtype=np.float64)
    ct = np.empty(n, dtype=np.int64)
    max_id = 0
    for i, ((a, b), (s, c, _m)) in enumerate(edges.items()):
        u[i], v[i] = a, b
        sm[i], ct[i] = s, c
        if a > max_id:
            max_id = a
        if b > max_id:
            max_id = b
    thrs = np.asarray(aff_thresholds, dtype=np.float64)
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    lib = ctypes.CDLL(str(_HEAP_SO))
    lib.heap_agglomerate_cpu.restype = ctypes.c_int
    lib.heap_agglomerate_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
    ]
    rc = lib.heap_agglomerate_cpu(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        n,
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(thrs),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
    )
    if rc != 1:
        return None
    out: dict[float, _UF] = {}
    for i, t in enumerate(aff_thresholds):
        uf = _UF()
        p = parents[i]
        uf.p = {int(j): int(p[j]) for j in range(1, max_id + 1) if p[j] != j}
        out[float(t)] = uf
    return out


def _heap_agglomerate(
    edges: dict[tuple[int, int], list],
    aff_thresholds: list[float],
) -> dict[float, _UF]:
    """S4. waterz mergeUntil: merge while score < (1-aff_thr). Ascending score."""
    c = _heap_agglomerate_c(edges, aff_thresholds)
    if c is not None:
        return c
    # elist: u, v, sum, count, mean, deleted, stale, score
    elist = []
    adj: dict[int, set[int]] = defaultdict(set)
    pair: dict[tuple[int, int], int] = {}
    for (u, v), (s, n, mean) in edges.items():
        eid = len(elist)
        sc = 1.0 - mean
        elist.append([u, v, float(s), int(n), float(mean), False, False, sc])
        adj[u].add(eid)
        adj[v].add(eid)
        pair[(u, v)] = eid

    heap: list[tuple[float, int]] = [(e[7], i) for i, e in enumerate(elist)]
    heapq.heapify(heap)
    uf = _UF()

    def score_of(e):
        return 1.0 - e[4]

    def find_pair(a: int, b: int) -> int | None:
        a, b = uf.find(a), uf.find(b)
        if a > b:
            a, b = b, a
        return pair.get((a, b))

    def set_pair(a: int, b: int, eid: int):
        a, b = uf.find(a), uf.find(b)
        if a > b:
            a, b = b, a
        pair[(a, b)] = eid

    def del_pair(a: int, b: int):
        a, b = uf.find(a), uf.find(b)
        if a > b:
            a, b = b, a
        pair.pop((a, b), None)

    def merge_one(eid: int):
        e = elist[eid]
        a = uf.find(e[0])
        b = uf.find(e[1])
        if a == b:
            e[5] = True
            return
        del_pair(a, b)
        uf.union(a, b)  # b -> a, new root is a
        e[5] = True
        for neid in list(adj[b]):
            if neid == eid or elist[neid][5]:
                continue
            ne = elist[neid]
            nb = uf.find(ne[0]) if uf.find(ne[0]) != a else uf.find(ne[1])
            # nb is the opposite of b's endpoint = neighbour
            if nb == a:
                ne[5] = True
                continue
            existing = find_pair(a, nb)
            if existing is None or existing == neid:
                ne[0], ne[1] = a, nb
                adj[a].add(neid)
                set_pair(a, nb, neid)
                ne[6] = True
                ne[7] = score_of(ne)
            else:
                oe = elist[existing]
                if oe[7] <= ne[7]:
                    keep, drop = oe, ne
                    keep_id, drop_id = existing, neid
                else:
                    keep, drop = ne, oe
                    keep_id, drop_id = neid, existing
                keep[2] += drop[2]
                keep[3] += drop[3]
                keep[4] = keep[2] / keep[3]
                keep[7] = score_of(keep)
                keep[0], keep[1] = a, nb
                keep[6] = True
                drop[5] = True
                del_pair(a, nb)
                set_pair(a, nb, keep_id)
                adj[a].add(keep_id)
                heapq.heappush(heap, (keep[7], keep_id))
        adj[a].update(adj[b])
        adj.pop(b, None)

    snapshots: dict[float, _UF] = {}
    for aff_thr in sorted(aff_thresholds, reverse=True):
        # high aff first = low score first (waterz ascending score if we go
        # 0.5,0.4,0.3,0.2). Caller may pass any set; we do each independently
        # from the current mergedUntil by continuing the heap.
        thr = 1.0 - aff_thr
        while heap:
            sc, eid = heapq.heappop(heap)
            e = elist[eid]
            if e[5]:
                continue
            nsc = score_of(e)
            e[7] = nsc
            if e[6] or nsc > sc + 1e-15:
                e[6] = False
                heapq.heappush(heap, (nsc, eid))
                continue
            if nsc >= thr:
                heapq.heappush(heap, (nsc, eid))
                break
            merge_one(eid)
        snap = _UF()
        snap.p = dict(uf.p)
        snapshots[aff_thr] = snap
    return snapshots


def extract(seg: np.ndarray, uf: _UF) -> np.ndarray:
    flat = np.ascontiguousarray(seg).reshape(-1)
    uniq, inv = np.unique(flat, return_inverse=True)
    roots = np.array([0 if int(u) == 0 else uf.find(int(u)) for u in uniq], dtype=np.uint32)
    return roots[inv].reshape(seg.shape)


def extract_parent(seg: np.ndarray, parent: np.ndarray) -> np.ndarray:
    """parent[i] = root of fragment i (already compressed). parent[0] unused."""
    flat = np.ascontiguousarray(seg).reshape(-1)
    out = parent[flat]
    out[flat == 0] = 0
    return out.reshape(seg.shape)


def heap_from_arrays(
    u: np.ndarray,
    v: np.ndarray,
    sm: np.ndarray,
    ct: np.ndarray,
    aff_thresholds: list[float],
    max_id: int | None = None,
) -> dict[float, np.ndarray]:
    """C++ heap. Returns {thr: parent_array} with parent[id]=root."""
    import ctypes

    u = np.ascontiguousarray(u, dtype=np.uint32)
    v = np.ascontiguousarray(v, dtype=np.uint32)
    sm = np.ascontiguousarray(sm, dtype=np.float64)
    ct = np.ascontiguousarray(ct, dtype=np.int64)
    edge_max = int(max(int(u.max()), int(v.max())))
    max_id = edge_max if max_id is None else max(int(max_id), edge_max)
    thrs = np.asarray(aff_thresholds, dtype=np.float64)
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    lib = ctypes.CDLL(str(_HEAP_SO))
    lib.heap_agglomerate_cpu.restype = ctypes.c_int
    lib.heap_agglomerate_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
    ]
    rc = lib.heap_agglomerate_cpu(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        len(u),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(thrs),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
    )
    if rc != 1:
        raise RuntimeError("heap_agglomerate_cpu failed")
    return {float(t): parents[i] for i, t in enumerate(aff_thresholds)}


def agglomerate_exact(
    aff: np.ndarray,
    aff_thresholds: list[float],
    low: float = 1e-4,
    high: float = 0.9999,
    fragments: np.ndarray | None = None,
) -> dict[float, np.ndarray]:
    if fragments is None:
        fragments = watershed(aff, low, high)
    edges = region_graph(aff, fragments)
    snaps = _heap_agglomerate(edges, aff_thresholds)
    return {t: extract(fragments, uf) for t, uf in snaps.items()}


def boruvka_from_arrays(
    u: np.ndarray,
    v: np.ndarray,
    sm: np.ndarray,
    ct: np.ndarray,
    aff_thresholds: list[float],
    max_id: int | None = None,
) -> dict[float, np.ndarray]:
    import ctypes

    u = np.ascontiguousarray(u, dtype=np.uint32)
    v = np.ascontiguousarray(v, dtype=np.uint32)
    sm = np.ascontiguousarray(sm, dtype=np.float64)
    ct = np.ascontiguousarray(ct, dtype=np.int64)
    edge_max = int(max(int(u.max()), int(v.max())))
    max_id = edge_max if max_id is None else max(int(max_id), edge_max)
    thrs = np.asarray(aff_thresholds, dtype=np.float64)
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    lib = ctypes.CDLL(str(_BORUVKA_SO))
    lib.boruvka_cpu.restype = ctypes.c_int
    lib.boruvka_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
    ]
    rc = lib.boruvka_cpu(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        len(u),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(thrs),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
    )
    if rc != 1:
        raise RuntimeError("boruvka_cpu failed")
    return {float(t): parents[i] for i, t in enumerate(aff_thresholds)}


def agglomerate_boruvka(
    aff: np.ndarray,
    aff_thresholds: list[float],
    low: float = 1e-4,
    high: float = 0.9999,
    fragments: np.ndarray | None = None,
) -> dict[float, np.ndarray]:
    """Live-mean Boruvka. Independent of heap order."""
    if fragments is None:
        fragments = watershed(aff, low, high)
    if _BORUVKA_SO.is_file() and _RAG_SO.is_file():
        u, v, sm, ct = rag_arrays(aff, fragments)
        snaps = boruvka_from_arrays(u, v, sm, ct, aff_thresholds, max_id=int(fragments.max()))
        return {t: extract_parent(fragments, p) for t, p in snaps.items()}
    edges = region_graph(aff, fragments)
    out = {}
    for thr in aff_thresholds:
        uf = _UF()
        # working copy of edges keyed by current roots
        cur: dict[tuple[int, int], list] = {
            (u, v): [s, n, mean] for (u, v), (s, n, mean) in edges.items()
        }
        changed = True
        while changed:
            changed = False
            best: dict[int, tuple[float, int, int, tuple[int, int]]] = {}
            for (u0, v0), (s, n, mean) in cur.items():
                u, v = uf.find(u0), uf.find(v0)
                if u == v or mean <= thr:
                    continue
                sc = 1.0 - mean
                key_uv = (min(u, v), max(u, v))
                cand = (sc, key_uv[0], key_uv[1], key_uv)
                for node in (u, v):
                    prev = best.get(node)
                    if prev is None or cand < prev:
                        best[node] = cand
            pairs = []
            seen = set()
            for node, (sc, a, b, key) in best.items():
                if key in seen:
                    continue
                oa, ob = uf.find(a), uf.find(b)
                if oa == ob:
                    continue
                # mutual or lower absorbs
                ba = best.get(oa)
                bb = best.get(ob)
                if ba is None or bb is None:
                    continue
                if ba[3] != key or bb[3] != key:
                    continue
                # consistent orientation: lower root absorbs
                lo, hi = (oa, ob) if oa < ob else (ob, oa)
                pairs.append((lo, hi, key))
                seen.add(key)
            if not pairs:
                break
            changed = True
            for lo, hi, _ in pairs:
                uf.union(lo, hi)
            nxt: dict[tuple[int, int], list] = {}
            for (u0, v0), (s, n, mean) in cur.items():
                u, v = uf.find(u0), uf.find(v0)
                if u == v:
                    continue
                key = (min(u, v), max(u, v))
                if key in nxt:
                    nxt[key][0] += s
                    nxt[key][1] += n
                    nxt[key][2] = nxt[key][0] / nxt[key][1]
                else:
                    nxt[key] = [s, n, s / n]
            cur = nxt
        out[thr] = extract(fragments, uf)
    return out
