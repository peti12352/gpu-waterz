#!/usr/bin/env python3
"""P0s leftover after SDSL; P0t relative-contact; P0u tiles. Writes branch json."""
from __future__ import annotations

import ctypes
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import CACHE, SO, compile_so, frag_sizes, load_rag  # noqa: E402

E4 = {0.2: 294518, 0.3: 322167, 0.4: 345376, 0.5: 379876}
TILES = ((8, 32, 32), (16, 64, 64), (32, 128, 128))


def fragment_centroids(fr, max_id):
    zdim, ydim, xdim = fr.shape
    flat = fr.ravel()
    n = max_id + 1
    cnt = np.bincount(flat, minlength=n).astype(np.float64)
    idx = np.arange(flat.size, dtype=np.float64)
    plane = ydim * xdim
    zz = np.floor(idx / plane)
    yy = np.floor((idx / xdim) % ydim)
    xx = idx % xdim
    sz = np.bincount(flat, weights=zz, minlength=n)
    sy = np.bincount(flat, weights=yy, minlength=n)
    sx = np.bincount(flat, weights=xx, minlength=n)
    den = np.maximum(cnt, 1.0)
    return (sz / den).astype(np.float32), (sy / den).astype(np.float32), (sx / den).astype(np.float32)


def tile_ids(cz, cy, cx, dz, dy, dx, shape):
    _, ydim, xdim = shape
    ny = int(np.ceil(ydim / dy)) + 1
    nx = int(np.ceil(xdim / dx)) + 1
    tz = np.floor(cz / dz).astype(np.int32)
    ty = np.floor(cy / dy).astype(np.int32)
    tx = np.floor(cx / dx).astype(np.int32)
    np.maximum(tz, 0, out=tz)
    np.maximum(ty, 0, out=ty)
    np.maximum(tx, 0, out=tx)
    return (tz * ny + ty) * nx + tx


def parse(text: str) -> dict:
    p0s = []
    for m in re.finditer(
        r"P0s S0=(\d+) T=([0-9.]+) n_residual=(\d+) n_large_large=(\d+) n_small_touch=(\d+)",
        text,
    ):
        p0s.append({
            "S0": int(m.group(1)),
            "T": float(m.group(2)),
            "n_residual": int(m.group(3)),
            "n_large_large": int(m.group(4)),
            "n_small_touch": int(m.group(5)),
        })
    p0t = []
    for m in re.finditer(
        r"P0t gamma=([0-9.]+) alpha=([0-9.]+) T=([0-9.]+) giant=([0-9.]+) ncc=(\d+) nmerge=(\d+) max_vox=(\d+)",
        text,
    ):
        p0t.append({
            "gamma": float(m.group(1)),
            "alpha": float(m.group(2)),
            "T": float(m.group(3)),
            "giant": float(m.group(4)),
            "ncc": int(m.group(5)),
            "nmerge": int(m.group(6)),
            "max_vox": int(m.group(7)),
        })
    p0t_b16 = []
    for m in re.finditer(
        r"P0t_b16 gamma=([0-9.]+) alpha=([0-9.]+) T=([0-9.]+) giant=([0-9.]+) ncc=(\d+) nmerge=(\d+) rounds=(\d+)",
        text,
    ):
        p0t_b16.append({
            "gamma": float(m.group(1)),
            "alpha": float(m.group(2)),
            "T": float(m.group(3)),
            "giant": float(m.group(4)),
            "ncc": int(m.group(5)),
            "nmerge": int(m.group(6)),
            "rounds": int(m.group(7)),
        })
    p0u = []
    for m in re.finditer(
        r"P0u tile=(\S+) T=([0-9.]+) n_intra=(\d+) n_inter=(\d+) intra_frac=([0-9.]+) frozen=(\d+) n_residual_inter=(\d+)",
        text,
    ):
        dz, dy, dx = (int(x) for x in m.group(1).split("x"))
        p0u.append({
            "tile": [dz, dy, dx],
            "T": float(m.group(2)),
            "n_intra": int(m.group(3)),
            "n_inter": int(m.group(4)),
            "intra_frac": float(m.group(5)),
            "frozen": int(m.group(6)),
            "n_residual_inter": int(m.group(7)),
        })
    return decide_branch(p0s, p0t, p0t_b16, p0u)


def _near_e4(ncc, t, frac=0.20):
    e4 = E4[round(t, 1)]
    return (1.0 - frac) * e4 <= ncc <= (1.0 + frac) * e4


def _p0t_candidate(rows):
    by = {}
    for r in rows:
        by.setdefault((r["gamma"], r["alpha"]), {})[round(r["T"], 1)] = r
    hits = []
    for key, d in by.items():
        def ok_t(t):
            if t not in d:
                return False
            return d[t]["giant"] <= 0.05 and _near_e4(d[t]["ncc"], t)

        all4 = all(ok_t(t) for t in (0.2, 0.3, 0.4, 0.5))
        mid = ok_t(0.3) and ok_t(0.4)
        if all4 or mid:
            hits.append({
                "gamma": key[0],
                "alpha": key[1],
                "all4": all4,
                "rows": d,
            })
    hits.sort(key=lambda h: (not h["all4"], h["rows"].get(0.2, {}).get("giant", 1.0)))
    return hits


def decide_branch(p0s, p0t, p0t_b16, p0u):
    l33_s0 = None
    l33_s0s = []
    by_s0 = {}
    for r in p0s:
        by_s0.setdefault(r["S0"], []).append(r)
    ok_s0 = []
    for s0, rows in by_s0.items():
        if any(r["n_residual"] < 50_000 for r in rows):
            mx = max(r["n_residual"] for r in rows)
            t02 = next((r["n_residual"] for r in rows if abs(r["T"] - 0.2) < 1e-9), mx)
            # lock-eligible first (all T residual <=5k), largest leftover first
            lock = 0 if mx <= 5000 else 1
            ok_s0.append((lock, -t02, s0, mx))
    if ok_s0:
        ok_s0.sort()
        l33_s0 = ok_s0[0][2]
        l33_s0s = [t[2] for t in ok_s0]

    r32_hits = _p0t_candidate(p0t)
    r32 = r32_hits[0] if r32_hits else None

    b34_tile = None
    by_tile = {}
    for r in p0u:
        by_tile.setdefault(tuple(r["tile"]), []).append(r)
    for tile, rows in by_tile.items():
        r02 = next((r for r in rows if abs(r["T"] - 0.2) < 1e-9), rows[0])
        if r02["frozen"] < 50_000 or r02["intra_frac"] >= 0.7:
            b34_tile = list(tile)
            break

    if l33_s0 is not None:
        branch = ["L33"]
    elif r32 is not None:
        branch = ["R32"]
    elif b34_tile is not None:
        branch = ["B34"]
    else:
        branch = ["V35"]

    return {
        "p0s": p0s,
        "p0t": p0t,
        "p0t_b16": p0t_b16,
        "p0u": p0u,
        "branch": branch,
        "l33_s0": l33_s0,
        "l33_s0s": l33_s0s,
        "r32_gamma": None if r32 is None else r32["gamma"],
        "r32_alpha": None if r32 is None else r32["alpha"],
        "b34_tile": b34_tile,
        "l33_kill": l33_s0 is None,
        "p0t_candidates": len(r32_hits),
    }


def main():
    compile_so()
    u, v, sm, ct, fr, max_id = load_rag()
    sz = frag_sizes(fr, max_id)
    total = int(sz[1:].sum())
    print(f"P0 leftover edges={len(u)} max_id={max_id} total_vox={total}", flush=True)
    cz, cy, cx = fragment_centroids(fr, max_id)
    tids = [
        np.ascontiguousarray(tile_ids(cz, cy, cx, *t, fr.shape), dtype=np.int32)
        for t in TILES
    ]
    lib = ctypes.CDLL(str(SO))
    lib.p0_leftover_block_cpu.restype = ctypes.c_int
    lib.p0_leftover_block_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int64, ctypes.c_uint32, ctypes.c_int64,
    ]
    rc = lib.p0_leftover_block_cpu(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        sz.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        tids[0].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        tids[1].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        tids[2].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int64(len(u)),
        ctypes.c_uint32(max_id),
        ctypes.c_int64(total),
    )
    print(f"P0 leftover rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(0 if main() == 1 else 1)
