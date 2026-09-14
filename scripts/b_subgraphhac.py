#!/usr/bin/env python3
"""Optional CPU: static SubgraphHAC good-merge on rag.npz at T=0.3.

TeraHAC Def. 1 / DynHAC Def. 2: an edge is (1+ε)-good iff
    max(wmax(u), wmax(v)) / mean(uv) <= 1+ε
(equivalent: mean >= max(best[u], best[v]) / (1+ε)).

Union currently-good above-T edges, remap, recompute best, repeat.
Gate: VOI at T=0.3 vs TASK slack, and work vs locked E6s T=0.3 (235 inners).
Close the class if VOI fails or there is no 1.96x work cut. No CUDA.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import CACHE, load_rag  # noqa: E402
from ref_cpu import extract_parent  # noqa: E402
from task_gate import BASE_MERGE, BASE_SPLIT, SLACK  # noqa: E402
from voi_numpy import voi_split_merge  # noqa: E402

OUT = CACHE / "b_subgraphhac.json"
GT = ROOT / "data/ws_bounty/cremiA_val/gt.h5"
T = 0.3
EPS = 0.10  # TeraHAC default
LOCKED_INNERS_T03 = 235  # b_dead_agg E6s T=76.50
LOCKED_NLIVE_T03 = 898175


def compress(p):
    p = p.copy()
    changed = True
    while changed:
        nxt = p[p]
        changed = not np.array_equal(nxt, p)
        p = nxt
    return p


def main():
    print("B SubgraphHAC good-merge CPU. Not a 2 Gvox/s claim.", flush=True)
    u, v, sm, ct, fr, max_id = load_rag()
    n = max_id + 1
    mean = sm / np.maximum(ct, 1)
    parent = np.arange(n, dtype=np.int64)
    work_edges = 0
    nmerge = 0
    rounds = 0
    t0 = time.perf_counter()
    for rounds in range(1, 65):
        parent = compress(parent)
        ru = parent[u.astype(np.int64)]
        rv = parent[v.astype(np.int64)]
        live = ru != rv
        work_edges += int(u.size)
        if not np.any(live):
            break
        w = mean[live]
        a = ru[live]
        b = rv[live]
        best = np.zeros(n, dtype=np.float64)
        np.maximum.at(best, a, w)
        np.maximum.at(best, b, w)
        thr = np.maximum(best[a], best[b]) / (1.0 + EPS)
        good = (w >= T) & (w >= thr)
        ng = int(good.sum())
        if ng == 0:
            break
        ga = a[good]
        gb = b[good]
        # Sequential UF on the good snapshot (order: higher mean first).
        order = np.argsort(-w[good], kind="stable")
        p = parent
        merged = 0
        for i in order:
            x, y = int(ga[i]), int(gb[i])
            while p[x] != x:
                p[x] = p[p[x]]
                x = int(p[x])
            while p[y] != y:
                p[y] = p[p[y]]
                y = int(p[y])
            if x == y:
                continue
            if x > y:
                x, y = y, x
            p[y] = x
            merged += 1
        parent = p
        nmerge += merged
        print(f"B SGHAC r={rounds} live={int(live.sum())} good={ng} "
              f"merged={merged}", flush=True)
        if merged == 0:
            break
    wall = time.perf_counter() - t0
    parent = compress(parent)
    parent32 = parent.astype(np.uint32)
    lab = extract_parent(fr, parent32).astype(np.uint32, copy=False)
    import h5py
    with h5py.File(GT, "r") as f:
        key = "gt" if "gt" in f else ("neuronIds" if "neuronIds" in f else "label")
        gt = f[key][:]
    if gt.shape != lab.shape:
        # CREMI-A val gt is [Z,Y,X] matching fragments.
        gt = np.asarray(gt)
    split, merge = voi_split_merge(lab, gt)
    lim_s, lim_m = BASE_SPLIT[T] + SLACK, BASE_MERGE[T] + SLACK
    voi_ok = bool(split <= lim_s and merge <= lim_m)
    # Locked work proxy: inners * leftover-live. Ours: rounds * nedge.
    locked_work = LOCKED_INNERS_T03 * LOCKED_NLIVE_T03
    our_work = work_edges
    cut = locked_work / max(our_work, 1)
    close = (not voi_ok) or (cut < 1.96)
    doc = {
        "claim": "not a 2 Gvox/s number",
        "algo": "static SubgraphHAC good-merge",
        "T": T,
        "eps": EPS,
        "rounds": rounds,
        "nmerge": nmerge,
        "work_edge_visits": work_edges,
        "wall_s": wall,
        "voi_split": split,
        "voi_merge": merge,
        "limit_split": lim_s,
        "limit_merge": lim_m,
        "voi_pass": voi_ok,
        "locked_work_proxy": locked_work,
        "work_cut_vs_locked": cut,
        "has_1p96x": bool(cut >= 1.96),
        "close_class": close,
        "start_cuda": False,
        "nseg": int((np.unique(lab) != 0).sum()),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(
        f"B SGHAC VOI {split:.4f}/{merge:.4f} "
        f"{'PASS' if voi_ok else 'FAIL'} "
        f"work_cut={cut:.3f}x close={close} -> {OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
