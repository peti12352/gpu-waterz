"""Chunked T=0.3 VOI. Never materializes a 180 Mvox label volume."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
from task_gate import BASE_MERGE, BASE_SPLIT, SLACK  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / "data/cache"
T = 0.3
CHUNK = 1 << 20  # 1M voxels, ~8 MB of uint32 pairs


def grade_t3(split, merge):
    sl, ml = BASE_SPLIT[T] + SLACK, BASE_MERGE[T] + SLACK
    return split <= sl and merge <= ml, sl, ml


def voi_parent_mmap(parent, fr_path=None, gt_path=None, chunk=CHUNK):
    """H(seg|gt), H(gt|seg) from parent[frag] vs gt, streamed."""
    fr_path = Path(fr_path or CACHE / "wz_fragments.npy")
    gt_path = Path(gt_path or CACHE / "gt.npy")
    fr = np.load(fr_path, mmap_mode="r").reshape(-1)
    gt = np.load(gt_path, mmap_mode="r").reshape(-1)
    n = int(fr.size)
    pair = {}
    t_cnt = {}
    s_cnt = {}
    nkeep = 0
    parent = np.asarray(parent, dtype=np.uint32)
    for i in range(0, n, chunk):
        g = np.asarray(gt[i:i + chunk])
        f = np.asarray(fr[i:i + chunk])
        m = g != 0
        if not m.any():
            continue
        g = g[m]
        s = parent[f[m]]
        nkeep += int(g.size)
        keys = (g.astype(np.uint64) << np.uint64(32)) | s.astype(np.uint64)
        uk, ck = np.unique(keys, return_counts=True)
        for k, c in zip(uk.tolist(), ck.tolist()):
            pair[k] = pair.get(k, 0) + int(c)
        ug, cg = np.unique(g, return_counts=True)
        for k, c in zip(ug.tolist(), cg.tolist()):
            t_cnt[int(k)] = t_cnt.get(int(k), 0) + int(c)
        us, cs = np.unique(s, return_counts=True)
        for k, c in zip(us.tolist(), cs.tolist()):
            s_cnt[int(k)] = s_cnt.get(int(k), 0) + int(c)
        del g, f, m, s, keys, uk, ck, ug, cg, us, cs
    if nkeep == 0:
        return 0.0, 0.0, 0
    invn = 1.0 / nkeep

    def h_from_counts(d):
        acc = 0.0
        for c in d.values():
            p = c * invn
            acc -= p * np.log2(p)
        return acc

    h_st = h_from_counts(pair)
    h_t = h_from_counts(t_cnt)
    h_s = h_from_counts(s_cnt)
    nseg = sum(1 for k in s_cnt if k != 0)
    return float(h_st - h_t), float(h_st - h_s), int(nseg)
