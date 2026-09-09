"""VOI matching funkey/waterz evaluate.hpp: skip gt==0, pred 0 is a label.

voi_split = H(seg|gt) = H(s,t) - H(t)
voi_merge = H(gt|seg) = H(s,t) - H(s)
in bits (log2).
"""
from __future__ import annotations

import numpy as np


def voi_split_merge(seg: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    mask = np.asarray(gt).ravel() != 0
    g = np.asarray(gt).ravel()[mask]
    s = np.asarray(seg).ravel()[mask]
    n = float(g.size)
    if n == 0:
        return 0.0, 0.0
    _, ginv = np.unique(g, return_inverse=True)
    _, sinv = np.unique(s, return_inverse=True)
    ns = int(sinv.max()) + 1
    pair = ginv.astype(np.int64) * ns + sinv.astype(np.int64)
    pcnt = np.unique(pair, return_counts=True)[1].astype(np.float64)
    t = np.bincount(ginv).astype(np.float64)
    si = np.bincount(sinv).astype(np.float64)
    p = pcnt / n
    tn = t[t > 0] / n
    sn = si[si > 0] / n
    h_st = float(-(p * np.log2(p)).sum())
    h_t = float(-(tn * np.log2(tn)).sum())
    h_s = float(-(sn * np.log2(sn)).sum())
    return h_st - h_t, h_st - h_s
