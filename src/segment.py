"""TASK.md API: affinity -> final uint32 labels. GPU WS + RAG + G1 S4 heap."""
from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_WS = ROOT / "src/libws_gpu.so"
_RAG = ROOT / "src/librag_gpu.so"
_HEAP = ROOT / "src/libheap_cpu.so"
_RAC = ROOT / "src/librac_agg.so"
_PARHAC_D = ROOT / "src/libparhac_d.so"


def _e6r_locked():
    p = ROOT / "data/cache/e6r_pass.txt"
    if not p.is_file():
        return False
    t = p.read_text()
    return t.startswith("PASS") and "LOCK" in t


def _as_u8(aff) -> np.ndarray:
    if hasattr(aff, "detach"):
        aff = aff.detach().cpu().numpy()
    a = np.ascontiguousarray(aff)
    if a.dtype == np.uint8:
        return a
    a = a.astype(np.float32, copy=False)
    return np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)


def _watershed(aff_u8, low, high):
    z, y, x = aff_u8.shape[1:]
    out = np.zeros((z, y, x), dtype=np.uint32)
    lib = ctypes.CDLL(str(_WS))
    lib.watershed_gpu_e9.restype = ctypes.c_int
    lib.watershed_gpu_e9.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.watershed_gpu_e9(
        aff_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        z, y, x, ctypes.c_float(low), ctypes.c_float(high),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
    )
    return out


# Measured on val: 7505458 edges from 180 Mvox = 0.0417 edges/voxel
# (scripts/c1_memory_budget.py). A fixed 20M cap makes rag_gpu return -1 for
# anything past ~480 Mvox, so the 2.16 Gvox target would fail on edge capacity
# before it failed on memory.
#
# 0.055 keeps 32% headroom over the measured density. Headroom is expensive
# here: rag.cu sizes its table as next_pow2(2 * max_edges) * 16 B, so crossing
# a power-of-two boundary doubles it. At 2.16 Gvox, 0.055 lands under 2^28
# slots for a 4.29 GiB table, where 0.08 tips into 2^29 and 8.00 GiB. Overflow
# is reported as -1 rather than silently truncated, so a too-small cap fails
# loudly.
EDGES_PER_VOX = 0.055
MIN_MAX_EDGES = 20_000_000


def _max_edges(nvox):
    return max(MIN_MAX_EDGES, int(EDGES_PER_VOX * nvox))


def _rag(aff_u8, seg):
    z, y, x = aff_u8.shape[1:]
    max_e = _max_edges(z * y * x)
    u = np.empty(max_e, np.uint32)
    v = np.empty(max_e, np.uint32)
    sm = np.empty(max_e, np.float64)
    ct = np.empty(max_e, np.int64)
    lib = ctypes.CDLL(str(_RAG))
    lib.rag_gpu.restype = ctypes.c_int64
    lib.rag_gpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
    ]
    n = lib.rag_gpu(
        aff_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        seg.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        z, y, x,
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        max_e,
    )
    if n < 0:
        raise RuntimeError("rag overflow")
    return u[:n].copy(), v[:n].copy(), sm[:n].copy(), ct[:n].copy()


def _heap(u, v, sm, ct, thresholds, max_id):
    sys_path = str(ROOT / "src")
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from ref_cpu import heap_from_arrays
    return heap_from_arrays(u, v, sm, ct, list(thresholds), max_id=max_id)


def _parhac(u, v, sm, ct, thresholds, max_id, eps=0.08):
    """Paper ParHAC Alg. 1+2, waterz means. AGG=parhac-paper-ε 0.08 (E12)."""
    u = np.ascontiguousarray(u, dtype=np.uint32)
    v = np.ascontiguousarray(v, dtype=np.uint32)
    sm = np.ascontiguousarray(sm, dtype=np.float64)
    ct = np.ascontiguousarray(ct, dtype=np.int64)
    thrs = np.asarray(list(thresholds), dtype=np.float64)
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(thrs), 3), dtype=np.int64)
    lib = ctypes.CDLL(str(_RAC))
    lib.parhac_paper_cpu.restype = ctypes.c_int
    lib.parhac_paper_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    rc = lib.parhac_paper_cpu(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(thrs)),
        ctypes.c_double(eps),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    if rc != 1:
        raise RuntimeError("parhac_paper_cpu failed")
    return {float(t): parents[i] for i, t in enumerate(thresholds)}


def _parhac_d_dev(u_ptr, v_ptr, sm_ptr, ct_ptr, nedge, thresholds, max_id, eps=0.08):
    """Device paper-ParHAC on already-resident RAG. Used only if E6r LOCK."""
    thrs = np.asarray(list(thresholds), dtype=np.float64)
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(thrs), 3), dtype=np.int64)
    lib = ctypes.CDLL(str(_PARHAC_D))
    lib.parhac_paper_d_dev.restype = ctypes.c_int
    lib.parhac_paper_d_dev.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    rc = lib.parhac_paper_d_dev(
        u_ptr, v_ptr, sm_ptr, ct_ptr,
        ctypes.c_int64(nedge),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(thrs)),
        ctypes.c_double(eps),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    if rc != 1:
        raise RuntimeError("parhac_paper_d_dev failed")
    return {float(t): parents[i] for i, t in enumerate(thresholds)}


def _rac(u, v, sm, ct, thresholds, max_id):
    """Y1 exact RAC + global-min fallback. AGG=rac."""
    u = np.ascontiguousarray(u, dtype=np.uint32)
    v = np.ascontiguousarray(v, dtype=np.uint32)
    sm = np.ascontiguousarray(sm, dtype=np.float64)
    ct = np.ascontiguousarray(ct, dtype=np.int64)
    thrs = np.asarray(list(thresholds), dtype=np.float64)
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(thrs), 3), dtype=np.int64)
    lib = ctypes.CDLL(str(_RAC))
    lib.rac_agg_cpu.restype = ctypes.c_int
    lib.rac_agg_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    rc = lib.rac_agg_cpu(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(thrs)),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    if rc != 1:
        raise RuntimeError("rac_agg_cpu failed")
    return {float(t): parents[i] for i, t in enumerate(thresholds)}


def _extract_gpu(fr, parent):
    """E10/R14: labels[i] = parent[seg[i]] on GPU."""
    fr = np.ascontiguousarray(fr.ravel(), dtype=np.uint32)
    parent = np.ascontiguousarray(parent, dtype=np.uint32)
    out = np.empty_like(fr)
    ms = ctypes.c_float(0)
    lib = ctypes.CDLL(str(_WS))
    lib.extract_gpu.restype = ctypes.c_int
    lib.extract_gpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int64,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_float),
    ]
    rc = lib.extract_gpu(
        fr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        parent.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_int64(fr.size),
        ctypes.c_uint32(int(parent.size) - 1),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.byref(ms),
    )
    if rc != 1:
        raise RuntimeError("extract_gpu failed")
    return out.reshape(-1)


def _ensure_sv7():
    lib = ctypes.CDLL(str(_WS))
    lib.ws_set_sv_rounds.argtypes = [ctypes.c_int]
    lib.ws_set_sv_rounds(7)


def segment(aff, thresholds, aff_low=1e-4, aff_high=0.9999):
    """Return list of uint32 [Z,Y,X] final labels, one per affinity threshold.

    Host API copies aff to GPU for flow/RAG. Event window is device work only
    when using watershed_gpu_e9_d / rag_gpu_d / extract_gpu_d.
    WS = E9c (GPU divideplateaus + UF basins, W14 adaptive SV=7).
    AGG=parhac-paper-ε 0.08 until a replacement locks.
    Extract = E10 extract_gpu.
    """
    aff_u8 = _as_u8(aff)
    if aff_u8.ndim != 4 or aff_u8.shape[0] != 3:
        raise ValueError(f"aff must be [3,Z,Y,X], got {aff_u8.shape}")
    fr = _watershed(aff_u8, aff_low, aff_high)
    u, v, sm, ct = _rag(aff_u8, fr)
    snaps = _parhac(u, v, sm, ct, thresholds, max_id=int(fr.max()))
    return [
        _extract_gpu(fr, snaps[float(t)]).reshape(fr.shape).astype(np.uint32, copy=False)
        for t in thresholds
    ]


def segment_d(aff, thresholds, aff_low=1e-4, aff_high=0.9999):
    """Device-resident WS+RAG+extract. AGG is host paper-ε unless E6r LOCK.

    `aff` already on CUDA (uint8 or float32 [3,Z,Y,X] torch) or numpy (copied
    outside the timed window). Returns list of uint32 [Z,Y,X] on CPU.
    """
    import torch

    _ensure_sv7()
    if hasattr(aff, "is_cuda") and aff.is_cuda:
        aff_t = aff
        if aff_t.dtype != torch.uint8:
            aff_t = (aff_t.float().clamp(0, 1) * 255.0).round().to(torch.uint8)
        aff_t = aff_t.contiguous()
    else:
        aff_u8 = _as_u8(aff)
        aff_t = torch.from_numpy(aff_u8).to("cuda")
    if aff_t.ndim != 4 or aff_t.shape[0] != 3:
        raise ValueError(f"aff must be [3,Z,Y,X], got {tuple(aff_t.shape)}")
    z, y, x = int(aff_t.shape[1]), int(aff_t.shape[2]), int(aff_t.shape[3])
    n = z * y * x
    seg_t = torch.empty((z, y, x), dtype=torch.int32, device="cuda")
    nfrag = ctypes.c_uint32(0)
    ms = ctypes.c_float(0)
    libw = ctypes.CDLL(str(_WS))
    libw.watershed_gpu_e9_d.restype = ctypes.c_int
    libw.watershed_gpu_e9_d.argtypes = [
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_float),
    ]
    libw.watershed_gpu_e9_d(
        aff_t.data_ptr(), z, y, x,
        ctypes.c_float(aff_low), ctypes.c_float(aff_high),
        seg_t.data_ptr(), ctypes.byref(nfrag), ctypes.byref(ms),
    )
    max_e = _max_edges(n)
    u_t = torch.empty(max_e, dtype=torch.int32, device="cuda")
    v_t = torch.empty(max_e, dtype=torch.int32, device="cuda")
    sm_t = torch.empty(max_e, dtype=torch.float64, device="cuda")
    ct_t = torch.empty(max_e, dtype=torch.int64, device="cuda")
    libr = ctypes.CDLL(str(_RAG))
    libr.rag_gpu_d.restype = ctypes.c_int64
    libr.rag_gpu_d.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64,
    ]
    nedge = libr.rag_gpu_d(
        aff_t.data_ptr(), seg_t.data_ptr(), z, y, x,
        u_t.data_ptr(), v_t.data_ptr(), sm_t.data_ptr(), ct_t.data_ptr(), max_e,
    )
    if nedge < 0:
        raise RuntimeError("rag_gpu_d overflow")
    if _e6r_locked() and _PARHAC_D.is_file():
        snaps = _parhac_d_dev(
            u_t.data_ptr(), v_t.data_ptr(), sm_t.data_ptr(), ct_t.data_ptr(),
            int(nedge), thresholds, max_id=int(nfrag.value),
        )
    else:
        u = u_t[:nedge].cpu().numpy().astype(np.uint32, copy=False)
        v = v_t[:nedge].cpu().numpy().astype(np.uint32, copy=False)
        sm = sm_t[:nedge].cpu().numpy()
        ct = ct_t[:nedge].cpu().numpy()
        snaps = _parhac(u, v, sm, ct, thresholds, max_id=int(nfrag.value))
    out = []
    libw.extract_gpu_d.restype = ctypes.c_int
    libw.extract_gpu_d.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
    ]
    seg_flat = seg_t.reshape(-1)
    for t in thresholds:
        par = np.ascontiguousarray(snaps[float(t)], dtype=np.uint32)
        par_t = torch.from_numpy(par).to("cuda")
        lab_t = torch.empty(n, dtype=torch.int32, device="cuda")
        rc = libw.extract_gpu_d(seg_flat.data_ptr(), par_t.data_ptr(), n, lab_t.data_ptr())
        if rc != 1:
            raise RuntimeError("extract_gpu_d failed")
        out.append(lab_t.cpu().numpy().astype(np.uint32, copy=False).reshape(z, y, x))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("affinity_h5")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.2, 0.3, 0.4, 0.5])
    p.add_argument("--dataset", default="affinity")
    args = p.parse_args()
    import h5py
    with h5py.File(args.affinity_h5, "r") as f:
        aff = f[args.dataset][:]
    labs = segment(aff, args.thresholds)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for t, lab in zip(args.thresholds, labs):
        dest = out / f"mine_thr{t}.h5"
        with h5py.File(dest, "w") as h:
            h.create_dataset("labels", data=lab)
        print("wrote", dest, "nseg", int((np.unique(lab) != 0).sum()))


if __name__ == "__main__":
    main()
