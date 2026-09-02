#!/usr/bin/env python3
"""E1: GPU UF watershed + GPU RAG + locked Y2 ParHAC ε=0.01.

TASK.md accuracy only: shipped run_baseline.py must print ACCURACY GATE: PASS.
nfrag vs 2175400 is logged, not a fail.
"""
from __future__ import annotations

import ctypes
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ref_cpu import extract_parent  # noqa: E402
from task_gate import (  # noqa: E402
    AFF_HIGH,
    AFF_LOW,
    AFF_THRESHOLDS,
    BG_VAL_MEASURED,
    FRAGMENTS_VAL,
    print_contract,
)

AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
OUT = ROOT / "data/ws_bounty"
WS_SO = ROOT / "src/libws_gpu.so"
RAG_SO = ROOT / "src/librag_gpu.so"
RAC_SO = ROOT / "src/librac_agg.so"
NVCC = "/usr/local/cuda-12.8/bin/nvcc"
PY = ROOT / ".venv-cuda/bin/python"
if not PY.is_file():
    PY = ROOT / ".venv/bin/python"


def compile_all():
    subprocess.check_call([
        NVCC, "-O3", "-arch=sm_120", "--shared", "-Xcompiler", "-fPIC",
        "-I/usr/local/cuda-12.8/targets/x86_64-linux/include",
        "-o", str(WS_SO), str(ROOT / "csrc/ws.cu"),
    ])
    subprocess.check_call([
        NVCC, "-O3", "-arch=sm_120", "--shared", "-Xcompiler", "-fPIC",
        "-I/usr/local/cuda-12.8/targets/x86_64-linux/include",
        "-o", str(RAG_SO), str(ROOT / "csrc/rag.cu"),
    ])
    subprocess.check_call([
        "g++", "-O3", "-DNDEBUG", "-shared", "-fPIC",
        "-o", str(RAC_SO), str(ROOT / "src/rac_agg.cpp"),
    ])


def gpu_ws(aff_u8):
    z, y, x = aff_u8.shape[1:]
    aff_t = torch.from_numpy(np.ascontiguousarray(aff_u8)).to("cuda")
    out = torch.zeros((z, y, x), dtype=torch.int32, device="cuda")
    lib = ctypes.CDLL(str(WS_SO))
    lib.watershed_gpu_d.restype = ctypes.c_int
    lib.watershed_gpu_d.argtypes = [
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float, ctypes.c_void_p,
    ]
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    n = lib.watershed_gpu_d(
        aff_t.data_ptr(), z, y, x, AFF_LOW, AFF_HIGH, out.data_ptr())
    t1.record()
    torch.cuda.synchronize()
    ms = t0.elapsed_time(t1)
    fr = out.cpu().numpy().astype(np.uint32, copy=False)
    return fr, int(n), float(ms)


def rag_gpu(aff_u8, seg):
    z, y, x = aff_u8.shape[1:]
    max_e = 40_000_000
    u = np.empty(max_e, dtype=np.uint32)
    v = np.empty(max_e, dtype=np.uint32)
    sm = np.empty(max_e, dtype=np.float64)
    ct = np.empty(max_e, dtype=np.int64)
    lib = ctypes.CDLL(str(RAG_SO))
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
    a = np.ascontiguousarray(aff_u8)
    s = np.ascontiguousarray(seg, dtype=np.uint32)
    n = lib.rag_gpu(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
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


def parhac(u, v, sm, ct, max_id, eps=0.01):
    thrs = np.asarray(AFF_THRESHOLDS, dtype=np.float64)
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(thrs), 3), dtype=np.int64)
    lib = ctypes.CDLL(str(RAC_SO))
    lib.parhac_agg_cpu.restype = ctypes.c_int
    lib.parhac_agg_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    rc = lib.parhac_agg_cpu(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(thrs)),
        ctypes.c_double(eps),
        ctypes.c_int(0),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    if rc != 1:
        raise RuntimeError("parhac_agg_cpu failed")
    return parents, stats


def main():
    print_contract()
    compile_all()
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    print(f"aff {aff.shape} {aff.dtype}", flush=True)

    t0 = time.time()
    fr, nlib, ws_ms = gpu_ws(aff)
    nfrag = int((np.unique(fr) != 0).sum())
    bg = int((fr == 0).sum())
    rel = abs(nfrag - FRAGMENTS_VAL) / FRAGMENTS_VAL
    print(
        f"E1 WS nfrag={nfrag} lib={nlib} bg={bg} "
        f"rel_vs_TASK={rel:.6f} bg_vs_measured={bg == BG_VAL_MEASURED} "
        f"device_ms={ws_ms:.2f} wall={time.time()-t0:.2f}",
        flush=True,
    )

    t1 = time.time()
    u, v, sm, ct = rag_gpu(aff, fr)
    print(f"E1 RAG edges={len(u)} wall={time.time()-t1:.2f}", flush=True)

    max_id = int(max(int(u.max()) if len(u) else 0, int(v.max()) if len(v) else 0, int(fr.max())))
    t2 = time.time()
    parents, stats = parhac(u, v, sm, ct, max_id, eps=0.01)
    print(f"E1 AGG wall={time.time()-t2:.2f}", flush=True)

    dest_dir = OUT / "e1_gpuws"
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, thr in enumerate(AFF_THRESHOLDS):
        lab = extract_parent(fr, parents[i]).astype(np.uint32, copy=False)
        dest = dest_dir / f"mine_thr{thr}.h5"
        with h5py.File(dest, "w") as h:
            h.create_dataset("labels", data=lab)
        nseg = int((np.unique(lab) != 0).sum())
        print(
            f"  T={thr} nseg={nseg} rounds={int(stats[i,0])} merges={int(stats[i,1])}",
            flush=True,
        )
        paths.append(str(dest))

    cmd = [
        str(PY),
        str(OUT / "baseline/run_baseline.py"),
        "--candidate",
        *paths,
    ]
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(OUT), capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.stderr:
        sys.stderr.write(r.stderr)
    ok = "ACCURACY GATE: PASS" in r.stdout
    print("E1", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
