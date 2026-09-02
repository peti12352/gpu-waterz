#!/usr/bin/env python3
"""P0x: per-inner live-edge histogram + GPU microbench. Writes val_proj + branch."""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import AFF_THRESHOLDS, CACHE, SO, compile_so, load_rag  # noqa: E402

NVCC = "/usr/local/cuda-12.8/bin/nvcc"
P0X_SO = ROOT / "src/libp0x_inner.so"


def compile_p0x():
    src = ROOT / "csrc/p0x_inner.cu"
    if P0X_SO.exists() and P0X_SO.stat().st_mtime >= src.stat().st_mtime:
        return
    subprocess.check_call([
        NVCC, "-O3", "-arch=sm_120", "--shared", "-Xcompiler", "-fPIC",
        "-o", str(P0X_SO), str(src),
    ])


def interp_ms(n, sizes, times):
    n = max(float(n), 1.0)
    xs = np.log(np.asarray(sizes, dtype=np.float64))
    ys = np.asarray(times, dtype=np.float64)
    ln = np.log(n)
    if ln <= xs[0]:
        return float(ys[0] * n / sizes[0])
    if ln >= xs[-1]:
        return float(ys[-1] * n / sizes[-1])
    return float(np.interp(ln, xs, ys))


def main():
    compile_so()
    compile_p0x()
    u, v, sm, ct, fr, max_id = load_rag()
    thrs = np.asarray(AFF_THRESHOLDS, dtype=np.float64)
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(thrs), 3), dtype=np.int64)
    os.environ["WATERZ_P0"] = "1"
    lib = ctypes.CDLL(str(SO))
    lib.parhac_paper_cpu.restype = ctypes.c_int
    lib.parhac_paper_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    print("P0x running paper-ε 0.08 with WATERZ_P0=1", flush=True)
    rc = lib.parhac_paper_cpu(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(thrs)),
        ctypes.c_double(0.08),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    lib.parhac_p0_count.restype = ctypes.c_int
    n_in = int(lib.parhac_p0_count())
    layers = np.zeros(max(n_in, 1), dtype=np.int32)
    lib.parhac_p0_copy_layers.restype = ctypes.c_int
    lib.parhac_p0_copy_layers.argtypes = [
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
    ]
    n_in = int(lib.parhac_p0_copy_layers(
        layers.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(len(layers)),
    ))
    layers = layers[:n_in]
    print(
        f"P0x rc={rc} ninner={n_in} layer_mean={float(layers.mean()) if n_in else 0:.1f} "
        f"layer_max={int(layers.max()) if n_in else 0} "
        f"iners_per_T={list(map(int, stats[:, 2]))}",
        flush=True,
    )

    sizes = np.array([40_000, 400_000, 2_400_000], dtype=np.int32)
    launch = np.zeros(3, dtype=np.float32)
    fused = np.zeros(3, dtype=np.float32)
    gl = ctypes.CDLL(str(P0X_SO))
    gl.p0x_microbench.restype = ctypes.c_int
    gl.p0x_microbench.argtypes = [
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ]
    brc = gl.p0x_microbench(
        sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(3), ctypes.c_int(200),
        launch.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        fused.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    print(f"P0x bench rc={brc} launch_ms={list(map(float, launch))} fused_ms={list(map(float, fused))}", flush=True)

    val_launch = sum(interp_ms(int(k), sizes, launch) for k in layers)
    val_fused = sum(interp_ms(int(k), sizes, fused) for k in layers)
    # one-shot T=0.3 ≈ first three T inners (0.5+0.4+0.3) if sorted high-first
    n03 = int(stats[2, 2] + stats[1, 2] + stats[0, 2]) if len(stats) >= 3 else n_in
    # stats rows follow AFF_THRESHOLDS order 0.2,0.3,0.4,0.5; paper runs high-T first
    # use all inners for val_proj (four T); also report T=0.3-only share
    t03_frac = (int(stats[1, 2]) + int(stats[2, 2]) + int(stats[3, 2])) / max(n_in, 1)
    val_fused_t03 = val_fused * t03_frac
    track_a = val_fused <= 50.0
    info = {
        "ninner": n_in,
        "layer_mean": float(layers.mean()) if n_in else 0,
        "layer_max": int(layers.max()) if n_in else 0,
        "sizes": [int(x) for x in sizes],
        "launch_ms_each": [float(x) for x in launch],
        "fused_ms_each": [float(x) for x in fused],
        "val_proj_launch_ms": val_launch,
        "val_proj_fused_ms": val_fused,
        "val_proj_fused_t03_ms": val_fused_t03,
        "track": "A" if track_a else "B",
        "budget_ms": 50.0,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "p0_gpu_inner.json").write_text(json.dumps(info, indent=2))
    print(
        f"P0x val_proj launch={val_launch:.3f}ms fused={val_fused:.3f}ms "
        f"t03={val_fused_t03:.3f}ms track={info['track']}",
        flush=True,
    )
    return 0 if brc == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
