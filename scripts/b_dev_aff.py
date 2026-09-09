#!/usr/bin/env python3
"""B: device-path aff park after k_flow on watershed_gpu_e9_d / segment_d.

Splits flow / label so e9b peak excludes caller aff (E5: only k_flow reads it).
Gate: nfrag 2175400, wz_fragments.npy byte-identical, fused WS peak drop.
Idle-5090 only. Not a 2 Gvox/s claim. No 2.16 Gvox allocation.
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from task_gate import AFF_HIGH, AFF_LOW, FRAGMENTS_VAL  # noqa: E402
from p1_make_big_indep import (  # noqa: E402
    AFF, CACHE, card_busy, fingerprint, gpu_state, nfrag_bg,
)
import segment as S  # noqa: E402

OUT = CACHE / "b_dev_aff.json"
WS = ROOT / "src/libws_gpu.so"
NVCC = "/usr/local/cuda-12.8/bin/nvcc"
GIB = 1024.0 ** 3
HOST_AFTER_FREE = 3256881152  # b_w2_free_aff.json peak_bytes


def nvcc_arch_flags():
    return [
        "-gencode", "arch=compute_86,code=sm_86",
        "-gencode", "arch=compute_120,code=sm_120",
    ]


def build(extra=None, out=None):
    dest = Path(out or WS)
    if os.environ.get("WATERZ_SKIP_BUILD") == "1" and dest.is_file():
        return
    cmd = [
        NVCC, "-O3", *nvcc_arch_flags(), "--shared", "-Xcompiler", "-fPIC",
        "-I/usr/local/cuda-12.8/targets/x86_64-linux/include",
        "-o", str(dest), str(ROOT / "csrc/ws.cu"),
    ]
    if extra:
        cmd[2:2] = list(extra)
    subprocess.check_call(cmd)


def bind(lib):
    lib.ws_flow_d.restype = ctypes.c_int
    lib.ws_flow_d.argtypes = [
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float, ctypes.c_void_p,
    ]
    lib.ws_label_d.restype = ctypes.c_int
    lib.ws_label_d.argtypes = [
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.watershed_gpu_e9_d.restype = ctypes.c_int
    lib.watershed_gpu_e9_d.argtypes = [
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_float),
    ]
    lib.ws_mem_reset.restype = None
    lib.ws_mem_peak.restype = ctypes.c_size_t
    lib.ws_mem_cur.restype = ctypes.c_size_t
    lib.ws_mem_credit.argtypes = [ctypes.c_size_t]
    lib.ws_mem_debit.argtypes = [ctypes.c_size_t]


def run_split(lib, aff_u8, park):
    z, y, x = (int(v) for v in aff_u8.shape[1:])
    n = z * y * x
    aff_bytes = 3 * n
    lib.ws_mem_reset()
    aff_d = S.DevBuf.from_host(aff_u8)
    bits_d = S.DevBuf((z, y, x), np.uint8)
    seg_d = S.DevBuf((z, y, x), np.uint32)
    lib.ws_mem_credit(ctypes.c_size_t(aff_bytes))
    lib.ws_flow_d(
        ctypes.c_void_p(aff_d.ptr), z, y, x,
        ctypes.c_float(AFF_LOW), ctypes.c_float(AFF_HIGH),
        ctypes.c_void_p(bits_d.ptr),
    )
    if park:
        lib.ws_mem_debit(ctypes.c_size_t(aff_bytes))
        aff_d.free()
        aff_d = None
    nfrag = ctypes.c_uint32(0)
    ms = ctypes.c_float(0)
    rc = lib.ws_label_d(
        ctypes.c_void_p(bits_d.ptr), z, y, x,
        ctypes.c_void_p(seg_d.ptr), ctypes.byref(nfrag), ctypes.byref(ms),
    )
    peak = int(lib.ws_mem_peak())
    leaked = int(lib.ws_mem_cur())
    seg = seg_d.to_host()
    bits_d.free()
    seg_d.free()
    if aff_d is not None:
        lib.ws_mem_debit(ctypes.c_size_t(aff_bytes))
        aff_d.free()
    return seg, {
        "nfrag_ret": int(rc),
        "nfrag_out": int(nfrag.value),
        "peak_bytes": peak,
        "leaked_bytes": leaked,
        "park": park,
        "aff_bytes": aff_bytes,
        "divide_ms": float(ms.value),
    }


def main():
    print("B device-path aff park. Not a 2 Gvox/s claim.", flush=True)
    busy = card_busy()
    if busy:
        print(f"B REFUSE card busy: {busy}", flush=True)
        raise SystemExit(2)
    print(f"GPU idle: {gpu_state()}", flush=True)
    build()
    lib = ctypes.CDLL(str(WS))
    bind(lib)
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    held, held_meta = run_split(lib, aff, park=False)
    parked, park_meta = run_split(lib, aff, park=True)
    nfrag, bg = nfrag_bg(parked)
    fp, _, _ = fingerprint(parked)
    oracle = None
    op = CACHE / "wz_fragments.npy"
    if op.exists():
        ref = np.ascontiguousarray(np.load(op), dtype=np.uint32)
        oracle = bool(ref.shape == parked.shape and np.array_equal(ref, parked))
    identity = bool(np.array_equal(held, parked))
    nvox = int(parked.size)
    saved = held_meta["peak_bytes"] - park_meta["peak_bytes"]
    pred_216 = (park_meta["peak_bytes"] / nvox) * 2.16e9 / GIB
    pred_216_fused = (
        (park_meta["peak_bytes"] + 4 * nvox) / nvox * 2.16e9 / GIB
    )
    # fused = tracked (e9b + credited aff if held) ; park drops aff.
    ok = bool(
        nfrag == FRAGMENTS_VAL
        and oracle is not False
        and identity
        and park_meta["leaked_bytes"] == 0
        and saved > 0
    )
    doc = {
        "claim": "not a 2 Gvox/s number",
        "lever": "split ws_flow_d / ws_label_d; park aff after k_flow",
        "nfrag": nfrag,
        "bg": bg,
        "fingerprint": fp,
        "oracle_array_equal": oracle,
        "held_vs_park_identical": identity,
        "held": held_meta,
        "parked": park_meta,
        "saved_bytes": saved,
        "saved_gib": saved / GIB,
        "host_after_free_aff_bytes": HOST_AFTER_FREE,
        "pred_2p16_tracked_gib": pred_216,
        "pred_2p16_tracked_plus_seg_gib": pred_216_fused,
        "gpu": gpu_state(),
        "pass": ok,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(
        f"B nfrag={nfrag} oracle={oracle} ident={identity} "
        f"held={held_meta['peak_bytes'] / GIB:.3f} GiB "
        f"park={park_meta['peak_bytes'] / GIB:.3f} GiB "
        f"saved={saved / GIB:.3f} GiB "
        f"{'PASS' if ok else 'FAIL'} -> {OUT}",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
