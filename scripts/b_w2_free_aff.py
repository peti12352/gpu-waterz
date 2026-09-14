#!/usr/bin/env python3
"""B: W2 first lever: free the host-path aff copy after k_flow.

Measures val WS peak + nfrag/fingerprint vs wz_fragments.npy.
Does not touch watershed_gpu_e9_d (caller aff lives for RAG).
Rebuilds libws_gpu.so. Idle-5090 only. Not a 2 Gvox/s claim.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from task_gate import FRAGMENTS_VAL  # noqa: E402
from p1_make_big_indep import (  # noqa: E402
    AFF, CACHE, GpuWs, card_busy, fingerprint, gpu_state, nfrag_bg,
)

OUT = CACHE / "b_w2_free_aff.json"
WS = ROOT / "src/libws_gpu.so"
NVCC = "/usr/local/cuda-12.8/bin/nvcc"
GIB = 1024.0 ** 3
P1_PEAK = 3796962448  # P1 val, before this lever


def build():
    subprocess.check_call([
        NVCC, "-O3", "-arch=sm_120", "--shared", "-Xcompiler", "-fPIC",
        "-I/usr/local/cuda-12.8/targets/x86_64-linux/include",
        "-o", str(WS), str(ROOT / "csrc/ws.cu"),
    ])


def main():
    print("B W2 free-aff-after-k_flow. Not a 2 Gvox/s claim.", flush=True)
    busy = card_busy()
    if busy:
        print(f"B REFUSE card busy: {busy}", flush=True)
        raise SystemExit(2)
    print(f"GPU idle: {gpu_state()}", flush=True)
    build()
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    ws = GpuWs()
    seg, meta = ws.run(aff)
    nfrag, bg = nfrag_bg(seg)
    fp, _, _ = fingerprint(seg)
    oracle = None
    op = CACHE / "wz_fragments.npy"
    if op.exists():
        ref = np.ascontiguousarray(np.load(op), dtype=np.uint32)
        oracle = bool(ref.shape == seg.shape and np.array_equal(ref, seg))
    nvox = int(seg.size)
    peak = int(meta["peak_bytes"])
    saved = P1_PEAK - peak
    pred_216 = (peak / nvox) * 2.16e9 / GIB
    ok = bool(nfrag == FRAGMENTS_VAL and oracle is not False and meta["leaked_bytes"] == 0)
    doc = {
        "claim": "not a 2 Gvox/s number",
        "lever": "cudaFree(aff) after k_flow in e9c_watershed",
        "nfrag": nfrag,
        "bg": bg,
        "fingerprint": fp,
        "oracle_array_equal": oracle,
        "peak_bytes": peak,
        "peak_gib": peak / GIB,
        "p1_peak_bytes": P1_PEAK,
        "saved_bytes": saved,
        "saved_gib": saved / GIB,
        "pred_2p16_ws_only_gib": pred_216,
        "leaked": meta["leaked_bytes"],
        "ws": meta,
        "gpu": gpu_state(),
        "pass": ok,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(
        f"B nfrag={nfrag} oracle={oracle} peak={peak / GIB:.3f} GiB "
        f"(was {P1_PEAK / GIB:.3f}) saved={saved / GIB:.3f} GiB "
        f"pred_2.16_scratch*nvox={pred_216:.2f} GiB "
        f"{'PASS' if ok else 'FAIL'} -> {OUT}",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
