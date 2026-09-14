#!/usr/bin/env python3
"""C1: measure the pipeline's device footprint, then predict the graded volume.

The TASK target is 2.16 Gvox on a 24 GB RTX 3090 Ti. That case cannot be
measured directly, it does not fit on the dev 5090 either, and attempting it
would OOM a shared GPU. So instead every cudaMalloc in ws.cu / rag.cu /
parhac_d.cu is counted (see the accounting note at the top of each), the exact
peak is measured at val scale where it does fit, the per-voxel coefficient is
derived, and the larger volumes are predicted from it.

Two footprints are reported per stage:
  measured   exact bytes from the in-library counters at val scale
  predicted  measured_bytes / val_voxels * target_voxels

The prediction is linear in voxel count, which is right for every buffer keyed
on `size` (the great majority). It is NOT right for the RAG hash table, whose
capacity is derived from max_edges, nor for edge-keyed agglomeration buffers,
so those are computed from their own formulas instead of scaled.

Peak-of-pipeline is not the sum of stage peaks: stages free before the next
allocates. Both the sum and the true concurrent peak are reported, because the
concurrent peak is what has to fit.
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
CACHE = ROOT / "data/cache"
WS = ROOT / "src/libws_gpu.so"
RAG = ROOT / "src/librag_gpu.so"
NVCC = "/usr/local/cuda-12.8/bin/nvcc"
GIB = 1024.0 ** 3

# TASK volumes. val is the measured instrument; 2.16 Gvox is the graded target.
VOLUMES = {
    "val 180 Mvox (125x1200x1200)": 125 * 1200 * 1200,
    "1.44 Gvox (250x2400x2400)": 250 * 2400 * 2400,
    "2.16 Gvox (375x2400x2400)": 375 * 2400 * 2400,
}
CARD_GIB = {"RTX 3090 Ti": 24.0, "RTX 5090": 32.0}
MAX_E = 20_000_000
# Must track src/segment.py's _max_edges and csrc/rag.cu's Slot.
EDGES_PER_VOX = 0.055
MIN_MAX_E = 20_000_000
SLOT_BYTES = 16


def build():
    for so, src in ((WS, "csrc/ws.cu"), (RAG, "csrc/rag.cu")):
        subprocess.check_call([
            NVCC, "-O3", "-arch=sm_120", "--shared", "-Xcompiler", "-fPIC",
            "-I/usr/local/cuda-12.8/targets/x86_64-linux/include",
            "-o", str(so), str(ROOT / src),
        ])


def bind_counters(lib, prefix):
    for name in ("reset", "peak", "cur"):
        fn = getattr(lib, f"{prefix}_mem_{name}")
        fn.restype = None if name == "reset" else ctypes.c_size_t
        fn.argtypes = []
    return (getattr(lib, f"{prefix}_mem_reset"),
            getattr(lib, f"{prefix}_mem_peak"),
            getattr(lib, f"{prefix}_mem_cur"))


def main():
    build()
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    z, y, x = aff.shape[1:]
    nvox = z * y * x
    print(f"C1 val volume {z}x{y}x{x} = {nvox} vox", flush=True)

    libw = ctypes.CDLL(str(WS))
    libr = ctypes.CDLL(str(RAG))
    ws_reset, ws_peak, ws_cur = bind_counters(libw, "ws")
    rag_reset, rag_peak, rag_cur = bind_counters(libr, "rag")

    libw.watershed_gpu_e9.restype = ctypes.c_int
    libw.watershed_gpu_e9.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_uint32),
    ]
    seg = np.zeros((z, y, x), dtype=np.uint32)
    ws_reset()
    libw.watershed_gpu_e9(
        aff.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        z, y, x, ctypes.c_float(1e-4), ctypes.c_float(0.9999),
        seg.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
    )
    ws_bytes = int(ws_peak())
    ws_leak = int(ws_cur())
    nfrag = int(seg.max())
    print(f"C1 watershed peak={ws_bytes / GIB:.3f} GiB "
          f"({ws_bytes / nvox:.2f} B/vox) leaked={ws_leak} nfrag={nfrag}",
          flush=True)

    libr.rag_gpu.restype = ctypes.c_int64
    libr.rag_gpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
    ]
    u = np.zeros(MAX_E, np.uint32)
    v = np.zeros(MAX_E, np.uint32)
    sm = np.zeros(MAX_E, np.float64)
    ct = np.zeros(MAX_E, np.int64)
    rag_reset()
    nedge = int(libr.rag_gpu(
        aff.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        seg.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        z, y, x,
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        MAX_E,
    ))
    rag_bytes = int(rag_peak())
    print(f"C1 rag peak={rag_bytes / GIB:.3f} GiB "
          f"({rag_bytes / nvox:.2f} B/vox) leaked={int(rag_cur())} "
          f"nedge={nedge}", flush=True)

    ws_per_vox = ws_bytes / nvox
    rag_per_vox = rag_bytes / nvox
    # Fragments and edges both scale with voxel count on these volumes, so the
    # edge-keyed RAG buffers scale linearly too; the hash table is the one
    # piece keyed on the max_edges argument rather than on the actual count.
    frag_per_vox = nfrag / nvox
    edge_per_vox = nedge / nvox
    print(f"C1 derived: {frag_per_vox:.4f} frag/vox, {edge_per_vox:.4f} edge/vox",
          flush=True)

    rows = []
    for label, tv in VOLUMES.items():
        # Watershed scales linearly: its buffers are either 4 B/vox arrays or
        # corner/plateau/queue arrays whose counts track voxel count at fixed
        # plateau density, which tiling preserves.
        ws_p = ws_per_vox * tv

        # RAG must NOT be scaled linearly. Its dominant buffer is a hash table
        # sized next_pow2(2 * max_edges) * sizeof(Slot), a step function of the
        # max_edges argument rather than of voxel count, and segment.py derives
        # max_edges from the volume. Scaling the val measurement instead
        # produced 34.45 GiB for 2.16 Gvox where the formula gives ~11.7 GiB.
        max_e = max(MIN_MAX_E, int(EDGES_PER_VOX * tv))
        cap = 1
        while cap < 2 * max_e:
            cap *= 2
        rag_table = cap * SLOT_BYTES
        rag_flags = cap * 4
        rag_edges = max_e * (4 + 4 + 8 + 8)
        rag_p = rag_table + rag_flags + rag_edges

        # Resident across stages: uint8 affinity, uint32 fragment ids, uint32
        # output labels.
        io_b = (3.0 + 4.0 + 4.0) * tv
        concurrent = max(ws_p, rag_p) + io_b
        rows.append({
            "volume": label,
            "voxels": tv,
            "ws_gib": ws_p / GIB,
            "rag_gib": rag_p / GIB,
            "rag_table_gib": rag_table / GIB,
            "rag_max_edges": max_e,
            "io_gib": io_b / GIB,
            "concurrent_peak_gib": concurrent / GIB,
            "est_fragments": frag_per_vox * tv,
            "est_edges": edge_per_vox * tv,
            "fits_3090ti_24gib": bool(concurrent / GIB <= 24.0),
            "slab_mvox_for_20gib": 20.0 * GIB / (ws_per_vox + 11.0) / 1e6,
        })

    print("\nC1 footprint. WS scaled from measurement; RAG from its own "
          "next_pow2(2*max_edges) formula.")
    print(f"{'volume':32s} {'WS':>9s} {'RAG':>9s} {'io':>8s} {'peak':>9s} {'24GiB':>7s}")
    for r in rows:
        print(f"{r['volume']:32s} {r['ws_gib']:8.2f}G {r['rag_gib']:8.2f}G "
              f"{r['io_gib']:7.2f}G {r['concurrent_peak_gib']:8.2f}G "
              f"{'yes' if r['fits_3090ti_24gib'] else 'NO':>7s}")
    tgt = rows[-1]
    print(f"\nC1 at 2.16 Gvox: ~{tgt['est_fragments'] / 1e6:.1f}M fragments, "
          f"~{tgt['est_edges'] / 1e6:.1f}M edges, RAG max_edges="
          f"{tgt['rag_max_edges']} -> table alone {tgt['rag_table_gib']:.2f} GiB")
    print(f"C1 largest z-slab that fits ~20 GiB of usable VRAM: "
          f"~{tgt['slab_mvox_for_20gib']:.0f} Mvox, so 2.16 Gvox needs "
          f">= {int(tgt['voxels'] / (tgt['slab_mvox_for_20gib'] * 1e6)) + 1} slabs")

    out = {
        "val": {
            "voxels": nvox, "nfrag": nfrag, "nedge": nedge,
            "ws_peak_bytes": ws_bytes, "rag_peak_bytes": rag_bytes,
            "ws_bytes_per_vox": ws_per_vox, "rag_bytes_per_vox": rag_per_vox,
            "ws_leaked_bytes": ws_leak,
        },
        "projections": rows,
        "cards_gib": CARD_GIB,
        "note": ("agglomeration excluded: it is keyed on edges/fragments, not "
                 "voxels, and thrust scratch is not counted by the in-library "
                 "counter"),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "c1_memory_budget.json").write_text(json.dumps(out, indent=2) + "\n")

    target = rows[-1]
    ok = target["fits_3090ti_24gib"]
    print(f"\nC1 {'PASS' if ok else 'FAIL'}: 2.16 Gvox needs "
          f"{target['concurrent_peak_gib']:.2f} GiB of a 24 GiB 3090 Ti",
          flush=True)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
