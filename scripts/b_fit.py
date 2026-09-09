#!/usr/bin/env python3
"""B: measured 24 GB fit table from val, predicted at 2.16 Gvox.

Does not allocate 2.16 Gvox. Uses parked WS, thinned edges, dead-agg cut.
Idle-5090. Not a 2 Gvox/s claim.
"""
from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from task_gate import AFF_HIGH, AFF_LOW  # noqa: E402
from p1_make_big_indep import AFF, CACHE, card_busy, gpu_state  # noqa: E402
import segment as S  # noqa: E402

OUT = CACHE / "b_fit.json"
GIB = 1024.0 ** 3
USABLE = 23.0
NVOX_216 = 2_160_000_000
NEDGE_216_OFFICIAL = 90_323_139


def next_pow2(x):
    p = 1
    while p < x:
        p <<= 1
    return p


def main():
    print("B fit table. Not a 2 Gvox/s claim. No 2.16 allocation.", flush=True)
    busy = card_busy()
    if busy:
        print(f"B REFUSE card busy: {busy}", flush=True)
        raise SystemExit(2)
    print(f"GPU idle: {gpu_state()}", flush=True)
    libw = ctypes.CDLL(str(S._WS))
    libr = ctypes.CDLL(str(S._RAG))
    libp = ctypes.CDLL(str(S._PARHAC_D))
    for lib, name in ((libw, "ws"), (libr, "rag"), (libp, "agg")):
        getattr(lib, f"{name}_mem_reset").restype = None
        getattr(lib, f"{name}_mem_peak").restype = ctypes.c_size_t
    libw.ws_mem_reset()
    libr.rag_mem_reset()
    libp.agg_mem_reset()
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    nvox = int(aff[0].size)
    aff_d = S.DevBuf.from_host(aff)
    out = S.segment_d(aff_d, [0.3], return_device=True)
    nedge = int(S.LAST_NEDGE)
    parked = bool(S.LAST_AFF_PARKED)
    for b in out:
        b.free()
    aff_d.free()
    ws_p = int(libw.ws_mem_peak())
    rag_p = int(libr.rag_mem_peak())
    agg_p = int(libp.agg_mem_peak())
    max_e = S._max_edges(nvox)
    table = next_pow2(2 * max_e) * 20  # Slot 16 B + uint32 flag
    wide = max_e * 24
    thin = nedge * 24
    seg_b = 4 * nvox
    aff_b = 3 * nvox
    bits_b = nvox
    # Fused stage maxima at this volume. Aff parked out of WS.
    peak_ws = ws_p + seg_b + bits_b
    peak_rag = aff_b + seg_b + rag_p + wide
    peak_agg = seg_b + thin + agg_p
    scale_v = NVOX_216 / nvox
    scale_e = NEDGE_216_OFFICIAL / nedge
    pred = {
        "nvox": NVOX_216,
        "nedge_official": NEDGE_216_OFFICIAL,
        "ws_tracked_gib": ws_p * scale_v / GIB,
        "rag_tracked_gib": rag_p * scale_e / GIB,
        "agg_tracked_gib": agg_p * scale_e / GIB,
        "peak_ws_gib": peak_ws * scale_v / GIB,
        "peak_rag_gib": (
            3 * NVOX_216 + 4 * NVOX_216
            + next_pow2(2 * S._max_edges(NVOX_216)) * 20
            + S._max_edges(NVOX_216) * 24
        ) / GIB,
        "peak_agg_gib": (
            4 * NVOX_216 + NEDGE_216_OFFICIAL * 24 + agg_p * scale_e
        ) / GIB,
    }
    worst = max(pred["peak_ws_gib"], pred["peak_rag_gib"], pred["peak_agg_gib"])
    binding = max(
        (pred["peak_ws_gib"], "ws"),
        (pred["peak_rag_gib"], "rag"),
        (pred["peak_agg_gib"], "agg"),
    )[1]
    doc = {
        "claim": "not a 2 Gvox/s number",
        "measured_on": "val 180 Mvox, idle 5090",
        "no_2p16_allocation": True,
        "nvox_val": nvox,
        "nedge_val": nedge,
        "max_edges_val": max_e,
        "edges_per_vox_cap": S.EDGES_PER_VOX,
        "aff_parked": parked,
        "ws_tracked_bytes": ws_p,
        "rag_tracked_bytes": rag_p,
        "agg_tracked_bytes": agg_p,
        "ws_tracked_gib": ws_p / GIB,
        "rag_tracked_gib": rag_p / GIB,
        "agg_tracked_gib": agg_p / GIB,
        "peak_ws_val_gib": peak_ws / GIB,
        "peak_rag_val_gib": peak_rag / GIB,
        "peak_agg_val_gib": peak_agg / GIB,
        "pred_2p16": pred,
        "usable_gib": USABLE,
        "worst_2p16_gib": worst,
        "binding": binding,
        "fits": bool(worst <= USABLE),
        "gpu": gpu_state(),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(
        f"B fit val ws={ws_p / GIB:.3f} rag={rag_p / GIB:.3f} "
        f"agg={agg_p / GIB:.3f} GiB parked={parked} nedge={nedge}\n"
        f"B fit 2.16 pred ws={pred['peak_ws_gib']:.2f} "
        f"rag={pred['peak_rag_gib']:.2f} agg={pred['peak_agg_gib']:.2f} "
        f"worst={worst:.2f} ({binding}) "
        f"{'FITS' if doc['fits'] else 'OVER'} {USABLE} usable -> {OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
