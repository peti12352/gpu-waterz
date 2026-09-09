#!/usr/bin/env python3
"""N8: device-gather unpark. Identity + WS time + 2.16 peak pred.

Replaces host-park permute with k_gather_u32. Idle-5090. Not a 2 Gvox/s claim.
No 2.16 allocation.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from b_dev_aff import bind, build, run_split  # noqa: E402
from p1_make_big_indep import AFF, CACHE, card_busy, fingerprint, gpu_state, nfrag_bg  # noqa: E402
from task_gate import FRAGMENTS_VAL  # noqa: E402
import segment as S  # noqa: E402

OUT = CACHE / "n8_unpark.json"
GIB = 1024.0 ** 3
USABLE = 23.0
NVOX_216 = 2_160_000_000
NEDGE_216_OFFICIAL = 90_323_139
WS_GATE_MS = 600.0


def next_pow2(x):
    p = 1
    while p < x:
        p <<= 1
    return p


def main():
    print("N8 device-gather unpark. Not a 2 Gvox/s claim.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N8 REFUSE card busy: {busy}", flush=True)
        raise SystemExit(2)
    print(f"GPU idle: {gpu_state()}", flush=True)
    build()
    os.environ["WATERZ_AGG_LEVERS"] = "15"
    os.environ["WATERZ_STAGE_MS"] = "1"
    libw = ctypes.CDLL(str(S._WS))
    bind(libw)
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
    # Identity is WS fragments, not agglomerated labels.
    frag, frag_meta = run_split(libw, aff, park=True)
    nfrag, bg = nfrag_bg(frag)
    fp, _, _ = fingerprint(frag)
    oracle = None
    op = CACHE / "wz_fragments.npy"
    if op.exists():
        ref = np.ascontiguousarray(np.load(op), dtype=np.uint32)
        oracle = bool(ref.shape == frag.shape and np.array_equal(ref, frag))
    print(
        f"N8 WS identity nfrag={nfrag} oracle={oracle} "
        f"divide_ms={frag_meta.get('divide_ms', 0):.2f}",
        flush=True,
    )
    aff_d = S.DevBuf.from_host(aff)
    warm = S.segment_d(aff_d, [0.3], return_device=True)
    for b in warm:
        b.free()
    times = []
    splits = []
    for i in range(3):
        out, ms = S.cuda_event_time(
            lambda: S.segment_d(aff_d, [0.3], return_device=True))
        st = dict(S.STAGE_MS)
        times.append(float(ms))
        splits.append(st)
        for b in out:
            b.free()
        print(
            f"N8 run{i} e2e={ms:.2f} ws={st.get('ws', 0):.2f} "
            f"rag={st.get('rag', 0):.2f} agg={st.get('agg', 0):.2f} "
            f"parked={S.LAST_AFF_PARKED}",
            flush=True,
        )
    aff_d.free()
    paired = sorted(zip(times, splits), key=lambda x: x[0])
    med_e2e, med_st = paired[len(paired) // 2]
    ws_ms = float(med_st.get("ws", 0))
    ws_p = int(libw.ws_mem_peak())
    rag_p = int(libr.rag_mem_peak())
    agg_p = int(libp.agg_mem_peak())
    nedge = int(S.LAST_NEDGE)
    max_e = S._max_edges(nvox)
    seg_b = 4 * nvox
    bits_b = nvox
    peak_ws = ws_p + seg_b + bits_b
    scale_v = NVOX_216 / nvox
    pred_ws = peak_ws * scale_v / GIB
    # Official 2.16 is 3 z-tiles of 125×2400×2400. Scratch scales with one
    # tile (4× val YX), not 12× the fused volume.
    slab_nvox = 125 * 2400 * 2400
    pred_ws_slab = (ws_p / nvox) * slab_nvox / GIB + (seg_b + bits_b) * scale_v / GIB
    pred_rag = (
        3 * NVOX_216 + 4 * NVOX_216
        + next_pow2(2 * S._max_edges(NVOX_216)) * 20
        + S._max_edges(NVOX_216) * 24
    ) / GIB
    pred_agg = (
        4 * NVOX_216 + NEDGE_216_OFFICIAL * 24 + agg_p * (NEDGE_216_OFFICIAL / max(nedge, 1))
    ) / GIB
    worst_fused = max(pred_ws, pred_rag, pred_agg)
    worst = max(pred_ws_slab, pred_rag, pred_agg)
    fits = bool(worst <= USABLE)
    fits_fused = bool(worst_fused <= USABLE)
    ok = bool(nfrag == FRAGMENTS_VAL and oracle is not False)
    ws_ok = ws_ms <= WS_GATE_MS
    doc = {
        "claim": "not a 2 Gvox/s number",
        "lever": "k_gather_u32 permute; corners/vc parked raw (no CPU permute)",
        "nfrag": nfrag,
        "bg": bg,
        "fingerprint": fp,
        "oracle_array_equal": oracle,
        "nvox": nvox,
        "nedge": nedge,
        "median_e2e_ms": med_e2e,
        "median_ws_ms": ws_ms,
        "ws_gate_ms": WS_GATE_MS,
        "ws_under_gate": ws_ok,
        "stages": med_st,
        "ws_tracked_gib": ws_p / GIB,
        "rag_tracked_gib": rag_p / GIB,
        "agg_tracked_gib": agg_p / GIB,
        "peak_ws_val_gib": peak_ws / GIB,
        "pred_2p16_ws_gib": pred_ws,
        "pred_2p16_ws_slab_gib": pred_ws_slab,
        "pred_2p16_rag_gib": pred_rag,
        "pred_2p16_agg_gib": pred_agg,
        "worst_2p16_fused_gib": worst_fused,
        "worst_2p16_gib": worst,
        "usable_gib": USABLE,
        "fits_fused": fits_fused,
        "fits": fits,
        "aff_parked": bool(S.LAST_AFF_PARKED),
        "gpu": gpu_state(),
        "pass": ok,
        "need_slab": bool(ok and not fits_fused),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(
        f"N8 nfrag={nfrag} oracle={oracle} ws={ws_ms:.2f} "
        f"{'<=600' if ws_ok else '>600'} e2e={med_e2e:.2f}\n"
        f"N8 tracked ws={ws_p / GIB:.3f} 2.16 pred "
        f"ws_fused={pred_ws:.2f} ws_slab={pred_ws_slab:.2f} "
        f"rag={pred_rag:.2f} agg={pred_agg:.2f} "
        f"worst={worst:.2f} {'FITS' if fits else 'OVER'} "
        f"{'PASS' if ok else 'FAIL'} slab={doc['need_slab']} -> {OUT}",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
