#!/usr/bin/env python3
"""N7: post-fit idle-5090 remesure of segment_d stages.

Parked aff + DoubleBuffer + WATERZ_AGG_LEVERS=15 + WATERZ_STAGE_MS=1.
Replaces hardcoded WS 536 / RAG 9 / leftover 96. Not a 2 Gvox/s claim.
No 2.16 allocation. Greengoblin only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from p1_make_big_indep import AFF, CACHE, card_busy, gpu_state  # noqa: E402
import segment as S  # noqa: E402

OUT = CACHE / "n7_postfit_stages.json"
NV_VAL = 180e6
NV_BIG = 2.16e9
BW_5090 = 1792e9
BW_3090 = 1008e9
SCALE = (NV_BIG / NV_VAL) * (BW_5090 / BW_3090)
BUDGET = 1080.0
VFAC = 2.883813137549246
E2_FACTOR = 0.2588850437504452
OLD = {
    "ws": 536.0,
    "rag": 9.0,
    "agg": 317.614975896955,
    "leftover": 96.08137420070125,
    "e2e": 958.6963500976562,
}


def to_3090(ms):
    return ms * SCALE


def main():
    print("N7 post-fit remesure. Not a 2 Gvox/s claim.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N7 REFUSE card busy: {busy}", flush=True)
        raise SystemExit(2)
    print(f"GPU idle: {gpu_state()}", flush=True)
    os.environ["WATERZ_AGG_LEVERS"] = "15"
    os.environ["WATERZ_STAGE_MS"] = "1"
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    nvox = int(aff[0].size)
    aff_d = S.DevBuf.from_host(aff)
    warm = S.segment_d(aff_d, [0.3], return_device=True)
    for b in warm:
        b.free()
    runs = []
    for i in range(5):
        out, ms = S.cuda_event_time(
            lambda: S.segment_d(aff_d, [0.3], return_device=True))
        st = dict(S.STAGE_MS)
        runs.append({
            "e2e_ms": float(ms),
            "ws": float(st.get("ws", 0)),
            "rag": float(st.get("rag", 0)),
            "agg": float(st.get("agg", 0)),
            "extract": float(st.get("extract", 0)),
            "parked": bool(S.LAST_AFF_PARKED),
            "nedge": int(S.LAST_NEDGE),
            "backend": str(S.AGG_BACKEND),
        })
        for b in out:
            b.free()
        print(
            f"N7 run{i} e2e={ms:.2f} ms ws={st.get('ws', 0):.2f} "
            f"rag={st.get('rag', 0):.2f} agg={st.get('agg', 0):.2f} "
            f"extract={st.get('extract', 0):.2f} parked={S.LAST_AFF_PARKED}",
            flush=True,
        )
    aff_d.free()
    runs.sort(key=lambda r: r["e2e_ms"])
    med = runs[2]
    stage_sum = med["ws"] + med["rag"] + med["agg"] + med["extract"]
    leftover = med["e2e_ms"] - stage_sum
    # N0 applied e2_factor to compact only (218 of 318). Same share here.
    compact_share = 218.83 / 317.614975896955
    compact_ms = med["agg"] * compact_share
    g15_agg_e2 = med["agg"] - compact_ms * (1.0 - E2_FACTOR)
    honest = {
        "ws": to_3090(med["ws"] + max(leftover, 0.0)),
        "rag": to_3090(med["rag"]),
        "agg": to_3090(g15_agg_e2) / VFAC,
        "extract": to_3090(med["extract"]),
    }
    honest["total_ms"] = (
        honest["ws"] + honest["rag"] + honest["agg"] + honest["extract"]
    )
    honest["gvox_s"] = NV_BIG / (honest["total_ms"] / 1e3) / 1e9
    honest["vs_budget"] = honest["total_ms"] / BUDGET
    need_agg = BUDGET - honest["ws"] - honest["rag"] - honest["extract"]
    need_x = honest["agg"] / need_agg if need_agg > 0 else None
    raw_no_e2 = to_3090(med["agg"]) / VFAC
    doc = {
        "claim": "not a 2 Gvox/s number",
        "measured_on": "val 180 Mvox, idle 5090, post-fit park+DoubleBuffer",
        "no_2p16_allocation": True,
        "nvox": nvox,
        "gpu": gpu_state(),
        "scale": SCALE,
        "vfac": VFAC,
        "e2_factor": E2_FACTOR,
        "old_prefit": OLD,
        "runs": runs,
        "median": med,
        "leftover_val_ms": leftover,
        "stage_sum_val_ms": stage_sum,
        "delta_vs_old": {
            "e2e": med["e2e_ms"] - OLD["e2e"],
            "ws": med["ws"] - OLD["ws"],
            "rag": med["rag"] - OLD["rag"],
            "agg": med["agg"] - OLD["agg"],
            "leftover": leftover - OLD["leftover"],
        },
        "honest_2p16_3090": honest,
        "agg_no_e2_credit_ms": raw_no_e2,
        "agg_budget_ms": need_agg,
        "need_agg_x": need_x,
        "parked": all(r["parked"] for r in runs),
        "backend": med["backend"],
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(
        f"N7 median e2e={med['e2e_ms']:.2f} ws={med['ws']:.2f} "
        f"rag={med['rag']:.2f} agg={med['agg']:.2f} extract={med['extract']:.2f} "
        f"leftover={leftover:.2f}\n"
        f"N7 vs old e2e {OLD['e2e']:.1f}->{med['e2e_ms']:.1f} "
        f"ws {OLD['ws']:.1f}->{med['ws']:.1f} leftover {OLD['leftover']:.1f}->{leftover:.1f}\n"
        f"N7 honest 2.16 pred {honest['total_ms']:.0f} ms "
        f"{honest['gvox_s']:.3f} Gvox/s {honest['vs_budget']:.2f}x "
        f"need_agg_x={need_x if need_x is None else f'{need_x:.2f}'} -> {OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
