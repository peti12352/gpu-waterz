#!/usr/bin/env python3
"""N0: recalibrate E6 without dumping leftover into extract.

E6-as-written set extract_val = e2e - 536 - 9 - agg (~96 ms) and scaled it
by 21.333 → 2049 ms. Real extract on val was 3.434 ms (LOG E10). The 96 ms
is alloc / e9c / sync leftover. This file attributes that leftover three
ways and writes the honest stack the plan uses (leftover absorbed into WS).

Not a TASK 2 Gvox/s claim.
"""
from __future__ import annotations

import json
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "data/cache"
GOB = CACHE / "greengoblin_20260905"

NV_VAL = 180e6
NV_BIG = 2.16e9
BW_5090 = 1792e9
BW_3090 = 1008e9
SCALE = (NV_BIG / NV_VAL) * (BW_5090 / BW_3090)  # 21.333
BUDGET = 1080.0
EXTRACT_VAL_MS = 3.434  # LOG: extract_gpu device_ms=3.434
WS_UNTILED_VAL = 536.0
RAG_VAL = 9.0
E2E_G0 = 1082.174560546875
E2E_G15 = 958.6963500976562


def finish(s):
    tot = s["ws"] + s["rag"] + s["agg"] + s["extract"] + s.get("leftover", 0.0)
    s["total_ms"] = tot
    s["gvox_s"] = NV_BIG / (tot / 1e3) / 1e9
    s["vs_budget"] = tot / BUDGET
    return s


def main():
    g = json.loads((GOB / "g_levers.json").read_text()
                   if (GOB / "g_levers.json").is_file()
                   else (CACHE / "g_levers.json").read_text())
    if (GOB / "d_bench_dev_g15.json").is_file():
        e2e_g15 = json.loads((GOB / "d_bench_dev_g15.json").read_text())["median_ms"]
    else:
        e2e_g15 = E2E_G15
    e1 = json.loads((CACHE / "e1_t3_voi.json").read_text())
    e2 = json.loads((CACHE / "e2_csr_full.json").read_text())
    e4 = json.loads((CACHE / "e4_rep_uf.json").read_text())
    e6 = json.loads((CACHE / "e6_recal_m1.json").read_text())

    g15_agg = g["levers"]["15"]["p0aa"]["phase_sum_ms"]
    g15_compact = g["levers"]["15"]["phases"]["compact"]
    leftover_val = e2e_g15 - WS_UNTILED_VAL - RAG_VAL - g15_agg
    print(f"N0 e2e_g15={e2e_g15:.2f} agg={g15_agg:.2f} "
          f"ws={WS_UNTILED_VAL} rag={RAG_VAL} extract_meas={EXTRACT_VAL_MS} "
          f"leftover={leftover_val:.2f}", flush=True)

    e2_factor = 1.0 / e2["cut"] if e2.get("identical") and e2.get("cut", 1) > 1 else 1.0
    g15_agg_e2 = g15_agg - g15_compact * (1.0 - e2_factor)
    vfac = float(e6.get("vfac") or 1.0)
    vsrc = e6.get("vsrc", "")
    w5_model_3090 = 281.0
    e4_ws = w5_model_3090 * (e4["max_rep_frac"] / 0.46) + 80.0

    def to_3090(ms):
        return ms * SCALE

    agg_s3 = to_3090(g15_agg_e2) / vfac
    rag_s3 = to_3090(RAG_VAL)
    extract_honest = to_3090(EXTRACT_VAL_MS)
    leftover_scaled = to_3090(max(leftover_val, 0.0))

    attributions = {
        "leftover_into_ws": {
            "note": "e9c/alloc-as-WS; absorbed by W5/E4. Plan honest stack.",
            "ws": e4_ws, "rag": rag_s3, "agg": agg_s3,
            "extract": extract_honest, "leftover": 0.0,
        },
        "leftover_into_alloc": {
            "note": "leftover scales like volume (E6-as-written extract dump).",
            "ws": e4_ws, "rag": rag_s3, "agg": agg_s3,
            "extract": extract_honest, "leftover": leftover_scaled,
        },
        "leftover_into_sync": {
            "note": "leftover is launch/sync and does not scale with volume.",
            "ws": e4_ws, "rag": rag_s3, "agg": agg_s3,
            "extract": extract_honest, "leftover": max(leftover_val, 0.0),
        },
    }
    for k, s in attributions.items():
        s["name"] = k
        finish(s)
        print(f"N0 {k}: ws={s['ws']:.0f} rag={s['rag']:.0f} agg={s['agg']:.0f} "
              f"extract={s['extract']:.0f} leftover={s['leftover']:.0f} "
              f"TOTAL={s['total_ms']:.0f} ms {s['gvox_s']:.3f} Gvox/s "
              f"{s['vs_budget']:.2f}x", flush=True)

    honest = attributions["leftover_into_ws"]
    need_agg = BUDGET - honest["ws"] - honest["rag"] - honest["extract"]
    need_x = honest["agg"] / need_agg if need_agg > 0 else None
    reflect_ws_rag = {
        "ws": honest["ws"] / 12.0,
        "rag": 30.0,
        "agg": honest["agg"],
        "extract": honest["extract"] / 12.0,
        "leftover": 0.0,
        "name": "honest_plus_ws_rag_reflect",
        "note": "N4 may cut WS/RAG; does not cut merge count.",
    }
    finish(reflect_ws_rag)
    need_agg_reflect = BUDGET - reflect_ws_rag["ws"] - reflect_ws_rag["rag"] - reflect_ws_rag["extract"]
    need_x_reflect = honest["agg"] / need_agg_reflect if need_agg_reflect > 0 else None

    out = {
        "scale": SCALE, "budget_ms": BUDGET,
        "e2e_g15_val": e2e_g15, "g15_agg_val": g15_agg,
        "leftover_val_ms": leftover_val,
        "extract_measured_val_ms": EXTRACT_VAL_MS,
        "e2_factor": e2_factor, "vfac": vfac, "vsrc": vsrc,
        "e6_written_stack3_ms": e6["stacks"][2]["total_ms"],
        "attributions": attributions,
        "honest": honest,
        "agg_ms": honest["agg"],
        "agg_budget_ms": need_agg,
        "need_agg_x": need_x,
        "honest_plus_reflect": reflect_ws_rag,
        "need_agg_x_if_reflect": need_x_reflect,
        "gate_n5_cuda": {
            "need_x_without_n4": need_x,
            "need_x_with_n4": need_x_reflect,
            "threshold_without_n4": 1.95,
            "threshold_with_n4": 1.18,
        },
        "stack3_under_budget": honest["total_ms"] <= BUDGET,
    }
    dest = CACHE / "e6_recal_honest.json"
    dest.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nN0 honest TOTAL={honest['total_ms']:.0f} ms "
          f"{honest['gvox_s']:.3f} Gvox/s {honest['vs_budget']:.2f}x budget",
          flush=True)
    print(f"N0 need another {need_x:.2f}x on agg to hit 1080 "
          f"(or {need_x_reflect:.2f}x if N4 WS+RAG reflect lands)", flush=True)
    print(f"N0 wrote {dest.name}", flush=True)
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
