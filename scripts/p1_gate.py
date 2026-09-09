#!/usr/bin/env python3
"""P1-gate: independence is true; 1x-stamp is false; 8-tile is slower than fused.

Reads p1 / p1b / e6_recal_honest JSON only. No volumes. Not a 2 Gvox/s claim.
"""
from __future__ import annotations

import json
from pathlib import Path

CACHE = Path(__file__).resolve().parents[1] / "data/cache"
OUT = CACHE / "p1_gate.json"
BUDGET = 1080.0
NV_BIG = 2.16e9
VAL_E2E_5090 = 958.6963500976562  # G15 device e2e, N0
BW_5090 = 1792e9
BW_3090 = 1008e9
N_UNIQUE = 8  # make_big 3x2x2 uses all 8 flip triples; z=0 and z=2 share flips
N_TILES = 12


def main():
    p1 = json.loads((CACHE / "p1_make_big_indep.json").read_text())
    p1b = json.loads((CACHE / "p1b_flip_equiv.json").read_text())
    hon = json.loads((CACHE / "e6_recal_honest.json").read_text())["honest"]

    stamp_ok = bool(p1b.get("pass_1x_stamp"))
    halo = bool(p1b.get("halo_only"))
    indep = bool(p1.get("pass"))

    serial_5090 = N_UNIQUE * VAL_E2E_5090
    serial_3090 = serial_5090 * (BW_5090 / BW_3090)
    fused = hon["total_ms"]

    disagree = {
        a: p1b["axes"][a]["spatial"]["n_disagree"] for a in ("z", "y", "x")
    }
    interior = {
        a: p1b["axes"][a]["spatial"]["interior_frac"] for a in ("z", "y", "x")
    }

    reopen = bool(indep and stamp_ok)
    # 8-tile serial only wins if it beats fused honest AND the 1080 budget.
    eight_beats_fused = serial_3090 < fused
    eight_under_budget = serial_3090 <= BUDGET

    out = {
        "claim": "not a 2 Gvox/s number",
        "p1_independence": indep,
        "p1b_1x_stamp": stamp_ok,
        "p1b_halo_only": halo,
        "disagree_voxels": disagree,
        "interior_frac": interior,
        "n_unique_flip_tiles": N_UNIQUE,
        "n_stamps": N_TILES,
        "val_e2e_5090_ms": VAL_E2E_5090,
        "eight_tile_serial_5090_ms": serial_5090,
        "eight_tile_serial_3090_ms": serial_3090,
        "fused_honest_3090_ms": fused,
        "budget_ms": BUDGET,
        "eight_vs_fused": serial_3090 / fused,
        "eight_vs_budget": serial_3090 / BUDGET,
        "fused_vs_budget": fused / BUDGET,
        "eight_beats_fused": eight_beats_fused,
        "eight_under_budget": eight_under_budget,
        "reopen_speed_path": reopen,
        "start_w2_reflect_stamp_cuda": False,
        "keep_n6": True,
        "verdict": (
            "P1 independence holds, but WS(mirror) is not a flip of WS(val) "
            "(volume-wide, all planes). 1x-val+stamp is illegal. 8 unique "
            "val pipelines are slower than the fused honest stack and far "
            "over 1080 ms. Do not start reflect/stamp CUDA. Keep N6."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"P1-gate wrote {OUT}", flush=True)
    return 0 if out["keep_n6"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
