#!/usr/bin/env python3
"""E6: recalibrate the e2e projection from the idle 5090 G0/G15 phases.

Does not use a4_sync_cost 667.6 ms. G2 residual is the endpoint scan, not a
CSR the device does not run. Three stacks only. No 'reachable' sentence
unless stack 3 is under 1080 ms with measured factors.
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
SCALE = (NV_BIG / NV_VAL) * (BW_5090 / BW_3090)  # 12 * 1.778 = 21.333
BUDGET = 1080.0


def loadj(*cands):
    for p in cands:
        if p.is_file():
            return json.loads(p.read_text()), p
    raise SystemExit(f"E6 missing {' or '.join(str(p) for p in cands)}")


def main():
    g, gp = loadj(GOB / "g_levers.json", CACHE / "g_levers.json")
    # Idle 5090 val, 2026-09-05. Local data/cache/d_bench_dev.json is an
    # older contended run (4.2 s) and must not be used.
    e2e_g0 = 1082.174560546875
    e2e_g15 = 958.6963500976562
    if (GOB / "d_bench_dev_g15.json").is_file():
        e2e_g15 = json.loads(
            (GOB / "d_bench_dev_g15.json").read_text())["median_ms"]
    e1p = CACHE / "e1_t3_voi.json"
    e2p = CACHE / "e2_csr_full.json"
    if not e2p.is_file():
        e2p = CACHE / "e2_csr.json"
    e4p = CACHE / "e4_rep_uf.json"
    e1 = json.loads(e1p.read_text()) if e1p.is_file() else None
    e2 = json.loads(e2p.read_text()) if e2p.is_file() else None
    e4 = json.loads(e4p.read_text()) if e4p.is_file() else None

    g0_agg = g["reference"]["phase_sum_ms"]
    g15_agg = g["levers"]["15"]["p0aa"]["phase_sum_ms"]
    g15_compact = g["levers"]["15"]["phases"]["compact"]
    print(f"E6 e2e idle val G0={e2e_g0:.1f} G15={e2e_g15:.1f}", flush=True)
    # residual = e2e - agg. Includes WS + RAG + extract + alloc on val.
    resid_g15 = e2e_g15 - g15_agg
    ws_untiled_val = 536.0  # LOG idle 5090; e9b~446 of it
    rag_val = 9.0
    extract_val = e2e_g15 - ws_untiled_val - rag_val - g15_agg
    if extract_val < 0:
        extract_val = 0.0
        ws_untiled_val = e2e_g15 - rag_val - g15_agg

    def to_3090(val_ms):
        return val_ms * SCALE

    # W5 byte-model 281 ms at 2.16 on 3090 Ti (m1). Unmeasured.
    w5_model_3090 = 281.0
    # E4: if rep_frac << 0.46, compress traffic scales by rep_frac/0.46
    e4_ws = None
    if e4 and e4.get("pass"):
        e4_ws = w5_model_3090 * (e4["max_rep_frac"] / 0.46) + 80.0  # floor

    # E2: compact 219 ms of G15 becomes lookup-sized
    e2_factor = 1.0
    if e2 and e2.get("identical") and e2.get("cut", 1) > 1:
        e2_factor = 1.0 / e2["cut"]
        if not e2.get("pass"):
            print(f"E6 note: E2 cut {e2['cut']:.1f}x below the 10x gate; "
                  f"using it as a measured factor anyway", flush=True)
    g15_agg_e2 = g15_agg - g15_compact * (1.0 - e2_factor)

    # E1 best T=0.3 PASS work factor vs locked sum_nlive
    vfac = 1.0
    vsrc = "none"
    if e1:
        locked = next((r for r in e1["rows"]
                       if abs(r["eps"] - 0.08) < 1e-12 and r["size_asym"]), None)
        cands = [r for r in e1["rows"] if r.get("pass")
                 and not (abs(r["eps"] - 0.08) < 1e-12 and r["size_asym"])]
        if locked and locked.get("sum_nlive") and cands:
            best = min(cands, key=lambda r: r.get("sum_nlive") or 1e30)
            if best.get("sum_nlive"):
                vfac = locked["sum_nlive"] / best["sum_nlive"]
                vsrc = f"{best['tag']} eps={best['eps']} asym={best['size_asym']}"
        elif cands:
            # device-reused rows have no sum_nlive; use v_levers.json
            vl = json.loads((CACHE / "v_levers.json").read_text())["rows"]
            base = vl[0]["sum_nlive"]
            for r in cands:
                match = next((x for x in vl if abs(x["eps"] - r["eps"]) < 1e-12
                              and x["size_asym"] == r["size_asym"]), None)
                if match:
                    fac = base / match["sum_nlive"]
                    if fac > vfac:
                        vfac, vsrc = fac, (f"{r['tag']} eps={r['eps']} "
                                           f"from v_levers")

    stack1 = {
        "name": "locked G15, no V, untiled WS",
        "ws": to_3090(ws_untiled_val),
        "rag": to_3090(rag_val),
        "agg": to_3090(g15_agg),
        "extract": to_3090(max(extract_val, 0)),
    }
    stack2 = {
        "name": "G15 + E2 + W4/W5 byte model, no V",
        "ws": e4_ws if e4_ws is not None else w5_model_3090,
        "rag": to_3090(rag_val),
        "agg": to_3090(g15_agg_e2),
        "extract": to_3090(max(extract_val, 0)),
    }
    stack3 = {
        "name": "G15 + E2 + W4/W5 + best E1 T=0.3",
        "ws": stack2["ws"],
        "rag": stack2["rag"],
        "agg": stack2["agg"] / vfac,
        "extract": stack2["extract"],
        "vfac": vfac, "vsrc": vsrc,
    }

    def finish(s):
        tot = s["ws"] + s["rag"] + s["agg"] + s["extract"]
        s["total_ms"] = tot
        s["gvox_s"] = NV_BIG / (tot / 1e3) / 1e9
        s["vs_budget"] = tot / BUDGET
        return s

    stacks = [finish(stack1), finish(stack2), finish(stack3)]
    print(f"E6 scale val->2.16 Gvox 3090 Ti = {SCALE:.3f}  from {gp.name}")
    print(f"E6 idle val G0 e2e={e2e_g0:.1f} G15 e2e={e2e_g15:.1f} "
          f"G15 agg={g15_agg:.1f} compact={g15_compact:.1f}")
    print(f"E6 E2 factor={e2_factor:.4f}  E1 vfac={vfac:.2f} [{vsrc}]")
    for s in stacks:
        print(f"E6 {s['name']}")
        print(f"E6   ws={s['ws']:.0f} rag={s['rag']:.0f} agg={s['agg']:.0f} "
              f"extract={s['extract']:.0f}  TOTAL={s['total_ms']:.0f} ms "
              f"= {s['gvox_s']:.3f} Gvox/s  {s['vs_budget']:.2f}x budget")

    s3 = stacks[2]
    under = s3["total_ms"] <= BUDGET
    print(f"\nE6 verdict: stack 3 is {s3['total_ms']:.0f} ms vs {BUDGET:.0f} ms")
    if under:
        print("E6 stack 3 is under budget on this recalibration. That is "
              "still not a TASK claim: W5/E2/E1 factors must be measured "
              "on a card before anyone writes reachable.")
    else:
        print("E6 DESIGN SHORT: stack 3 still over 1.08 s. Do not start "
              "CUDA for levers that cannot close it.")

    dest = CACHE / "e6_recal_m1.json"
    dest.write_text(json.dumps({
        "scale": SCALE, "budget_ms": BUDGET,
        "e2e_g0_val": e2e_g0, "e2e_g15_val": e2e_g15,
        "g15_agg_val": g15_agg, "g15_compact_val": g15_compact,
        "e2_factor": e2_factor, "vfac": vfac, "vsrc": vsrc,
        "stacks": stacks, "stack3_under_budget": under,
    }, indent=2) + "\n")
    print(f"E6 wrote {dest.name}")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
