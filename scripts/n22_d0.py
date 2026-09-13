#!/usr/bin/env python3
"""N22 D0 leftover inventory from N21 pins + I0 nsys. No GPU.

Not a 2 Gvox/s number; not 3090 Ti.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys_path_note = ROOT / "scripts"
import sys
sys.path.insert(0, str(sys_path_note))
from n22_dead import stamp  # noqa: E402

CACHE = ROOT / "data/cache"
CLAIM = "N22 D0 leftover inventory; not a throughput claim; not a 2 Gvox/s number; not 3090 Ti"
VAL_PIN_MS = 456.7932924255729
PIN_AGG_216 = 1679.9063761718571


def pct(xs, p):
    if not xs:
        return None
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, int(round((p / 100.0) * (len(ys) - 1)))))
    return ys[i]


def main():
    d0 = json.loads((CACHE / "N21_D0.json").read_text())
    owners = json.loads((CACHE / "N21_D0_OWNERS.json").read_text())
    nsys = json.loads((CACHE / "n19_nsys.json").read_text())
    dirty = d0["dirty"]
    head = dirty["rows_head"]
    tail = dirty["rows_tail"]
    first = head[0]
    ntab0 = first["ntab"]
    nuniq0 = first["nuniq"]
    ndirty0 = first["ndirty"]
    nact0 = first["nact"]
    nscan0 = first["nscan"]
    empty0 = 1.0 - (nuniq0 / ntab0) if ntab0 else None
    below_tl = ndirty0 - nact0

    emit = {
        "nsys_total_ms": 161.122414,
        "instances": 438,
        "avg_ns": 367859.4,
        "med_ns": 42272.0,
        "max_ns": 6765011,
        "avg_over_med": 367859.4 / 42272.0,
        "first_inner_ntab": ntab0,
        "first_inner_nuniq": nuniq0,
        "first_inner_empty_frac": empty0,
        "n21_a3_skip_reason": "ntab already next_pow2(ndirty*2+1024); that skipped occupied emit",
        "go_slot_emit": True,
        "not_cub_deviceselect": "DeviceSelect over ntab is still O(ntab)",
        "not_hash_clear": True,
    }
    copy_sz = {
        "nsys_total_ms": 114.511686,
        "instances": 246,
        "avg_ns": 465494.7,
        "med_ns": 478445.0,
        "uniform": True,
        "already_listed": True,
        "sticky_sz0_voi_dead": True,
        "necessary_for_freeze_eps": True,
        "go_kernel": False,
    }
    rewrite = {
        "nsys_total_ms": 272.950646,
        "first_inner_ndirty": ndirty0,
        "first_inner_nact": nact0,
        "first_inner_nscan": nscan0,
        "below_tl_live_dirty": below_tl,
        "holes_are_output": True,
        "needs_csr": True,
        "go_kernel_without_csr": False,
        "do_not_launch_e6t": True,
    }
    skip = [
        {
            "id": "N22_PROPOSE_ALREADY_LISTED",
            "kernel": "k_propose_listed",
            "ms": 146.784784,
            "avg_over_med": 215860.0 / 8400.0,
            "reason": "already listed; tail is atomicMax on large nact, not empty-slot scan",
        },
        {
            "id": "N22_RAG_FACES_BELOW_GATE",
            "kernel": "k_hash_faces",
            "ms": 47.879648,
            "reason": "RAG 48 ms of 85; below closer gates",
        },
        {
            "id": "N22_COUNT_V2_WS_SKIP",
            "kernel": "k_count_v2",
            "ms": 148.142816,
            "reason": "WS; N19 W1/W3 owner <200 ms hook/vcount skip; WS frozen N21_W3",
        },
        {
            "id": "N22_COMPACT_ROOTS_LISTED",
            "kernel": "k_compact_roots",
            "ms": 77.364002,
            "reason": "already listed over live roots; uniform",
        },
        {
            "id": "N22_PACK_AMASK_A2",
            "kernel": "k_pack_amask",
            "ms": 100.200579,
            "reason": "N21 A2 listed rebuild already attacked this domain",
        },
    ]
    doc = {
        "claim": CLAIM,
        "pin_agg_216_ms": PIN_AGG_216,
        "val_pin_ms": VAL_PIN_MS,
        "emit": emit,
        "copy_sz": copy_sz,
        "rewrite": rewrite,
        "skip": skip,
        "gpu_trial": "N22_A3_SLOT_EMIT then stack if identity PASS",
        "owners_four_kernel_ms": owners["floors_vs_1679.9"]["four_contract_rebuild_fuse_insert_emit_ms"],
        "nsys_source": str(CACHE / "n19_nsys.json"),
        "d0_source": str(CACHE / "N21_D0.json"),
        "n_records_d0": dirty["n_records"],
        "rows_head_n": len(head),
        "rows_tail_n": len(tail),
        "full_315_rows_not_in_d0_json": True,
        "head_ntab_over_nuniq": [r["ntab"] / r["nuniq"] if r["nuniq"] else None for r in head],
    }
    (CACHE / "N22_D0.json").write_text(json.dumps(doc, indent=2) + "\n")

    stamp("N22_D0_EMIT", "go_slot_emit", emit,
          note="nsys avg/med 8.7x; first inner ntab 16.8M vs nuniq 4.4M; N21_A3 skipped occupied emit")
    stamp("N22_COPY_SZ_NECESSARY", "necessary_work_already_listed", copy_sz,
          note="114.5 ms listed over roots; 246 x ~465 us uniform; STICKY_SZ0 VOI-dead; not a closer")
    stamp("N22_REWRITE_NEEDS_CSR", "structural_no_kernel", rewrite,
          note="first inner ndirty 5.6M vs nact 2.8M; holes are rewrite output; do not CONT E6t")
    for s in skip:
        stamp(s["id"], "skip_unrun_low_ev", s, note=s["reason"])

    lines = [
        "# N22 D0 leftover inventory",
        "",
        CLAIM + ".",
        "",
        "## Emit (the GPU trial)",
        "",
        f"- nsys k_hash_emit_holes 161.1 ms / 438 launches; avg 368 us vs med 42 us "
        f"(avg/med={emit['avg_over_med']:.1f}); max 6.8 ms",
        f"- first inner ntab={ntab0} nuniq={nuniq0} empty_frac={empty0:.3f}",
        "- N21_A3 skipped because ntab is already sized for load 0.5; that is not occupied emit",
        "- CUB DeviceSelect over ntab is still O(ntab); do not resurrect k_hash_clear",
        "- GPU trial: WATERZ_SLOT_EMIT=1 (N22_A3), then stack with listed insert/rebuild",
        "",
        "## copy_sz (stamp, no kernel)",
        "",
        "- k_copy_sz_list 114.5 ms, 246 launches, avg 465 us med 478 us (uniform)",
        "- already listed over live roots (AGG_LEVERS bit 8); required for freeze eps",
        "- STICKY_SZ0 is VOI-dead; do not grind",
        "",
        "## rewrite (stamp, no kernel)",
        "",
        f"- k_rewrite_dirty_fuse 273.0 ms; first inner ndirty={ndirty0} nact={nact0} "
        f"below-TL live dirty={below_tl}",
        "- holes are the output of rewrite; listed insert/rebuild cannot delete the scan",
        "- no dirty-edge CSR on the legal path; adj_off is E6t StarMerge-only; do not launch",
        "",
        "## skipped (lurking but not unrun-high-EV)",
        "",
    ]
    for s in skip:
        lines.append(f"- {s['id']}: {s['kernel']} {s['ms']:.1f} ms; {s['reason']}")
    (ROOT / "notes" / "N22_D0.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "tag": "N22_D0",
        "go_slot_emit": True,
        "copy_sz_kernel": False,
        "rewrite_kernel": False,
        "first_inner_empty_frac": empty0,
        "emit_avg_over_med": emit["avg_over_med"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
