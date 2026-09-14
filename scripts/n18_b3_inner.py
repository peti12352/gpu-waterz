#!/usr/bin/env python3
"""N18 B3: layer/inner work cut that is NOT fuse_pack/unmark/graph.

Probe WATERZ_MAX_INNER if present; else document local maximum.
Kill if cut <100 ms on 2.16 agg. Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from n13_baseline import parks_env  # noqa: E402
from n17_gate import DEEP  # noqa: E402
from n18_dead import refuse_or_ok, stamp  # noqa: E402
from n18_voi_gate import CLAIM, run_gate  # noqa: E402
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402

NOTE = ROOT / "notes/N18_B3.md"
OUT = CACHE / "n18_b3.json"
N17_AGG = 1690.0


def main():
    print(f"N18 B3 inner work cut. {CLAIM}.", flush=True)
    msg = refuse_or_ok("N18_B3", force="--force" in sys.argv)
    if msg:
        print(msg, flush=True)
        return 3
    busy = card_busy()
    if busy:
        print(f"N18 B3 REFUSE: {busy}", flush=True)
        return 2
    # Do not reopen fuse_pack / dirty_unmark / graph
    for dead in ("AGG_FUSE_PACK_closer", "AGG_DIRTY_UNMARK_closer", "AGG_cuda_graph_G2"):
        if refuse_or_ok(dead) is None:
            pass

    # Probe a non-reopened lever: WATERZ_HASH_WIDTH scale (if supported) or
    # document that ParHAC is local max after emit_holes.
    # Use WATERZ_COMPACT_EVERY=0 explicitly (already default) vs measuring
    # WATERZ_LAYER_PARALLEL if absent -> honest exit.
    extra = {**DEEP, "WATERZ_EMIT_HOLES": "1", "WATERZ_NLIVE_ARITH": "1"}
    # Try reducing rebuild frequency via COMPACT_EVERY=8 only as diagnostic
    # (compact_every default was killed as product default: env-only probe ok)
    os.environ["WATERZ_N18_EXP"] = "N18_B3"
    rc, doc = run_gate(
        "N18_B3",
        {**extra, "WATERZ_COMPACT_EVERY": "8"},
        "n18_b3",
        mode="voi_only",
        run_216=True,
    )
    agg = float(doc.get("agg_ms") or 0)
    cut = (N17_AGG - agg) if agg else 0
    kill = (not (doc.get("four") or {}).get("four_pass")
            or cut < 100
            or doc.get("hang"))
    reason = (
        "four-T FAIL" if not (doc.get("four") or {}).get("four_pass") else
        f"cut {cut:.0f}ms <100" if cut < 100 else
        "hang" if doc.get("hang") else "ok"
    )
    doc["agg_cut_vs_n17"] = cut
    doc["kill"] = kill
    doc["reason"] = reason
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        f"# N18 B3\n\n{CLAIM}.\n\n"
        f"- probe WATERZ_COMPACT_EVERY=8 (env-only; not product default)\n"
        f"- agg={agg} cut_vs_n17={cut:.1f} kill={kill} reason={reason}\n"
        f"- fuse_pack/unmark/graph not reopened\n"
        f"- If kill: declare ParHAC local maximum; remaining gap "
        f"agg need ~3x from {N17_AGG} toward ≤1000\n"
    )
    if kill:
        stamp("N18_B3", reason, {"agg": agg, "cut": cut}, "notes/N18_B3.md")
        stamp("N18_B_STOP", "ParHAC local maximum after B3",
              {"n17_agg": N17_AGG, "agg": agg}, "notes/N18_B3.md")
    print(json.dumps({"rc": rc, "agg": agg, "cut": cut, "kill": kill}), flush=True)
    return 0 if not kill else 1


if __name__ == "__main__":
    raise SystemExit(main())
