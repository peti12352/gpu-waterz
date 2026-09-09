#!/usr/bin/env python3
"""N19 H3: batch RNN / safe parallel merges probe inside ParHAC layer.

Env WATERZ_BATCH_RNN=1 if implemented in parhac_d.cu; else stamp incomplete.
Kill if four-T FAIL or agg cut <100 ms.
Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from n17_gate import DEEP  # noqa: E402
from n19_dead import refuse_or_ok, stamp  # noqa: E402
from n19_voi_gate import CLAIM, run_gate  # noqa: E402
from p1_make_big_indep import CACHE, card_busy  # noqa: E402

NOTE = ROOT / "notes/N19_H3.md"
OUT = CACHE / "N19_H3.json"
N18_AGG = 1683.0


def main():
    print(f"N19 H3 batch RNN. {CLAIM}.", flush=True)
    msg = refuse_or_ok("N19_H3", force="--force" in sys.argv)
    if msg:
        print(msg, flush=True)
        return 3
    busy = card_busy()
    if busy:
        print(f"N19 H3 REFUSE: {busy}", flush=True)
        return 2

    # Check if lever exists in source
    src = (ROOT / "csrc/parhac_d.cu").read_text()
    if "WATERZ_BATCH_RNN" not in src and "batch_rnn" not in src:
        reason = "BATCH_RNN not implemented in parhac_d.cu; stamp incomplete (no silent skip as speed win)"
        doc = {
            "claim": CLAIM, "kill": True, "reason": reason,
            "implemented": False, "ran_216": False,
        }
        CACHE.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=2) + "\n")
        NOTE.write_text(
            f"# N19 H3 batch RNN\n\n{CLAIM}.\n\n- {reason}\n"
            f"- ParChain-inspired; deferred to avoid unmeasured stack\n"
        )
        stamp("N19_H3", reason, doc, "notes/N19_H3.md")
        print(json.dumps(doc), flush=True)
        return 1

    os.environ["WATERZ_N19_EXP"] = "N19_H3"
    extra = {**DEEP, "WATERZ_EMIT_HOLES": "1", "WATERZ_BATCH_RNN": "1"}
    rc, doc = run_gate("N19_H3", extra, "n19_h3", mode="voi_only", run_216=True)
    agg = float(doc.get("agg_ms") or 0)
    cut = (N18_AGG - agg) if agg else 0
    kill = (
        not (doc.get("four") or {}).get("four_pass")
        or cut < 100
        or doc.get("hang")
    )
    reason = (
        "four-T FAIL" if not (doc.get("four") or {}).get("four_pass") else
        f"cut {cut:.0f}<100" if cut < 100 else
        "hang" if doc.get("hang") else "ok"
    )
    doc["agg_cut_vs_n18"] = cut
    doc["kill"] = kill
    doc["reason"] = reason
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        f"# N19 H3\n\n{CLAIM}.\n\n- agg={agg} cut={cut:.1f} kill={kill} {reason}\n"
    )
    if kill:
        stamp("N19_H3", reason, {"agg": agg, "cut": cut}, "notes/N19_H3.md")
    print(json.dumps({"rc": rc, "agg": agg, "cut": cut, "kill": kill}), flush=True)
    return 0 if not kill else 1


if __name__ == "__main__":
    raise SystemExit(main())
