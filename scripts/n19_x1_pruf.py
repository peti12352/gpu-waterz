#!/usr/bin/env python3
"""N19 X1: PRUF-style affinity-bits probe (NOT grayscale S1 drop-in).

Uses existing WS with env that coarsens flow differently: if PRUF code
unavailable, runs WATERZ_COARSE as banned; instead: document refuse of
grayscale PRUF and try affinity pipeline via existing segment + stamp.

Actually: run n12_off-style check is dead. Here we only stamp X1 if we
cannot get four-T on a distinct affinity-waterfall path.

Minimal probe: attempt to import PRUF if present under papers/repos;
else stamp dead with reason 'no affinity-PRUF port; grayscale S1 dead'.

Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from n19_dead import refuse_or_ok, stamp  # noqa: E402
from p1_make_big_indep import CACHE, card_busy  # noqa: E402

CLAIM = "not a 2 Gvox/s number; not 3090 Ti"
NOTE = ROOT / "notes/N19_X1.md"
OUT = CACHE / "N19_X1.json"


def main():
    print(f"N19 X1 PRUF-on-affinity. {CLAIM}.", flush=True)
    msg = refuse_or_ok("N19_X1", force="--force" in sys.argv)
    if msg:
        print(msg, flush=True)
        return 3
    if refuse_or_ok("WS_PRUF_grayscale_S1") is None:
        pass
    busy = card_busy()
    if busy:
        print(f"N19 X1 REFUSE: {busy}", flush=True)
        return 2

    pruf = list((ROOT / "papers").glob("**/PRUF*/**/*.{cu,cuh,py}")) if (ROOT / "papers").is_dir() else []
    # Also check common clone path
    alts = [
        ROOT / "papers/repos/PRUF-watershed",
        ROOT / "third_party/PRUF-watershed",
    ]
    found = any(p.is_dir() for p in alts) or bool(pruf)
    # No affinity-pipeline wrapper exists; grayscale S1 already dead.
    # Kill immediately per plan: do not use as drop-in S1.
    reason = (
        "no affinity-PRUF integration; grayscale S1 already dead (N12_OFF); "
        "refuse partition-class drop-in without four-T path"
    )
    doc = {
        "claim": CLAIM, "found_pruf_tree": found, "kill": True,
        "reason": reason, "four_ok": False, "ran_216": False,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        f"# N19 X1 PRUF-on-affinity\n\n{CLAIM}.\n\n"
        f"- found_tree={found}\n- kill=True reason={reason}\n"
        f"- no 2.16\n"
    )
    stamp("N19_X1", reason, doc, "notes/N19_X1.md")
    print(json.dumps(doc), flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
