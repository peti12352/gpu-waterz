#!/usr/bin/env python3
"""N19 X3: RAMA one-shot with 60s abort (prior 600s voided).

Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from n19_dead import refuse_or_ok, stamp  # noqa: E402
from p1_make_big_indep import CACHE, card_busy  # noqa: E402

CLAIM = "not a 2 Gvox/s claim. Not 3090 Ti."
# fix claim string
CLAIM = "not a 2 Gvox/s number; not 3090 Ti"
NOTE = ROOT / "notes/N19_X3.md"
OUT = CACHE / "N19_X3.json"
ABORT = 60


def main():
    print(f"N19 X3 RAMA {ABORT}s abort. {CLAIM}.", flush=True)
    msg = refuse_or_ok("N19_X3", force="--force" in sys.argv)
    if msg:
        print(msg, flush=True)
        return 3
    if refuse_or_ok("AGG_mutex_Kruskal_X1_SubgraphHAC_RAMA") is None:
        pass
    busy = card_busy()
    if busy:
        print(f"N19 X3 REFUSE: {busy}", flush=True)
        return 2

    # Look for prior rama runner
    candidates = [
        ROOT / "scripts/n13_rama.py",
        ROOT / "scripts/rama_val.py",
    ]
    runner = next((p for p in candidates if p.is_file()), None)
    if runner is None:
        reason = "no rama runner script; class already task_legal=False; stamp dead"
        doc = {"claim": CLAIM, "kill": True, "reason": reason, "ran_216": False}
        CACHE.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=2) + "\n")
        NOTE.write_text(f"# N19 X3 RAMA\n\n{CLAIM}.\n\n- {reason}\n")
        stamp("N19_X3", reason, doc, "notes/N19_X3.md")
        print(json.dumps(doc), flush=True)
        return 1

    try:
        r = subprocess.run(
            [sys.executable, "-u", str(runner)],
            cwd=str(ROOT), timeout=ABORT, capture_output=True, text=True,
        )
        reason = f"rama rc={r.returncode} (timeout not hit); VOI expected FAIL class"
        kill = True
        doc = {
            "claim": CLAIM, "kill": kill, "reason": reason,
            "stdout_tail": (r.stdout or "")[-1500:],
            "stderr_tail": (r.stderr or "")[-1500:],
            "ran_216": False,
        }
    except subprocess.TimeoutExpired:
        reason = f"timeout {ABORT}s (prior N13 was 600s void)"
        doc = {"claim": CLAIM, "kill": True, "reason": reason, "ran_216": False}

    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(f"# N19 X3 RAMA\n\n{CLAIM}.\n\n- {doc['reason']}\n- no 2.16\n")
    stamp("N19_X3", doc["reason"], doc, "notes/N19_X3.md")
    print(json.dumps({"kill": True, "reason": doc["reason"]}), flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
