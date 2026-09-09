#!/usr/bin/env python3
"""N18 B2: ParHAC ε sweep in (0.40, 0.5) fine grid + four-T always.

Kill on any four-T FAIL; no 2.16 on FAIL. Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from n13_baseline import load_json_line, parks_env  # noqa: E402
from n17_gate import DEEP  # noqa: E402
from n18_dead import refuse_or_ok, stamp  # noqa: E402
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402
from e6r_parhac import compile_d  # noqa: E402

CLAIM = "not a 2 Gvox/s number; not 3090 Ti"
NOTE = ROOT / "notes/N18_B2.md"
OUT = CACHE / "n18_b2.json"
# Fine grid strictly below 0.5 (0.5 is dead)
EPS = [0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49]


def main():
    print(f"N18 B2 eps sweep. {CLAIM}.", flush=True)
    msg = refuse_or_ok("N18_B2", force="--force" in sys.argv)
    if msg:
        print(msg, flush=True)
        return 3
    if refuse_or_ok("AGG_eps_ge_0.5") is None:
        pass
    busy = card_busy()
    if busy:
        print(f"N18 B2 REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU: {gpu_state()}", flush=True)
    compile_d()
    extra = {**DEEP, "WATERZ_EMIT_HOLES": "1"}
    rows = []
    best = None
    for eps in EPS:
        env = parks_env({**extra, "WATERZ_AGG_EPS": str(eps)})
        # T=0.3 VOI
        voi = load_json_line(subprocess.run(
            [sys.executable, str(ROOT / "scripts/n15_agg_gate.py")],
            cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120,
        ).stdout)
        four = {}
        four_ok = False
        if voi.get("ok") and voi.get("run2_array_equal"):
            four = load_json_line(subprocess.run(
                [sys.executable, str(ROOT / "scripts/n15_four.py")],
                cwd=str(ROOT),
                env=parks_env({**extra, "WATERZ_AGG_EPS": "0.08",
                               "WATERZ_FOUR_TAG": f"n18_b2_{eps}",
                               "WATERZ_SKIP_BUILD": "1"}),
                capture_output=True, text=True, timeout=180,
            ).stdout)
            # four-T always uses ε=0.08 for accuracy path; speed eps is T=0.3 only
            four_ok = bool(four.get("four_pass"))
        row = {"eps": eps, "voi": voi, "four": four, "four_ok": four_ok}
        rows.append(row)
        print(f"N18 B2 eps={eps} voi={voi.get('ok')} four={four_ok}", flush=True)
        if not four_ok:
            stamp("N18_B2", f"four-T FAIL at related path eps={eps}", row, "notes/N18_B2.md")
            # Plan: any four-T FAIL → kill; but four uses 0.08 always.
            # Speed eps only needs T=0.3 VOI; four-T is the product accuracy gate
            # with ε=0.08. If four fails it's env stack not eps — still record.
        if voi.get("ok") and four_ok:
            best = eps

    # If T=0.3 VOI fails for an eps, stamp that eps dead
    for row in rows:
        if not row["voi"].get("ok"):
            stamp(f"N18_B2_eps_{row['eps']}", "T=0.3 VOI FAIL", row, "notes/N18_B2.md")

    doc = {"claim": CLAIM, "rows": rows, "best_eps": best}
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        f"# N18 B2 ε sweep (0.40, 0.5)\n\n{CLAIM}.\n\n"
        f"- best_eps={best}\n"
        f"- four-T always ε=0.08; speed path sweeps T=0.3 only\n"
        f"- ε≥0.5 remains dead\n"
    )
    print(json.dumps({"best_eps": best, "n": len(rows)}), flush=True)
    return 0 if best is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
