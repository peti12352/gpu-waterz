#!/usr/bin/env python3
"""N15 exp8: layer-sticky sz0. WATERZ_STICKY_SZ0=1. ε stays 0.40.

VOI + 2-run, then 2.16. Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from n13_baseline import load_json_line, parks_env  # noqa: E402
from n15_agg_gate import AGG_BASE, AGG_GATE  # noqa: E402
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402
from e6r_parhac import compile_d  # noqa: E402

NOTE = ROOT / "notes/N15_T8.md"
OUT = CACHE / "n15_t8.json"
EXTRA = {"WATERZ_STICKY_SZ0": "1"}


def main():
    print("N15 T8 sticky sz0. Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N15 T8 REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)
    os.environ.update(EXTRA)
    compile_d()
    env = parks_env(EXTRA)
    vr = subprocess.run(
        [sys.executable, str(ROOT / "scripts/n15_agg_gate.py")],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    )
    voi = load_json_line(vr.stdout)
    print(f"N15 T8 voi {voi} stderr_tail={vr.stderr[-800:]}", flush=True)
    voi_ok = bool(voi.get("ok") and voi.get("run2_array_equal"))
    run216, agg, rc216 = {}, 0.0, None
    if voi_ok:
        r216 = subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts/n8_run216.py")],
            cwd=str(ROOT), env=parks_env({**EXTRA, "WATERZ_SKIP_BUILD": "1"}),
        )
        rc216 = r216.returncode
        p216 = CACHE / "n8_216.json"
        if p216.is_file():
            run216 = json.loads(p216.read_text())
        agg = float((run216.get("stages") or {}).get("agg") or 0)
    else:
        print("N15 T8 skip 2.16 (VOI or 2-run failed)", flush=True)
    cut = (AGG_BASE / agg) if agg > 0 else 0.0
    keep = bool(voi_ok and rc216 == 0 and agg > 0 and agg <= AGG_GATE)
    doc = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "exp": "T8 sticky sz0",
        "voi": voi,
        "run216": run216,
        "agg_ms": agg,
        "agg_base": AGG_BASE,
        "gate_ms": AGG_GATE,
        "cut": cut,
        "keep_default": keep,
        "rc216": rc216,
        "skipped_216": not voi_ok,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    m1 = voi.get("m1") or {}
    NOTE.write_text(
        "# N15 exp8 sticky sz0\n\n"
        "Not a 2 Gvox/s claim. Not 3090 Ti. WATERZ_STICKY_SZ0=1. ε=0.40.\n\n"
        f"- voi ok={voi.get('ok')} run2={voi.get('run2_array_equal')} "
        f"split={m1.get('split')} merge={m1.get('merge')} "
        f"inner={m1.get('inner')} merges={m1.get('merges')}\n"
        f"- 2.16 skipped={doc['skipped_216']} agg={agg:.1f} vs base {AGG_BASE:.1f} "
        f"gate {AGG_GATE:.1f} cut={cut:.3f} rc216={rc216}\n"
        f"- keep_default={keep}\n"
    )
    print(f"N15 T8 keep_default={keep} voi={voi_ok} agg={agg:.1f} -> {OUT}",
          flush=True)
    return 0 if voi_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
