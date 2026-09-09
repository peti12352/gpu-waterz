#!/usr/bin/env python3
"""N15 exp6: fuse dirty rewrite holes (drop k_count_u8 + k_list_holes).

WATERZ_FUSE_DIRTY=1. VOI + 2-run parents, then 2.16 agg vs 2234/1.2.
Not a 2 Gvox/s claim. Not 3090 Ti. Default WS (no T2T5 stack).
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
from n15_agg_gate import AGG_BASE, AGG_GATE  # noqa: E402
from n13_baseline import load_json_line  # noqa: E402
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402
from e6r_parhac import compile_d  # noqa: E402

NOTE = ROOT / "notes/N15_T6.md"
OUT = CACHE / "n15_t6.json"
EXTRA = {"WATERZ_FUSE_DIRTY": "1"}


def main():
    print("N15 T6 fuse dirty. Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N15 T6 REFUSE card busy: {busy}", flush=True)
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
    print(f"N15 T6 voi {voi} stderr_tail={vr.stderr[-800:]}", flush=True)
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
        print("N15 T6 skip 2.16 (VOI or 2-run failed)", flush=True)
    cut = (AGG_BASE / agg) if agg > 0 else 0.0
    keep = bool(voi_ok and rc216 == 0 and agg > 0 and agg <= AGG_GATE)
    doc = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "exp": "T6 fuse dirty holes",
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
        "# N15 exp6 fuse dirty\n\n"
        "Not a 2 Gvox/s claim. Not 3090 Ti. WATERZ_FUSE_DIRTY=1. "
        "Default WS. Drop k_count_u8 + k_list_holes; keep k_count_live.\n\n"
        f"- voi ok={voi.get('ok')} run2={voi.get('run2_array_equal')} "
        f"split={m1.get('split')} merge={m1.get('merge')} "
        f"inner={m1.get('inner')} merges={m1.get('merges')}\n"
        f"- 2.16 skipped={doc['skipped_216']} agg={agg:.1f} vs base {AGG_BASE:.1f} "
        f"gate {AGG_GATE:.1f} cut={cut:.3f} rc216={rc216}\n"
        f"- keep_default={keep} (need VOI, 2-run, agg <= {AGG_GATE:.1f})\n"
    )
    print(f"N15 T6 keep_default={keep} voi={voi_ok} agg={agg:.1f} -> {OUT}",
          flush=True)
    return 0 if voi_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
