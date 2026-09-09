#!/usr/bin/env python3
"""N16 T11 stitch arena. WATERZ_STITCH_ARENA=1 plus T2+T5 WS stack.

Identity 2-run then 2.16 WS vs 1313 (stacked) and 2579 (N13).
Not a 2 Gvox/s claim. Not 3090 Ti.
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
from n15_gate import WS_BASE, WS_GATE  # noqa: E402
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402
from b_dev_aff import build  # noqa: E402

NOTE = ROOT / "notes/N16_T11.md"
OUT = CACHE / "n16_t11.json"
T2T5_WS = 1311.4086668938398
EXTRA = {
    "WATERZ_STITCH_ARENA": "1",
    "WATERZ_FOLD_FLATTEN": "1",
    "WATERZ_SHARE_OFF": "1",
    "WATERZ_HOOK_ROOT": "1",
}


def main():
    print("N16 T11 stitch arena. Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N16 T11 REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)
    os.environ.update(EXTRA)
    build()
    env = parks_env(EXTRA)
    idr = subprocess.run(
        [sys.executable, str(ROOT / "scripts/n15_gate.py")],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    )
    ident = load_json_line(idr.stdout)
    print(f"N16 T11 ident {ident} stderr_tail={idr.stderr[-500:]}", flush=True)
    ok = bool(ident.get("identity") and ident.get("run2_array_equal"))
    run216, ws, rc216 = {}, 0.0, None
    if ok:
        r216 = subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts/n8_run216.py")],
            cwd=str(ROOT), env=parks_env({**EXTRA, "WATERZ_SKIP_BUILD": "1"}),
        )
        rc216 = r216.returncode
        p216 = CACHE / "n8_216.json"
        if p216.is_file():
            run216 = json.loads(p216.read_text())
        ws = float((run216.get("stages") or {}).get("ws") or 0)
    cut_n13 = (WS_BASE / ws) if ws else 0.0
    cut_st = (T2T5_WS / ws) if ws else 0.0
    keep = bool(ok and rc216 == 0 and ws > 0 and ws <= WS_GATE and cut_st >= 1.2)
    doc = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "ident": ident, "run216": run216, "ws_ms": ws,
        "t2t5_ws": T2T5_WS, "cut_vs_n13": cut_n13, "cut_vs_t2t5": cut_st,
        "keep_default": keep, "rc216": rc216,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        "# N16 T11 stitch arena\n\n"
        "Not a 2 Gvox/s claim. Not 3090 Ti. "
        "WATERZ_STITCH_ARENA=1 + T2+T5.\n\n"
        f"- ident={ident.get('identity')} run2={ident.get('run2_array_equal')} "
        f"nfrag={ident.get('nfrag')} peak={ident.get('peak_bytes')}\n"
        f"- 2.16 WS={ws:.1f} vs T2T5 {T2T5_WS:.1f} cut={cut_st:.3f} "
        f"vs N13 {WS_BASE:.1f} cut={cut_n13:.3f} nlab={run216.get('nlab')}\n"
        f"- keep_default={keep} (need extra 1.2x vs already-stacked T2T5)\n"
    )
    print(f"N16 T11 keep={keep} ident={ok} ws={ws:.1f} -> {OUT}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
