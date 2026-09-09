#!/usr/bin/env python3
"""N15 stack: T2 fold+share-off AND T5 hook-to-root.

Each passed ident and WS 1.2x alone. Composition is not additive until timed.
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

NOTE = ROOT / "notes/N15_T2T5.md"
OUT = CACHE / "n15_t2t5.json"
EXTRA = {
    "WATERZ_FOLD_FLATTEN": "1",
    "WATERZ_SHARE_OFF": "1",
    "WATERZ_HOOK_ROOT": "1",
}


def main():
    print("N15 T2+T5 stack. Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N15 T2T5 REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)
    os.environ.update(EXTRA)
    build()
    env = parks_env(EXTRA)
    idr = subprocess.run(
        [sys.executable, str(ROOT / "scripts/n15_gate.py")],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    )
    ident_doc = load_json_line(idr.stdout)
    print(f"N15 T2T5 ident {ident_doc} stderr_tail={idr.stderr[-800:]}", flush=True)
    ident_ok = bool(ident_doc.get("identity"))
    det_ok = bool(ident_doc.get("run2_array_equal"))
    run216, ws, rc216 = {}, 0.0, None
    if ident_ok and det_ok:
        r216 = subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts/n8_run216.py")],
            cwd=str(ROOT), env=parks_env({**EXTRA, "WATERZ_SKIP_BUILD": "1"}),
        )
        rc216 = r216.returncode
        p216 = CACHE / "n8_216.json"
        if p216.is_file():
            run216 = json.loads(p216.read_text())
        ws = float((run216.get("stages") or {}).get("ws") or 0)
    else:
        print("N15 T2T5 skip 2.16 (identity or 2-run failed)", flush=True)
    cut = (WS_BASE / ws) if ws > 0 else 0.0
    keep = bool(ident_ok and det_ok and rc216 == 0 and ws > 0 and ws <= WS_GATE)
    doc = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "exp": "T2+T5 fold+share-off+hook-root",
        "ident": ident_doc,
        "run216": run216,
        "ws_ms": ws,
        "ws_base": WS_BASE,
        "t2_ws": 1785.3229460306466,
        "t5_ws": 2105.3744489327073,
        "gate_ms": WS_GATE,
        "cut": cut,
        "keep_default": keep,
        "rc216": rc216,
        "skipped_216": not (ident_ok and det_ok),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        "# N15 T2+T5 stack\n\n"
        "Not a 2 Gvox/s claim. Not 3090 Ti. "
        "WATERZ_FOLD_FLATTEN=1 SHARE_OFF=1 HOOK_ROOT=1.\n\n"
        f"- ident identity={ident_doc.get('identity')} "
        f"run2={ident_doc.get('run2_array_equal')} "
        f"nfrag={ident_doc.get('nfrag')} bg={ident_doc.get('bg')} "
        f"ndiff={ident_doc.get('ndiff_raw')} peak={ident_doc.get('peak_bytes')}\n"
        f"- 2.16 skipped={doc['skipped_216']} WS={ws:.1f} vs base {WS_BASE:.1f} "
        f"T2={1785.3:.1f} T5={2105.4:.1f} gate {WS_GATE:.1f} cut={cut:.3f} "
        f"rc216={rc216}\n"
        f"- keep_default={keep} (need ident, 2-run, WS <= {WS_GATE:.1f})\n"
        "- C++ product defaults stay off until 2.16 peak vs 3090 24GB is logged\n"
    )
    print(f"N15 T2T5 keep_default={keep} ident={ident_ok} ws={ws:.1f} -> {OUT}",
          flush=True)
    return 0 if ident_ok and det_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
