#!/usr/bin/env python3
"""N16 T12 pinned changed. WATERZ_PIN_CHANGED=1 + T2+T5."""
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

NOTE = ROOT / "notes/N16_T12.md"
OUT = CACHE / "n16_t12.json"
T2T5_WS = 1311.4086668938398
EXTRA = {
    "WATERZ_PIN_CHANGED": "1",
    "WATERZ_FOLD_FLATTEN": "1",
    "WATERZ_SHARE_OFF": "1",
    "WATERZ_HOOK_ROOT": "1",
}


def main():
    print("N16 T12 pin changed. Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N16 T12 REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)
    os.environ.update(EXTRA)
    build()
    ident = load_json_line(subprocess.run(
        [sys.executable, str(ROOT / "scripts/n15_gate.py")],
        cwd=str(ROOT), env=parks_env(EXTRA), capture_output=True, text=True,
    ).stdout)
    print(f"N16 T12 ident {ident}", flush=True)
    ok = bool(ident.get("identity") and ident.get("run2_array_equal"))
    run216, ws, rc216 = {}, 0.0, None
    if ok:
        r = subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts/n8_run216.py")],
            cwd=str(ROOT), env=parks_env({**EXTRA, "WATERZ_SKIP_BUILD": "1"}),
        )
        rc216 = r.returncode
        p = CACHE / "n8_216.json"
        if p.is_file():
            run216 = json.loads(p.read_text())
        ws = float((run216.get("stages") or {}).get("ws") or 0)
    cut_st = (T2T5_WS / ws) if ws else 0.0
    keep = bool(ok and rc216 == 0 and ws > 0 and ws <= WS_GATE and cut_st >= 1.2)
    doc = {"claim": "not a 2 Gvox/s number; not 3090 Ti", "ident": ident,
           "run216": run216, "ws_ms": ws, "cut_vs_t2t5": cut_st,
           "keep_default": keep, "rc216": rc216}
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        "# N16 T12 pinned changed\n\n"
        "Not a 2 Gvox/s claim. Not 3090 Ti. WATERZ_PIN_CHANGED=1 + T2+T5.\n\n"
        f"- ident={ident.get('identity')} run2={ident.get('run2_array_equal')}\n"
        f"- 2.16 WS={ws:.1f} vs T2T5 {T2T5_WS:.1f} cut={cut_st:.3f} "
        f"nlab={run216.get('nlab')}\n"
        f"- keep_default={keep}\n"
    )
    print(f"N16 T12 keep={keep} ws={ws:.1f} -> {OUT}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
