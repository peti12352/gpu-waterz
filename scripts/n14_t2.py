#!/usr/bin/env python3
"""N14 T2: path-halving inside k_w5_compress_list. WATERZ_LIST_HALVING=1.

Identity vs wz_fragments.npy. 2.16 WS vs N13 2579/1.2. Default stays off
unless keep_default. Not a 2 Gvox/s claim. Not 3090 Ti.
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
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402
from b_dev_aff import build  # noqa: E402

NOTE = ROOT / "notes/N14_T2.md"
OUT = CACHE / "n14_t2.json"
WS_BASE = 2579.0453841909766
GATE = WS_BASE / 1.2


def main():
    print("N14 T2 list halving. Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N14 T2 REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)
    extra = {"WATERZ_LIST_HALVING": "1"}
    os.environ["WATERZ_LIST_HALVING"] = "1"
    os.environ.pop("WATERZ_FOLD_FLATTEN", None)
    build()
    env = parks_env(extra)
    env.pop("WATERZ_FOLD_FLATTEN", None)
    idr = subprocess.run(
        [sys.executable, str(ROOT / "scripts/n13_baseline.py"), "--ident"],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    )
    ident_doc = load_json_line(idr.stdout)
    print(f"N14 T2 ident {ident_doc} stderr_tail={idr.stderr[-500:]}", flush=True)
    r216 = subprocess.run(
        [sys.executable, "-u", str(ROOT / "scripts/n8_run216.py")],
        cwd=str(ROOT), env=parks_env({**extra, "WATERZ_SKIP_BUILD": "1"}),
    )
    run216 = {}
    p216 = CACHE / "n8_216.json"
    if p216.is_file():
        run216 = json.loads(p216.read_text())
    ws = float((run216.get("stages") or {}).get("ws") or 0)
    ident_ok = bool(ident_doc.get("identity"))
    cut = (WS_BASE / ws) if ws > 0 else 0.0
    keep = bool(ident_ok and r216.returncode == 0 and ws > 0 and ws <= GATE)
    doc = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "ident": ident_doc,
        "run216": run216,
        "ws_ms": ws,
        "ws_base": WS_BASE,
        "gate_ms": GATE,
        "cut": cut,
        "keep_default": keep,
        "rc216": r216.returncode,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        "# N14 T2 list path-halving\n\n"
        "Not a 2 Gvox/s claim. Not 3090 Ti. WATERZ_LIST_HALVING=1. "
        "Not UF_ALGO=1. Default UF=3 unchanged unless keep_default.\n\n"
        f"- ident identity={ident_doc.get('identity')} nfrag={ident_doc.get('nfrag')} "
        f"bg={ident_doc.get('bg')}\n"
        f"- 2.16 WS={ws:.1f} ms vs base {WS_BASE:.1f} gate {GATE:.1f} cut={cut:.3f}\n"
        f"- keep_default={keep} (need ident and WS <= {GATE:.1f})\n"
    )
    print(f"N14 T2 keep_default={keep} ws={ws:.1f} -> {OUT}", flush=True)
    return 0 if ident_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
