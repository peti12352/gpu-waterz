#!/usr/bin/env python3
"""N17 stacked-gate runner. Identity / T=0.3 VOI 2-run / four-T / 2.16.

Not a 2 Gvox/s claim. Not 3090 Ti. Caller sets EXTRA (process-cached).
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
from n15_gate import WS_BASE, WS_GATE  # noqa: E402
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402
from b_dev_aff import build  # noqa: E402
from e6r_parhac import compile_d  # noqa: E402

DEEP = {
    "WATERZ_FOLD_FLATTEN": "1",
    "WATERZ_SHARE_OFF": "1",
    "WATERZ_HOOK_ROOT": "1",
    "WATERZ_FUSE_DIRTY": "1",
    "WATERZ_NLIVE_ARITH": "1",
}
N13_E2E = 4915.3
DEEP_AGG = 1928.9960050955415
DEEP_WS = 1312.015887349844


def run_stack(name: str, extra: dict, four_tag: str, need_ident: bool = True):
    print(f"N17 {name}. Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N17 {name} REFUSE card busy: {busy}", flush=True)
        return 2, {}
    print(f"GPU idle: {gpu_state()}", flush=True)
    os.environ.update(extra)
    compile_d()
    if need_ident:
        build()
    env = parks_env(extra)
    ident = {}
    ident_ok = True
    if need_ident:
        ident = load_json_line(subprocess.run(
            [sys.executable, str(ROOT / "scripts/n15_gate.py")],
            cwd=str(ROOT), env=env, capture_output=True, text=True,
        ).stdout)
        print(f"N17 {name} ident {ident}", flush=True)
        ident_ok = bool(ident.get("identity") and ident.get("run2_array_equal"))
    voi = {}
    voi_ok = False
    if ident_ok:
        voi = load_json_line(subprocess.run(
            [sys.executable, str(ROOT / "scripts/n15_agg_gate.py")],
            cwd=str(ROOT), env=env, capture_output=True, text=True,
        ).stdout)
        print(f"N17 {name} voi {voi}", flush=True)
        voi_ok = bool(voi.get("ok") and voi.get("run2_array_equal"))
    four = {}
    four_ok = False
    if voi_ok:
        four = load_json_line(subprocess.run(
            [sys.executable, str(ROOT / "scripts/n15_four.py")],
            cwd=str(ROOT), env=parks_env({**extra, "WATERZ_FOUR_TAG": four_tag,
                                         "WATERZ_SKIP_BUILD": "1"}),
            capture_output=True, text=True,
        ).stdout)
        print(f"N17 {name} four {four}", flush=True)
        four_ok = bool(four.get("four_pass"))
    run216, agg, ws, rc216 = {}, 0.0, 0.0, None
    if ident_ok and voi_ok and four_ok:
        r = subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts/n8_run216.py")],
            cwd=str(ROOT), env=parks_env({**extra, "WATERZ_SKIP_BUILD": "1"}),
        )
        rc216 = r.returncode
        p = CACHE / "n8_216.json"
        if p.is_file():
            run216 = json.loads(p.read_text())
        st = run216.get("stages") or {}
        agg = float(st.get("agg") or 0)
        ws = float(st.get("ws") or 0)
    cut_agg = (AGG_BASE / agg) if agg else 0.0
    cut_ws = (WS_BASE / ws) if ws else 0.0
    keep = bool(
        ident_ok and voi_ok and four_ok and rc216 == 0
        and agg > 0 and agg <= AGG_GATE
        and (not need_ident or (ws > 0 and ws <= WS_GATE))
    )
    doc = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "ident": ident, "voi": voi, "four": four, "run216": run216,
        "agg_ms": agg, "ws_ms": ws, "cut_agg": cut_agg, "cut_ws": cut_ws,
        "agg_gate": AGG_GATE, "deep_agg": DEEP_AGG, "deep_ws": DEEP_WS,
        "keep_default": keep, "rc216": rc216, "env": extra,
    }
    return (0 if ident_ok and voi_ok and four_ok else 1), doc
