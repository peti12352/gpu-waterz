#!/usr/bin/env python3
"""N15 legal stack: T2+T5 WS and T6 fuse dirty.

T8 sticky excluded (four-T FAIL). Not a 2 Gvox/s claim. Not 3090 Ti.
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
from n15_agg_gate import AGG_BASE, AGG_GATE  # noqa: E402
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402
from b_dev_aff import build  # noqa: E402
from e6r_parhac import compile_d  # noqa: E402

NOTE = ROOT / "notes/N15_STACK.md"
OUT = CACHE / "n15_stack.json"
EXTRA = {
    "WATERZ_FOLD_FLATTEN": "1",
    "WATERZ_SHARE_OFF": "1",
    "WATERZ_HOOK_ROOT": "1",
    "WATERZ_FUSE_DIRTY": "1",
}
E2E_BASE = 4915.3


def main():
    print("N15 T2T5+T6 stack. Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N15 STACK REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)
    os.environ.update(EXTRA)
    build()
    compile_d()
    env = parks_env(EXTRA)
    idr = subprocess.run(
        [sys.executable, str(ROOT / "scripts/n15_gate.py")],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    )
    ident_doc = load_json_line(idr.stdout)
    print(f"N15 STACK ident {ident_doc}", flush=True)
    vr = subprocess.run(
        [sys.executable, str(ROOT / "scripts/n15_agg_gate.py")],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    )
    voi = load_json_line(vr.stdout)
    print(f"N15 STACK voi {voi} stderr_tail={vr.stderr[-400:]}", flush=True)
    ident_ok = bool(ident_doc.get("identity") and ident_doc.get("run2_array_equal"))
    voi_ok = bool(voi.get("ok") and voi.get("run2_array_equal"))
    run216, ws, agg, e2e, rc216 = {}, 0.0, 0.0, 0.0, None
    if ident_ok and voi_ok:
        r216 = subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts/n8_run216.py")],
            cwd=str(ROOT), env=parks_env({**EXTRA, "WATERZ_SKIP_BUILD": "1"}),
        )
        rc216 = r216.returncode
        p216 = CACHE / "n8_216.json"
        if p216.is_file():
            run216 = json.loads(p216.read_text())
        ws = float((run216.get("stages") or {}).get("ws") or 0)
        agg = float((run216.get("stages") or {}).get("agg") or 0)
        e2e = float(run216.get("e2e_ms") or 0)
    keep = bool(
        ident_ok and voi_ok and rc216 == 0
        and ws > 0 and ws <= WS_GATE
        and agg > 0 and agg <= AGG_GATE
    )
    doc = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "ident": ident_doc,
        "voi": voi,
        "run216": run216,
        "ws_ms": ws,
        "agg_ms": agg,
        "e2e_ms": e2e,
        "ws_cut": (WS_BASE / ws) if ws else 0,
        "agg_cut": (AGG_BASE / agg) if agg else 0,
        "e2e_cut": (E2E_BASE / e2e) if e2e else 0,
        "keep_default": keep,
        "rc216": rc216,
        "nlab": run216.get("nlab"),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        "# N15 legal stack T2+T5+T6\n\n"
        "Not a 2 Gvox/s claim. Not 3090 Ti. "
        "FOLD_FLATTEN=1 SHARE_OFF=1 HOOK_ROOT=1 FUSE_DIRTY=1. "
        "T8 excluded (four-T FAIL).\n\n"
        f"- ident={ident_doc.get('identity')} run2={ident_doc.get('run2_array_equal')} "
        f"nfrag={ident_doc.get('nfrag')} peak={ident_doc.get('peak_bytes')}\n"
        f"- voi ok={voi.get('ok')} run2={voi.get('run2_array_equal')}\n"
        f"- 2.16 WS={ws:.1f} (base {WS_BASE:.1f} cut {doc['ws_cut']:.3f}) "
        f"agg={agg:.1f} (base {AGG_BASE:.1f} cut {doc['agg_cut']:.3f}) "
        f"e2e={e2e:.1f} (base {E2E_BASE:.1f} cut {doc['e2e_cut']:.3f}) "
        f"nlab={run216.get('nlab')} rc={rc216}\n"
        f"- keep_default={keep} (need WS<=2149 AND agg<=1862; T6 agg alone misses 1.2x)\n"
        "- C++ defaults stay off: share_off is +4 B/vox; 3090 24GB peak not measured\n"
    )
    print(f"N15 STACK keep={keep} ws={ws:.1f} agg={agg:.1f} e2e={e2e:.1f} -> {OUT}",
          flush=True)
    return 0 if ident_ok and voi_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
