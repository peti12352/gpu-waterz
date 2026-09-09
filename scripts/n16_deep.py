#!/usr/bin/env python3
"""N16 deep verify of legal stack: four-T, 2-run 2.16 label hash, WS peak.

Env: T2+T5+T6 and T7 if WATERZ_NLIVE_ARITH already proven. Default includes T7
so this script is run AFTER n16_t7.py. Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import ctypes
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from n13_baseline import load_json_line, parks_env  # noqa: E402
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402
from n8_run216 import official_216  # noqa: E402
from b_dev_aff import build  # noqa: E402
from e6r_parhac import compile_d  # noqa: E402
import segment as S  # noqa: E402

NOTE = ROOT / "notes/N16_DEEP.md"
OUT = CACHE / "n16_deep.json"
EXTRA = {
    "WATERZ_FOLD_FLATTEN": "1",
    "WATERZ_SHARE_OFF": "1",
    "WATERZ_HOOK_ROOT": "1",
    "WATERZ_FUSE_DIRTY": "1",
    "WATERZ_NLIVE_ARITH": "1",
}
E2E_BASE = 4915.3
WS_BASE = 2579.0453841909766
AGG_BASE = 2234.0503642335534


def _peak():
    lib = ctypes.CDLL(str(ROOT / "src/libws_gpu.so"))
    lib.ws_mem_peak.restype = ctypes.c_size_t
    return int(lib.ws_mem_peak())


def hash_216(env):
    aff = official_216()
    aff_d = S.DevBuf.from_host(aff)
    del aff
    out, ms = S.cuda_event_time(
        lambda: S.segment_d(aff_d, [0.3], return_device=True))
    st = dict(S.STAGE_MS)
    peak = _peak()
    h = None
    nlab = 0
    if out:
        host = out[0].to_host()
        nlab = int(np.unique(host).size)
        h = hashlib.sha256(host.tobytes()).hexdigest()
        for b in out:
            b.free()
        del host
    aff_d.free()
    return {
        "e2e_ms": float(ms),
        "stages": st,
        "nlab": nlab,
        "sha256": h,
        "ws_peak_bytes": peak,
    }


def main():
    print("N16 DEEP stack verify. Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N16 DEEP REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)
    os.environ.update(EXTRA)
    build()
    compile_d()
    env = parks_env(EXTRA)
    ident = load_json_line(subprocess.run(
        [sys.executable, str(ROOT / "scripts/n15_gate.py")],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    ).stdout)
    print(f"N16 DEEP ident {ident}", flush=True)
    voi = load_json_line(subprocess.run(
        [sys.executable, str(ROOT / "scripts/n15_agg_gate.py")],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    ).stdout)
    print(f"N16 DEEP voi {voi}", flush=True)
    four = load_json_line(subprocess.run(
        [sys.executable, str(ROOT / "scripts/n15_four.py")],
        cwd=str(ROOT), env=parks_env({**EXTRA, "WATERZ_FOUR_TAG": "n16_deep_four",
                                     "WATERZ_SKIP_BUILD": "1"}),
        capture_output=True, text=True,
    ).stdout)
    print(f"N16 DEEP four {four}", flush=True)
    ident_ok = bool(ident.get("identity") and ident.get("run2_array_equal"))
    voi_ok = bool(voi.get("ok") and voi.get("run2_array_equal"))
    four_ok = bool(four.get("four_pass"))
    r1, r2 = {}, {}
    if ident_ok and voi_ok and four_ok:
        os.environ.update(parks_env({**EXTRA, "WATERZ_SKIP_BUILD": "1",
                                     "WATERZ_STAGE_MS": "1",
                                     "WATERZ_AGG_LEVERS": "15"}))
        r1 = hash_216(None)
        print(f"N16 DEEP 2.16a {r1}", flush=True)
        r2 = hash_216(None)
        print(f"N16 DEEP 2.16b {r2}", flush=True)
    det216 = bool(r1.get("sha256") and r1.get("sha256") == r2.get("sha256"))
    ws = float((r1.get("stages") or {}).get("ws") or 0)
    agg = float((r1.get("stages") or {}).get("agg") or 0)
    e2e = float(r1.get("e2e_ms") or 0)
    peak = int(r1.get("ws_peak_bytes") or 0)
    keep = bool(
        ident_ok and voi_ok and four_ok and det216
        and ws > 0 and ws <= WS_BASE / 1.2
        and agg > 0 and agg <= AGG_BASE / 1.2
    )
    doc = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "ident": ident, "voi": voi, "four": four,
        "run_a": r1, "run_b": r2, "det_216": det216,
        "ws_ms": ws, "agg_ms": agg, "e2e_ms": e2e,
        "ws_peak_bytes": peak,
        "ws_peak_gib": peak / 1073741824.0,
        "keep_default": keep,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        "# N16 deep stack verify\n\n"
        "Not a 2 Gvox/s claim. Not 3090 Ti. "
        "FOLD+SHARE_OFF+HOOK_ROOT+FUSE_DIRTY+NLIVE_ARITH.\n\n"
        f"- ident={ident.get('identity')} run2={ident.get('run2_array_equal')} "
        f"peak_val={ident.get('peak_bytes')}\n"
        f"- voi ok={voi.get('ok')} run2={voi.get('run2_array_equal')}\n"
        f"- four_pass={four.get('four_pass')}\n"
        f"- 2.16 det sha equal={det216} a={r1.get('sha256')} b={r2.get('sha256')}\n"
        f"- 2.16a e2e={e2e:.1f} WS={ws:.1f} agg={agg:.1f} nlab={r1.get('nlab')} "
        f"ws_peak={peak} ({peak/1073741824.0:.2f} GiB)\n"
        f"- keep_default={keep} (need four-T + 2.16 det + WS 1.2x AND agg 1.2x)\n"
        "- C++ defaults stay off unless keep_default and 3090 24GB fits peak\n"
    )
    print(f"N16 DEEP keep={keep} det216={det216} e2e={e2e:.1f} peak={peak} -> {OUT}",
          flush=True)
    return 0 if ident_ok and voi_ok and four_ok and det216 else 1


if __name__ == "__main__":
    raise SystemExit(main())
