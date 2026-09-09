#!/usr/bin/env python3
"""N17 T2 deep: 2-run 2.16 sha256 + WS peak. Ident/VOI/four already gated.

WATERZ_EMIT_HOLES=1 + N16 deep env. Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import ctypes
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from n13_baseline import parks_env  # noqa: E402
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402
from n8_run216 import official_216  # noqa: E402
from n17_gate import DEEP  # noqa: E402
import segment as S  # noqa: E402

NOTE = ROOT / "notes/N17_DEEP.md"
OUT = CACHE / "n17_deep.json"
EXTRA = {**DEEP, "WATERZ_EMIT_HOLES": "1"}
WS_BASE = 2579.0453841909766
AGG_BASE = 2234.0503642335534


def _peak():
    lib = ctypes.CDLL(str(ROOT / "src/libws_gpu.so"))
    lib.ws_mem_peak.restype = ctypes.c_size_t
    return int(lib.ws_mem_peak())


def hash_216():
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
    print("N17 DEEP T2 2-run hash. Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N17 DEEP REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)
    os.environ.update(parks_env({**EXTRA, "WATERZ_SKIP_BUILD": "1",
                                 "WATERZ_STAGE_MS": "1",
                                 "WATERZ_AGG_LEVERS": "15"}))
    r1 = hash_216()
    print(f"N17 DEEP 2.16a {r1}", flush=True)
    r2 = hash_216()
    print(f"N17 DEEP 2.16b {r2}", flush=True)
    det216 = bool(r1.get("sha256") and r1.get("sha256") == r2.get("sha256"))
    ws = float((r1.get("stages") or {}).get("ws") or 0)
    agg = float((r1.get("stages") or {}).get("agg") or 0)
    e2e = float(r1.get("e2e_ms") or 0)
    peak = int(r1.get("ws_peak_bytes") or 0)
    keep = bool(
        det216 and r1.get("nlab") == 3860788
        and ws > 0 and ws <= WS_BASE / 1.2
        and agg > 0 and agg <= AGG_BASE / 1.2
    )
    doc = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "run_a": r1, "run_b": r2, "det_216": det216,
        "ws_ms": ws, "agg_ms": agg, "e2e_ms": e2e,
        "ws_peak_bytes": peak,
        "ws_peak_gib": peak / 1073741824.0,
        "keep_default": keep,
        "cpp_default": False,
        "cpp_default_reason": "3090 24GB peak not measured",
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        "# N17 T2 two-run 2.16 labels\n\n"
        "Not a 2 Gvox/s claim. Not 3090 Ti. EMIT_HOLES + N16 deep env.\n\n"
        f"- 2.16 det sha equal={det216} a={r1.get('sha256')} b={r2.get('sha256')}\n"
        f"- 2.16a e2e={e2e:.1f} WS={ws:.1f} agg={agg:.1f} nlab={r1.get('nlab')} "
        f"ws_peak={peak} ({peak/1073741824.0:.2f} GiB)\n"
        f"- 2.16b e2e={float(r2.get('e2e_ms') or 0):.1f} "
        f"agg={float((r2.get('stages') or {}).get('agg') or 0):.1f} "
        f"nlab={r2.get('nlab')}\n"
        f"- keep_default={keep} (2-run det + nlab lock + WS 1.2x AND agg 1.2x)\n"
        "- C++ EMIT_HOLES default stays 0 until 3090 24GB peak is measured\n"
    )
    print(f"N17 DEEP keep={keep} det216={det216} e2e={e2e:.1f} peak={peak} -> {OUT}",
          flush=True)
    return 0 if det216 else 1


if __name__ == "__main__":
    raise SystemExit(main())
