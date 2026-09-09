#!/usr/bin/env python3
"""N16 T15 fuse rebuild+pack. WATERZ_FUSE_PACK=1.

VOI 2-run, four-T, 2.16 agg. Not a 2 Gvox/s claim. Not 3090 Ti.
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

NOTE = ROOT / "notes/N16_T15.md"
OUT = CACHE / "n16_t15.json"
EXTRA = {"WATERZ_FUSE_PACK": "1"}


def main():
    print("N16 T15 fuse pack. Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N16 T15 REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)
    os.environ.update(EXTRA)
    compile_d()
    env = parks_env(EXTRA)
    voi = load_json_line(subprocess.run(
        [sys.executable, str(ROOT / "scripts/n15_agg_gate.py")],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    ).stdout)
    print(f"N16 T15 voi {voi}", flush=True)
    voi_ok = bool(voi.get("ok") and voi.get("run2_array_equal"))
    four = {}
    if voi_ok:
        four = load_json_line(subprocess.run(
            [sys.executable, str(ROOT / "scripts/n15_four.py")],
            cwd=str(ROOT), env=parks_env({**EXTRA, "WATERZ_FOUR_TAG": "n16_t15_four",
                                         "WATERZ_SKIP_BUILD": "1"}),
            capture_output=True, text=True,
        ).stdout)
        print(f"N16 T15 four {four}", flush=True)
    four_ok = bool(four.get("four_pass"))
    run216, agg, rc216 = {}, 0.0, None
    if voi_ok and four_ok:
        r = subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts/n8_run216.py")],
            cwd=str(ROOT), env=parks_env({**EXTRA, "WATERZ_SKIP_BUILD": "1"}),
        )
        rc216 = r.returncode
        p = CACHE / "n8_216.json"
        if p.is_file():
            run216 = json.loads(p.read_text())
        agg = float((run216.get("stages") or {}).get("agg") or 0)
    cut = (AGG_BASE / agg) if agg else 0.0
    keep = bool(voi_ok and four_ok and rc216 == 0 and agg > 0 and agg <= AGG_GATE)
    doc = {"claim": "not a 2 Gvox/s number; not 3090 Ti", "voi": voi, "four": four,
           "run216": run216, "agg_ms": agg, "cut": cut, "keep_default": keep,
           "rc216": rc216}
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    m1 = voi.get("m1") or {}
    NOTE.write_text(
        "# N16 T15 fuse rebuild+pack\n\n"
        "Not a 2 Gvox/s claim. Not 3090 Ti. WATERZ_FUSE_PACK=1.\n\n"
        f"- voi ok={voi.get('ok')} run2={voi.get('run2_array_equal')} "
        f"split={m1.get('split')} merge={m1.get('merge')}\n"
        f"- four_pass={four.get('four_pass')}\n"
        f"- 2.16 agg={agg:.1f} vs {AGG_BASE:.1f} cut={cut:.3f} "
        f"nlab={run216.get('nlab')}\n"
        f"- keep_default={keep}\n"
    )
    print(f"N16 T15 keep={keep} voi={voi_ok} four={four_ok} agg={agg:.1f} -> {OUT}",
          flush=True)
    return 0 if voi_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
