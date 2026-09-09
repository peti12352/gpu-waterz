#!/usr/bin/env python3
"""N12 W5 tile A/B: 16x16x32 and 8x32x32 vs 8x16x32. Identity + 2.16 WS.

Kill as default if identity breaks or 2.16 WS not <=0.85x of 2582.
Keep both binaries. Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from b_dev_aff import WS, build  # noqa: E402
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402

NOTE = ROOT / "notes/N12_TILE.md"
OUT = CACHE / "n12_tile.json"
BASE_WS_MS = 2582.0
SHAPES = [
    ("8x16x32", ["-DW5_TZ=8", "-DW5_TY=16", "-DW5_TX=32"]),
    ("16x16x32", ["-DW5_TZ=16", "-DW5_TY=16", "-DW5_TX=32"]),
    ("8x32x32", ["-DW5_TZ=8", "-DW5_TY=32", "-DW5_TX=32"]),
]


def one(tag, defs):
    dest = ROOT / "src" / f"libws_gpu_{tag.replace('x', '_')}.so"
    env = os.environ.copy()
    os.environ.pop("WATERZ_SKIP_BUILD", None)
    build(extra=defs, out=dest)
    import shutil
    shutil.copy2(dest, WS)
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts/n11_e4.py"), "--algo", "3"],
        cwd=str(ROOT), capture_output=True, text=True,
        env={**env, "WATERZ_UF_ALGO": "3", "WATERZ_HOST_PARK": "0",
             "WATERZ_AFF_PARK": "0", "WATERZ_N11_CHILD": "1"},
    )
    doc = {}
    for ln in r.stdout.splitlines():
        if ln.startswith("{"):
            doc = json.loads(ln)
    doc["tag"] = tag
    doc["rc"] = r.returncode
    if not doc.get("identity"):
        print(f"N12 tile {tag} identity FAIL stderr={r.stderr[-800:]}", flush=True)
        return doc
    env216 = {**env, "WATERZ_UF_ALGO": "3", "WATERZ_HOST_PARK": "0",
              "WATERZ_AFF_PARK": "0", "WATERZ_STAGE_MS": "1",
              "WATERZ_SKIP_BUILD": "1"}
    r2 = subprocess.run(
        [sys.executable, "-u", str(ROOT / "scripts/n8_run216.py")],
        cwd=str(ROOT), env=env216, check=False,
    )
    p = CACHE / "n8_216.json"
    doc["run216"] = json.loads(p.read_text()) if p.is_file() else {"rc": r2.returncode}
    ws = (doc["run216"].get("stages") or {}).get("ws")
    doc["ws_216"] = ws
    doc["ws_ratio"] = (ws / BASE_WS_MS) if ws else None
    doc["keep_default"] = bool(
        doc.get("identity") and ws is not None and ws <= 0.85 * BASE_WS_MS
    )
    print(
        f"N12 tile {tag} identity={doc.get('identity')} stitch={doc.get('stitch_ms')} "
        f"ws216={ws} ratio={doc.get('ws_ratio')} keep={doc['keep_default']}",
        flush=True,
    )
    return doc


def main():
    print("Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N12 REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)
    rows = [one(tag, defs) for tag, defs in SHAPES]
    # restore baseline 8x16x32 as the product binary
    base = ROOT / "src" / "libws_gpu_8_16_32.so"
    if base.is_file():
        import shutil
        shutil.copy2(base, WS)
    keep = next((r for r in rows if r.get("keep_default") and r["tag"] != "8x16x32"), None)
    doc = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "gpu": gpu_state(),
        "rows": rows,
        "default_tile": keep["tag"] if keep else "8x16x32",
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    lines = ["# N12 W5 tile A/B", "", "Not a 2 Gvox/s claim. Not 3090 Ti.", ""]
    for r in rows:
        lines.append(
            f"- {r['tag']} identity={r.get('identity')} stitch={r.get('stitch_ms')} "
            f"ws216={r.get('ws_216')} keep_default={r.get('keep_default')}"
        )
    lines += ["", f"default stays **{doc['default_tile']}**.", ""]
    NOTE.write_text("\n".join(lines) + "\n")
    print(f"N12 tile default={doc['default_tile']} -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
