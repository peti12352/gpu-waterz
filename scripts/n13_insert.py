#!/usr/bin/env python3
"""N13 L4: grade WATERZ_HASH_INSERT_ONLY. Never default on FAIL.

Not a 2 Gvox/s claim. Not 3090 Ti. Idle-5090 only.
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
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402

NOTE = ROOT / "notes/N13_INSERT.md"
OUT = CACHE / "n13_insert.json"
WARM_VAL_AGG = 211.0


def parks_env(extra=None):
    e = os.environ.copy()
    e["WATERZ_UF_ALGO"] = "3"
    e["WATERZ_HOST_PARK"] = "0"
    e["WATERZ_AFF_PARK"] = "0"
    e["WATERZ_AGG_LEVERS"] = "15"
    e.pop("WATERZ_AGG_EPS", None)
    e.pop("WATERZ_PAPER_E6T", None)
    if extra:
        e.update(extra)
    return e


def load_json_line(stdout):
    for ln in stdout.splitlines():
        if ln.startswith("{"):
            return json.loads(ln)
    return {"raw": stdout[-1500:]}


def main():
    print("Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N13 L4 REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)
    r = subprocess.run(
        [sys.executable, "-u", str(ROOT / "scripts/n12_compact.py"),
         "--insert-only"],
        cwd=str(ROOT), env=parks_env(), capture_output=True, text=True,
    )
    row = load_json_line(r.stdout)
    print(
        f"N13 L4 insert-only {row} rc={r.returncode} "
        f"stderr_tail={r.stderr[-600:].replace(chr(10), ' | ')}",
        flush=True,
    )
    ms = float(row.get("ms") or 0)
    ok = bool(row.get("ok"))
    cut = (WARM_VAL_AGG / ms) if ms > 0 else None
    keep = bool(ok and cut is not None and cut >= 1.2)
    doc = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "gpu": gpu_state(),
        "insert_only": row,
        "warm_val_agg_ms": WARM_VAL_AGG,
        "wall_cut_vs_211": cut,
        "keep_default": keep,
        "rc": r.returncode,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        "# N13 L4 HASH_INSERT_ONLY\n\n"
        "Not a 2 Gvox/s claim. Not 3090 Ti.\n\n"
        f"- ms={row.get('ms')} split={row.get('split')} merge={row.get('merge')} "
        f"nseg={row.get('nseg')} voi_ok={ok}\n"
        f"- vs warm val agg 211 ms cut={cut} (need >=1.2 and VOI PASS to default)\n"
        f"- keep_default={keep}\n"
    )
    print(f"N13 L4 keep={keep} -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
