#!/usr/bin/env python3
"""N14 T4: fused 2.16 OOM check. WATERZ_Z_SLAB=0. Not a closer.

Cannot ship for 3090 24 GB. One attempt. OOM or no 1.2x -> stop.
Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from n13_baseline import parks_env  # noqa: E402
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402

NOTE = ROOT / "notes/N14_T4.md"
OUT = CACHE / "n14_t4.json"
E2E_BASE = 4915.3


def main():
    print("N14 T4 fused 2.16. Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N14 T4 REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)
    env = parks_env({
        "WATERZ_Z_SLAB": "0",
        "WATERZ_SKIP_BUILD": "1",
    })
    env.pop("WATERZ_FOLD_FLATTEN", None)
    env.pop("WATERZ_LIST_HALVING", None)
    r = subprocess.run(
        [sys.executable, "-u", str(ROOT / "scripts/n8_run216.py")],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    )
    run216 = {}
    p216 = CACHE / "n8_216.json"
    oom = ("out of memory" in (r.stderr + r.stdout).lower()
           or "oom" in (r.stderr + r.stdout).lower()
           or r.returncode != 0)
    if p216.is_file() and r.returncode == 0:
        run216 = json.loads(p216.read_text())
    e2e = float(run216.get("e2e_ms") or 0)
    keep = bool(r.returncode == 0 and e2e > 0 and e2e <= E2E_BASE / 1.2)
    doc = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti; fused cannot ship 3090 24GB",
        "gpu": gpu_state(),
        "rc": r.returncode,
        "oom": oom,
        "e2e_ms": e2e,
        "keep_default": keep,
        "stdout_tail": r.stdout[-1500:],
        "stderr_tail": r.stderr[-1500:],
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        "# N14 T4 fused 2.16 OOM check\n\n"
        "Not a 2 Gvox/s claim. Not 3090 Ti. WATERZ_Z_SLAB=0. "
        "Cannot ship for 3090 Ti 24 GB even if 5090 32 GB fits.\n\n"
        f"- rc={r.returncode} oom={oom} e2e={e2e:.1f} keep_default={keep}\n"
        f"- stderr_tail={r.stderr[-400:]!r}\n"
    )
    print(f"N14 T4 oom={oom} rc={r.returncode} e2e={e2e:.1f} keep={keep}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
