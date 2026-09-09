#!/usr/bin/env python3
"""N12 3090 Ti median-of-5. Do not run until that card is rented.

Asserts nvidia-smi name contains '3090 Ti'. Same env as N11 parks-off memlog.
Not a 2 Gvox/s claim until this script's e2e is measured on that card.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402

OUT = CACHE / "n12_3090.json"
NVOX = 2_160_000_000
N = 5


def gpu_name():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    )
    return r.stdout.strip()


def main():
    print("N12 3090 Ti. Not a 5090 number.", flush=True)
    name = gpu_name()
    print(f"nvidia-smi name={name!r}", flush=True)
    if "3090 Ti" not in name:
        print(f"N12 3090 REFUSE name {name!r} is not 3090 Ti", flush=True)
        return 2
    busy = card_busy()
    if busy:
        print(f"N12 REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)
    os.environ["WATERZ_UF_ALGO"] = "3"
    os.environ["WATERZ_HOST_PARK"] = "0"
    os.environ["WATERZ_AFF_PARK"] = "0"
    os.environ["WATERZ_STAGE_MS"] = "1"
    os.environ["WATERZ_AGG_LEVERS"] = "15"
    os.environ["WATERZ_FATBIN"] = "1"
    os.environ.pop("WATERZ_AGG_EPS", None)
    times = []
    last = None
    for i in range(N):
        os.environ.pop("WATERZ_SKIP_BUILD", None)
        if i > 0:
            os.environ["WATERZ_SKIP_BUILD"] = "1"
        r = subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts/n8_run216.py")],
            cwd=str(ROOT), check=False,
        )
        if r.returncode != 0:
            print(f"N12 3090 run {i} rc={r.returncode}", flush=True)
            return r.returncode
        last = json.loads((CACHE / "n8_216.json").read_text())
        times.append(float(last["e2e_ms"]))
        print(f"N12 3090 [{i}] e2e={times[-1]:.1f} ms", flush=True)
    times.sort()
    med = statistics.median(times)
    gvox = (NVOX / 1e9) / (med / 1000.0) if med > 0 else 0.0
    doc = {
        "card": name,
        "n": N,
        "e2e_ms": times,
        "median_ms": med,
        "gvox_s": gvox,
        "last_stages": last.get("stages") if last else None,
        "host_ram_gb_needed": 64,
        "gpu_gb_needed": 24,
        "claim": "3090 Ti median-of-5; still not a 2 Gvox/s claim unless median<=1080",
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"N12 3090 median={med:.1f} ms {gvox:.3f} Gvox/s -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
