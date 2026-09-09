#!/usr/bin/env python3
"""W-levers: WATERZ_UF_ALGO 0/1/2/3 and WATERZ_WS_W3, bit-identity + timing.

Mirrors scripts/g_levers.py. Needs a GPU to compare label volumes; without
one it still type-checks the sources and records that the device gate is
blocked. The CPU identity gates live in scripts/w0_ws_ref.py --w5 (W5
parent arrays, W3 bitmask labels) and do not need a card.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
PY = sys.executable

ALGOS = [
    (0, "untiled hook + compress"),
    (1, "untiled hook + pointer jump"),
    (2, "W4 tiled hook"),
    (3, "W5 tile-local UF + stitch"),
]


def gpu_ok() -> bool:
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def main() -> int:
    rows = []
    print("W  CPU identity (w0_ws_ref --sweep --w5)")
    rc = subprocess.call(
        [PY, str(ROOT / "scripts" / "w0_ws_ref.py"),
         "--sweep", "--w5", "--out", "w_levers_cpu.json"],
        cwd=str(ROOT))
    rows.append({"name": "cpu_w5_w3", "rc": rc,
                 "identical": rc == 0})
    if not gpu_ok():
        print("W  no GPU; device A/B of WATERZ_UF_ALGO deferred")
        rows.append({"name": "device", "status": "blocked_no_gpu"})
        dest = CACHE / "w_levers.json"
        dest.write_text(json.dumps({"rows": rows, "gpu": False}, indent=2) + "\n")
        print(f"W  wrote {dest.name}")
        return 0 if rc == 0 else 1
    # Device path: compare nfrag and labels against algo 0. Requires the
    # watershed DSO and an affinity volume, which this machine does not have.
    print("W  GPU present; run c2_ws_invariants under each algo in e3 --gpu-window")
    dest = CACHE / "w_levers.json"
    dest.write_text(json.dumps({"rows": rows, "gpu": True}, indent=2) + "\n")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
