#!/usr/bin/env python3
"""P0: measure only. No lock change.

P0a: per-inner |layer|, |Ec|, merges, us at locked ε=0.08
P0b: S3 reducibility violations (must be 0 or stop T14)
P0c: basin SV parent-change per round after e9b_divide
P0d: unique-bit vs multi-bit after divide
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv/bin/python"
PYC = ROOT / ".venv-cuda/bin/python"
NVCC = "/usr/local/cuda-12.8/bin/nvcc"


def main():
    env = os.environ.copy()
    env["WATERZ_P0"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    print("=== P0a/P0b parhac ε=0.08 ===", flush=True)
    r = subprocess.run(
        [str(PY), str(ROOT / "scripts/e3_paper_parhac.py"), "--eps", "0.08"],
        cwd=str(ROOT),
        env=env,
    )
    print(f"P0a/P0b rc={r.returncode}", flush=True)

    print("=== P0c/P0d WS ===", flush=True)
    py = PYC if PYC.is_file() else PY
    subprocess.check_call(
        [
            NVCC, "-O3", "-arch=sm_120", "--shared", "-Xcompiler", "-fPIC",
            "-I/usr/local/cuda-12.8/targets/x86_64-linux/include",
            "-o", str(ROOT / "src/libws_gpu.so"),
            str(ROOT / "csrc/ws.cu"),
        ]
    )
    r2 = subprocess.run([str(py), str(ROOT / "scripts/p0_ws.py")], cwd=str(ROOT))
    print(f"P0c/P0d rc={r2.returncode}", flush=True)
    if r.returncode != 0 or r2.returncode != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
