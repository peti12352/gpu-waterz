#!/usr/bin/env python3
"""Prepare GPU ports. Do not launch kernels.

Gates from CPU campaign:
  N20_D1 go_star False -> do not compile/run StarMerge rewrite
  N20_D3 go_gpu_rnn    -> only then would WATERZ_BATCH_RNN be legal
User instruction 2026-09-11: GPU tests prepare only, never run.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
NOTE = ROOT / "notes/N20_GPU_PREP.md"
CU = ROOT / "csrc/n20_rnn_gpu_prep.cu"
CLAIM = "N20 GPU prep only; not a run; not a throughput claim"


def loadj(name):
    p = CACHE / name
    if not p.is_file():
        return {}
    return json.loads(p.read_text())


def main():
    if os.environ.get("N20_GPU_RUN", "0") == "1":
        print("REFUSE: N20_GPU_RUN=1 is forbidden in this campaign", flush=True)
        return 2
    d1 = loadj("N20_D1.json")
    d3 = loadj("N20_D3.json")
    rnn = loadj("N20_RNN.json")
    host = socket.gethostname()
    nvcc = subprocess.run(["which", "nvcc"], capture_output=True, text=True)
    smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    go_star = bool(d1.get("go_star"))
    go_rnn = bool(d3.get("go_gpu_rnn"))
    recipe = {
        "claim": CLAIM,
        "host": host,
        "gpu_run": False,
        "go_star": go_star,
        "go_gpu_rnn": go_rnn,
        "nvcc": (nvcc.stdout or "").strip(),
        "nvidia_smi_L": (smi.stdout or smi.stderr or "").strip().splitlines()[:4],
        "star_action": "do_not_rewrite" if not go_star else "blocked_by_user_cpu_only",
        "rnn_action": (
            "do_not_launch; skeleton in csrc/n20_rnn_gpu_prep.cu"
        ),
        "would_compile": (
            "nvcc -O3 -std=c++17 --shared -Xcompiler -fPIC "
            "csrc/n20_rnn_gpu_prep.cu -o src/libn20_rnn_gpu.so"
        ),
        "would_launch_only_if": [
            "N20_D3.go_gpu_rnn is true",
            "N20_RNN four-T PASS",
            "user lifts CPU-only hold",
            "N20_GPU_RUN=1",
        ],
        "best_agg_ms": 1679.9,
        "reason": (
            "StarMerge EV-dead at production eps; RNN GPU port gated on D3 "
            "round count. This script never launches a kernel."
        ),
        "rnn_cpu": {k: rnn.get(k) for k in ("rnn_ms", "four_ok", "kill_as_closer", "reason")},
        "cu_exists": CU.is_file(),
        "ran_gpu": False,
        "ran_216": False,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "N20_GPU_PREP.json").write_text(json.dumps(recipe, indent=2) + "\n")
    NOTE.write_text(
        "# N20 GPU prep (not a run)\n\n"
        f"{CLAIM}. host={host}.\n\n"
        f"- go_star={go_star} (must stay false; D1 layer-0 is not small-merge)\n"
        f"- go_gpu_rnn={go_rnn}\n"
        f"- nvcc={recipe['nvcc'] or 'missing'}\n"
        f"- smi: {recipe['nvidia_smi_L']}\n"
        f"- skeleton: `{CU.name}` exists={CU.is_file()}\n"
        "- Do not set N20_GPU_RUN. Do not nsys. Do not 2.16.\n"
        "- Product default stays E6s. Do not reopen AGG_E6t.\n"
    )
    print(json.dumps(recipe, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
