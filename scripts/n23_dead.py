#!/usr/bin/env python3
"""N23 dead-end registry. Do not write n19-n22 dead jsonl."""
from __future__ import annotations

import ctypes
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEAD = ROOT / "data/cache/n23_dead.jsonl"


def stamp(exp_id: str, reason: str, numbers=None, note: str = "") -> None:
    DEAD.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "exp_id": exp_id,
        "reason": reason,
        "numbers": numbers or {},
        "note": note,
        "claim": "idle RTX 5090; not a throughput claim",
    }
    with DEAD.open("a") as f:
        f.write(json.dumps(row) + "\n")


def release_gpu() -> int:
    """Drop the parent CUDA context so n8_run216 card_busy() sees an empty card."""
    for name in (
        "libcudart.so.12",
        "libcudart.so",
        "/usr/local/cuda-12.8/lib64/libcudart.so",
    ):
        try:
            rt = ctypes.CDLL(name)
            rt.cudaDeviceReset.restype = ctypes.c_int
            return int(rt.cudaDeviceReset())
        except OSError:
            continue
    return -1
