#!/usr/bin/env python3
"""N21 dead-end registry. Do not write n19_dead.jsonl.

Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEAD = ROOT / "data/cache/n21_dead.jsonl"


def stamp(exp_id: str, reason: str, numbers=None, note: str = "") -> None:
    DEAD.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "exp_id": exp_id,
        "reason": reason,
        "numbers": numbers or {},
        "note": note,
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
    }
    with DEAD.open("a") as f:
        f.write(json.dumps(row) + "\n")
