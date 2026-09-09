#!/usr/bin/env python3
"""N19 dead-end registry. Seeds n18 + N18 stamps + plan; refuse unless --force.

Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEAD = ROOT / "data/cache/n19_dead.jsonl"
N18_DEAD = ROOT / "data/cache/n18_dead.jsonl"

# Extra N19 plan refuse list (beyond n18 seed).
N19_SEED = [
    {"exp_id": "N18_A2", "reason": "VCOUNT_COMPACT four fail", "note": "plan"},
    {"exp_id": "N18_A3", "reason": "TIE_FLIP 2.16 rc=-3", "note": "plan"},
    {"exp_id": "N18_A4", "reason": "COARSE_DELTA 2.16 rc=-3", "note": "plan"},
    {"exp_id": "N18_A5", "reason": "BLOCK_VOI four fail", "note": "plan"},
    {"exp_id": "N18_B1", "reason": "parallel BinQueue VOI/wall kill", "note": "plan"},
    {"exp_id": "N18_B2", "reason": "eps 0.41-0.49 merge FAIL", "note": "plan"},
    {"exp_id": "N18_B3", "reason": "COMPACT_EVERY=8 slower", "note": "plan"},
    {"exp_id": "AGG_N18_parallel_BinQueue", "reason": "N18 B1 dead", "note": "plan"},
    {"exp_id": "AGG_eps_0.41_to_0.49", "reason": "all T=0.3 merge FAIL", "note": "plan"},
]


def ensure_seed():
    DEAD.parent.mkdir(parents=True, exist_ok=True)
    if DEAD.is_file() and DEAD.stat().st_size > 0:
        return
    seen = set()
    rows = []
    if N18_DEAD.is_file():
        for ln in N18_DEAD.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                row = json.loads(ln)
                eid = row.get("exp_id")
                if eid and eid not in seen:
                    seen.add(eid)
                    rows.append(row)
            except (json.JSONDecodeError, TypeError):
                continue
    else:
        # fallback: import n18 seed
        sys_path = str(ROOT / "scripts")
        import sys
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        from n18_dead import SEED as N18_SEED  # noqa: WPS433
        for row in N18_SEED:
            eid = row["exp_id"]
            if eid not in seen:
                seen.add(eid)
                rows.append(row)
    for row in N19_SEED:
        eid = row["exp_id"]
        if eid not in seen:
            seen.add(eid)
            rows.append(row)
    with DEAD.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_ids() -> set[str]:
    ensure_seed()
    ids = set()
    for ln in DEAD.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            ids.add(json.loads(ln)["exp_id"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return ids


def is_dead(exp_id: str) -> bool:
    return exp_id in load_ids()


def stamp(exp_id: str, reason: str, numbers=None, note: str = "") -> None:
    ensure_seed()
    row = {
        "exp_id": exp_id,
        "reason": reason,
        "numbers": numbers or {},
        "note": note,
    }
    with DEAD.open("a") as f:
        f.write(json.dumps(row) + "\n")


def refuse_or_ok(exp_id: str, force: bool = False):
    if force:
        return None
    if is_dead(exp_id):
        return f"N19 REFUSE dead exp_id={exp_id} (use --force)"
    return None
