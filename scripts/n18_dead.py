#!/usr/bin/env python3
"""N18 dead-end registry. Refuse re-run unless --force.

Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEAD = ROOT / "data/cache/n18_dead.jsonl"

# Seeded from plan dead-end registry (do not reopen).
SEED = [
    {"exp_id": "WS_fold_no_SHARE_OFF", "reason": "wrong basins 89% ndiff", "note": "plan"},
    {"exp_id": "WS_PIN_CHANGED", "reason": "WS~1e6 ms mapped host atomics", "note": "plan"},
    {"exp_id": "WS_STITCH_ARENA", "reason": "1.00x+leak", "note": "plan"},
    {"exp_id": "WS_SORT_PACK", "reason": "slower", "note": "plan"},
    {"exp_id": "WS_JUMP_FLATTEN", "reason": "exhausted", "note": "plan"},
    {"exp_id": "WS_LIST_HALVING", "reason": "exhausted", "note": "plan"},
    {"exp_id": "WS_E4_closer", "reason": "not a closer", "note": "plan"},
    {"exp_id": "WS_drop_k_count_v2", "reason": "BFS qsz corrupt; N16_T13", "note": "plan"},
    {"exp_id": "WS_Z_SLAB_0_fused", "reason": "OOM", "note": "plan"},
    {"exp_id": "WS_Playne_park_era", "reason": "park-era theorem", "note": "plan"},
    {"exp_id": "WS_PRUF_grayscale_S1", "reason": "Meyer!=waterz S1; N12_OFF", "note": "plan"},
    {"exp_id": "AGG_sticky_sz0", "reason": "T=0.2 merge 0.3585>0.3525", "note": "plan"},
    {"exp_id": "AGG_serial_BinQueue", "reason": "N14 T3 hang>180s; N11 1-thread void", "note": "plan"},
    {"exp_id": "AGG_cuda_graph_G2", "reason": "killed", "note": "plan"},
    {"exp_id": "AGG_FUSE_PACK_closer", "reason": "noise/illegal closer", "note": "plan"},
    {"exp_id": "AGG_DIRTY_UNMARK_closer", "reason": "noise/illegal closer", "note": "plan"},
    {"exp_id": "AGG_eps_ge_0.5", "reason": "four-T FAIL merge", "note": "plan"},
    {"exp_id": "AGG_MAX_OUTER_32", "reason": "product kill", "note": "plan"},
    {"exp_id": "AGG_compact_every_default", "reason": "killed", "note": "plan"},
    {"exp_id": "AGG_HASH_INSERT_ONLY", "reason": "killed", "note": "plan"},
    {"exp_id": "AGG_E6t_StarMerge_default", "reason": "killed", "note": "plan"},
    {"exp_id": "AGG_mutex_Kruskal_X1_SubgraphHAC_RAMA", "reason": "task_legal=False class", "note": "plan"},
    {"exp_id": "AGG_cpp_defaults_pre_3090_peak", "reason": "policy", "note": "plan"},
]


def ensure_seed():
    DEAD.parent.mkdir(parents=True, exist_ok=True)
    if DEAD.is_file() and DEAD.stat().st_size > 0:
        return
    with DEAD.open("w") as f:
        for row in SEED:
            f.write(json.dumps(row) + "\n")


def load_ids() -> set[str]:
    ensure_seed()
    ids = set()
    if not DEAD.is_file():
        return ids
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
    """Return None if ok to run, else refuse message."""
    if force:
        return None
    if is_dead(exp_id):
        return f"N18 REFUSE dead exp_id={exp_id} (use --force)"
    return None
