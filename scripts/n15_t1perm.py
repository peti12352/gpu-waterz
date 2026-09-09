#!/usr/bin/env python3
"""N15 exp1 T1-PERM: same partition vs ID permutation vs wrong basins.

N14 T1: nfrag/bg matched gold, array_equal False. That is necessary for a
relabel, not sufficient. TASK grades the WS partition; this stack still
locks byte-identical vs wz_fragments.npy. Decide before any 2.16 or CUDA.

WATERZ_FOLD_FLATTEN=1. No 2.16. Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from n13_baseline import parks_env  # noqa: E402
from p1_make_big_indep import (  # noqa: E402
    AFF, CACHE, card_busy, fingerprint, gpu_state, nfrag_bg, same_partition,
)
from b_dev_aff import bind, build, run_split  # noqa: E402
from task_gate import FRAGMENTS_VAL  # noqa: E402
import h5py

NOTE = ROOT / "notes/N15_T1PERM.md"
OUT = CACHE / "n15_t1perm.json"
BG_VAL = 506_568
GOLD = CACHE / "wz_fragments.npy"


def canon_min_index(seg: np.ndarray) -> np.ndarray:
    """Map each nonzero label to the min linear index of its voxels. bg stays 0."""
    lab = np.ascontiguousarray(seg, dtype=np.uint32).ravel()
    n = int(lab.size)
    mx = int(lab.max()) + 1
    mins = np.full(mx, n, dtype=np.int64)
    np.minimum.at(mins, lab, np.arange(n, dtype=np.int64))
    mins[0] = 0
    out = mins[lab]
    out = np.where(lab == 0, 0, out)
    return out.reshape(seg.shape)


def pair_split_merge(gold: np.ndarray, pred: np.ndarray) -> dict:
    """How many gold fragments split across pred ids, and vice versa.

    Ignores (0,0). A true ID permutation has n_gold_split=0 and n_pred_merge=0.
    """
    g = np.ascontiguousarray(gold, dtype=np.uint32).ravel()
    p = np.ascontiguousarray(pred, dtype=np.uint32).ravel()
    # unique (g,p) pairs via packing; val max ids are ~2.1e6 so 2^21+ fits uint64.
    key = (g.astype(np.uint64) << 32) | p.astype(np.uint64)
    uniq = np.unique(key)
    ug = (uniq >> 32).astype(np.uint32)
    up = (uniq & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    mask = (ug != 0) | (up != 0)
    ug, up = ug[mask], up[mask]
    _, gcnt = np.unique(ug, return_counts=True)
    _, pcnt = np.unique(up, return_counts=True)
    return {
        "n_pairs": int(ug.size),
        "n_gold_split": int((gcnt > 1).sum()),
        "n_pred_merge": int((pcnt > 1).sum()),
        "max_gold_fanout": int(gcnt.max()) if gcnt.size else 0,
        "max_pred_fanout": int(pcnt.max()) if pcnt.size else 0,
    }


def perm_once():
    os.environ.update({
        "WATERZ_UF_ALGO": "3",
        "WATERZ_HOST_PARK": "0",
        "WATERZ_AFF_PARK": "0",
        "WATERZ_FOLD_FLATTEN": "1",
    })
    import ctypes
    lib = ctypes.CDLL(str(ROOT / "src/libws_gpu.so"))
    bind(lib)
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    gold = np.load(GOLD)
    seg1, meta1 = run_split(lib, aff, park=True)
    seg2, meta2 = run_split(lib, aff, park=True)
    nfrag, bg = nfrag_bg(seg1)
    gnfrag, gbg = nfrag_bg(gold)
    fp1, nsz1, _ = fingerprint(seg1)
    fpg, nszg, _ = fingerprint(gold)
    canon1 = canon_min_index(seg1)
    canong = canon_min_index(gold)
    ndiff = int(np.count_nonzero(seg1 != gold))
    ndiff_canon = int(np.count_nonzero(canon1 != canong))
    bg_mask_eq = bool(np.array_equal(seg1 == 0, gold == 0))
    sm = pair_split_merge(gold, seg1)
    perm_same = bool(same_partition(seg1, gold))
    canon_eq = bool(ndiff_canon == 0)
    det = bool(np.array_equal(seg1, seg2))
    ident_byte = bool(np.array_equal(seg1, gold) and nfrag == FRAGMENTS_VAL
                      and bg == BG_VAL)
    # Revive T1 only if the partition is gold's (perm or byte-identical).
    # Size-histogram match alone is not a partition proof.
    verdict = (
        "byte_identical" if ident_byte else
        "same_partition_relabel" if (perm_same and canon_eq) else
        "wrong_basins"
    )
    return {
        "claim": "not a 2 Gvox/s number; not 3090 Ti; no 2.16",
        "nfrag": nfrag,
        "bg": bg,
        "gold_nfrag": gnfrag,
        "gold_bg": gbg,
        "nfrag_match": bool(nfrag == FRAGMENTS_VAL and nfrag == gnfrag),
        "bg_match": bool(bg == BG_VAL and bg == gbg),
        "bg_mask_eq": bg_mask_eq,
        "array_equal": bool(np.array_equal(seg1, gold)),
        "identity_byte": ident_byte,
        "run2_array_equal": det,
        "same_partition": perm_same,
        "canon_eq": canon_eq,
        "ndiff_raw": ndiff,
        "ndiff_canon": ndiff_canon,
        "fp_fold": fp1,
        "fp_gold": fpg,
        "fp_eq": bool(fp1 == fpg),
        "n_nonzero_sizes_fold": nsz1,
        "n_nonzero_sizes_gold": nszg,
        "pairs": sm,
        "verdict": verdict,
        "revive_t1": bool(verdict in ("byte_identical", "same_partition_relabel")
                          and det),
        "kill_t1_as_speed": bool(verdict == "wrong_basins"),
        "meta1": meta1,
        "meta2": meta2,
        "vox": int(seg1.size),
    }


def write_note(doc: dict) -> None:
    p = doc.get("pairs") or {}
    NOTE.write_text(
        "# N15 exp1 T1-PERM\n\n"
        "Not a 2 Gvox/s claim. Not 3090 Ti. No 2.16. "
        "WATERZ_FOLD_FLATTEN=1 vs wz_fragments.npy.\n\n"
        f"- identity_byte={doc.get('identity_byte')} array_equal={doc.get('array_equal')} "
        f"nfrag={doc.get('nfrag')} bg={doc.get('bg')} "
        f"(gold {doc.get('gold_nfrag')}/{doc.get('gold_bg')})\n"
        f"- run2_array_equal={doc.get('run2_array_equal')} "
        f"bg_mask_eq={doc.get('bg_mask_eq')}\n"
        f"- same_partition={doc.get('same_partition')} canon_eq={doc.get('canon_eq')} "
        f"fp_eq={doc.get('fp_eq')}\n"
        f"- ndiff_raw={doc.get('ndiff_raw')} ndiff_canon={doc.get('ndiff_canon')} "
        f"vox={doc.get('vox')}\n"
        f"- pairs n={p.get('n_pairs')} gold_split={p.get('n_gold_split')} "
        f"pred_merge={p.get('n_pred_merge')} "
        f"max_gold_fanout={p.get('max_gold_fanout')} "
        f"max_pred_fanout={p.get('max_pred_fanout')}\n"
        f"- verdict={doc.get('verdict')} revive_t1={doc.get('revive_t1')} "
        f"kill_t1_as_speed={doc.get('kill_t1_as_speed')}\n"
        "- keep_default=False (exp1 is a diagnosis; no product-default change)\n"
        "- next: share-off+fold find only if kill_t1_as_speed; "
        "min-index relabel only if revive_t1\n"
    )


def main():
    print("N15 T1-PERM. Not a 2 Gvox/s claim. Not 3090 Ti. No 2.16.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N15 T1-PERM REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)
    extra = {"WATERZ_FOLD_FLATTEN": "1"}
    os.environ["WATERZ_FOLD_FLATTEN"] = "1"
    build()
    env = parks_env(extra)
    r = subprocess.run(
        [sys.executable, str(Path(__file__)), "--perm"],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    )
    doc = {}
    for ln in r.stdout.splitlines():
        if ln.startswith("{"):
            doc = json.loads(ln)
            break
    if not doc:
        print(f"N15 T1-PERM no json rc={r.returncode} stderr={r.stderr[-1500:]}",
              flush=True)
        return 1
    print(f"N15 T1-PERM {json.dumps(doc, sort_keys=True)}", flush=True)
    if r.stderr:
        print(f"N15 T1-PERM stderr_tail={r.stderr[-800:]}", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    write_note(doc)
    print(f"N15 T1-PERM verdict={doc.get('verdict')} -> {OUT}", flush=True)
    return 0 if doc.get("run2_array_equal") else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--perm":
        print(json.dumps(perm_once()), flush=True)
        raise SystemExit(0)
    raise SystemExit(main())
