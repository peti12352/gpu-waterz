#!/usr/bin/env python3
"""Public quality gate on the gpu_waterz API.

Default (CREMI-A val, if present):
  - two-run label identity at affinity 0.3 (dual-eps 0.40)
  - four-T VOI at 0.2/0.3/0.4/0.5 (dual-eps 0.08)
  - fragment count vs the published 2,175,400 / 506,568 numbers

VOI is `scripts/voi_numpy.py` (same formula as funkey/waterz
`evaluate.hpp`: skip gt==0, predicted 0 is a label, bits). Does not
import waterz. Set WATERZ_USE_WATERZ_EVAL=1 to grade with
`waterz.evaluate` instead when that package is installed.

Optional --216 times the [3,375,2400,2400] volume when that
HDF5 exists. Needs h5py: `uv sync --extra eval`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gpu_waterz.limits import (  # noqa: E402
    AFF_THRESHOLDS,
    BG_VAL,
    FRAGMENTS_VAL,
    grade_voi,
)
from voi_numpy import voi_split_merge  # noqa: E402


def voi_pair(seg: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    if os.environ.get("WATERZ_USE_WATERZ_EVAL") == "1":
        import waterz

        scores = waterz.evaluate(
            np.asarray(seg, dtype=np.uint64),
            np.asarray(gt, dtype=np.uint64),
        )
        return float(scores["voi_split"]), float(scores["voi_merge"])
    return voi_split_merge(seg, gt)


SHAPE_216 = (3, 375, 2400, 2400)

VAL_DIR = Path(os.environ.get("WATERZ_VAL_DIR", ROOT / "data/cremiA_val"))
AFF_H5 = VAL_DIR / "affinity.h5"
GT_H5 = VAL_DIR / "gt.h5"


def _big_candidates() -> list[Path]:
    out = []
    env = os.environ.get("WATERZ_AFF_216")
    if env:
        out.append(Path(env))
    out.extend((
        ROOT / "data/cremiA_216/affinity.h5",
        ROOT / "data/cache/big_216.h5",
    ))
    return out


PRODUCT_ENV = {
    "WATERZ_UF_ALGO": "3",
    "WATERZ_HOST_PARK": "0",
    "WATERZ_AFF_PARK": "0",
    "WATERZ_AGG_LEVERS": "15",
    "WATERZ_FOLD_FLATTEN": "1",
    "WATERZ_SHARE_OFF": "1",
    "WATERZ_HOOK_ROOT": "1",
    "WATERZ_FUSE_DIRTY": "1",
    "WATERZ_NLIVE_ARITH": "1",
    "WATERZ_EMIT_HOLES": "1",
}


def apply_product_env() -> None:
    for k, v in PRODUCT_ENV.items():
        os.environ.setdefault(k, v)
    os.environ.pop("WATERZ_AGG_EPS", None)


def load_h5(path: Path, keys: tuple[str, ...]) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as f:
        for k in keys:
            if k in f:
                return np.ascontiguousarray(f[k][:])
        raise KeyError(f"{path}: none of {keys} found; have {list(f.keys())}")


def nfrag_bg(lab: np.ndarray) -> tuple[int, int]:
    flat = np.asarray(lab).reshape(-1)
    bg = int(np.count_nonzero(flat == 0))
    nfrag = int(flat.max()) if flat.size else 0
    return nfrag, bg


def gpu_busy() -> str | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not lines:
        return None
    return "; ".join(lines[:8])


def grade_val() -> dict:
    try:
        import h5py  # noqa: F401
    except ImportError:
        return {
            "ok": False,
            "error": "h5py missing",
            "hint": "uv sync --extra eval",
        }
    import gpu_waterz as wz

    wz.require_cuda_libs()
    if not AFF_H5.is_file() or not GT_H5.is_file():
        return {
            "ok": False,
            "error": "missing CREMI-A val HDF5",
            "affinity": str(AFF_H5),
            "gt": str(GT_H5),
            "hint": "place affinity.h5 and gt.h5 in data/cremiA_val/ (or WATERZ_VAL_DIR)",
        }

    aff = load_h5(AFF_H5, ("affinity",))
    gt = load_h5(GT_H5, ("gt", "labels"))
    fr = wz.fragments(aff)
    nfrag, bg = nfrag_bg(fr)
    run_a = wz.segment(aff, [0.3])[0]
    run_b = wz.segment(aff, [0.3])[0]
    ident = bool(np.array_equal(run_a, run_b))
    split_t3, merge_t3 = voi_pair(run_a, gt)
    t3_ok, sl3, ml3 = grade_voi(split_t3, merge_t3, 0.3)

    four = wz.segment(aff, list(AFF_THRESHOLDS))
    four_rows = []
    four_ok = True
    for t, lab in zip(AFF_THRESHOLDS, four):
        split, merge = voi_pair(lab, gt)
        ok, sl, ml = grade_voi(split, merge, t)
        four_ok = four_ok and ok
        four_rows.append({
            "aff": t,
            "split": split,
            "merge": merge,
            "limit_split": sl,
            "limit_merge": ml,
            "ok": bool(ok),
        })

    ok = bool(ident and t3_ok and four_ok)
    return {
        "ok": ok,
        "identity_t03": ident,
        "nfrag": nfrag,
        "bg": bg,
        "nfrag_published": FRAGMENTS_VAL,
        "bg_published": BG_VAL,
        "t03": {
            "split": split_t3,
            "merge": merge_t3,
            "limit_split": sl3,
            "limit_merge": ml3,
            "ok": bool(t3_ok),
        },
        "four": four_rows,
        "four_ok": bool(four_ok),
    }


def time_216() -> dict:
    import gpu_waterz as wz
    from gpu_waterz._backend import seg

    wz.require_cuda_libs()
    path = next((p for p in _big_candidates() if p.is_file()), None)
    if path is None:
        return {
            "ok": False,
            "error": "missing 2.16 Gvox affinity HDF5",
            "tried": [str(p) for p in _big_candidates()],
        }
    aff = load_h5(path, ("affinity",))
    if tuple(aff.shape) != SHAPE_216:
        return {"ok": False, "error": f"{path} shape {aff.shape} != {SHAPE_216}"}
    os.environ["WATERZ_STAGE_MS"] = "1"
    aff_d = seg.DevBuf.from_host(aff)
    del aff
    _out, ms = seg.cuda_event_time(
        lambda: seg.segment_d(aff_d, [0.3], return_device=True)
    )
    return {
        "ok": True,
        "path": str(path),
        "e2e_ms": float(ms),
        "stages_ms": dict(seg.STAGE_MS),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--216", dest="run_216", action="store_true",
                   help="also time [3,375,2400,2400] if present")
    args = p.parse_args()
    apply_product_env()

    busy = gpu_busy()
    if busy:
        print(json.dumps({"ok": False, "refuse": "gpu_busy", "busy": busy}))
        return 2

    doc: dict = {"env": {k: os.environ.get(k) for k in PRODUCT_ENV}}
    val = grade_val()
    doc["val"] = val
    rc = 0 if val.get("ok") else 1
    if args.run_216:
        big = time_216()
        doc["t216"] = big
        if not big.get("ok"):
            rc = 1 if rc == 0 else rc
    print(json.dumps(doc, default=float), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
