#!/usr/bin/env python3
"""N8: one official 2.16 segment_d at T=0.3 on idle 5090.

Not TASK-grade (wrong card). Writes the e2e number that decides N8_IMPOSSIBLE.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from b_dev_aff import build  # noqa: E402
from e6r_parhac import compile_d  # noqa: E402
from p1_make_big_indep import CACHE, MAKE_BIG, card_busy, gpu_state  # noqa: E402
import segment as S  # noqa: E402

OUT = CACHE / "n8_216.json"
SHAPE = (3, 375, 2400, 2400)
NVOX = 2_160_000_000
IMPOSSIBLE_MS = 2000.0


def official_216():
    """Official make_big 3×2×2, already-written h5, or build it here."""
    candidates = [
        ROOT / "data/ws_bounty/big/affinity.h5",
        ROOT / "data/cache/big_216.h5",
        Path("/tmp/wz_big_216.h5"),
    ]
    for p in candidates:
        if p.is_file():
            print(f"N8 2.16 load {p}", flush=True)
            with h5py.File(p, "r") as f:
                aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
            if tuple(aff.shape) != SHAPE:
                raise RuntimeError(f"{p} shape {aff.shape} != {SHAPE}")
            return aff
    if not MAKE_BIG.is_file():
        raise FileNotFoundError(f"official make_big.py missing at {MAKE_BIG}")
    out = ROOT / "data/cache/big_216.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"N8 2.16 running official make_big.py -> {out}", flush=True)
    spec = importlib.util.spec_from_file_location("ws_bounty_make_big", MAKE_BIG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
    mod.build(str(src), str(out), (3, 2, 2))
    with h5py.File(out, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    if tuple(aff.shape) != SHAPE:
        raise RuntimeError(f"make_big shape {aff.shape} != {SHAPE}")
    return aff


def main():
    print("N8 one 2.16 segment_d. Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N8 REFUSE card busy: {busy}", flush=True)
        raise SystemExit(2)
    print(f"GPU idle: {gpu_state()}", flush=True)
    build()
    compile_d()
    os.environ["WATERZ_AGG_LEVERS"] = "15"
    os.environ["WATERZ_STAGE_MS"] = "1"
    # Speed path: single T=0.3 uses ε=0.40 unless overridden.
    os.environ.pop("WATERZ_AGG_EPS", None)
    aff = official_216()
    print(f"N8 2.16 aff {aff.shape} {aff.nbytes / 2**30:.2f} GiB", flush=True)
    aff_d = S.DevBuf.from_host(aff)
    del aff
    out, ms = S.cuda_event_time(
        lambda: S.segment_d(aff_d, [0.3], return_device=True))
    st = dict(S.STAGE_MS)
    nlab = 0
    if out:
        host = out[0].to_host()
        nlab = int(np.unique(host).size)
        for b in out:
            b.free()
        del host
    aff_d.free()
    e2e = float(ms)
    gvox_s = (NVOX / 1e9) / (e2e / 1000.0) if e2e > 0 else 0.0
    impossible = e2e > IMPOSSIBLE_MS
    doc = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "gpu": gpu_state(),
        "shape": list(SHAPE),
        "nvox": NVOX,
        "e2e_ms": e2e,
        "stages": st,
        "nlab": nlab,
        "gvox_s_5090": gvox_s,
        "eps": 0.40,
        "impossible_vs_2s": impossible,
        "aff_parked": bool(S.LAST_AFF_PARKED),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(
        f"N8 2.16 e2e={e2e:.1f} ms stages={st} nlab={nlab} "
        f"gvox/s={gvox_s:.3f} {'>>2s' if impossible else '~1-2s or under'} "
        f"-> {OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
