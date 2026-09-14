#!/usr/bin/env python3
"""N8: A/B WATERZ_UF_ALGO 0/2/3 on idle 5090. Keep W5 only if identity + >=1.3x.

Not a 2 Gvox/s claim. No 2.16 allocation.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from b_dev_aff import bind, build, run_split  # noqa: E402
from p1_make_big_indep import AFF, CACHE, card_busy, fingerprint, gpu_state, nfrag_bg  # noqa: E402
from task_gate import FRAGMENTS_VAL  # noqa: E402

OUT = CACHE / "n8_w5.json"
SPEEDUP_GATE = 1.3


def one(algo: int):
    os.environ["WATERZ_UF_ALGO"] = str(algo)
    build()
    import ctypes
    lib = ctypes.CDLL(str(ROOT / "src/libws_gpu.so"))
    bind(lib)
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    # uf_algo() caches on first call; this process sees only this algo.
    import time
    times = []
    last = None
    for _ in range(3):
        t0 = time.perf_counter()
        seg, meta = run_split(lib, aff, park=True)
        times.append((time.perf_counter() - t0) * 1000.0)
        last = (seg, meta)
    seg, meta = last
    nfrag, bg = nfrag_bg(seg)
    fp, _, _ = fingerprint(seg)
    oracle = None
    op = CACHE / "wz_fragments.npy"
    if op.exists():
        ref = np.ascontiguousarray(np.load(op), dtype=np.uint32)
        oracle = bool(ref.shape == seg.shape and np.array_equal(ref, seg))
    times.sort()
    return {
        "algo": algo,
        "nfrag": nfrag,
        "bg": bg,
        "fingerprint": list(fp) if not isinstance(fp, (str, int, float)) else fp,
        "oracle_array_equal": oracle,
        "ws_ms_median": times[len(times) // 2],
        "ws_ms": times,
        "divide_ms": float(meta["divide_ms"]),
        "nfrag_ok": nfrag == FRAGMENTS_VAL,
        "identity": bool(oracle is not False and nfrag == FRAGMENTS_VAL),
    }


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--algo":
        doc = one(int(sys.argv[2]))
        print(json.dumps(doc), flush=True)
        return 0 if doc["identity"] else 1

    print("N8 W5 A/B. Not a 2 Gvox/s claim.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N8 REFUSE card busy: {busy}", flush=True)
        raise SystemExit(2)
    print(f"GPU idle: {gpu_state()}", flush=True)
    rows = []
    for algo in (0, 2, 3):
        r = subprocess.run(
            [sys.executable, str(Path(__file__)), "--algo", str(algo)],
            cwd=str(ROOT), capture_output=True, text=True, check=False,
        )
        # last JSON line is the result; nvcc/E9b chatter is stderr+stdout
        line = None
        for ln in r.stdout.splitlines():
            if ln.startswith("{"):
                line = ln
        if line is None:
            print(f"N8 W5 algo={algo} failed rc={r.returncode}\n"
                  f"stdout={r.stdout[-2000:]}\nstderr={r.stderr[-2000:]}",
                  flush=True)
            raise SystemExit(1)
        doc = json.loads(line)
        rows.append(doc)
        print(
            f"N8 W5 algo={algo} identity={doc['identity']} "
            f"ws={doc['ws_ms_median']:.2f} nfrag={doc['nfrag']}",
            flush=True,
        )
    base = next(x["ws_ms_median"] for x in rows if x["algo"] == 0)
    w5 = next(x for x in rows if x["algo"] == 3)
    speedup = base / w5["ws_ms_median"] if w5["ws_ms_median"] else 0.0
    keep = bool(w5["identity"] and speedup >= SPEEDUP_GATE)
    out = {
        "claim": "not a 2 Gvox/s number",
        "gpu": gpu_state(),
        "rows": rows,
        "speedup_w5_vs_0": speedup,
        "gate": SPEEDUP_GATE,
        "keep_w5_default": keep,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"N8 W5 speedup={speedup:.2f}x keep={keep} -> {OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
