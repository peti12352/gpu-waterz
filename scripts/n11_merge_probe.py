#!/usr/bin/env python3
"""N11 C: val RAG serial merge-cost probe. No new agglomerator.

Device loop: find two roots + atomicMin/size add. Then +neighbor walk.
Kill GPU BinQueue if wall >= val ParHAC agg (~202 ms).
Idle-5090. Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import load_rag  # noqa: E402
from p1_make_big_indep import CACHE, card_busy, gpu_state  # noqa: E402

NVCC = "/usr/local/cuda-12.8/bin/nvcc"
DSO = ROOT / "src/libn11_probe.so"
OUT = CACHE / "n11_probe.json"
NOTE = ROOT / "notes/N11_PROBE.md"
PARHAC_VAL_AGG_MS = 202.0
TARGET_MERGES = 1_853_086


def compile_probe():
    src = ROOT / "csrc/n11_merge_probe.cu"
    if DSO.exists() and DSO.stat().st_mtime >= src.stat().st_mtime:
        return
    subprocess.check_call([
        NVCC, "-O3", "-arch=sm_120", "--shared", "-Xcompiler", "-fPIC",
        "-o", str(DSO), str(src),
    ])


def run_probe(lib, u, v, nnode, walk):
    wall = ctypes.c_double(0)
    n_pop = ctypes.c_int64(0)
    n_merge = ctypes.c_int64(0)
    rc = lib.n11_merge_probe(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_int64(len(u)),
        ctypes.c_uint32(nnode),
        ctypes.c_int(TARGET_MERGES),
        ctypes.c_int(1 if walk else 0),
        ctypes.byref(wall),
        ctypes.byref(n_pop),
        ctypes.byref(n_merge),
    )
    ms = float(wall.value)
    pops = int(n_pop.value)
    ns_per = (ms * 1e6 / pops) if pops else None
    return {
        "rc": int(rc),
        "walk": bool(walk),
        "wall_ms": ms,
        "n_pop": pops,
        "n_merge": int(n_merge.value),
        "ns_per_pop": ns_per,
        "parhac_val_agg_ms": PARHAC_VAL_AGG_MS,
        "kill": rc != 0 or ms >= PARHAC_VAL_AGG_MS,
    }


def write_note(doc):
    p1 = doc["probe_union"]
    p2 = doc.get("probe_walk")
    lines = [
        "# N11 C: serial merge-cost probe",
        "",
        "Val RAG. One device thread. Not a 2 Gvox/s number. Not 3090 Ti.",
        f"Kill if wall >= ParHAC val agg ({PARHAC_VAL_AGG_MS} ms).",
        "",
        "## find+union (no neighbor rewrite)",
        "",
        f"- wall={p1['wall_ms']:.3f} ms n_pop={p1['n_pop']} n_merge={p1['n_merge']}",
        f"- ns/pop={p1['ns_per_pop']}",
        f"- kill={p1['kill']}",
        "",
    ]
    if p2:
        lines += [
            "## + neighbor walk (notifyEdgeMerge-class)",
            "",
            f"- wall={p2['wall_ms']:.3f} ms n_pop={p2['n_pop']} n_merge={p2['n_merge']}",
            f"- ns/pop={p2['ns_per_pop']}",
            f"- kill={p2['kill']}",
            "",
        ]
    else:
        lines += [
            "## + neighbor walk",
            "",
            "Skipped (find+union already >= ParHAC val agg).",
            "",
        ]
    lines += [
        "## Verdict",
        "",
        "GPU MEAN BinQueue alive." if doc["bins_alive"] else "GPU BinQueue dead. Skip E.",
        "",
    ]
    NOTE.write_text("\n".join(lines) + "\n")


def main():
    print("N11 merge probe. Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N11 REFUSE card busy: {busy}", flush=True)
        raise SystemExit(2)
    print(f"GPU idle: {gpu_state()}", flush=True)
    compile_probe()
    u, v, sm, ct, fr, max_id = load_rag()
    mean = sm / np.maximum(ct.astype(np.float64), 1.0)
    order = np.argsort(-mean, kind="mergesort")
    u = np.ascontiguousarray(u[order], dtype=np.uint32)
    v = np.ascontiguousarray(v[order], dtype=np.uint32)
    nnode = max_id + 1
    print(
        f"N11 RAG nedge={len(u)} nnode={nnode} target_merges={TARGET_MERGES}",
        flush=True,
    )
    lib = ctypes.CDLL(str(DSO))
    lib.n11_merge_probe.restype = ctypes.c_int
    lib.n11_merge_probe.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int64, ctypes.c_uint32, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    ]
    p1 = run_probe(lib, u, v, nnode, walk=False)
    print(
        f"N11 probe union wall={p1['wall_ms']:.3f} ms n_pop={p1['n_pop']} "
        f"n_merge={p1['n_merge']} ns/pop={p1['ns_per_pop']} kill={p1['kill']}",
        flush=True,
    )
    p2 = None
    if not p1["kill"]:
        p2 = run_probe(lib, u, v, nnode, walk=True)
        print(
            f"N11 probe walk wall={p2['wall_ms']:.3f} ms n_pop={p2['n_pop']} "
            f"n_merge={p2['n_merge']} ns/pop={p2['ns_per_pop']} kill={p2['kill']}",
            flush=True,
        )
    bins_alive = (not p1["kill"]) and p2 is not None and (not p2["kill"])
    doc = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "gpu": gpu_state(),
        "nedge": int(len(u)),
        "nnode": int(nnode),
        "target_merges": TARGET_MERGES,
        "parhac_val_agg_ms": PARHAC_VAL_AGG_MS,
        "probe_union": p1,
        "probe_walk": p2,
        "bins_alive": bins_alive,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    write_note(doc)
    print(f"N11 probe bins_alive={bins_alive} -> {OUT} {NOTE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
