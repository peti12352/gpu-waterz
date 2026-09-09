#!/usr/bin/env python3
"""N10: split the WS remainder. Host-park vs device-resident corners.

Idle-5090. Val identity. Not a 2 Gvox/s claim. No 2.16 unless val passes.
"""
from __future__ import annotations

import ctypes
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
from p1_make_big_indep import AFF, CACHE, card_busy, gpu_state, nfrag_bg  # noqa: E402
from task_gate import FRAGMENTS_VAL  # noqa: E402

OUT = CACHE / "n10_ws_split.json"


def n10_stats(lib):
    park = ctypes.c_float(0)
    vcount = ctypes.c_float(0)
    sort = ctypes.c_float(0)
    unpark = ctypes.c_float(0)
    scan = ctypes.c_float(0)
    lib.ws_n10_stats(
        ctypes.byref(park), ctypes.byref(vcount), ctypes.byref(sort),
        ctypes.byref(unpark), ctypes.byref(scan),
    )
    return {
        "park_ms": float(park.value),
        "vcount_ms": float(vcount.value),
        "sort_ms": float(sort.value),
        "unpark_ms": float(unpark.value),
        "scan_ms": float(scan.value),
    }


def one(host_park: int):
    os.environ["WATERZ_UF_ALGO"] = "3"
    os.environ["WATERZ_HOST_PARK"] = str(host_park)
    os.environ["WATERZ_AFF_PARK"] = "0"
    build()
    lib = ctypes.CDLL(str(ROOT / "src/libws_gpu.so"))
    bind(lib)
    lib.ws_n9_reset.restype = None
    lib.ws_n10_stats.restype = None
    lib.ws_n10_stats.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.ws_n9_reset()
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    times = []
    last = None
    for _ in range(3):
        lib.ws_n9_reset()
        import time
        t0 = time.perf_counter()
        seg, meta = run_split(lib, aff, park=True)
        times.append((time.perf_counter() - t0) * 1000.0)
        last = (seg, meta, n10_stats(lib))
    seg, meta, phases = last
    nfrag, bg = nfrag_bg(seg)
    oracle = None
    op = CACHE / "wz_fragments.npy"
    if op.exists():
        ref = np.ascontiguousarray(np.load(op), dtype=np.uint32)
        oracle = bool(ref.shape == seg.shape and np.array_equal(ref, seg))
    times.sort()
    return {
        "host_park": host_park,
        "nfrag": nfrag,
        "bg": bg,
        "oracle_array_equal": oracle,
        "identity": bool(oracle is not False and nfrag == FRAGMENTS_VAL),
        "ws_ms_median": times[len(times) // 2],
        "ws_ms": times,
        "divide_ms": float(meta.get("divide_ms", 0)),
        "phases": phases,
    }


def main():
    print("N10 WS split. Not a 2 Gvox/s claim.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N10 REFUSE card busy: {busy}", flush=True)
        raise SystemExit(2)
    print(f"GPU idle: {gpu_state()}", flush=True)
    rows = []
    for park in (1, 0):
        r = subprocess.run(
            [sys.executable, str(Path(__file__)), "--park", str(park)],
            cwd=str(ROOT), capture_output=True, text=True, check=False,
        )
        line = None
        for ln in r.stdout.splitlines():
            if ln.startswith("{"):
                line = ln
        if line is None:
            print(f"N10 park={park} failed rc={r.returncode}\n"
                  f"stdout={r.stdout[-2000:]}\nstderr={r.stderr[-2000:]}",
                  flush=True)
            raise SystemExit(1)
        doc = json.loads(line)
        rows.append(doc)
        print(
            f"N10 park={park} identity={doc['identity']} "
            f"ws={doc['ws_ms_median']:.2f} phases={doc['phases']}",
            flush=True,
        )
        sys.stderr.write(r.stderr[-3000:])
    on = next(x for x in rows if x["host_park"] == 1)
    off = next(x for x in rows if x["host_park"] == 0)
    out = {
        "claim": "not a 2 Gvox/s number",
        "gpu": gpu_state(),
        "rows": rows,
        "speedup_no_host_park": (
            on["ws_ms_median"] / off["ws_ms_median"]
            if off["ws_ms_median"] else None
        ),
        "both_identity": bool(on["identity"] and off["identity"]),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"N10 speedup={out['speedup_no_host_park']} "
        f"ident={out['both_identity']} -> {OUT}",
        flush=True,
    )
    return 0 if out["both_identity"] else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--park":
        doc = one(int(sys.argv[2]))
        print(json.dumps(doc), flush=True)
        raise SystemExit(0 if doc["identity"] else 1)
    raise SystemExit(main())
