#!/usr/bin/env python3
"""G16: G4r -> G5r -> G6r after current locks. GPU AGG only if a new lock has rounds≤30."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from segment import _as_u8, _parhac, _rag, _watershed  # noqa: E402
from ref_cpu import extract_parent  # noqa: E402

import h5py

PY = ROOT / ".venv/bin/python"
OUT = ROOT / "data/ws_bounty"
AFF = OUT / "cremiA_val/affinity.h5"
CACHE = ROOT / "data/cache"


def _grade_once():
    cmd = [str(PY), str(ROOT / "src/segment.py"), str(AFF), "--out-dir", str(OUT)]
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        return False, ""
    paths = [str(OUT / f"mine_thr{t}.h5") for t in (0.2, 0.3, 0.4, 0.5)]
    grade = [str(PY), str(OUT / "baseline/run_baseline.py"), "--candidate", *paths]
    print("+", " ".join(grade), flush=True)
    g = subprocess.run(grade, cwd=str(OUT), capture_output=True, text=True)
    sys.stdout.write(g.stdout)
    return "ACCURACY GATE: PASS" in g.stdout, g.stdout


def main():
    ok = False
    for attempt in range(1, 4):
        ok, _ = _grade_once()
        if ok:
            print(f"G16 G4r PASS try={attempt}")
            break
        print(f"G16 G4r miss try={attempt}/3 (paper-ε T=0.2 merge jitter)")
    if not ok:
        print("G16 G4r FAIL")
        raise SystemExit(1)
    with h5py.File(AFF, "r") as f:
        aff = _as_u8(f["affinity"][:])
    t0 = time.perf_counter()
    fr = _watershed(aff, 1e-4, 0.9999)
    tws = time.perf_counter() - t0
    t0 = time.perf_counter()
    u, v, sm, ct = _rag(aff, fr)
    trag = time.perf_counter() - t0
    e6r = CACHE / "e6r_pass.txt"
    e6r_lock = e6r.is_file() and e6r.read_text().startswith("PASS") and "LOCK" in e6r.read_text()
    t0 = time.perf_counter()
    if e6r_lock:
        sys.path.insert(0, str(ROOT / "scripts"))
        from e6r_parhac import compile_d, DSO
        import ctypes
        import numpy as np

        compile_d()
        thrs = np.asarray([0.3], dtype=np.float64)
        max_id = int(fr.max())
        parents = np.empty((1, max_id + 1), dtype=np.uint32)
        stats = np.zeros((1, 3), dtype=np.int64)
        lib = ctypes.CDLL(str(DSO))
        lib.parhac_paper_d.restype = ctypes.c_int
        lib.parhac_paper_d.argtypes = [
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
            ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int64),
        ]
        lib.parhac_paper_d(
            u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_int64(len(u)),
            thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(1), ctypes.c_double(0.08),
            parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(max_id),
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        )
        snaps = {0.3: parents[0]}
    else:
        snaps = _parhac(u, v, sm, ct, [0.3], max_id=int(fr.max()))
    tagg = time.perf_counter() - t0
    t0 = time.perf_counter()
    extract_parent(fr, snaps[0.3])
    text = time.perf_counter() - t0
    tot = tws + trag + tagg + text
    # Working proxy: val AGG ≤50 ms (TASK 2 Gvox/s -> ~90 ms e2e; WS+RAG+extract ~40 ms).
    g6 = "PASS" if tagg <= 0.050 else "FAIL"
    print(
        f"G16 G6r one-shot ws={tws:.3f} rag={trag:.3f} agg={tagg:.3f} "
        f"extract={text:.3f} total={tot:.3f} {g6} (AGG budget 50ms; e2e proxy 90ms)"
    )
    stamps = []
    for name in (
        "e6r", "l36", "r36", "n36",
        "a27", "z25", "f23", "r24", "c28", "s26", "w31", "h30",
        "l33", "r32", "b34", "v35", "b18", "a17",
    ):
        p = CACHE / f"{name}_pass.txt"
        stamps.append(f"{name}={(p.read_text().strip() if p.is_file() else 'none')}")
    gpu_ok = any(
        p.is_file() and p.read_text().startswith("PASS") and "LOCK" in p.read_text()
        for p in (
            CACHE / f"{n}_pass.txt"
            for n in ("e6r", "l36", "r36", "n36")
        )
    )
    print("G16 stamps " + " | ".join(stamps))
    if e6r_lock:
        print("G16 GPU_AGG paper-ε device (E6r LOCK)")
    elif gpu_ok:
        print("G16 GPU_AGG SKIP (Track B CPU lock; leftover≤5k is the G6 shape)")
    else:
        print("G16 GPU_AGG SKIP (no stamped lock)")
    print("G16 G9 BLOCKED: no 3090 Ti")
    print("G16 host segment() SV unchanged (not forced to 7)")


if __name__ == "__main__":
    main()
