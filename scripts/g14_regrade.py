#!/usr/bin/env python3
"""G14: G4r → G5r → G6r after current WS/AGG locks."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ref_cpu import extract_parent  # noqa: E402
from segment import _as_u8, _parhac, _rag, _watershed  # noqa: E402

import h5py

PY = ROOT / ".venv/bin/python"
OUT = ROOT / "data/ws_bounty"
AFF = OUT / "cremiA_val/affinity.h5"


def main():
    cmd = [str(PY), str(ROOT / "src/segment.py"), str(AFF), "--out-dir", str(OUT)]
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(r.returncode)
    paths = [str(OUT / f"mine_thr{t}.h5") for t in (0.2, 0.3, 0.4, 0.5)]
    grade = [str(PY), str(OUT / "baseline/run_baseline.py"), "--candidate", *paths]
    print("+", " ".join(grade), flush=True)
    g = subprocess.run(grade, cwd=str(OUT), capture_output=True, text=True)
    sys.stdout.write(g.stdout)
    if "ACCURACY GATE: PASS" not in g.stdout:
        print("G14 G4r FAIL")
        raise SystemExit(1)
    print("G14 G4r PASS")
    with h5py.File(AFF, "r") as f:
        aff = _as_u8(f["affinity"][:])
    t0 = time.perf_counter()
    fr = _watershed(aff, 1e-4, 0.9999)
    tws = time.perf_counter() - t0
    t0 = time.perf_counter()
    u, v, sm, ct = _rag(aff, fr)
    trag = time.perf_counter() - t0
    t0 = time.perf_counter()
    snaps = _parhac(u, v, sm, ct, [0.3], max_id=int(fr.max()))
    tagg = time.perf_counter() - t0
    t0 = time.perf_counter()
    extract_parent(fr, snaps[0.3])
    text = time.perf_counter() - t0
    tot = tws + trag + tagg + text
    print(
        f"G14 G6r one-shot ws={tws:.3f} rag={trag:.3f} agg={tagg:.3f} "
        f"extract={text:.3f} total={tot:.3f} "
        f"{'PASS' if tot < 0.050 else 'FAIL'} (need <0.050s)"
    )
    print("G14 G9 BLOCKED — no 3090 Ti")


if __name__ == "__main__":
    main()
