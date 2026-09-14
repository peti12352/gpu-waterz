#!/usr/bin/env python3
"""E9c: GPU findbasins after E9b divide. nfrag/bg gate, then G4r if exact."""
from __future__ import annotations

import ctypes
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ref_cpu import extract_parent  # noqa: E402
from task_gate import (  # noqa: E402
    AFF_HIGH,
    AFF_LOW,
    AFF_THRESHOLDS,
    BG_VAL_MEASURED,
    FRAGMENTS_VAL,
    print_contract,
)

AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
OUT = ROOT / "data/ws_bounty"
WS_SO = ROOT / "src/libws_gpu.so"
RAG_SO = ROOT / "src/librag_gpu.so"
RAC_SO = ROOT / "src/librac_agg.so"
NVCC = "/usr/local/cuda-12.8/bin/nvcc"
PY = ROOT / ".venv/bin/python"
JITTER = 20


def compile_ws():
    subprocess.check_call([
        NVCC, "-O3", "-arch=sm_120", "--shared", "-Xcompiler", "-fPIC",
        "-I/usr/local/cuda-12.8/targets/x86_64-linux/include",
        "-o", str(WS_SO), str(ROOT / "csrc/ws.cu"),
    ])


def main():
    print_contract()
    compile_ws()
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    z, y, x = aff.shape[1:]
    seg = np.zeros((z, y, x), dtype=np.uint32)
    lib = ctypes.CDLL(str(WS_SO))
    lib.e9c_watershed.restype = ctypes.c_int
    lib.e9c_watershed.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    t0 = time.time()
    nfrag = lib.e9c_watershed(
        aff.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        z, y, x, ctypes.c_float(AFF_LOW), ctypes.c_float(AFF_HIGH),
        seg.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
    )
    wall = time.time() - t0
    bg = int((seg == 0).sum())
    dn = abs(int(nfrag) - FRAGMENTS_VAL)
    print(f"E9c nfrag={nfrag} task={FRAGMENTS_VAL} d={dn} bg={bg} "
          f"bg_task={BG_VAL_MEASURED} wall={wall:.3f}")
    if dn > JITTER:
        print("E9c FAIL nfrag: do not re-run E1")
        raise SystemExit(1)
    if bg != BG_VAL_MEASURED:
        print("E9c FAIL bg")
        raise SystemExit(1)
    print("E9c nfrag/bg PASS: G4r with locked RAG+AGG")
    sys.path.insert(0, str(ROOT / "src"))
    from segment import _parhac, _rag
    u, v, sm, ct = _rag(aff, seg)
    snaps = _parhac(u, v, sm, ct, AFF_THRESHOLDS, max_id=int(seg.max()))
    dest_dir = OUT / "e9c"
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for t in AFF_THRESHOLDS:
        lab = extract_parent(seg, snaps[float(t)]).astype(np.uint32, copy=False)
        dest = dest_dir / f"mine_thr{t}.h5"
        with h5py.File(dest, "w") as h:
            h.create_dataset("labels", data=lab)
        paths.append(str(dest))
        print(f"  wrote {dest}")
    cmd = [str(PY), str(OUT / "baseline/run_baseline.py"), "--candidate", *paths]
    r = subprocess.run(cmd, cwd=str(OUT), capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    ok = "ACCURACY GATE: PASS" in r.stdout
    print(f"E9c G4r {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
