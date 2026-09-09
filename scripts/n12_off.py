#!/usr/bin/env python3
"""N12 off-contract speed probes. Always grade, never default on FAIL.

Not a 2 Gvox/s claim. Not 3090 Ti. Skip GPU FIFO hist-q (BinQueue dead).
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import (  # noqa: E402
    AFF_THRESHOLDS, CACHE, grade_parents, load_rag, run_kruskal,
)
from e6r_parhac import DSO, compile_d  # noqa: E402
from p1_make_big_indep import AFF, card_busy, gpu_state  # noqa: E402
from t3_memsafe import grade_t3, voi_parent_mmap  # noqa: E402

NOTE = ROOT / "notes/N12_OFF.md"
OUT = CACHE / "n12_off.json"
EPS = (0.5, 0.8, 1.0, 2.0)
PRED_SDSL = 1


def parhac(eps, thrs):
    u, v, sm, ct, fr, max_id = load_rag()
    lib = ctypes.CDLL(str(DSO))
    lib.parhac_paper_d.restype = ctypes.c_int
    lib.parhac_paper_d.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(thrs), 3), dtype=np.int64)
    t0 = time.perf_counter()
    rc = lib.parhac_paper_d(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(thrs)),
        ctypes.c_double(eps),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    ms = (time.perf_counter() - t0) * 1000.0
    return rc, parents, fr, ms, max_id


def write_pruf_slices(aff_u8, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    vol = aff_u8.max(axis=0)
    z, y, x = vol.shape
    for zi in range(z):
        p = dest / f"{zi:04d}.pgm"
        sl = np.ascontiguousarray(vol[zi], dtype=np.uint8)
        with p.open("wb") as f:
            f.write(f"P5\n{x} {y}\n255\n".encode("ascii"))
            f.write(sl.tobytes())
    return z, y, x


def main():
    print("Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N12 REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)
    os.environ["WATERZ_AGG_LEVERS"] = "15"
    compile_d()
    rows = []
    t3 = np.asarray([0.3], dtype=np.float64)

    for eps in EPS:
        rc, parents, fr, ms, max_id = parhac(eps, t3)
        split, merge, nseg = voi_parent_mmap(parents[0])
        ok, _, _ = grade_t3(split, merge)
        row = {
            "id": f"eps_{eps}",
            "wall_ms": ms,
            "nseg": nseg,
            "voi_split": split,
            "voi_merge": merge,
            "task_legal": bool(ok),
            "rc": rc,
        }
        rows.append(row)
        print(
            f"N12 off eps={eps} {ms:.1f}ms split={split:.4f} merge={merge:.4f} "
            f"legal={ok}",
            flush=True,
        )

    t0 = time.perf_counter()
    rc_k, parents_k, fr_k, stats_k, max_id_k = run_kruskal(PRED_SDSL, 1, 0.3)
    ms_k = (time.perf_counter() - t0) * 1000.0
    split, merge, nseg = voi_parent_mmap(parents_k[1] if parents_k.shape[0] > 1 else parents_k[0])
    ok, _, _ = grade_t3(split, merge)
    four_ok = False
    try:
        four_ok = grade_parents(parents_k, fr_k, "N12 kruskal", "n12_kruskal")
    except Exception as e:
        four_ok = False
        print(f"N12 kruskal four-T grade err {e}", flush=True)
    rows.append({
        "id": "kruskal_frozen",
        "wall_ms": ms_k,
        "nseg": nseg,
        "voi_split": split,
        "voi_merge": merge,
        "task_legal": bool(ok and four_ok),
        "vs_parhac_val_202ms": ms_k,
        "rc": int(rc_k),
        "four_pass": bool(four_ok),
    })
    print(f"N12 off kruskal {ms_k:.1f}ms legal={ok and four_ok}", flush=True)

    mutex = {"id": "mutex_hop", "task_legal": False}
    try:
        t0 = time.perf_counter()
        r = subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts/m16b_hop.py")],
            cwd=str(ROOT), capture_output=True, text=True, timeout=600,
        )
        mutex.update({
            "wall_ms": (time.perf_counter() - t0) * 1000.0,
            "rc": r.returncode,
            "stdout_tail": r.stdout[-1500:],
            "task_legal": False,
            "expect": "M16 FAIL split 0.91",
        })
    except Exception as e:
        mutex["error"] = str(e)
    rows.append(mutex)
    print(f"N12 off mutex {mutex.get('rc')} {mutex.get('error')}", flush=True)

    pruf = {"id": "PRUF3D", "task_legal": False}
    pruf_src = ROOT / "papers/repos/PRUF-watershed"
    try:
        import h5py
        with h5py.File(AFF, "r") as f:
            aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
        slas = Path("/tmp/n12_pruf_in")
        outd = Path("/tmp/n12_pruf_out")
        outd.mkdir(parents=True, exist_ok=True)
        z, y, x = write_pruf_slices(aff, slas)
        mk = subprocess.run(
            ["make", "PRUF3D", "NVCCFLAGS=-O3 -arch=sm_120 --use_fast_math --std=c++17"],
            cwd=str(pruf_src), capture_output=True, text=True, timeout=180,
        )
        binp = pruf_src / "PRUF3D"
        if mk.returncode != 0 or not binp.is_file():
            pruf.update({"build_fail": mk.stderr[-1500:], "rc": mk.returncode})
        else:
            t0 = time.perf_counter()
            run = subprocess.run(
                [str(binp), str(slas), str(outd)],
                capture_output=True, text=True, timeout=180,
            )
            pruf.update({
                "wall_ms": (time.perf_counter() - t0) * 1000.0,
                "rc": run.returncode,
                "stdout": run.stdout[-800:],
                "shape": [z, y, x],
                "vs_ws_val_344ms": True,
                "task_legal": False,
                "note": "grayscale Meyer, not waterz fragments",
            })
    except Exception as e:
        pruf["error"] = str(e)
    rows.append(pruf)
    print(f"N12 off PRUF {pruf}", flush=True)

    rama = {"id": "RAMA", "task_legal": False, "built": False}
    rama_src = ROOT / "papers/repos/RAMA"
    try:
        cm = subprocess.run(
            ["cmake", "-S", str(rama_src), "-B", "/tmp/n12_rama_build"],
            capture_output=True, text=True, timeout=120,
        )
        rama["cmake_rc"] = cm.returncode
        rama["cmake_tail"] = (cm.stdout + cm.stderr)[-1500:]
        rama["built"] = cm.returncode == 0
    except Exception as e:
        rama["error"] = str(e)
    rows.append(rama)

    rows.append({
        "id": "cuSLINK",
        "task_legal": False,
        "note": "cuVS build_sorted_mst is frozen MST / single-linkage; Kruskal row is the executable analog",
    })
    rows.append({
        "id": "hist_q_gpu_fifo",
        "skipped": True,
        "note": "N11 BinQueue dead; CPU Q20 already four-T PASS",
    })

    doc = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "gpu": gpu_state(),
        "rows": rows,
        "default_any": False,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2, default=str) + "\n")
    lines = ["# N12 off-contract", "", "Not a 2 Gvox/s claim. Not 3090 Ti.",
             "Never default on FAIL.", ""]
    for r in rows:
        lines.append(
            f"- {r.get('id')} legal={r.get('task_legal')} wall={r.get('wall_ms')} "
            f"nseg={r.get('nseg')} voi={r.get('voi_split')}/{r.get('voi_merge')}"
        )
    NOTE.write_text("\n".join(lines) + "\n")
    print(f"N12 off -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
