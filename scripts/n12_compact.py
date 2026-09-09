#!/usr/bin/env python3
"""N12: WATERZ_COMPACT_EVERY k=2/4/8 after nsys. T=0.3 VOI + four-T ε=0.08.

Kill as default if T=0.3 split>0.4738 or merge>0.2611 or wall cut <1.2x.
Keep the env lever. Not a 2 Gvox/s claim. Not 3090 Ti.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import AFF_THRESHOLDS, CACHE, grade_parents, load_rag  # noqa: E402
from e6r_parhac import DSO, compile_d  # noqa: E402
from p1_make_big_indep import card_busy, gpu_state  # noqa: E402
from t3_memsafe import grade_t3, voi_parent_mmap  # noqa: E402

NOTE = ROOT / "notes/N12_COMPACT.md"
OUT = CACHE / "n12_compact.json"
KS = (0, 2, 4, 8)


def bind(lib):
    lib.parhac_paper_d.restype = ctypes.c_int
    lib.parhac_paper_d.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    return lib


def run_parhac(lib, u, v, sm, ct, max_id, thrs, eps):
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
    return rc, parents, stats, ms


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--insert-only":
        os.environ["WATERZ_HASH_INSERT_ONLY"] = "1"
        os.environ["WATERZ_AGG_LEVERS"] = "15"
        os.environ["WATERZ_COMPACT_EVERY"] = "0"
        compile_d()
        lib = bind(ctypes.CDLL(str(DSO)))
        u, v, sm, ct, fr, max_id = load_rag()
        t3 = np.asarray([0.3], dtype=np.float64)
        rc, parents, stats, ms = run_parhac(lib, u, v, sm, ct, max_id, t3, 0.40)
        split, merge, nseg = voi_parent_mmap(parents[0])
        ok, _, _ = grade_t3(split, merge)
        print(json.dumps({
            "ms": ms, "split": split, "merge": merge, "nseg": nseg,
            "ok": bool(ok), "rc": rc,
        }), flush=True)
        return 0

    print("Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N12 REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)
    os.environ["WATERZ_AGG_LEVERS"] = "15"
    compile_d()
    lib = bind(ctypes.CDLL(str(DSO)))
    u, v, sm, ct, fr, max_id = load_rag()
    t3 = np.asarray([0.3], dtype=np.float64)
    four = np.asarray(AFF_THRESHOLDS, dtype=np.float64)
    rows = []
    base_ms = None
    for k in KS:
        os.environ["WATERZ_COMPACT_EVERY"] = str(k)
        # parhac getenv is per process; reload so compact_k is fresh
        # compact_k is read once at parhac_e6s_dev entry, not cached across calls
        rc, parents, stats, ms = run_parhac(lib, u, v, sm, ct, max_id, t3, 0.40)
        split, merge, nseg = voi_parent_mmap(parents[0])
        ok, sl, ml = grade_t3(split, merge)
        if k == 0:
            base_ms = ms
        cut = (base_ms / ms) if (base_ms and ms > 0) else None
        rc4, p4, st4, ms4 = run_parhac(lib, u, v, sm, ct, max_id, four, 0.08)
        four_ok = False
        if rc4 == 1:
            four_ok = grade_parents(p4, fr, f"N12 compact k={k}", f"n12_compact_k{k}")
        row = {
            "k": k,
            "t03_ms": ms,
            "t03_rc": rc,
            "voi_split": split,
            "voi_merge": merge,
            "nseg": nseg,
            "t03_pass": bool(ok),
            "wall_cut": cut,
            "four_ms": ms4,
            "four_pass": bool(four_ok),
            "task_legal": bool(ok and four_ok),
            "keep_default": bool(
                ok and four_ok and cut is not None and cut >= 1.2
            ),
        }
        rows.append(row)
        print(
            f"N12 compact k={k} t03={ms:.1f}ms split={split:.4f} merge={merge:.4f} "
            f"pass={ok} cut={cut} four={four_ok} keep={row['keep_default']}",
            flush=True,
        )
    insert = None
    r = __import__("subprocess").run(
        [sys.executable, str(Path(__file__)), "--insert-only"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    for ln in r.stdout.splitlines():
        if ln.startswith("{"):
            insert = json.loads(ln)
    if insert is None:
        insert = {"stderr": r.stderr[-1500:], "rc": r.returncode}
    doc = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "gpu": gpu_state(),
        "rows": rows,
        "insert_only": insert,
        "keep_default_k": next((r["k"] for r in rows if r["keep_default"]), 0),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    lines = ["# N12 compact-every", "", "Not a 2 Gvox/s claim. Not 3090 Ti.", ""]
    for r in rows:
        lines.append(
            f"- k={r['k']} t03={r['t03_ms']:.1f}ms split={r['voi_split']:.4f} "
            f"merge={r['voi_merge']:.4f} cut={r['wall_cut']} four={r['four_pass']} "
            f"keep_default={r['keep_default']}"
        )
    lines += ["", f"insert_only={insert}", "",
              f"default compact stays k={doc['keep_default_k']} (0 = layer-only).", ""]
    NOTE.write_text("\n".join(lines) + "\n")
    print(f"N12 compact -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
