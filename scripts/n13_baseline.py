#!/usr/bin/env python3
"""N13 P0: freeze UF=3 parks-off against N12 fatbin 4923.9 ms ±2%.

Not a 2 Gvox/s claim. Not 3090 Ti. Idle-5090 only.
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
from e6r_parhac import DSO, compile_d  # noqa: E402
from p1_make_big_indep import AFF, CACHE, card_busy, gpu_state, nfrag_bg  # noqa: E402
from task_gate import FRAGMENTS_VAL  # noqa: E402
from t3_memsafe import grade_t3, voi_parent_mmap  # noqa: E402
from _agg_common import load_rag  # noqa: E402

NOTE = ROOT / "notes/N13_BASELINE.md"
OUT = CACHE / "n13_baseline.json"
N12_E2E = 4923.9
TOL = 0.02
BG_VAL = 506_568


def parks_env(extra=None):
    e = os.environ.copy()
    e["WATERZ_UF_ALGO"] = "3"
    e["WATERZ_HOST_PARK"] = "0"
    e["WATERZ_AFF_PARK"] = "0"
    e["WATERZ_STAGE_MS"] = "1"
    e["WATERZ_AGG_LEVERS"] = "15"
    e.pop("WATERZ_AGG_EPS", None)
    e.pop("WATERZ_PAPER_E6T", None)
    if extra:
        e.update(extra)
    return e


def ident():
    os.environ.update({
        "WATERZ_UF_ALGO": "3",
        "WATERZ_HOST_PARK": "0",
        "WATERZ_AFF_PARK": "0",
    })
    lib = ctypes.CDLL(str(ROOT / "src/libws_gpu.so"))
    bind(lib)
    with h5py.File(AFF, "r") as f:
        aff = np.ascontiguousarray(f["affinity"][:], dtype=np.uint8)
    seg, _ = run_split(lib, aff, park=True)
    nfrag, bg = nfrag_bg(seg)
    gold = np.load(CACHE / "wz_fragments.npy")
    eq = bool(np.array_equal(seg, gold))
    return {
        "identity": bool(eq and nfrag == FRAGMENTS_VAL and bg == BG_VAL),
        "array_equal": eq,
        "nfrag": nfrag,
        "bg": bg,
    }


def val_t03():
    os.environ["WATERZ_AGG_LEVERS"] = "15"
    os.environ.pop("WATERZ_PAPER_E6T", None)
    compile_d()
    lib = ctypes.CDLL(str(DSO))
    lib.parhac_paper_d.restype = ctypes.c_int
    lib.parhac_paper_d.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    u, v, sm, ct, _fr, max_id = load_rag()
    thrs = np.asarray([0.3], dtype=np.float64)
    parents = np.empty((1, max_id + 1), dtype=np.uint32)
    stats = np.zeros((1, 3), dtype=np.int64)
    rc = lib.parhac_paper_d(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(1),
        ctypes.c_double(0.40),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    split, merge, nseg = voi_parent_mmap(parents[0])
    ok, sl, ml = grade_t3(split, merge)
    return {
        "rc": int(rc),
        "split": split,
        "merge": merge,
        "nseg": nseg,
        "ok": bool(ok),
        "limit_split": sl,
        "limit_merge": ml,
        "inner": int(stats[0, 2]),
        "merges": int(stats[0, 1]),
    }


def load_json_line(stdout: str):
    for ln in stdout.splitlines():
        if ln.startswith("{"):
            return json.loads(ln)
    return {}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--ident":
        print(json.dumps(ident()), flush=True)
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--val-t03":
        print(json.dumps(val_t03()), flush=True)
        return 0

    print("Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N13 P0 REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)
    build()
    compile_d()

    idr = subprocess.run(
        [sys.executable, str(Path(__file__)), "--ident"],
        cwd=str(ROOT), env=parks_env(), capture_output=True, text=True,
    )
    ident_doc = load_json_line(idr.stdout)
    print(f"N13 P0 ident {ident_doc} stderr_tail={idr.stderr[-400:]}", flush=True)

    t03r = subprocess.run(
        [sys.executable, str(Path(__file__)), "--val-t03"],
        cwd=str(ROOT), env=parks_env(), capture_output=True, text=True,
    )
    t03 = load_json_line(t03r.stdout)
    print(f"N13 P0 T=0.3 {t03} stderr_tail={t03r.stderr[-400:]}", flush=True)

    r216 = subprocess.run(
        [sys.executable, "-u", str(ROOT / "scripts/n8_run216.py")],
        cwd=str(ROOT), env=parks_env({"WATERZ_SKIP_BUILD": "1"}),
    )
    run216 = {}
    p216 = CACHE / "n8_216.json"
    if p216.is_file():
        run216 = json.loads(p216.read_text())
    e2e = float(run216.get("e2e_ms") or 0)
    lo, hi = N12_E2E * (1 - TOL), N12_E2E * (1 + TOL)
    within = bool(lo <= e2e <= hi)
    stack_ok = bool(
        ident_doc.get("identity") and t03.get("ok") and within and r216.returncode == 0
    )
    doc = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "gpu": gpu_state(),
        "ident": ident_doc,
        "val_t03": t03,
        "run216": run216,
        "n12_e2e_ms": N12_E2E,
        "e2e_ms": e2e,
        "within_2pct": within,
        "stack_ok": stack_ok,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        "# N13 P0 baseline\n\n"
        "Not a 2 Gvox/s claim. Not 3090 Ti. Parks-off UF=3.\n\n"
        f"- ident identity={ident_doc.get('identity')} nfrag={ident_doc.get('nfrag')} "
        f"bg={ident_doc.get('bg')}\n"
        f"- T=0.3 split={t03.get('split')} merge={t03.get('merge')} ok={t03.get('ok')}\n"
        f"- 2.16 e2e={e2e:.1f} ms vs N12 {N12_E2E} within_2pct={within} "
        f"stages={run216.get('stages')}\n"
        f"- stack_ok={stack_ok}\n"
    )
    print(f"N13 P0 stack_ok={stack_ok} e2e={e2e:.1f} -> {OUT}", flush=True)
    if not stack_ok:
        print("N13 P0 STOP campaign: card/bin drifted from N12.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
