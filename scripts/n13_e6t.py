#!/usr/bin/env python3
"""N13 L2: idle E6t StarMerge. No second compact kernel.

Not a 2 Gvox/s claim. Not 3090 Ti. Idle-5090 only.
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import AFF_THRESHOLDS, CACHE, grade_parents, load_rag  # noqa: E402
from e6r_parhac import DSO, compile_d  # noqa: E402
from p1_make_big_indep import card_busy, gpu_state  # noqa: E402
from t3_memsafe import grade_t3, voi_parent_mmap  # noqa: E402

NOTE = ROOT / "notes/N13_E6T.md"
OUT = CACHE / "n13_e6t.json"
N12_AGG = 2230.0


def parks_env(e6t: bool):
    e = os.environ.copy()
    e["WATERZ_UF_ALGO"] = "3"
    e["WATERZ_HOST_PARK"] = "0"
    e["WATERZ_AFF_PARK"] = "0"
    e["WATERZ_STAGE_MS"] = "1"
    e["WATERZ_AGG_LEVERS"] = "15"
    e.pop("WATERZ_AGG_EPS", None)
    if e6t:
        e["WATERZ_PAPER_E6T"] = "1"
    else:
        e.pop("WATERZ_PAPER_E6T", None)
    return e


def bind_timed(lib):
    lib.parhac_paper_d_timed.restype = ctypes.c_int
    lib.parhac_paper_d_timed.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_double),
    ]


def run_one(eps, thrs, want_parents=False):
    compile_d()
    u, v, sm, ct, fr, max_id = load_rag()
    lib = ctypes.CDLL(str(DSO))
    bind_timed(lib)
    t = np.asarray(thrs, dtype=np.float64)
    parents = np.empty((len(t), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(t), 3), dtype=np.int64)
    device_ms = ctypes.c_double(0.0)
    rc = lib.parhac_paper_d_timed(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        t.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(len(t)),
        ctypes.c_double(eps),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.byref(device_ms),
    )
    out = {
        "rc": int(rc),
        "device_ms": float(device_ms.value),
        "outer": [int(x) for x in stats[:, 0]],
        "merges": [int(x) for x in stats[:, 1]],
        "inner": [int(x) for x in stats[:, 2]],
        "eps": eps,
        "thrs": list(thrs),
        "e6t": bool(os.environ.get("WATERZ_PAPER_E6T")),
    }
    if want_parents and rc == 1 and len(t) == 1:
        split, merge, nseg = voi_parent_mmap(parents[0])
        ok, sl, ml = grade_t3(split, merge)
        out.update({
            "split": split, "merge": merge, "nseg": nseg,
            "ok": bool(ok), "limit_split": sl, "limit_merge": ml,
        })
        np.save(CACHE / "n13_e6t_parents.npy", parents[0])
    elif want_parents and rc == 1:
        out["four_pass"] = bool(
            grade_parents(parents, fr, "N13 E6t four-T", "n13_e6t_four")
        )
    print(json.dumps(out), flush=True)
    return 0 if rc == 1 else 1


def child():
    mode = sys.argv[2]
    if mode == "t03":
        return run_one(0.40, [0.3], want_parents=True)
    if mode == "t03-noparents":
        return run_one(0.40, [0.3], want_parents=False)
    if mode == "four":
        return run_one(0.08, list(AFF_THRESHOLDS), want_parents=True)
    raise SystemExit(f"unknown mode {mode}")


def load_json_line(stdout: str):
    last = {}
    for ln in stdout.splitlines():
        if ln.startswith("{"):
            last = json.loads(ln)
    return last


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        return child()

    print("Not a 2 Gvox/s claim. Not 3090 Ti.", flush=True)
    busy = card_busy()
    if busy:
        print(f"N13 L2 REFUSE card busy: {busy}", flush=True)
        return 2
    print(f"GPU idle: {gpu_state()}", flush=True)
    compile_d()

    def sub(e6t, mode):
        r = subprocess.run(
            [sys.executable, str(Path(__file__)), "--child", mode],
            cwd=str(ROOT), env=parks_env(e6t), capture_output=True, text=True,
        )
        doc = load_json_line(r.stdout)
        print(
            f"N13 L2 e6t={e6t} mode={mode} rc={r.returncode} {doc} "
            f"stderr_tail={r.stderr[-500:].replace(chr(10), ' | ')}",
            flush=True,
        )
        return r, doc

    s_run, e6s = sub(False, "t03-noparents")
    t_run, e6t = sub(True, "t03")
    voi_ok = bool(e6t.get("ok"))
    e6s_ms = float(e6s.get("device_ms") or 0)
    e6t_ms = float(e6t.get("device_ms") or 0)
    not_slower = bool(e6s_ms > 0 and e6t_ms <= 0.85 * e6s_ms)

    r1, d1 = sub(True, "t03")
    r2, d2 = sub(True, "t03")
    p1 = CACHE / "n13_e6t_parents.npy"
    # second run overwrote the npy; re-run once saving both
    a = np.asarray(d1.get("nseg"))
    # parents file is from last child; run two dedicated saves
    ident = False
    ndiff = None
    pa = CACHE / "n13_e6t_p0.npy"
    pb = CACHE / "n13_e6t_p1.npy"
    for i, dest in enumerate((pa, pb)):
        r = subprocess.run(
            [sys.executable, str(Path(__file__)), "--child", "t03"],
            cwd=str(ROOT), env=parks_env(True), capture_output=True, text=True,
        )
        src = CACHE / "n13_e6t_parents.npy"
        if src.is_file():
            np.save(dest, np.load(src))
        print(f"N13 L2 det run{i} {load_json_line(r.stdout)}", flush=True)
    if pa.is_file() and pb.is_file():
        a0, a1 = np.load(pa), np.load(pb)
        ident = bool(np.array_equal(a0, a1))
        ndiff = int((a0 != a1).sum())

    four = None
    run216 = None
    keep = False
    if voi_ok and ident:
        _, four = sub(True, "four")
        four_ok = bool(four.get("four_pass"))
        if four_ok:
            r = subprocess.run(
                [sys.executable, "-u", str(ROOT / "scripts/n8_run216.py")],
                cwd=str(ROOT), env=parks_env(True),
            )
            p216 = CACHE / "n8_216.json"
            if p216.is_file():
                run216 = json.loads(p216.read_text())
            agg = float((run216 or {}).get("stages", {}).get("agg") or 0)
            keep = bool(
                voi_ok and ident and four_ok
                and agg > 0 and agg <= N12_AGG / 1.2
            )
    else:
        four_ok = False

    doc = {
        "claim": "not a 2 Gvox/s number; not 3090 Ti",
        "gpu": gpu_state(),
        "e6s_t03": e6s,
        "e6t_t03": e6t,
        "voi_ok": voi_ok,
        "e6s_ms": e6s_ms,
        "e6t_ms": e6t_ms,
        "not_slower_0p85": not_slower,
        "byte_identical": ident,
        "ndiff": ndiff,
        "four": four,
        "run216": run216,
        "keep_default": keep,
        "ship": bool(voi_ok and ident),
        "dirty_scan_ceiling_ms": 1598.5,
        "e2e_if_zero_dirty_ms": 4924.0 - 1598.5,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        "# N13 L2 idle E6t\n\n"
        "Not a 2 Gvox/s claim. Not 3090 Ti. No second StarMerge.\n\n"
        f"- E6s T=0.3 device_ms={e6s_ms:.1f}\n"
        f"- E6t T=0.3 device_ms={e6t_ms:.1f} split={e6t.get('split')} "
        f"merge={e6t.get('merge')} voi_ok={voi_ok}\n"
        f"- determinism byte_identical={ident} ndiff={ndiff}\n"
        f"- not_slower_0.85x={not_slower} four={four}\n"
        f"- 2.16={run216}\n"
        f"- keep_default={keep} (need VOI+four-T+ident+agg<=2230/1.2)\n"
        f"- ship={doc['ship']} (TASK line 118 blocks if ident False)\n"
        f"- dirty-scan ceiling 1598.5 ms; zeroing it leaves ~3326 ms e2e on 5090\n"
    )
    print(
        f"N13 L2 voi={voi_ok} ident={ident} ndiff={ndiff} "
        f"keep={keep} -> {OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
