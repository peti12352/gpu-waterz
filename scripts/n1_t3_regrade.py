#!/usr/bin/env python3
"""N1: T=0.3-only regrade. Incremental JSON. Chunked mmap VOI. No 180 Mvox labels."""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from t3_memsafe import CACHE, T, grade_t3, voi_parent_mmap  # noqa: E402

SO_CC = ROOT / "src/libfrozen_cc.so"
SO_AGG = ROOT / "src/librac_agg.so"


def compile_one(so, src):
    if so.exists() and so.stat().st_mtime >= src.stat().st_mtime:
        return
    subprocess.check_call(
        ["g++", "-O3", "-DNDEBUG", "-shared", "-fPIC", "-o", str(so), str(src)]
    )


def row(tag, split, merge, **extra):
    ok, sl, ml = grade_t3(split, merge)
    shallow = extra.pop("shallow", False)
    keep = bool(ok and shallow)
    out = {
        "tag": tag, "threshold": T,
        "voi_split": float(split), "voi_merge": float(merge),
        "pass": bool(ok), "sl": sl, "ml": ml,
        "shallow": bool(shallow), "keep": keep, **extra,
    }
    print(f"N1 {tag:28s} split={split:.4f} merge={merge:.4f}  "
          f"{'PASS' if ok else 'FAIL'}  shallow={int(shallow)}  "
          f"{'KEEP' if keep else 'drop'}", flush=True)
    return out


def save(dest, rows):
    keep = [r for r in rows if r["keep"]]
    dest.write_text(json.dumps({
        "threshold": T, "rows": rows, "n_keep": len(keep),
        "keep_tags": [r["tag"] for r in keep],
        "note": "KEEP = T=0.3 VOI PASS and shallow. Incremental write.",
    }, indent=2, default=float) + "\n")


def already(rows, tag):
    return any(r["tag"] == tag for r in rows)


def frozen_cc(u, v, mean, thr, max_id):
    import ctypes
    compile_one(SO_CC, ROOT / "src/frozen_cc.cpp")
    parent = np.empty(max_id + 1, dtype=np.uint32)
    lib = ctypes.CDLL(str(SO_CC))
    lib.frozen_cc_cpu.restype = ctypes.c_int
    lib.frozen_cc_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.c_int64, ctypes.c_double,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
    ]
    lib.frozen_cc_cpu(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        mean.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(len(u)), ctypes.c_double(thr),
        parent.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
    )
    return parent


def mutex_t3(u, v, sm, ct, max_id):
    import ctypes
    compile_one(SO_AGG, ROOT / "src/rac_agg.cpp")
    thrs = np.asarray([T], dtype=np.float64)
    parents = np.empty((1, max_id + 1), dtype=np.uint32)
    stats = np.zeros((1, 3), dtype=np.int64)
    lib = ctypes.CDLL(str(SO_AGG))
    lib.mutex_rag_cpu.restype = ctypes.c_int
    lib.mutex_rag_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    rc = lib.mutex_rag_cpu(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(1),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    return rc, parents[0], stats[0]


def kruskal_t3(u, v, sm, ct, sz, max_id, pred, n_bins, p0, p1):
    import ctypes
    compile_one(SO_AGG, ROOT / "src/rac_agg.cpp")
    thrs = np.asarray([T], dtype=np.float64)
    parents = np.empty((1, max_id + 1), dtype=np.uint32)
    stats = np.zeros((1, 3), dtype=np.int64)
    lib = ctypes.CDLL(str(SO_AGG))
    lib.kruskal_pred_cpu.restype = ctypes.c_int
    lib.kruskal_pred_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    rc = lib.kruskal_pred_cpu(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        sz.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(1),
        ctypes.c_int(pred), ctypes.c_int(n_bins),
        ctypes.c_double(p0), ctypes.c_double(p1),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    return rc, parents[0], stats[0]


def _compress(parent):
    while True:
        nxt = parent[parent]
        if np.array_equal(nxt, parent):
            return nxt
        parent = nxt


def waterfall_passes(u, v, mean, max_id, thr, npass):
    n = max_id + 1
    parent = np.arange(n, dtype=np.uint32)
    live = (mean > thr) & (u != 0) & (v != 0) & (u != v)
    uu, vv, mm = u[live], v[live], mean[live]
    nmerge = used = 0
    for _ in range(npass):
        parent = _compress(parent)
        fu, fv = parent[uu], parent[vv]
        ok = (fu != fv) & (fu != 0) & (fv != 0)
        if not ok.any():
            used += 1
            break
        best = np.full(n, -1.0, dtype=np.float64)
        np.maximum.at(best, fu[ok], mm[ok])
        np.maximum.at(best, fv[ok], mm[ok])
        take = ok & ((np.abs(mm - best[fu]) < 1e-12) | (np.abs(mm - best[fv]) < 1e-12))
        nm = 0
        for a, b in zip(fu[take].tolist(), fv[take].tolist()):
            while parent[a] != a:
                a = int(parent[a])
            while parent[b] != b:
                b = int(parent[b])
            if a == b:
                continue
            if a < b:
                parent[b] = a
            else:
                parent[a] = b
            nm += 1
        nmerge += nm
        used += 1
        del best, take, fu, fv, ok
        if nm == 0:
            break
    return _compress(parent), nmerge, used


def frag_sizes_mmap(max_id):
    fr = np.load(CACHE / "wz_fragments.npy", mmap_mode="r").reshape(-1)
    sz = np.zeros(max_id + 1, dtype=np.int64)
    step = 1 << 20
    for i in range(0, fr.size, step):
        sl = np.asarray(fr[i:i + step], dtype=np.uint32)
        sz += np.bincount(sl, minlength=max_id + 1)
        del sl
    return sz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="n1_t3_regrade.json")
    args = ap.parse_args()
    dest = CACHE / args.out
    rows = []
    if dest.is_file():
        try:
            rows = list(json.loads(dest.read_text()).get("rows") or [])
        except (OSError, json.JSONDecodeError):
            rows = []

    z = np.load(CACHE / "rag.npz")
    u = np.ascontiguousarray(z["keys"][:, 0], dtype=np.uint32)
    v = np.ascontiguousarray(z["keys"][:, 1], dtype=np.uint32)
    sm = np.ascontiguousarray(z["stats"][:, 0], dtype=np.float64)
    ct = np.ascontiguousarray(np.rint(z["stats"][:, 1]), dtype=np.int64)
    mean = sm / np.maximum(ct, 1)
    del z
    max_id = int(max(int(u.max()), int(v.max())))
    print(f"N1 nedge={u.size} max_id={max_id} T={T} resume={len(rows)}", flush=True)

    e1 = json.loads((CACHE / "e1_t3_voi.json").read_text())
    locked = next(r for r in e1["rows"]
                  if abs(r["eps"] - 0.08) < 1e-12 and r["size_asym"])
    no_asym = next(r for r in e1["rows"]
                   if abs(r["eps"] - 0.08) < 1e-12 and not r["size_asym"])
    if not already(rows, "locked_eps0.08"):
        rows.append(row(
            "locked_eps0.08", locked["voi_split"], locked["voi_merge"],
            shallow=False, src="e1", note="control; ParHAC not shallow",
            nseg=locked.get("nseg"), nmerge=locked.get("nmerge"),
        ))
        save(dest, rows)
    if not already(rows, "no_asym_eps0.08"):
        rows.append(row(
            "no_asym_eps0.08", no_asym["voi_split"], no_asym["voi_merge"],
            shallow=False, src="e1",
            note="T=0.3 PASS already; still ParHAC depth",
            nseg=no_asym.get("nseg"),
        ))
        save(dest, rows)

    def grade_parent(tag, parent, **extra):
        split, merge, nseg = voi_parent_mmap(parent)
        del parent
        gc.collect()
        rows.append(row(tag, split, merge, nseg=nseg, **extra))
        save(dest, rows)

    if not already(rows, "x0_frozen_cc"):
        t0 = time.perf_counter()
        p = frozen_cc(u, v, mean, T, max_id)
        grade_parent("x0_frozen_cc", p, shallow=True,
                     wall_s=time.perf_counter() - t0,
                     note="Kruskal/CC mean>T; one UF")
    x0 = next(r for r in rows if r["tag"] == "x0_frozen_cc")
    for qtag in ("hist_q_p50_frozen", "hist_q_p85_frozen"):
        if already(rows, qtag):
            continue
        rows.append(row(
            qtag, x0["voi_split"], x0["voi_merge"], shallow=True,
            nseg=x0.get("nseg"), src="degenerate_x0",
            note="no per-face hist on rag.npz; degenerate delta at mean; ≡ X0",
        ))
        save(dest, rows)

    sz = None
    jobs = (
        ("zlateski_s0_256", 1, 256.0, 0.0, "SDSL mean>T and min(S)<256"),
        ("zlateski_s0_1024", 1, 1024.0, 0.0, "SDSL mean>T and min(S)<1024"),
        ("relcontact_g0.05_a0.67", 9, 0.05, 2.0 / 3.0,
         "mean>T and area >= 0.05*min(S)^(2/3)"),
        ("relcontact_g0.10_a0.67", 9, 0.10, 2.0 / 3.0,
         "mean>T and area >= 0.10*min(S)^(2/3)"),
        ("waterfall_full", 7, 1.0, 0.0, "C++ waterfall to fixpoint (<=30)"),
    )
    for tag, pred, p0, p1, note in jobs:
        if already(rows, tag):
            continue
        if sz is None:
            print("N1 building frag sizes from mmap", flush=True)
            sz = frag_sizes_mmap(max_id)
        t0 = time.perf_counter()
        rc, p, st = kruskal_t3(u, v, sm, ct, sz, max_id, pred, 1, p0, p1)
        grade_parent(tag, p, shallow=True, rc=int(rc),
                     nmerge=int(st[1]), rounds=int(st[0]),
                     wall_s=time.perf_counter() - t0, note=note)

    for npass in (1, 2):
        tag = f"waterfall_{npass}pass"
        if already(rows, tag):
            continue
        t0 = time.perf_counter()
        p, nmerge, used = waterfall_passes(u, v, mean, max_id, T, npass)
        grade_parent(tag, p, shallow=True, nmerge=nmerge, rounds=used,
                     wall_s=time.perf_counter() - t0,
                     note=f"lowest-pass union, {npass} pass cap")

    if not already(rows, "mutex_absmax"):
        t0 = time.perf_counter()
        try:
            rc, p, st = mutex_t3(u, v, sm, ct, max_id)
            grade_parent(
                "mutex_absmax", p, shallow=True, rc=int(rc),
                nmerge=int(st[1]) if st.size > 1 else None,
                wall_s=time.perf_counter() - t0,
                note="Wolf/GASP AbsMax; sort+|w|+UF",
            )
        except MemoryError:
            rows.append(row(
                "mutex_absmax", 9.0, 9.0, shallow=True, pass_override=False,
                note="MemoryError in mutex_rag_cpu; treated FAIL",
            ))
            rows[-1]["pass"] = False
            rows[-1]["keep"] = False
            save(dest, rows)
            print("N1 mutex_absmax MemoryError FAIL", flush=True)

    keep = [r["tag"] for r in rows if r["keep"]]
    print(f"\nN1 keep={len(keep)} {keep}", flush=True)
    print(f"N1 wrote {dest.name}", flush=True)
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
