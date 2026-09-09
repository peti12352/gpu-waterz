#!/usr/bin/env python3
"""N3: RAG structure. Counts first. One ParHAC at a time. Fingerprint vs E2."""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from g0_agg_ref import load_rag, run  # noqa: E402
from t3_memsafe import CACHE, T, grade_t3, voi_parent_mmap  # noqa: E402

EPS = 0.08
LOCKED = {
    "nmerge": 1853427,
    "nseg": 321973,
    "n_layer": 17,
    "nouter": 653,
    "ninner": 1405,
    "sum_nlive": 2691304379,
}


def fp_match(r):
    return (
        r["nmerge"] == LOCKED["nmerge"]
        and r["nseg"] == LOCKED["nseg"]
        and r["n_layer"] == LOCKED["n_layer"]
        and r["nouter"] == LOCKED["nouter"]
        and r["ninner"] == LOCKED["ninner"]
        and r["sum_nlive"] == LOCKED["sum_nlive"]
    )


def remap_edges(u, v, sm, ct, parent):
    a, b = parent[u], parent[v]
    keep = (a != b) & (a != 0) & (b != 0)
    a, b, sm, ct = a[keep], b[keep], sm[keep], ct[keep]
    lo = np.minimum(a, b).astype(np.uint32)
    hi = np.maximum(a, b).astype(np.uint32)
    key = (lo.astype(np.uint64) << np.uint64(32)) | hi.astype(np.uint64)
    order = np.argsort(key, kind="stable")
    key, sm, ct, lo, hi = key[order], sm[order], ct[order], lo[order], hi[order]
    same = np.ones(key.size, dtype=bool)
    if key.size:
        same[1:] = key[1:] != key[:-1]
    idx = np.flatnonzero(same)
    return lo[idx], hi[idx], np.add.reduceat(sm, idx), np.add.reduceat(ct, idx)


def main():
    dest = CACHE / "n3_rag_structure.json"
    out = {}
    if dest.is_file():
        try:
            out = json.loads(dest.read_text())
        except (OSError, json.JSONDecodeError):
            out = {}

    u, v, sm, ct, max_id = load_rag()
    mean = sm / np.maximum(ct, 1)
    nedge = int(u.size)
    print(f"N3 nedge={nedge} nnode={max_id + 1}", flush=True)

    if "counts" not in out:
        bins = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                         0.99, 0.999, 1.0, 1.0000001])
        hist, _ = np.histogram(mean, bins=bins)
        n_eq1 = int(np.sum(mean == 1.0))
        n_gt09 = int(np.sum(mean > 0.9))
        n_gtT = int(np.sum(mean > T))
        deg = np.zeros(max_id + 1, dtype=np.int64)
        np.add.at(deg, u, 1)
        np.add.at(deg, v, 1)
        mean_deg = float(deg[1:].mean()) if max_id else 0.0
        nmerge_locked = LOCKED["nmerge"]
        visits_locked = 78e6 + 197e6 + 356e6 + 1505e6
        out["counts"] = {
            "nedge": nedge,
            "hist_bins": bins.tolist(),
            "hist": [int(x) for x in hist],
            "n_mean_eq_1": n_eq1,
            "n_mean_gt_0.9": n_gt09,
            "n_mean_gt_T": n_gtT,
            "n_mean_le_T": nedge - n_gtT,
            "prefilter_edge_cut": (nedge / n_gtT) if n_gtT else None,
            "mean_degree": mean_deg,
            "locked_visits_per_merge": visits_locked / nmerge_locked,
            "essential_visits_per_merge": mean_deg,
            "slack_vs_essential": (visits_locked / nmerge_locked) / mean_deg
            if mean_deg else None,
        }
        dest.write_text(json.dumps(out, indent=2, default=float) + "\n")
        print(f"N3 mean==1 {n_eq1} mean>0.9 {n_gt09} mean>T {n_gtT} "
              f"cut={out['counts']['prefilter_edge_cut']:.2f}x "
              f"deg={mean_deg:.2f}", flush=True)
    else:
        print(f"N3 counts resume {out['counts']}", flush=True)

    if "prefilter" not in out:
        mask = mean > T
        t0 = time.perf_counter()
        r = run("fast", u[mask], v[mask], sm[mask], ct[mask],
                max_id, T, EPS, 64, 0, 0, False, True)
        ident = fp_match(r)
        out["prefilter"] = {
            "parent_identical_via_fingerprint": ident,
            "nmerge": r["nmerge"], "nseg": r["nseg"],
            "n_layer": r["n_layer"], "nouter": r["nouter"],
            "ninner": r["ninner"], "sum_nlive": r["sum_nlive"],
            "work_cut": LOCKED["sum_nlive"] / r["sum_nlive"] if r["sum_nlive"] else None,
            "wall_s": time.perf_counter() - t0,
        }
        dest.write_text(json.dumps(out, indent=2, default=float) + "\n")
        print(f"N3 prefilter nmerge={r['nmerge']} nseg={r['nseg']} "
              f"fp={ident} work_x={out['prefilter']['work_cut']} "
              f"{out['prefilter']['wall_s']:.1f}s", flush=True)
        if not ident:
            split, merge, nseg = voi_parent_mmap(r["root"])
            ok, sl, ml = grade_t3(split, merge)
            out["prefilter"]["voi"] = {
                "voi_split": split, "voi_merge": merge, "pass": ok,
                "sl": sl, "ml": ml, "nseg": nseg,
            }
            dest.write_text(json.dumps(out, indent=2, default=float) + "\n")
            print(f"N3 prefilter VOI split={split:.4f} merge={merge:.4f} "
                  f"{'PASS' if ok else 'FAIL'}", flush=True)
        del r
        gc.collect()

    pf = out.get("prefilter")
    if pf and not pf.get("parent_identical_via_fingerprint") and "voi" not in pf:
        mask = mean > T
        print("N3 prefilter VOI resume (fingerprint missed)", flush=True)
        r = run("fast", u[mask], v[mask], sm[mask], ct[mask],
                max_id, T, EPS, 64, 0, 0, False, True)
        split, merge, nseg = voi_parent_mmap(r["root"])
        ok, sl, ml = grade_t3(split, merge)
        pf["voi"] = {"voi_split": split, "voi_merge": merge, "pass": ok,
                     "sl": sl, "ml": ml, "nseg": nseg}
        dest.write_text(json.dumps(out, indent=2, default=float) + "\n")
        print(f"N3 prefilter VOI split={split:.4f} merge={merge:.4f} "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        del r
        gc.collect()

    if "sat_prefix" not in out:
        n_eq1 = int(out["counts"]["n_mean_eq_1"])
        if n_eq1 == 0:
            out["sat_prefix"] = {
                "parent_identical_via_fingerprint": True,
                "nmerge": 0, "note": "no mean==1 edges; prefix is a no-op",
                "work_cut": 1.0, "voi": None,
            }
            print("N3 sat-prefix skip (0 edges at mean==1)", flush=True)
        else:
            import ctypes
            so = ROOT / "src/libfrozen_cc.so"
            parent = np.empty(max_id + 1, dtype=np.uint32)
            lib = ctypes.CDLL(str(so))
            lib.frozen_cc_cpu.restype = ctypes.c_int
            lib.frozen_cc_cpu.argtypes = [
                ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_double), ctypes.c_int64, ctypes.c_double,
                ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
            ]
            mean_c = np.ascontiguousarray(mean, dtype=np.float64)
            lib.frozen_cc_cpu(
                u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                mean_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int64(len(u)), ctypes.c_double(1.0 - 1e-15),
                parent.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                ctypes.c_uint32(max_id),
            )
            u2, v2, sm2, ct2 = remap_edges(u, v, sm, ct, parent)
            print(f"N3 sat contracted leftover edges={u2.size}", flush=True)
            t0 = time.perf_counter()
            r = run("fast", u2, v2, sm2, ct2, max_id, T, EPS, 64, 0, 0, False, True)
            composed = r["root"][parent]
            ident = fp_match(r) and int(np.unique(composed[1:]).size) == LOCKED["nseg"]
            voi = None
            if not ident:
                split, merge, nseg = voi_parent_mmap(composed)
                ok, sl, ml = grade_t3(split, merge)
                voi = {"voi_split": split, "voi_merge": merge, "pass": ok,
                       "sl": sl, "ml": ml, "nseg": nseg}
                print(f"N3 sat VOI split={split:.4f} merge={merge:.4f} "
                      f"{'PASS' if ok else 'FAIL'}", flush=True)
            out["sat_prefix"] = {
                "parent_identical_via_fingerprint": ident,
                "nmerge": r["nmerge"], "nseg": r["nseg"],
                "sum_nlive": r["sum_nlive"],
                "contracted_edges": int(u2.size),
                "work_cut": LOCKED["sum_nlive"] / r["sum_nlive"] if r["sum_nlive"] else None,
                "voi": voi,
                "wall_s": time.perf_counter() - t0,
            }
            del r, composed, parent, u2, v2, sm2, ct2
            gc.collect()
        dest.write_text(json.dumps(out, indent=2, default=float) + "\n")

    sat = out["sat_prefix"]
    out["kill_sat_prefix"] = (
        not sat.get("parent_identical_via_fingerprint")
        and not ((sat.get("voi") or {}).get("pass"))
    )
    dest.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"N3 wrote {dest.name}", flush=True)
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
