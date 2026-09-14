#!/usr/bin/env python3
"""G-levers: prove each WATERZ_AGG_LEVERS bit is bit-identical, and time it.

The levers in parhac_e6s_dev are all work-efficiency rewrites that are meant to
change nothing about the result. That claim is checked three ways here, in
increasing strength, against the same build with the lever off:

  1. the four-threshold segment fingerprint [294165, 322000, 345131, 379293]
     from a1_e6s_voi.json, plus outer/inner/merge counts per threshold;
  2. the full parent array, compared with np.array_equal, which is the
     byte-identity TASK.md actually requires;
  3. at T=0.3, the 17-element layer_outers and layer_merges vectors from
     parhac_e6s_p0aa, which pin the trajectory and not just the destination.

A lever that changes any of those is reverted, not tuned: the point of the
G series is that it is free of accuracy risk, and anything that is not free of
accuracy risk belongs behind the VOI gate in the V series instead.

The CUDA-event phase split comes along for free from the same p0aa entry, so
the measured speedup is reported next to the equality result rather than
needing a second run. Note that phase timings on a shared card are dominated by
launch count rather than work, see the header of m1_cost_model.py, so treat
them as an upper bound unless the card is idle.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import AFF_THRESHOLDS, CACHE, load_rag  # noqa: E402
from e6r_parhac import DSO, compile_d  # noqa: E402

PHASES = ("compact", "propose", "pack", "sort", "accept",
          "compress", "freeze", "color", "memset", "d2h", "host")

# Bit -> (name, what it replaces). Keep in step with parhac_e6s_dev.
LEVERS = [
    (1, "g1 freeze_reds", "k_freeze + k_color + dcolor clear, per inner/outer"),
    (2, "g2 dirty-scan", "hash_combine_live over the whole live edge array"),
    (4, "g3 active-lists", "k_propose over all live edges, pack over all nnode"),
    (8, "g4 root-list", "incremental roots; skip per-outer nnode rebuild"),
]


def bind(lib):
    lib.parhac_e6s.restype = ctypes.c_int
    lib.parhac_e6s.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64),
    ]
    lib.parhac_e6s_p0aa.restype = ctypes.c_int
    lib.parhac_e6s_p0aa.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int64),
    ]


def p(a, t):
    return a.ctypes.data_as(ctypes.POINTER(t))


def run_four(lib, u, v, sm, ct, max_id, eps):
    thrs = np.asarray(AFF_THRESHOLDS, dtype=np.float64)
    parents = np.empty((len(thrs), max_id + 1), dtype=np.uint32)
    stats = np.zeros((len(thrs), 3), dtype=np.int64)
    rc = lib.parhac_e6s(
        p(u, ctypes.c_uint32), p(v, ctypes.c_uint32), p(sm, ctypes.c_double),
        p(ct, ctypes.c_int64), ctypes.c_int64(u.size),
        p(thrs, ctypes.c_double), ctypes.c_int(len(thrs)),
        ctypes.c_double(eps), p(parents, ctypes.c_uint32),
        ctypes.c_uint32(max_id), p(stats, ctypes.c_int64))
    nseg = [int(np.unique(parents[i]).size) for i in range(len(thrs))]
    return {
        "rc": int(rc),
        "parents": parents,
        "nseg_unique_parents": nseg,
        "outer": [int(x) for x in stats[:, 0]],
        "merges": [int(x) for x in stats[:, 1]],
        "inner": [int(x) for x in stats[:, 2]],
    }


def run_p0aa(lib, u, v, sm, ct, max_id, eps):
    thrs = np.asarray([0.3], dtype=np.float64)
    parents = np.empty((1, max_id + 1), dtype=np.uint32)
    stats = np.zeros((1, 3), dtype=np.int64)
    phase = np.zeros(11, dtype=np.float64)
    n_layer = ctypes.c_int(0)
    lo = np.zeros(64, dtype=np.int32)
    lm = np.zeros(64, dtype=np.int32)
    lfz = np.full(64, -1, dtype=np.int32)
    work = np.zeros(2, dtype=np.int64)
    rc = lib.parhac_e6s_p0aa(
        p(u, ctypes.c_uint32), p(v, ctypes.c_uint32), p(sm, ctypes.c_double),
        p(ct, ctypes.c_int64), ctypes.c_int64(u.size),
        p(thrs, ctypes.c_double), ctypes.c_int(1), ctypes.c_double(eps),
        p(parents, ctypes.c_uint32), ctypes.c_uint32(max_id),
        p(stats, ctypes.c_int64), p(phase, ctypes.c_double),
        ctypes.byref(n_layer), p(lo, ctypes.c_int), p(lm, ctypes.c_int),
        p(lfz, ctypes.c_int), p(work, ctypes.c_int64))
    nl = int(n_layer.value)
    return {
        "rc": int(rc),
        "n_layer": nl,
        "layer_outers": [int(x) for x in lo[:nl]],
        "layer_merges": [int(x) for x in lm[:nl]],
        "nouter": int(stats[0, 0]),
        "nmerge": int(stats[0, 1]),
        "ninner": int(stats[0, 2]),
        "sum_nlive": int(work[0]),
        "sum_above": int(work[1]),
        "phases": {k: float(phase[i]) for i, k in enumerate(PHASES)},
        "phase_sum_ms": float(phase.sum()),
    }


def diff(ref, got, keys):
    return [k for k in keys if ref[k] != got[k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.08)
    ap.add_argument("--bits", default=None,
                    help="comma-separated lever bits to test; default all, "
                         "then the combination of all of them")
    args = ap.parse_args()

    compile_d()
    u, v, sm, ct, fr, max_id = load_rag()
    lib = ctypes.CDLL(str(DSO))
    bind(lib)

    def at(bits):
        os.environ["WATERZ_AGG_LEVERS"] = str(bits)
        return (run_four(lib, u, v, sm, ct, max_id, args.eps),
                run_p0aa(lib, u, v, sm, ct, max_id, args.eps))

    print("G-levers reference run, WATERZ_AGG_LEVERS=0", flush=True)
    r4, ra = at(0)
    locked = json.loads((CACHE / "a1_e6s_voi.json").read_text())
    fp_ok = r4["nseg_unique_parents"] == locked["nseg_unique_parents"]
    print(f"G-levers   nseg={r4['nseg_unique_parents']} "
          f"vs locked {locked['nseg_unique_parents']} "
          f"{'ok' if fp_ok else 'MISMATCH - the baseline itself moved'}")
    print(f"G-levers   T=0.3 layers={ra['n_layer']} outers={ra['nouter']} "
          f"inners={ra['ninner']} merges={ra['nmerge']} "
          f"phase_sum={ra['phase_sum_ms']:.1f} ms")

    todo = [int(b) for b in args.bits.split(",")] if args.bits \
        else [b for b, _, _ in LEVERS]
    if not args.bits and len(LEVERS) > 1:
        todo.append(sum(b for b, _, _ in LEVERS))

    results = {}
    ok_all = fp_ok
    for bits in todo:
        names = " + ".join(n for b, n, _ in LEVERS if bits & b) or "none"
        print(f"\nG-levers WATERZ_AGG_LEVERS={bits}  ({names})", flush=True)
        g4, ga = at(bits)
        bad = []
        if g4["rc"] != 1 or ga["rc"] != 1:
            bad.append(f"rc={g4['rc']}/{ga['rc']}")
        if not np.array_equal(r4["parents"], g4["parents"]):
            n = int((r4["parents"] != g4["parents"]).sum())
            bad.append(f"parent array differs in {n} entries")
        bad += diff(r4, g4, ["nseg_unique_parents", "outer", "inner", "merges"])
        bad += diff(ra, ga, ["n_layer", "layer_outers", "layer_merges",
                             "nouter", "ninner", "nmerge", "sum_nlive",
                             "sum_above"])
        speed = ra["phase_sum_ms"] / ga["phase_sum_ms"] if ga["phase_sum_ms"] else 0
        print(f"G-levers   {'BIT-IDENTICAL' if not bad else 'DIFFERS: ' + ', '.join(bad)}")
        print(f"G-levers   phase_sum {ra['phase_sum_ms']:.1f} -> "
              f"{ga['phase_sum_ms']:.1f} ms  {speed:.2f}x")
        for k in PHASES:
            b0, b1 = ra["phases"][k], ga["phases"][k]
            if max(b0, b1) > 1.0:
                print(f"G-levers     {k:9s} {b0:8.1f} -> {b1:8.1f} ms")
        results[str(bits)] = {
            "names": names, "identical": not bad, "why": bad,
            "speedup_phase_sum": speed,
            "phases": ga["phases"], "p0aa": {k: v for k, v in ga.items()
                                             if k != "phases"},
        }
        ok_all &= not bad

    os.environ["WATERZ_AGG_LEVERS"] = "0"
    dest = CACHE / "g_levers.json"
    dest.write_text(json.dumps(
        {"eps": args.eps, "baseline_fingerprint_ok": fp_ok,
         "reference": {k: v for k, v in ra.items()},
         "reference_four": {k: v for k, v in r4.items() if k != "parents"},
         "levers": results, "pass": bool(ok_all)}, indent=2, default=float) + "\n")
    print(f"\nG-levers {'PASS' if ok_all else 'FAIL'}; wrote {dest.name}")
    return ok_all


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
