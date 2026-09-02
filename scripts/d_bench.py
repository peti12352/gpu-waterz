#!/usr/bin/env python3
"""D: the graded speed measurement for segment_d, plus determinism, one command.

TASK asks for a median of 5 timings of segment() at threshold 0.3, reported as
voxels per second, with byte-identical labels run to run. Everything measured
before this script lived one layer down, on the agglomeration alone or on the
watershed alone, which is useful for deciding what to work on but is not the
number the task is graded on. This produces that number.

Times segment(), the entry point TASK names, which takes a host array and
returns host labels. That includes the host-to-device copy of the affinities
and the device-to-host copy of the labels, so it is the honest end-to-end cost
of asking for a segmentation. segment_d, the device-resident variant that
skips both copies, needs torch, which is not installed on this machine.

The stage split comes from segment.STAGE_MS. It is free to collect here: every
stage helper is a blocking ctypes call that ends on a device-to-host copy, so
wall clock around it is already device time.

Contention is checked, because every timing taken while another process is on
the card is junk, as this project learned the hard way: a host round-trip
measured 883 us on a shared card and 2.5 us on an idle one.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

AFF = ROOT / "data/ws_bounty/cremiA_val/affinity.h5"
CACHE = ROOT / "data/cache"
THR = 0.3
RUNS = 5
GVOX_TARGET = 2.0


def gpu_state():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.used,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def other_procs():
    """PIDs with memory on the card, excluding this process."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    if not out:
        return []
    me = str(os.getpid())
    return [ln for ln in out.splitlines() if ln.split(",")[0].strip() != me]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aff", default=str(AFF))
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--threshold", type=float, default=THR)
    args = ap.parse_args()

    import segment as S

    with h5py.File(args.aff, "r") as f:
        aff = f["affinity"][:]
    aff_u8 = S._as_u8(aff)
    z, y, x = aff_u8.shape[1:]
    nvox = int(z) * int(y) * int(x)

    print(f"D bench {z}x{y}x{x} = {nvox / 1e6:.1f} Mvox, T={args.threshold}\n"
          f"D gpu: {gpu_state()}", flush=True)
    before = other_procs()
    if before:
        print(f"D WARNING card is shared, timings are not gradeable: {before}",
              flush=True)

    print("D warmup", flush=True)
    labs0 = S.segment(aff_u8, [args.threshold])
    print(f"D agglomeration backend: {S.AGG_BACKEND}", flush=True)
    if S.AGG_BACKEND != "gpu":
        print("D FAIL ran the host agglomeration fallback; a speed number "
              "from that is meaningless", flush=True)
        return False

    times = []
    splits = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        labs = S.segment(aff_u8, [args.threshold])
        dt = time.perf_counter() - t0
        times.append(dt)
        splits.append(dict(S.STAGE_MS))
        print(f"D run{i} {dt * 1000:.1f} ms  {nvox / dt / 1e9:.3f} Gvox/s",
              flush=True)

    med = float(np.median(times))
    gvox = nvox / med / 1e9

    stages = sorted({k for d in splits for k in d})
    split_med = {k: float(np.median([d.get(k, 0.0) for d in splits]))
                 for k in stages}

    det = bool(np.array_equal(labs0[0], labs[0]))
    nseg = int((np.unique(labs[0]) != 0).sum())
    after = other_procs()

    print(
        f"D median={med * 1000:.1f} ms  min={min(times) * 1000:.1f}  "
        f"max={max(times) * 1000:.1f}\n"
        f"D throughput={gvox:.3f} Gvox/s (target {GVOX_TARGET} on a 3090 Ti)\n"
        f"D stage medians: "
        + " ".join(f"{k}={split_med[k]:.1f}" for k in stages)
        + f" sum={sum(split_med.values()):.1f} ms\n"
        f"D deterministic={det} nseg={nseg}\n"
        f"D card idle throughout: {not before and not after}",
        flush=True,
    )
    out = {
        "shape": [int(z), int(y), int(x)],
        "nvox": nvox,
        "threshold": args.threshold,
        "runs": args.runs,
        "times_ms": [t * 1000 for t in times],
        "median_ms": med * 1000,
        "gvox_per_s": gvox,
        "gvox_target": GVOX_TARGET,
        "stage_ms_median": split_med,
        "deterministic": det,
        "agg_backend": S.AGG_BACKEND,
        "nseg": nseg,
        "gpu": gpu_state(),
        "other_procs_before": before,
        "other_procs_after": after,
        "gradeable": not before and not after,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / "d_bench.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"D wrote {dest.name}", flush=True)
    return det


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
