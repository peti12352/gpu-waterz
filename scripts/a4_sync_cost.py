#!/usr/bin/env python3
"""A4: what one device-to-host round-trip costs this loop, measured not guessed.

After the proposal sort and the compaction moved to CUB, the kernels in E6s
account for roughly 200 ms of the phase split, yet the path takes ~9750 ms.
The suspicion is that the loop is latency-bound rather than throughput-bound:
each inner iteration makes four blocking copies back to the host (proposal
count, merge count, and the two counts inside compact_radix), and each one
drains the stream and gives the co-tenant process a scheduling quantum.

That suspicion is worth exactly nothing until it has a number attached, and
the number decides a large piece of work. If a round-trip is expensive, the
answer is device-side loop control (CUDA graphs with conditional nodes, or
kernels reading their counts from device memory), which is a substantial
change. If it is cheap, the time is somewhere else entirely and that change
would be wasted.

So: WATERZ_SYNC_PROBE=k adds k extra copies per inner iteration. Sweeping k
and fitting a line gives the marginal cost of one round-trip directly. The
slope is robust to the co-tenant's load in a way that any single absolute
timing is not, because every point pays the same contention.

Reports the slope, and from it the predicted saving from removing the four
that the algorithm currently needs. Runs are interleaved across k values so a
drift in the co-tenant's load spreads over all points instead of tilting the
line.
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
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import CACHE, load_rag  # noqa: E402
from e6r_parhac import DSO, compile_d  # noqa: E402

T = 0.3
PROBES = (0, 1, 2, 4)
REPS = 2
# Copies per inner iteration the algorithm itself needs: nprop, nmerge, and
# the selected and run counts inside compact_radix.
SYNCS_PER_ITER = 4


def bind(lib):
    lib.parhac_paper_d_timed.restype = ctypes.c_int
    lib.parhac_paper_d_timed.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_double),
    ]


def one(lib, u, v, sm, ct, max_id):
    thrs = np.asarray([T], dtype=np.float64)
    parents = np.empty((1, max_id + 1), dtype=np.uint32)
    stats = np.zeros((1, 3), dtype=np.int64)
    device_ms = ctypes.c_double(0.0)
    t0 = time.perf_counter()
    lib.parhac_paper_d_timed(
        u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        sm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ct.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(u)),
        thrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(1), ctypes.c_double(0.08),
        parents.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(max_id),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.byref(device_ms),
    )
    return {
        "device_ms": float(device_ms.value),
        "wall_ms": (time.perf_counter() - t0) * 1000.0,
        "inner": int(stats[0, 2]),
        "merges": int(stats[0, 1]),
    }


def gpu_state():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def main():
    compile_d()
    u, v, sm, ct, fr, max_id = load_rag()
    print(f"A4 sync cost, E6s @T={T}. GPU: {gpu_state()}", flush=True)

    runs: dict[int, list[dict]] = {k: [] for k in PROBES}
    for rep in range(REPS):
        for k in PROBES:
            # The library reads the probe count once per call, so setting it in
            # the environment before loading a fresh handle is enough; the DSO
            # is already resident, so re-reading getenv per call is what makes
            # this work without reloading it.
            os.environ["WATERZ_SYNC_PROBE"] = str(k)
            lib = ctypes.CDLL(str(DSO))
            bind(lib)
            r = one(lib, u, v, sm, ct, max_id)
            runs[k].append(r)
            print(f"A4 rep{rep} k={k} device_ms={r['device_ms']:.1f} "
                  f"inner={r['inner']} merges={r['merges']}", flush=True)
    os.environ.pop("WATERZ_SYNC_PROBE", None)

    inner = {r["inner"] for rs in runs.values() for r in rs}
    merges = {r["merges"] for rs in runs.values() for r in rs}
    med = {k: float(np.median([r["device_ms"] for r in runs[k]])) for k in PROBES}
    # Extra copies per run is k * inner, so fit device_ms against that.
    ni = float(next(iter(inner))) if len(inner) == 1 else float(np.mean(list(inner)))
    xs = np.asarray([k * ni for k in PROBES], dtype=np.float64)
    ys = np.asarray([med[k] for k in PROBES], dtype=np.float64)
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = np.polyval([slope, intercept], xs)
    resid = float(np.max(np.abs(ys - pred)))
    algo_syncs = SYNCS_PER_ITER * ni
    attributable = slope * algo_syncs

    print(
        f"A4 iterations identical across k: {len(inner) == 1} {sorted(inner)}\n"
        f"A4 merges identical across k: {len(merges) == 1} {sorted(merges)}\n"
        f"A4 median device_ms by k: "
        + " ".join(f"k={k}:{med[k]:.0f}" for k in PROBES) + "\n"
        f"A4 slope={slope * 1000:.3f} us per round-trip "
        f"(max fit residual {resid:.0f} ms)\n"
        f"A4 the loop's own {SYNCS_PER_ITER} per iteration = "
        f"{algo_syncs:.0f} round-trips -> {attributable:.0f} ms, "
        f"{100 * attributable / med[0]:.0f}% of the {med[0]:.0f} ms baseline",
        flush=True,
    )
    out = {
        "threshold": T,
        "probes": list(PROBES),
        "reps": REPS,
        "device_ms_median_by_probe": {str(k): med[k] for k in PROBES},
        "device_ms_all": {str(k): [r["device_ms"] for r in runs[k]] for k in PROBES},
        "inner_iters": sorted(inner),
        "merges": sorted(merges),
        "inner_used": ni,
        "us_per_round_trip": slope * 1000,
        "fit_max_residual_ms": resid,
        "syncs_per_iter_in_algorithm": SYNCS_PER_ITER,
        "attributable_ms": attributable,
        "attributable_frac_of_baseline": attributable / med[0],
        "gpu": gpu_state(),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / "a4_sync_cost.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"A4 wrote {dest.name}", flush=True)
    return len(inner) == 1 and len(merges) == 1


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
