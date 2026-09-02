#!/usr/bin/env python3
"""A3: how much of the agglomeration's per-node work is on nodes that matter.

Agglomeration is the largest gap to the budget (1294 ms against 25 ms), but the
GPU is contended, so any wall-clock comparison is worthless right now. This
measures the same thing in counts instead, which contention cannot distort.

Every inner iteration of the loop in parhac_dev sweeps all nnode nodes four
times over, regardless of how much of the graph is still live:

    cudaMemset(dprop, 0, nnode * 8)   k_pack_prop   k_compress   k_freeze

while k_propose sweeps only the nlive live edges. Nodes merged away stay in
those sweeps forever, so as roots disappear the loop pays full price for
progressively less. The useful width is the live-root count nact, and

    sum(nnode) / sum(nact)

is the ceiling on what working off a list of live roots could win. Reporting a
ceiling is the point: if it is small, the Track A listing work is not worth
doing, and that is a cheap thing to learn.

nact needs an extra nnode sweep per iteration to count, so
parhac_paper_d_profile only gathers it when asked. Do not read the wall time
from a run with it enabled. An earlier version of this script bounded the
useful width by 2*nlive instead, on the grounds that nlive edges touch at most
that many endpoints. That is true but far too loose to be informative: it
reported a 1.0x ceiling while most of those endpoints were duplicate
references to the same few roots.

Device memory is just the RAG edge arrays.
"""
from __future__ import annotations

import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _agg_common import CACHE, load_rag  # noqa: E402
from e6r_parhac import DSO, compile_d  # noqa: E402

HIST_CAP = 200000
# Per inner iteration: the dprop memset, k_pack_prop, k_compress, k_freeze.
NODE_SWEEPS_PER_INNER = 4
# Per outer: compress, zero_sz, rebuild_sz, color and copy_sz memsets/kernels.
NODE_SWEEPS_PER_OUTER = 7
THR = 0.3


def main():
    compile_d()
    u, v, sm, ct, fr, max_id = load_rag()
    nnode = max_id + 1
    lib = ctypes.CDLL(str(DSO))
    lib.parhac_paper_d_profile.restype = ctypes.c_int
    lib.parhac_paper_d_profile.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64),
    ]

    thrs = np.asarray([THR], dtype=np.float64)
    parents = np.empty((1, nnode), dtype=np.uint32)
    stats = np.zeros((1, 3), dtype=np.int64)
    phase = np.zeros(6, dtype=np.float64)
    h_nlive = np.zeros(HIST_CAP, dtype=np.int64)
    h_nprop = np.zeros(HIST_CAP, dtype=np.int64)
    h_nmerge = np.zeros(HIST_CAP, dtype=np.int64)
    h_nact = np.zeros(HIST_CAP, dtype=np.int64)
    hist_n = ctypes.c_int(0)

    print(f"A3 nnode={nnode} nedge={len(u)} T={THR}", flush=True)
    t0 = time.perf_counter()
    rc = lib.parhac_paper_d_profile(
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
        phase.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        h_nlive.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        h_nprop.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        h_nmerge.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int(HIST_CAP), ctypes.byref(hist_n), ctypes.c_int(1),
        h_nact.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    n = int(hist_n.value)
    # This path returns a nonzero status even on a completed run, so the
    # histogram being populated is what says the measurement is usable.
    if n <= 0:
        print(f"A3 FAIL rc={rc} hist_n={n}", flush=True)
        return False
    truncated = n >= HIST_CAP
    nlive = h_nlive[:n].astype(np.int64)
    nact = h_nact[:n].astype(np.int64)
    nprop = h_nprop[:n].astype(np.int64)

    w_node = NODE_SWEEPS_PER_INNER * float(n) * nnode
    u_node = NODE_SWEEPS_PER_INNER * float(nact.sum())
    bound = w_node / u_node if u_node else float("inf")
    nouter, nmerge_tot, ninner = (int(stats[0, 0]), int(stats[0, 1]),
                                  int(stats[0, 2]))
    outer_node = NODE_SWEEPS_PER_OUTER * float(nouter) * nnode
    memset_gb = float(n) * nnode * 8 / 1e9
    edge_visits = float(nlive.sum())

    frac = [float((nact < nnode / d).mean()) for d in (2, 10, 100, 1000)]
    print(
        f"A3 inner iters={n} (stats ninner={ninner}) outers={nouter} "
        f"merges={nmerge_tot} truncated={truncated}\n"
        f"A3 nlive: max={int(nlive.max())} p50={int(np.median(nlive))} "
        f"p90={int(np.percentile(nlive, 90))} min={int(nlive.min())} "
        f"sum={edge_visits:.3e}\n"
        f"A3 nact: max={int(nact.max())} p50={int(np.median(nact))} "
        f"min={int(nact.min())} last={int(nact[-1])}\n"
        f"A3 nprop: sum={float(nprop.sum()):.4e} max={int(nprop.max())} "
        f"p50={int(np.median(nprop))} -- k_accept_serial walks this many "
        f"proposals on ONE thread, and each is sorted twice\n"
        f"A3 node-visits done={w_node:.4e} useful={u_node:.4e} "
        f"ceiling on listing speedup={bound:.1f}x\n"
        f"A3 outer node-visits={outer_node:.4e} "
        f"({100 * outer_node / (w_node + outer_node):.1f}% of node work)\n"
        f"A3 dprop memset alone={memset_gb:.2f} GB of writes\n"
        f"A3 iterations with nact below nnode/2, /10, /100, /1000: "
        f"{frac[0]:.3f} {frac[1]:.3f} {frac[2]:.3f} {frac[3]:.3f}\n"
        f"A3 wall={wall_ms:.1f} ms phase compact={phase[0]:.1f} "
        f"propose={phase[1]:.1f} accept={phase[2]:.1f} d2h={phase[3]:.1f} "
        f"memset={phase[4]:.1f} (contended, ratios only)",
        flush=True,
    )
    out = {
        "nnode": int(nnode),
        "nedge": int(len(u)),
        "threshold": THR,
        "inner_iters": n,
        "hist_truncated": bool(truncated),
        "outers": nouter,
        "merges": nmerge_tot,
        "nlive_max": int(nlive.max()),
        "nlive_p50": float(np.median(nlive)),
        "nlive_sum": edge_visits,
        "nact_max": int(nact.max()),
        "nact_p50": float(np.median(nact)),
        "nact_min": int(nact.min()),
        "nact_sum": float(nact.sum()),
        "nprop_sum": float(nprop.sum()),
        "nprop_max": int(nprop.max()),
        "nprop_p50": float(np.median(nprop)),
        "node_visits_done": w_node,
        "node_visits_useful": u_node,
        "listing_speedup_ceiling": bound,
        "outer_node_visits": outer_node,
        "dprop_memset_gb": memset_gb,
        "frac_iters_below_half_tenth_hundredth_thousandth": frac,
        "wall_ms_contended": wall_ms,
        "phase_ms_contended": {
            "compact": phase[0], "propose": phase[1], "accept": phase[2],
            "d2h": phase[3], "memset": phase[4], "debug": phase[5],
        },
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / "a3_work_accounting.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"A3 wrote {dest.name}", flush=True)
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
