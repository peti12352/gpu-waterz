#!/usr/bin/env python3
"""V1/V2: does raising epsilon, or dropping the size asymmetry, cut the work?

These are the two accuracy-affecting levers. Their accuracy half needs the
four-threshold VOI gate against ground truth, which needs the affinity volume
and the CREMI labels; neither is on this machine. But their *speed* half is a
pure function of the RAG, and that is cached, so the necessary condition can be
settled here for free: if raising epsilon does not actually reduce the outer
and inner counts, the lever is dead and no VOI run is worth spending a GPU
window on.

Worth settling because the arithmetic in the plan is suspect. It projects
layers = ln(wmax/T)/ln(1+eps) and reads off a 2-4x work reduction, but
p0aa_e6s.json shows layer 0 alone performing 71% of all merges while already
pinned at the 64-outer cap. Widening the first band cannot make that layer
cheaper; it can only make it wider. Whether the total falls is therefore an
empirical question about where the merges sit, not an algebraic one.

V2 is the untested half of A4. It removes `sz[r] >= sz[bl]` from k_propose,
which currently rejects any colour-valid pair whose red is the smaller side.

Both are run through the same CPU replica the G series is gated on, so the
counts are the counts the device would produce.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from g0_agg_ref import CACHE, load_rag, run  # noqa: E402

EPS_SWEEP = [0.08, 0.12, 0.16, 0.24, 0.32]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--eps", nargs="+", type=float, default=EPS_SWEEP)
    ap.add_argument("--sub", type=int, default=0,
                    help="induced subgraph for a fast directional sweep")
    ap.add_argument("--max-outer", type=int, default=64)
    args = ap.parse_args()

    u, v, sm, ct, max_id = load_rag()
    if args.sub:
        keep = (u < args.sub) & (v < args.sub)
        u, v, sm, ct = u[keep], v[keep], sm[keep], ct[keep]
        max_id = int(max(u.max(), v.max()))
    print(f"V nedge={u.size} nnode={max_id + 1} T={args.threshold}"
          f"{f' sub={args.sub}' if args.sub else ' (full val)'}", flush=True)

    configs = [("v1", e, True) for e in args.eps]
    # V2 at the locked epsilon, so the two effects stay separable, then both
    # together at the widest epsilon, because they need not compose: dropping
    # the asymmetry lets bigger clusters absorb smaller ones, and a wider band
    # gives them more candidates to absorb.
    configs.append(("v2", args.eps[0], False))
    if len(args.eps) > 1:
        configs.append(("v1+v2", args.eps[-1], False))

    reused = None
    g0p = CACHE / "g0_agg_ref.json"
    if not args.sub and g0p.is_file():
        g0 = json.loads(g0p.read_text())
        b = g0.get("base") or {}
        if (g0.get("pass")
                and abs(float(b.get("eps", -1)) - args.eps[0]) < 1e-12
                and b.get("size_asym", True)
                and abs(float(g0.get("threshold", args.threshold))
 - args.threshold) < 1e-12):
            reused = b
            print(f"V reusing locked row from {g0p.name} "
                  f"(nmerge={b['nmerge']} sum_nlive={b['sum_nlive']})",
                  flush=True)

    rows = []
    for tag, eps, asym in configs:
        t0 = time.perf_counter()
        if reused is not None and abs(eps - args.eps[0]) < 1e-12 and asym:
            r = reused
            dt = 0.0
        else:
            r = run("base", u, v, sm, ct, max_id, args.threshold, eps,
                    args.max_outer, 0, 0, False, asym)
            dt = time.perf_counter() - t0
        rows.append({"tag": tag, "eps": eps, "size_asym": asym,
                     "n_layer": r["n_layer"], "nouter": r["nouter"],
                     "ninner": r["ninner"], "nmerge": r["nmerge"],
                     "nseg": r["nseg"], "sum_nlive": r["sum_nlive"],
                     "sum_above": r["sum_above"],
                     "layer_outers": r["layer_outers"],
                     "layer_merges": r["layer_merges"],
                     "wall_s": dt})
        print(f"V {tag} eps={eps:.2f} asym={int(asym)}  "
              f"layers={r['n_layer']:3d} outers={r['nouter']:4d} "
              f"inners={r['ninner']:4d} merges={r['nmerge']:8d} "
              f"nseg={r['nseg']:7d}  {dt:6.1f}s", flush=True)

    base = rows[0]
    print(f"\nV relative to eps={base['eps']:.2f} with the asymmetry in place")
    print("V   config           layers  outers  inners   work_x   nseg    "
          "d_nseg")
    for r in rows:
        # sum_nlive is the total edge-visit count the propose kernel performs,
        # which is the quantity every per-inner cost scales with.
        wx = base["sum_nlive"] / r["sum_nlive"] if r["sum_nlive"] else 0
        tag = (f"{r['tag']} eps={r['eps']:.2f}"
               + ("" if r["size_asym"] else " no-asym"))
        print(f"V   {tag:16s} {r['n_layer']:6d} {r['nouter']:7d} "
              f"{r['ninner']:7d} {wx:7.2f}x {r['nseg']:8d} "
              f"{r['nseg'] - base['nseg']:+7d}")

    print("\nV   nseg is the segment count, not accuracy. A config that "
          "changes it has changed the segmentation and still owes a full "
          "four-threshold VOI run before it can be used.")
    l0 = [(r["tag"], r["eps"], r["layer_outers"][0] if r["layer_outers"] else 0,
           r["layer_merges"][0] if r["layer_merges"] else 0) for r in rows]
    print("V   layer 0, where 71% of the merges are:")
    for tag, e, o, m in l0:
        print(f"V     {tag} eps={e:.2f}  outers={o:3d} merges={m:8d}")

    dest = CACHE / ("v_levers_sub.json" if args.sub else "v_levers.json")
    dest.write_text(json.dumps(
        {"threshold": args.threshold, "sub": args.sub, "rows": rows},
        indent=2, default=int) + "\n")
    print(f"\nV wrote {dest.name}")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
