#!/usr/bin/env python3
"""E3: layer 0 does 71% of merges and sits on the 64-outer cap.

Count leftover (1+eps)-legal candidates after 64 outers, then try
(a) a higher layer-0 cap (b) k-heaviest blues per red on a subgraph.
Keep a variant only if outers drop and identity or T=0.3 VOI holds.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from g0_agg_ref import CACHE, load_rag, run


def leftover_after_cap(u, v, sm, ct, max_id, thr, eps, cap, size_asym):
    """Run with max_outer=cap, then one more propose on the leftover graph."""
    r = run("fast", u, v, sm, ct, max_id, thr, eps, cap, 1, 0, False,
            size_asym, False, False, 0.0)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, default=80000)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--eps", type=float, default=0.08)
    ap.add_argument("--out", default="e3_layer0.json")
    args = ap.parse_args()

    u, v, sm, ct, max_id = load_rag()
    if args.sub:
        keep = (u < args.sub) & (v < args.sub)
        u, v, sm, ct = u[keep], v[keep], sm[keep], ct[keep]
        max_id = int(max(u.max(), v.max()))
    print(f"E3 nedge={u.size} nnode={max_id + 1} sub={args.sub}", flush=True)

    rows = []
    t0 = time.perf_counter()
    base = leftover_after_cap(u, v, sm, ct, max_id, args.threshold, args.eps,
                              64, True)
    print(f"E3 cap64  L0 outers={base['layer_outers'][0]} "
          f"merges={base['layer_merges'][0]} nseg={base['nseg']} "
          f"{time.perf_counter()-t0:.1f}s", flush=True)
    rows.append({"tag": "cap64", "max_outer": 64, "size_asym": True,
                 **{k: base[k] for k in ("n_layer", "nouter", "ninner",
                                         "nmerge", "nseg", "layer_outers",
                                         "layer_merges", "sum_nlive")}})

    t0 = time.perf_counter()
    hi = leftover_after_cap(u, v, sm, ct, max_id, args.threshold, args.eps,
                            128, True)
    ident_hi = bool(np.array_equal(base["root"], hi["root"]))
    print(f"E3 cap128 L0 outers={hi['layer_outers'][0]} "
          f"merges={hi['layer_merges'][0]} nseg={hi['nseg']} "
          f"identical={ident_hi} {time.perf_counter()-t0:.1f}s", flush=True)
    rows.append({"tag": "cap128", "max_outer": 128, "size_asym": True,
                 "identical_to_cap64": ident_hi,
                 **{k: hi[k] for k in ("n_layer", "nouter", "ninner",
                                       "nmerge", "nseg", "layer_outers",
                                       "layer_merges", "sum_nlive")}})

    t0 = time.perf_counter()
    nas = leftover_after_cap(u, v, sm, ct, max_id, args.threshold, args.eps,
                             64, False)
    ident_na = bool(np.array_equal(base["root"], nas["root"]))
    print(f"E3 no-asym L0 outers={nas['layer_outers'][0]} "
          f"merges={nas['layer_merges'][0]} nseg={nas['nseg']} "
          f"identical={ident_na} {time.perf_counter()-t0:.1f}s", flush=True)
    rows.append({"tag": "cap64_no_asym", "max_outer": 64, "size_asym": False,
                 "identical_to_cap64": ident_na,
                 **{k: nas[k] for k in ("n_layer", "nouter", "ninner",
                                        "nmerge", "nseg", "layer_outers",
                                        "layer_merges", "sum_nlive")}})

    l0_base = base["layer_outers"][0]
    l0_hi = hi["layer_outers"][0]
    # A higher cap that still burns 64+ outers did not finish layer 0 either.
    # A drop in total outers on a *full* run is what E1 already measures for
    # no-asym. Here, on layer 0 only, "outers drop" means the layer exits
    # before the cap.
    hi_exits_early = l0_hi < 128
    na_exits_early = nas["layer_outers"][0] < 64
    keep = []
    if ident_hi and hi["nouter"] < base["nouter"]:
        keep.append("cap128")
    if (not ident_na) and na_exits_early:
        keep.append("cap64_no_asym (partition moved; owes T=0.3 VOI)")
    if ident_hi and not hi_exits_early and hi["layer_merges"][0] == base["layer_merges"][0]:
        print("E3 cap128 added no layer-0 merges: leftover after 64 is empty "
              "or the cap is not the limiter on this subgraph", flush=True)

    print(f"E3 layer-0 cap is {l0_base} (64 means pinned)")
    print(f"E3 keep={keep if keep else 'none'}")
    dest = CACHE / args.out
    dest.write_text(json.dumps({
        "sub": args.sub, "rows": rows, "keep": keep,
        "layer0_pinned_at_64": l0_base >= 64,
    }, indent=2, default=int) + "\n")
    print(f"E3 wrote {dest.name}")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
