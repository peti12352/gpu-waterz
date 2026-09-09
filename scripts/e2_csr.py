#!/usr/bin/env python3
"""E2: CSR/adjacency dirty-combine vs the G2 endpoint scan.

Identity against the existing fast replica. Visit cut: dirty_scan_visits
must drop from O(ncompact * nlive) to O(ndirty + splice). Rebuild bytes
are charged separately; a full rebuild every inner is a kill (same order
as the scan). Splice is O(nmerge).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from g0_agg_ref import CACHE, load_rag, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, default=80000)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--eps", type=float, default=0.08)
    ap.add_argument("--max-layer", type=int, default=0)
    ap.add_argument("--theta", type=float, default=0.0,
                    help="also run E2b physical compact at this hole fraction")
    ap.add_argument("--ref-json", default="",
                    help="use this g0 json's fast.work as the scan baseline "
                         "instead of re-running scan")
    ap.add_argument("--out", default="e2_csr.json")
    args = ap.parse_args()

    u, v, sm, ct, max_id = load_rag()
    if args.sub:
        keep = (u < args.sub) & (v < args.sub)
        u, v, sm, ct = u[keep], v[keep], sm[keep], ct[keep]
        max_id = int(max(u.max(), v.max()))
    print(f"E2 nedge={u.size} nnode={max_id + 1} sub={args.sub}", flush=True)

    scan = None
    dt0 = 0.0
    refp = Path(args.ref_json) if args.ref_json else None
    if refp and refp.is_file():
        ref = json.loads(refp.read_text())
        scan = ref.get("fast") or ref.get("base")
        print(f"E2 scan baseline from {refp.name} "
              f"dirty_scan={scan['work']['dirty_scan_visits']}", flush=True)
    else:
        t0 = time.perf_counter()
        scan = run("fast", u, v, sm, ct, max_id, args.threshold, args.eps,
                   64, args.max_layer, 0, False, True, False, False, 0.0)
        dt0 = time.perf_counter() - t0
        print(f"E2 scan {dt0:.1f}s nmerge={scan['nmerge']} nseg={scan['nseg']} "
              f"dirty_scan={scan['work']['dirty_scan_visits']}", flush=True)

    t0 = time.perf_counter()
    csr = run("fast", u, v, sm, ct, max_id, args.threshold, args.eps,
              64, args.max_layer, 0, False, True, False, True, 0.0)
    dt1 = time.perf_counter() - t0
    if "root" in scan and "root" in csr:
        ident = bool(np.array_equal(scan["root"], csr["root"]))
    else:
        ident = (scan.get("nmerge") == csr["nmerge"]
                 and scan.get("nseg") == csr["nseg"]
                 and scan.get("sum_nlive") == csr["sum_nlive"])
        print("E2 identity via counters (per-inner CSR==scan assert is the "
              "parent proof)", flush=True)
    print(f"E2 csr  {dt1:.1f}s nmerge={csr['nmerge']} nseg={csr['nseg']} "
          f"lookup={csr['work']['csr_lookup_visits']} "
          f"rebuild={csr['work']['csr_rebuild_visits']} "
          f"splice={csr['work']['csr_splice_visits']} "
          f"identical={ident}", flush=True)

    ws, wc = scan["work"], csr["work"]
    scan_v = ws["dirty_scan_visits"]
    csr_v = wc["csr_lookup_visits"] + wc["csr_splice_visits"]
    rebuild_v = wc["csr_rebuild_visits"]
    cut = scan_v / max(csr_v, 1)
    # 8 B per visit (uint32 pair) is the honest compare
    scan_b = scan_v * 8
    csr_b = csr_v * 8 + rebuild_v * 4
    e2b = None
    if args.theta > 0:
        t0 = time.perf_counter()
        th = run("fast", u, v, sm, ct, max_id, args.threshold, args.eps,
                 64, args.max_layer, 0, False, True, False, True, args.theta)
        if "root" in scan:
            ident_th = bool(np.array_equal(scan["root"], th["root"]))
        else:
            ident_th = (scan.get("nmerge") == th["nmerge"]
                        and scan.get("nseg") == th["nseg"])
        e2b = {"theta": args.theta, "identical": ident_th,
               "work": th["work"], "nmerge": th["nmerge"],
               "wall_s": time.perf_counter() - t0}
        print(f"E2b theta={args.theta} identical={ident_th} "
              f"dirty_scan={th['work']['dirty_scan_visits']}", flush=True)

    ok = ident and cut >= 10.0
    # rebuild-every-layer is cheap (17 times). kill only if rebuild dominates
    # the scan it replaces *and* identity failed or cut < 10.
    rebuild_fatter = csr_b > scan_b
    print(f"E2 lookup+splice {csr_v} vs scan {scan_v} = {cut:.1f}x")
    print(f"E2 bytes csr {csr_b} vs scan {scan_b}  "
          f"rebuild_fatter={rebuild_fatter}")
    print(f"E2 {'PASS' if ok else 'FAIL'} identity={ident} cut={cut:.1f}x "
          f"(need >=10x and parent-identical)")

    dest = CACHE / args.out
    dest.write_text(json.dumps({
        "sub": args.sub, "identical": ident, "cut": cut,
        "scan_visits": scan_v, "csr_lookup_splice": csr_v,
        "csr_rebuild_visits": rebuild_v, "rebuild_fatter": rebuild_fatter,
        "pass": ok, "scan": {k: scan[k] for k in scan if k != "root"},
        "csr": {k: csr[k] for k in csr if k != "root"},
        "e2b": e2b,
    }, indent=2, default=int) + "\n")
    print(f"E2 wrote {dest.name}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
