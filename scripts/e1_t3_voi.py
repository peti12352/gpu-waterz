#!/usr/bin/env python3
"""E1: T=0.3-only eps/asym VOI. Speed is graded at 0.3; T=0.2 stays locked.

Uniform V1/V2 already failed four-T VOI at T=0.2 on the idle 5090. This asks
the question the four-T gate could not: does any unlocked config pass at
T=0.3 alone?

CPU ParHAC replica (g0 fast, bit-identical to base) plus waterz-matching VOI
on wz_fragments.npy and cremiA_val/gt.h5. No GPU.
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
sys.path.insert(0, str(ROOT / "src"))
from g0_agg_ref import CACHE, load_rag, run  # noqa: E402
from ref_cpu import extract_parent  # noqa: E402
from task_gate import BASE_MERGE, BASE_SPLIT, SLACK  # noqa: E402
from voi_numpy import voi_split_merge  # noqa: E402

T = 0.3
EPS_SWEEP = [0.08, 0.12, 0.16, 0.24, 0.32]
# Device four-T runs already measured these T=0.3 halves on the same partition
# family. Recorded so a killed laptop does not re-pay 3 min for a known fact.
DEVICE_T3 = {
    (0.08, True): {"voi_split": 0.451219, "voi_merge": 0.250531,
                   "src": "greengoblin a1 locked eps=0.08"},
    (0.16, True): {"voi_split": 0.453594, "voi_merge": 0.239641,
                   "src": "greengoblin v1_voi_real eps=0.16"},
    (0.08, False): {"voi_split": 0.456980, "voi_merge": 0.242085,
                    "src": "greengoblin v2_voi WATERZ_SIZE_ASYM=0"},
}


def configs(eps_list):
    out = [("v1", e, True) for e in eps_list]
    out.append(("v2", eps_list[0], False))
    if len(eps_list) > 1:
        out.append(("v1+v2", eps_list[-1], False))
    return out


def grade_t3(split, merge):
    sl, ml = BASE_SPLIT[T] + SLACK, BASE_MERGE[T] + SLACK
    ok = split <= sl and merge <= ml
    return ok, sl, ml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", nargs="+", type=float, default=EPS_SWEEP)
    ap.add_argument("--sub", type=int, default=0)
    ap.add_argument("--force-cpu", action="store_true",
                    help="recompute even the device-known rows")
    ap.add_argument("--out", default="e1_t3_voi.json")
    args = ap.parse_args()

    gt_p = CACHE / "gt.npy"
    if not gt_p.is_file():
        gt_p = ROOT / "data/ws_bounty/cremiA_val/gt.h5"
    fr_p = CACHE / "wz_fragments.npy"
    if not fr_p.is_file():
        print("E1 FAIL missing wz_fragments.npy", flush=True)
        return False
    if not gt_p.is_file():
        print("E1 FAIL missing data/cache/gt.npy or cremiA_val/gt.h5", flush=True)
        return False

    u, v, sm, ct, max_id = load_rag()
    if args.sub:
        keep = (u < args.sub) & (v < args.sub)
        u, v, sm, ct = u[keep], v[keep], sm[keep], ct[keep]
        max_id = int(max(u.max(), v.max()))
    print(f"E1 nedge={u.size} nnode={max_id + 1} T={T} "
          f"{'sub=' + str(args.sub) if args.sub else 'full val'}", flush=True)

    dest = CACHE / args.out
    rows = []
    if dest.is_file():
        try:
            prev = json.loads(dest.read_text())
            rows = list(prev.get("rows") or [])
        except (OSError, json.JSONDecodeError):
            rows = []

    def already(eps, asym):
        for r in rows:
            if abs(r["eps"] - eps) < 1e-12 and bool(r["size_asym"]) == asym:
                return r
        return None

    gt = fr = None
    any_pass = False
    for tag, eps, asym in configs(args.eps):
        hit = already(eps, asym)
        if hit is not None:
            print(f"E1 skip {tag} eps={eps} asym={int(asym)} "
                  f"(already in {dest.name})", flush=True)
            any_pass |= bool(hit.get("pass"))
            continue

        known = DEVICE_T3.get((eps, asym)) if not args.force_cpu and not args.sub else None
        row = {"tag": tag, "eps": eps, "size_asym": asym, "threshold": T}
        if known:
            split, merge = known["voi_split"], known["voi_merge"]
            ok, sl, ml = grade_t3(split, merge)
            row.update(known, pass_=ok, pass_ok=ok, sl=sl, ml=ml,
                       work=None, nseg=None)
            # json key "pass"
            row["pass"] = ok
            print(f"E1 {tag} eps={eps:.2f} asym={int(asym)}  "
                  f"split={split:.4f} merge={merge:.4f}  "
                  f"{'PASS' if ok else 'FAIL'}  [{known['src']}]", flush=True)
        else:
            t0 = time.perf_counter()
            r = run("fast", u, v, sm, ct, max_id, T, eps, 64, 0, 0, False, asym)
            dt = time.perf_counter() - t0
            if gt is None:
                if gt_p.suffix == ".npy":
                    gt = np.load(gt_p)
                else:
                    import h5py
                    with h5py.File(gt_p, "r") as h:
                        gt = np.ascontiguousarray(h["gt"][:].astype(np.uint32))
                fr = np.load(fr_p)
            lab = extract_parent(fr, r["root"]).astype(np.uint32, copy=False)
            split, merge = voi_split_merge(lab, gt)
            del lab
            ok, sl, ml = grade_t3(split, merge)
            row.update({
                "voi_split": split, "voi_merge": merge, "pass": ok,
                "sl": sl, "ml": ml, "src": "g0 fast + voi_numpy",
                "n_layer": r["n_layer"], "nouter": r["nouter"],
                "ninner": r["ninner"], "nmerge": r["nmerge"],
                "nseg": r["nseg"], "sum_nlive": r["sum_nlive"],
                "layer_outers": r["layer_outers"],
                "layer_merges": r["layer_merges"],
                "work": r["work"], "wall_s": dt,
            })
            print(f"E1 {tag} eps={eps:.2f} asym={int(asym)}  "
                  f"layers={r['n_layer']} outers={r['nouter']} "
                  f"nseg={r['nseg']} split={split:.4f} merge={merge:.4f}  "
                  f"{'PASS' if ok else 'FAIL'}  {dt:.1f}s", flush=True)
            del r
        any_pass |= bool(row["pass"])
        rows.append(row)
        dest.write_text(json.dumps(
            {"threshold": T, "sub": args.sub, "rows": rows},
            indent=2, default=float) + "\n")

    print("\nE1 T=0.3 only  split<=0.4738  merge<=0.2611")
    print("E1   config                 split   merge   work_x  pass")
    base = next((r for r in rows if abs(r["eps"] - 0.08) < 1e-12
                 and r["size_asym"]), rows[0])
    b_live = base.get("sum_nlive")
    for r in rows:
        wx = (b_live / r["sum_nlive"]) if b_live and r.get("sum_nlive") else None
        tag = (f"{r['tag']} eps={r['eps']:.2f}"
               + ("" if r["size_asym"] else " no-asym"))
        wxs = f"{wx:6.2f}x" if wx else "    n/a"
        print(f"E1   {tag:22s} {r['voi_split']:.4f}  {r['voi_merge']:.4f}  "
              f"{wxs}  {'PASS' if r['pass'] else 'FAIL'}")
    print(f"\nE1 {'PASS (at least one unlocked T=0.3 config)' if any_pass else 'KILL (no T=0.3 pass)'}")
    print(f"E1 wrote {dest.name}")
    return any_pass


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
