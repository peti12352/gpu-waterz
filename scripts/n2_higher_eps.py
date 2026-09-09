#!/usr/bin/env python3
"""N2: T=0.3-only eps 0.40 / 0.48 / 0.64 with size_asym on. Stop at first FAIL."""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from g0_agg_ref import load_rag, run  # noqa: E402
from t3_memsafe import CACHE, T, grade_t3, voi_parent_mmap  # noqa: E402

EPS = [0.40, 0.48, 0.64]


def main():
    u, v, sm, ct, max_id = load_rag()
    e1 = json.loads((CACHE / "e1_t3_voi.json").read_text())
    locked = next(r for r in e1["rows"]
                  if abs(r["eps"] - 0.08) < 1e-12 and r.get("size_asym"))
    base_live = None
    best32 = next((r for r in e1["rows"]
                   if abs(r["eps"] - 0.32) < 1e-12 and r.get("size_asym")
                   and r.get("pass")), None)
    if best32 and best32.get("sum_nlive") and locked.get("sum_nlive"):
        base_live = locked["sum_nlive"]
    print(f"N2 nedge={u.size} T={T} sweep={EPS}", flush=True)

    dest = CACHE / "n2_higher_eps.json"
    rows = []
    stopped = None
    if dest.is_file():
        try:
            prev = json.loads(dest.read_text())
            rows = list(prev.get("rows") or [])
            stopped = prev.get("stopped")
        except (OSError, json.JSONDecodeError):
            rows = []
    done = {r["eps"] for r in rows}
    for eps in EPS:
        if eps in done:
            print(f"N2 skip eps={eps} (already in {dest.name})", flush=True)
            if rows and not rows[-1].get("pass"):
                stopped = eps
                break
            continue
        t0 = time.perf_counter()
        r = run("fast", u, v, sm, ct, max_id, T, eps, 64, 0, 0, False, True)
        split, merge, nseg_v = voi_parent_mmap(r["root"])
        ok, sl, ml = grade_t3(split, merge)
        wx = (base_live / r["sum_nlive"]) if base_live and r.get("sum_nlive") else None
        row = {
            "tag": "v1", "eps": eps, "size_asym": True, "threshold": T,
            "voi_split": split, "voi_merge": merge, "pass": ok,
            "sl": sl, "ml": ml, "src": "g0 fast + voi_numpy",
            "n_layer": r["n_layer"], "nouter": r["nouter"],
            "ninner": r["ninner"], "nmerge": r["nmerge"],
            "nseg": r["nseg"], "sum_nlive": r["sum_nlive"],
            "layer_outers": r["layer_outers"], "layer_merges": r["layer_merges"],
            "work": r["work"], "wall_s": time.perf_counter() - t0,
            "work_x_vs_locked": wx,
        }
        rows.append(row)
        dest.write_text(json.dumps(
            {"threshold": T, "rows": rows, "stopped": stopped},
            indent=2, default=float) + "\n")
        print(f"N2 eps={eps:.2f} split={split:.4f} merge={merge:.4f} "
              f"nlive={r['sum_nlive']} work_x={wx if wx else 'n/a'} "
              f"{'PASS' if ok else 'FAIL'} {row['wall_s']:.1f}s", flush=True)
        del r
        gc.collect()
        if not ok:
            stopped = eps
            print(f"N2 STOP at first FAIL eps={eps}", flush=True)
            break

    dest.write_text(json.dumps({
        "threshold": T, "rows": rows, "stopped": stopped,
        "any_pass": any(r["pass"] for r in rows),
        "best_pass": min((r for r in rows if r["pass"]),
                         key=lambda r: r["sum_nlive"], default=None),
    }, indent=2, default=float) + "\n")
    print(f"N2 wrote {dest.name}", flush=True)
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
