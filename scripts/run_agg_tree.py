#!/usr/bin/env python3
"""Parse P0 stderr, write p0_agg_shape.json, run B18/A17/C19/Q20/S21/T22 by branch rule."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
LOG = ROOT / "data/logs"
PY = ROOT / ".venv/bin/python"


def parse(text: str) -> dict:
    out = {"p0e": {}, "p0f": {}, "p0g": {}, "p0h": {}, "p0i": {}, "p0j": {}, "branch": []}
    for m in re.finditer(r"P0e n_mean_gt_([0-9.]+)=(\d+)", text):
        out["p0e"][f"gt_{m.group(1)}"] = int(m.group(2))
    for tag in ("f10", "f05"):
        m = re.search(rf"P0{tag} total_rounds=(\d+)", text)
        if m:
            out["p0f"][tag] = int(m.group(1))
    m = re.search(r"P0g total_rounds=(\d+)", text)
    if m:
        out["p0g"]["rounds"] = int(m.group(1))
    m = re.search(r"P0h drop_upgma=(\d+) drop_minface=(\d+)", text)
    if m:
        out["p0h"]["drop_upgma"] = int(m.group(1))
        out["p0h"]["drop_minface"] = int(m.group(2))
    m = re.search(r"P0i sub_edges=(\d+) merges=(\d+) unique_scores=(\d+) max_parent_chain=(\d+)", text)
    if m:
        out["p0i"] = {
            "sub_edges": int(m.group(1)),
            "merges": int(m.group(2)),
            "unique_scores": int(m.group(3)),
            "max_parent_chain": int(m.group(4)),
        }
    m = re.search(r"P0j area_lt2=(\d+) lt8=(\d+) lt32=(\d+) lt128=(\d+) ge128=(\d+) max=(\d+)", text)
    if m:
        out["p0j"] = {
            "lt2": int(m.group(1)), "lt8": int(m.group(2)), "lt32": int(m.group(3)),
            "lt128": int(m.group(4)), "ge128": int(m.group(5)), "max": int(m.group(6)),
        }
    f_r = min(out["p0f"].get("f10", 10**9), out["p0f"].get("f05", 10**9), out["p0g"].get("rounds", 10**9))
    if f_r <= 30:
        out["branch"].append("B18")
    if out["p0h"].get("drop_upgma") or out["p0h"].get("drop_minface"):
        out["branch"].append("A17")
    n95 = out["p0e"].get("gt_0.95", 10**9)
    if n95 < 50_000:
        out["branch"].append("C19")
    if not out["branch"]:
        out["branch"].append("Q20")
    out["min_match_rounds"] = f_r if f_r < 10**9 else None
    return out


def run(cmd, logname):
    log = LOG / logname
    LOG.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(cmd), flush=True)
    with log.open("w") as f:
        r = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    print(log.read_text()[-2000:], flush=True)
    return r.returncode


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    log = LOG / "p0_agg_shape.log"
    LOG.mkdir(parents=True, exist_ok=True)
    print("+ P0 agg shape", flush=True)
    with log.open("w") as f:
        r = subprocess.run(
            [str(PY), str(ROOT / "scripts/p0_agg_shape.py")],
            cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, text=True,
        )
    text = log.read_text()
    print(text[-4000:], flush=True)
    if r.returncode != 0:
        raise SystemExit(r.returncode)
    info = parse(text)
    (CACHE / "p0_agg_shape.json").write_text(json.dumps(info, indent=2))
    print("P0 branch", info["branch"], "min_match_rounds", info["min_match_rounds"], flush=True)
    subprocess.run([str(PY), str(ROOT / "scripts/p0k_official.py")], cwd=str(ROOT))

    did_lock = False
    if "B18" in info["branch"]:
        rc = run([str(PY), str(ROOT / "scripts/b18_binmatch.py"), "--bins", "256"], "b18.log")
        did_lock = rc == 0
        if rc != 0 and info["min_match_rounds"] and 30 < info["min_match_rounds"] <= 80:
            rc = run([str(PY), str(ROOT / "scripts/b18_binmatch.py"), "--bins", "64"], "b18_64.log")
            did_lock = rc == 0
    if "A17" in info["branch"] and not did_lock:
        pi = 0 if info["p0h"].get("drop_upgma") else 1
        rc = run([str(PY), str(ROOT / "scripts/a17_dual.py"), "--eps", "0.08", "--pi", str(pi)], "a17.log")
        did_lock = rc == 0
        if rc != 0:
            rc = run([str(PY), str(ROOT / "scripts/a17_dual.py"), "--eps", "0.05", "--pi", str(pi)], "a17_e05.log")
            did_lock = rc == 0
        if rc != 0:
            rc = run([str(PY), str(ROOT / "scripts/a17_dual.py"), "--eps", "0.02", "--pi", str(pi)], "a17_e02.log")
            did_lock = rc == 0
    if "C19" in info["branch"] and not did_lock:
        rc = run([str(PY), str(ROOT / "scripts/c19_coarsen.py"), "--tau", "0.95"], "c19.log")
        did_lock = rc == 0
    if not did_lock:
        rc = run([str(PY), str(ROOT / "scripts/q20_quantile.py")], "q20.log")
        did_lock = rc == 0
        if rc != 0:
            rc = run([str(PY), str(ROOT / "scripts/s21_gasp_mean.py")], "s21.log")
            did_lock = rc == 0
    run([str(PY), str(ROOT / "scripts/t22_skip.py")], "t22.log")
    print(f"TREE lock={'yes' if did_lock else 'none'}", flush=True)


if __name__ == "__main__":
    main()
