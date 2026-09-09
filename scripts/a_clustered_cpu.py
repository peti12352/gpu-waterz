#!/usr/bin/env python3
"""A: clustered-graph work gate after reading ParHAC §2.3 / p.23, not the abstract.

E2 CSR already is MultiMerge (dirty walk + splice). The paper 7–11× is
Affinity/SCCsim GBBS vs clustered-graph (p.23), not ParHAC. This script
records that from e2_csr_full.json + the pinned quotes. Optional --rerun
replays g0 --csr on greengoblin; default does not.

Not a 2 Gvox/s claim.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
OUT = CACHE / "a_clustered_cpu.json"
G0 = ROOT / "scripts/g0_agg_ref.py"
E2 = CACHE / "e2_csr_full.json"


def main():
    rerun = "--rerun" in sys.argv
    e2 = json.loads(E2.read_text()) if E2.is_file() else None
    doc = {
        "claim": "not a 2 Gvox/s number",
        "paper": {
            "parhac": "papers/parhac_dhulipala2022.pdf arXiv:2206.11654",
            "section_2_3": (
                "clustered graph for ParHAC when ε is small (e.g. 0.01) "
                "and rounds merge few vertices"
            ),
            "seven_to_eleven_x": (
                "Affinity and SCCsim: GBBS full-weight-recompute vs clustered "
                "graph. NOT ParHAC vs ParHAC. SOURCES S32."
            ),
            "cpam_vs_ht": "D.3: same runtime, CPAM 2.9x space",
        },
        "e2_is_multimerge": True,
        "e2": {
            "identical": e2.get("identical") if e2 else None,
            "cut": e2.get("cut") if e2 else None,
            "scan_visits": e2.get("scan_visits") if e2 else None,
            "csr_lookup_splice": e2.get("csr_lookup_splice") if e2 else None,
        },
        "layer0_merges": 1322268,
        "nmerge": 1853427,
        "layer0_outers": 64,
        "merges_per_outer_layer0": 1322268 / 64,
        "small_round_regime": False,
        "honest_stack_already_credits_e2": True,
        "start_gpu_starmarge_v2": False,
        "keep_n6_on_work": True,
        "rerun": None,
    }
    if rerun:
        rag = CACHE / "rag.npz"
        if not rag.is_file():
            print("A REFUSE --rerun: rag.npz missing (run on greengoblin)",
                  flush=True)
            raise SystemExit(2)
        cmd = [sys.executable, "-u", str(G0), "--mode", "fast", "--csr",
               "--out", "a_clustered_g0_csr.json"]
        print("A rerun", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT))
        doc["rerun"] = {"rc": r.returncode}
        g0p = CACHE / "a_clustered_g0_csr.json"
        if g0p.is_file():
            g = json.loads(g0p.read_text())
            doc["rerun"]["nmerge"] = g.get("fast", {}).get("nmerge")
            doc["rerun"]["work"] = g.get("fast", {}).get("work")

    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(
        f"A E2 cut={doc['e2']['cut']} identical={doc['e2']['identical']} "
        f"layer0 ~{doc['merges_per_outer_layer0']:.0f} merges/outer "
        f"(not paper ε=0.01 small-round). 7-11x is Affinity/SCC. "
        f"keep_n6_on_work=True -> {OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
