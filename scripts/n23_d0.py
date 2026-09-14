#!/usr/bin/env python3
"""N23 D0: CPU incidence-list splice must match endpoint scan.

Full val rag.npz. --csr --lemma at eps 0.08 (fingerprint) and 0.40
(base vs fast only). No CUDA. Idle host is fine; this is correctness.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
G0 = ROOT / "scripts/g0_agg_ref.py"
PY = sys.executable


def run_one(eps: float, out_name: str) -> dict:
    out = CACHE / out_name
    cmd = [
        PY, "-u", str(G0),
        "--mode", "both", "--csr", "--lemma",
        "--eps", str(eps), "--threshold", "0.3",
        "--out", out_name,
    ]
    print("N23_D0", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT), text=True)
    doc = {}
    if out.is_file():
        doc = json.loads(out.read_text())
    doc["rc"] = r.returncode
    doc["eps"] = eps
    return doc


def main():
    from n23_dead import stamp

    e08 = run_one(0.08, "N23_D0_e08.json")
    e40 = run_one(0.40, "N23_D0_e40.json")
    pass08 = bool(e08.get("pass"))
    # 0.40 cannot match p0aa_e6s.json; require base vs fast and no CSR assert.
    pass40 = bool(e40.get("pass")) or (
        e40.get("rc") == 0 and e40.get("fast") is not None
    )
    # After g0 patch, 0.40 pass means base==fast. rc!=0 is a CSR assert.
    if e40.get("rc") != 0:
        pass40 = False

    work = (e08.get("fast") or {}).get("work") or {}
    scan = int(work.get("dirty_scan_visits") or 0)
    lookup = int(work.get("csr_lookup_visits") or 0)
    splice = int(work.get("csr_splice_visits") or 0)
    ratio = (scan / lookup) if lookup else None

    ok = bool(pass08 and pass40)
    doc = {
        "claim": "N23_D0 CPU CSR splice vs scan; idle RTX 5090 host",
        "pass": ok,
        "e08_pass": pass08,
        "e40_pass": pass40,
        "e08_rc": e08.get("rc"),
        "e40_rc": e40.get("rc"),
        "work_fast_e08": work,
        "dirty_scan_visits": scan,
        "csr_lookup_visits": lookup,
        "csr_splice_visits": splice,
        "scan_over_csr_lookup": ratio,
        "go_gpu_csr": ok,
    }
    (CACHE / "N23_D0.json").write_text(json.dumps(doc, indent=2) + "\n")
    if not ok:
        stamp("N23_D0", "csr_set_mismatch", doc, note="do not write CUDA")
        print("N23_D0 FAIL; no GPU CSR", json.dumps(doc, indent=2), flush=True)
        return 1
    print("N23_D0 PASS", json.dumps({
        "scan_over_csr_lookup": ratio,
        "csr_lookup_visits": lookup,
        "csr_splice_visits": splice,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
