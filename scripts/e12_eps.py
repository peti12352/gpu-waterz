#!/usr/bin/env python3
"""E12: paper Alg. 1+2 ε in (0.033333, 0.1). Lock largest ε that PASS+G5."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv/bin/python"
EPS = [0.05, 0.06, 0.07, 0.08, 0.09]


def main():
    results = []
    for eps in EPS:
        cmd = [str(PY), str(ROOT / "scripts/e3_paper_parhac.py"), "--eps", str(eps)]
        print("+", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT))
        ok = r.returncode == 0
        results.append((eps, ok))
        print(f"E12 eps={eps} {'PASS' if ok else 'FAIL'} rc={r.returncode}", flush=True)
    print("E12 summary", results)
    passed = [e for e, ok in results if ok]
    if passed:
        print(f"E12 lock candidate largest PASS ε={max(passed)}")
    else:
        print("E12 none passed; keep locked ε=0.033333")


if __name__ == "__main__":
    main()
