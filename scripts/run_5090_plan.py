#!/usr/bin/env python3
"""P0aa → P0ab → E6uvw gate. Refuses to start if another process owns the GPU."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"


def _gpu_busy():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nvidia-smi failed"
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not lines:
        return None
    return out.strip()


def _run(script):
    cmd = [str(PY), str(ROOT / "scripts" / script)]
    print(f"+ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def main():
    busy = _gpu_busy()
    if busy:
        print("GPU busy; do not touch other jobs:\n" + busy, flush=True)
        return 2
    rc = _run("p0aa_e6s.py")
    if rc != 0:
        print(f"P0aa failed rc={rc}", flush=True)
        return rc
    rc = _run("p0ab_outer_cap.py")
    if rc != 0:
        print(f"P0ab failed rc={rc}", flush=True)
        return rc
    cap_path = ROOT / "data" / "cache" / "p0ab_outer_cap.json"
    if cap_path.exists():
        print("P0ab", cap_path.read_text()[:400], flush=True)
    rc = _run("e6s_parhac.py")
    print(f"gate rc={rc}", flush=True)
    stamp = ROOT / "data" / "cache" / "e6r_pass.txt"
    if stamp.exists():
        print("stamp", stamp.read_text().strip(), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
