#!/usr/bin/env python3
"""P0x -> E6r or Track B (P0v/L36/R36/P0w) -> G16."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
LOG = ROOT / "data/logs"
PY = ROOT / ".venv/bin/python"

sys.path.insert(0, str(ROOT / "scripts"))
from p0_r32_leftover import parse as parse_p0v  # noqa: E402


def run(cmd, logname, timeout=None):
    log = LOG / logname
    LOG.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(cmd), flush=True)
    with log.open("w") as f:
        try:
            r = subprocess.run(
                cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                text=True, timeout=timeout,
            )
            rc = r.returncode
        except subprocess.TimeoutExpired:
            f.write("\nTIMEOUT\n")
            rc = 124
    print(log.read_text()[-3500:], flush=True)
    return rc


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    locked = False
    rc = run(
        [str(PY), str(ROOT / "scripts/p0_gpu_inner.py")],
        "p0_gpu_inner.log",
        timeout=600,
    )
    info = {}
    jp = CACHE / "p0_gpu_inner.json"
    if jp.is_file():
        info = json.loads(jp.read_text())
    print("P0x", info, flush=True)
    track = info.get("track", "B")
    if rc == 0 and track == "A":
        erc = run(
            [str(PY), str(ROOT / "scripts/e6r_parhac.py")],
            "e6r_parhac.log",
            timeout=600,
        )
        locked = erc == 0
        stamp = CACHE / "e6r_pass.txt"
        st = stamp.read_text() if stamp.is_file() else ""
        # Track B only if A dies on time, not on VOI (stamp "FAIL PASS agg_ms=...").
        if (not locked) and ("FAIL PASS agg_ms" in st) and ("LOCK" not in st):
            track = "B"
        elif not locked:
            track = "STOP"

    if not locked and track == "B":
        run(
            [str(PY), str(ROOT / "scripts/p0_r32_leftover.py")],
            "p0_r32_leftover.log",
            timeout=300,
        )
        p0v = parse_p0v((LOG / "p0_r32_leftover.log").read_text())
        (CACHE / "p0_r32_leftover.json").write_text(json.dumps(p0v, indent=2))
        print("P0v", p0v["branch"], "l36", p0v["l36"], flush=True)
        if p0v["l36"]:
            for g in (0.10, 0.05):
                if any(r["gamma"] == g and r["n_residual"] < 50_000 for r in p0v["p0v"]):
                    lrc = run(
                        [str(PY), str(ROOT / "scripts/l36_rel_leftover.py"),
                         "--gamma", str(g)],
                        f"l36_g{g}.log",
                    )
                    if lrc == 0:
                        locked = True
                        break
        if not locked:
            rrc = run(
                [str(PY), str(ROOT / "scripts/r36_relcontact.py")],
                "r36.log",
            )
            locked = rrc == 0
        if not locked:
            nrc = run(
                [str(PY), str(ROOT / "scripts/p0w_nnchain.py")],
                "p0w_nnchain.log",
                timeout=300,
            )
            locked = nrc == 0

    print(f"TREE lock={'yes' if locked else 'none'} track={track}", flush=True)
    grc = run(
        [str(PY), str(ROOT / "scripts/g16_regrade.py")],
        "g16_e6r.log",
        timeout=900,
    )
    print(f"G16 rc={grc}", flush=True)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
