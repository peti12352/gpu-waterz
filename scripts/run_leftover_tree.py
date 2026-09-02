#!/usr/bin/env python3
"""P0 leftover/relative/tiles then L33 / R32 / B34 / V35 by branch rule; G16."""
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
from p0_leftover_block import parse  # noqa: E402


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
    print(log.read_text()[-3000:], flush=True)
    return rc


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    log = LOG / "p0_leftover_block.log"
    LOG.mkdir(parents=True, exist_ok=True)
    print("+ P0 leftover/relative/tiles", flush=True)
    with log.open("w") as f:
        r = subprocess.run(
            [str(PY), str(ROOT / "scripts/p0_leftover_block.py")],
            cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, text=True,
        )
    text = log.read_text()
    print(text[-5000:], flush=True)
    if r.returncode != 0:
        raise SystemExit(r.returncode)
    info = parse(text)
    (CACHE / "p0_leftover_block.json").write_text(json.dumps(info, indent=2))
    print(
        f"P0 branch={info['branch']} l33_s0={info['l33_s0']} "
        f"r32=({info['r32_gamma']},{info['r32_alpha']}) b34={info['b34_tile']}",
        flush=True,
    )

    locked = False
    tried = []

    s0_list = info.get("l33_s0s") or ([info["l33_s0"]] if info.get("l33_s0") is not None else [])
    if s0_list:
        for s0 in s0_list:
            tried.append(f"L33/s{s0}")
            rc = run(
                [str(PY), str(ROOT / "scripts/l33_leftover.py"), "--s0", str(s0)],
                f"l33_s{s0}.log",
            )
            if rc == 0:
                locked = True
                break

    if not locked and (info["r32_gamma"] is not None or "R32" in info["branch"]):
        if info["r32_gamma"] is None:
            print("R32 skip (no P0t candidate)", flush=True)
        else:
            tried.append("R32")
            rc = run(
                [
                    str(PY), str(ROOT / "scripts/r32_relcontact.py"),
                    "--gamma", str(info["r32_gamma"]),
                    "--alpha", str(info["r32_alpha"]),
                ],
                f"r32_g{info['r32_gamma']}_a{info['r32_alpha']}.log",
            )
            locked = rc == 0

    if not locked and info["b34_tile"] is not None:
        tried.append("B34")
        dz, dy, dx = info["b34_tile"]
        rc = run(
            [
                str(PY), str(ROOT / "scripts/b34_block_s3.py"),
                "--dz", str(dz), "--dy", str(dy), "--dx", str(dx),
            ],
            f"b34_{dz}_{dy}_{dx}.log",
        )
        locked = rc == 0

    if not locked:
        tried.append("V35")
        rc = run(
            [str(PY), str(ROOT / "scripts/v35_hysteresis.py")],
            "v35.log",
            timeout=600,
        )
        locked = rc == 0

    print(f"TREE tried={tried} lock={'yes' if locked else 'none'}", flush=True)
    rc = run(
        [str(PY), str(ROOT / "scripts/g16_regrade.py")],
        "g16_leftover.log",
        timeout=600,
    )
    print(f"G16 rc={rc}", flush=True)
    raise SystemExit(0 if locked or rc == 0 else 1)


if __name__ == "__main__":
    main()
