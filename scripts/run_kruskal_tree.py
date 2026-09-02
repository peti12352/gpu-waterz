#!/usr/bin/env python3
"""P0m/n/r then A27 / Z25 / F23 / R24 / C28 / S26 / W31 / H30 by branch rule."""
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
from p0_kruskal_shape import parse  # noqa: E402


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
    print(log.read_text()[-2500:], flush=True)
    return rc


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    log = LOG / "p0_kruskal_shape.log"
    LOG.mkdir(parents=True, exist_ok=True)
    print("+ P0 kruskal shape", flush=True)
    with log.open("w") as f:
        r = subprocess.run(
            [str(PY), str(ROOT / "scripts/p0_kruskal_shape.py")],
            cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, text=True,
        )
    text = log.read_text()
    print(text[-4000:], flush=True)
    if r.returncode != 0:
        raise SystemExit(r.returncode)
    info = parse(text)
    (CACHE / "p0_kruskal_shape.json").write_text(json.dumps(info, indent=2))
    print("P0 branch", info["branch"], "a27_hits", len(info["a27_hits"]), flush=True)

    locked = False
    if "A27" in info["branch"] and info["a27_hits"]:
        areas = sorted({h["a"] for h in info["a27_hits"]})
        for a in areas:
            rc = run(
                [str(PY), str(ROOT / "scripts/a27_area_cc.py"), "--area", str(a)],
                f"a27_a{a}.log",
            )
            if rc == 0:
                locked = True
                break

    if not locked:
        for s0 in (64, 256, 1024, 4096, 16384):
            rc = run(
                [str(PY), str(ROOT / "scripts/z25_sdsl.py"), "--s0", str(s0)],
                f"z25_s{s0}.log",
            )
            if rc == 0:
                locked = True
                break
            # close: try smooth if serial existed
            stamp = CACHE / "z25_pass.txt"
            if stamp.is_file() and "PASS-serial" in stamp.read_text():
                rc = run(
                    [str(PY), str(ROOT / "scripts/z25_sdsl.py"),
                     "--s0", str(s0), "--beta", "0.05"],
                    f"z25_s{s0}_sm.log",
                )
                if rc == 0:
                    locked = True
                    break
        print("Z25 max-face SKIP (no per-face list on rag.npz)", flush=True)

    if not locked:
        for k in (50, 200, 800, 2000):
            rc = run(
                [str(PY), str(ROOT / "scripts/f23_fh.py"), "--k", str(k)],
                f"f23_k{k}.log",
            )
            if rc == 0:
                locked = True
                break

    if not locked:
        for q in (16, 64, 256):
            rc = run(
                [str(PY), str(ROOT / "scripts/r24_srm.py"), "--Q", str(q)],
                f"r24_q{q}.log",
            )
            if rc == 0:
                locked = True
                break

    if not locked:
        for b, smax in ((16, 1024), (32, 1024), (16, 4096), (32, 4096)):
            rc = run(
                [str(PY), str(ROOT / "scripts/c28_x1_sizecap.py"),
                 "--bins", str(b), "--smax", str(smax)],
                f"c28_b{b}_s{smax}.log",
            )
            if rc == 0:
                locked = True
                break

    if not locked:
        for om in (0.05, 0.1, 0.2, 0.4):
            rc = run(
                [str(PY), str(ROOT / "scripts/s26_soille.py"), "--omega", str(om)],
                f"s26_w{om}.log",
                timeout=120,
            )
            if rc == 0:
                locked = True
                break

    if not locked:
        rc = run([str(PY), str(ROOT / "scripts/w31_waterfall.py")], "w31.log")
        locked = rc == 0
        if not locked:
            rc = run([str(PY), str(ROOT / "scripts/h30_hem.py")], "h30.log")
            locked = rc == 0

    print(f"TREE lock={'yes' if locked else 'none'}", flush=True)


if __name__ == "__main__":
    main()
