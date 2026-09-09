#!/usr/bin/env python3
"""N14 T5: 3090 Ti rent gate. Do not run n12_3090.py on a 5090.

Only after T1-T3 stacked keep_default. This script records the skip.
Not a 2 Gvox/s claim.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from p1_make_big_indep import CACHE, gpu_state  # noqa: E402

NOTE = ROOT / "notes/N14_T5.md"
OUT = CACHE / "n14_t5.json"


def stacked_closer():
    keep = []
    for name in ("n14_t1.json", "n14_t2.json", "n14_t3.json"):
        p = CACHE / name
        if not p.is_file():
            return False, keep
        d = json.loads(p.read_text())
        keep.append(bool(d.get("keep_default")))
    return all(keep) and len(keep) == 3, keep


def gpu_name():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    return (r.stdout or "").strip()


def main():
    print("N14 T5 3090 gate. Not a 5090 2 Gvox/s number.", flush=True)
    name = gpu_name()
    stacked, keeps = stacked_closer()
    ran = False
    rc = None
    if stacked and "3090 Ti" in name:
        r = subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts/n12_3090.py")],
            cwd=str(ROOT),
        )
        ran = True
        rc = r.returncode
    doc = {
        "claim": "not a 2 Gvox/s number unless n12_3090.py median on 3090 Ti",
        "gpu": gpu_state(),
        "name": name,
        "stacked_closer": stacked,
        "keep_t1_t2_t3": keeps,
        "ran_n12_3090": ran,
        "rc": rc,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    NOTE.write_text(
        "# N14 T5 3090 Ti\n\n"
        "scripts/n12_3090.py is ready. Run only on a rented 3090 Ti after a "
        "stacked T1-T3 closer. Will not turn 4.9 s into 1.1 s by itself.\n\n"
        f"- nvidia-smi name={name!r}\n"
        f"- stacked_closer={stacked} keep={keeps}\n"
        f"- ran_n12_3090={ran} rc={rc}\n"
    )
    print(f"N14 T5 stacked={stacked} name={name!r} ran={ran} -> {OUT}", flush=True)
    if "3090 Ti" not in name and stacked:
        print("N14 T5: stacked closer but this card is not 3090 Ti. Do not claim grade.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
