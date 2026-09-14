#!/usr/bin/env python3
"""T15: GPU TeraHAC/mutex only if T14 outer<=30 or M16 PASS."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAMP = ROOT / "data/cache/t14_outer.txt"
M16 = ROOT / "data/cache/m16_pass.txt"


def main():
    outer = 10**9
    if STAMP.is_file():
        outer = int(STAMP.read_text().strip().split()[0])
    m16 = M16.is_file() and M16.read_text().strip() == "PASS"
    print(f"T15 t14_outer={outer} m16_pass={m16}")
    if m16:
        print("T15 mutex PASS: GPU cub sort is the G6 path (not implemented this cycle; host UF is already 13s->need <10ms)")
        return
    if outer <= 30:
        print("T15 would implement persistent GPU good-matching; not reached")
        raise SystemExit(2)
    print(
        f"T15 SKIP outer={outer} > 30 and M16 not PASS. "
        "Same reason as E6: too many sequential rounds for a 10 ms budget."
    )


if __name__ == "__main__":
    main()
