#!/usr/bin/env python3
"""T22: official SubgraphHAC 4k timing only if P0k returned."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
stamp = ROOT / "data/cache/p0k_returned.txt"
v = stamp.read_text().strip() if stamp.is_file() else "0"
print(f"T22 p0k_returned={v}")
if v != "1":
    print("T22 SKIP P0k did not return. Official TeraHAC not a G6 path.")
else:
    print("T22 would time 4k official SubgraphHAC: no standalone binary, SKIP")
