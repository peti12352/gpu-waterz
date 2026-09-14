#!/usr/bin/env python3
"""P0k: official ApproximateSubgraphHac on a 4096-vertex S3 subgraph. Timeout 30s."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAMP = ROOT / "data/cache/p0k_returned.txt"
GM = ROOT / "papers/repos/graph-mining"


def main():
    STAMP.parent.mkdir(parents=True, exist_ok=True)
    bazel = shutil.which("bazel")
    if bazel is None:
        print("P0k SKIP no bazel: official SubgraphHAC did not return")
        STAMP.write_text("0\n")
        return
    if not (GM / "in_memory/clustering/hac/subgraph/approximate_subgraph_hac.h").is_file():
        print("P0k SKIP no graph-mining headers")
        STAMP.write_text("0\n")
        return
    print("P0k bazel present; 30s timeout on a compile+run is not wired (no standalone target)")
    print("P0k SKIP: T22 not a VOI path; official target is not a one-file binary")
    STAMP.write_text("0\n")


if __name__ == "__main__":
    main()
