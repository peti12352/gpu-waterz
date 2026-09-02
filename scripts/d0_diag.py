#!/usr/bin/env python3
"""D0: classify paper-ParHAC outers at locked ε=0.033333. No algorithm change."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from e3_paper_parhac import main as e3_main  # noqa: E402


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--eps", "0.033333"]
    e3_main()
