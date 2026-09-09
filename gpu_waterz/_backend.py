"""Load the in-tree CUDA/Python backend from ``src/segment.py``."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import segment as seg  # noqa: E402

REQUIRED_SOS = (
    SRC / "libws_gpu.so",
    SRC / "librag_gpu.so",
    SRC / "libparhac_d.so",
)
