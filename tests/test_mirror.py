"""Edge-aware affinity mirror: seam plane of the flipped axis is zero."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _mirror():
    spec = importlib.util.spec_from_file_location(
        "make_big", ROOT / "scripts/make_big.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.mirror


def test_seam_zero_each_axis():
    mirror = _mirror()
    rng = np.random.default_rng(0)
    aff = rng.integers(1, 255, size=(3, 8, 8, 8), dtype=np.uint8)
    for axis in range(3):
        flips = [False, False, False]
        flips[axis] = True
        out = mirror(aff, tuple(flips))
        sl = [slice(None)] * 3
        sl[axis] = 0
        assert np.all(out[axis][tuple(sl)] == 0)
        assert out.shape == aff.shape
        assert out.dtype == np.uint8
