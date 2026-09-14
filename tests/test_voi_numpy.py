"""VOI formula matches funkey/waterz evaluate.hpp (gt==0 skipped)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from voi_numpy import voi_split_merge  # noqa: E402


def test_identical_is_zero():
    rng = np.random.default_rng(0)
    lab = rng.integers(1, 8, size=(4, 4, 4), dtype=np.uint32)
    split, merge = voi_split_merge(lab, lab)
    assert split == 0.0
    assert merge == 0.0


def test_skips_gt_zero():
    gt = np.array([0, 1, 1, 2], dtype=np.uint32)
    seg = np.array([9, 1, 1, 2], dtype=np.uint32)
    split, merge = voi_split_merge(seg, gt)
    assert split == 0.0
    assert merge == 0.0


def test_split_and_merge_bits():
    # two voxels of neuron 1 split into labels 1 and 2; neuron 2 intact
    gt = np.array([1, 1, 2, 2], dtype=np.uint32)
    seg = np.array([1, 2, 3, 3], dtype=np.uint32)
    split, merge = voi_split_merge(seg, gt)
    assert split > 0.0
    assert merge == 0.0
