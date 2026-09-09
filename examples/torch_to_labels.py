#!/usr/bin/env python3
"""Minimal torch (or numpy) affinities -> gpu_waterz labels.

Uses a synthetic [3,Z,Y,X] cube so no CREMI download is required.
Requires built CUDA libs (bash scripts/build_cuda.sh) for a real run.
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    import numpy as np

    try:
        import gpu_waterz as wz
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import gpu_waterz as wz

    z, y, x = 16, 32, 32
    rng = np.random.default_rng(0)
    aff = rng.random((3, z, y, x), dtype=np.float32)
    # Soft walls so watershed has structure
    aff[:, z // 2, :, :] *= 0.05
    aff[:, :, y // 2, :] *= 0.05

    print("cuda_libs_ready:", wz.cuda_libs_ready())
    print("scores_to_affinity([0.7]) ->", wz.scores_to_affinity([0.7]))

    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    if has_torch and torch.cuda.is_available() and wz.cuda_libs_ready():
        t = torch.from_numpy(aff).cuda()
        labs = wz.segment_d(wz.from_torch(t), [0.3], return_device=True)
        out = wz.to_torch(labs[0])
        print("torch cuda labels:", tuple(out.shape), out.dtype, out.device)
        return 0

    if not wz.cuda_libs_ready():
        print("CUDA .so missing; run: bash scripts/build_cuda.sh")
        print("API import OK; skipping segment.")
        return 0

    labs = wz.segment(aff, [0.3])
    print("numpy labels:", labs[0].shape, labs[0].dtype, "nunique", len(np.unique(labs[0])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
