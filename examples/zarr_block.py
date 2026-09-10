#!/usr/bin/env python3
"""Zarr affinity block -> gpu_waterz labels.

LSD/daisy workers split fragments then agglomerate (see
https://github.com/funkelab/lsd/blob/tutorial/lsd/tutorial/scripts/workers/agglomerate_worker.py).
This script is the e2e drop-in for one block: read [3,Z,Y,X] affinities,
call gpu_waterz.segment, write uint32 labels. No Mongo.

Needs zarr (uv sync --extra zarr) and CUDA libs for a real decode.
Without .so files it still writes the synthetic affinity volume and skips
segment.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def main() -> int:
    try:
        import zarr
    except ImportError:
        print("zarr not installed; uv sync --extra zarr")
        return 0

    import numpy as np

    try:
        import gpu_waterz as wz
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import gpu_waterz as wz

    z, y, x = 16, 32, 32
    rng = np.random.default_rng(0)
    aff = rng.random((3, z, y, x), dtype=np.float32)
    aff[:, z // 2, :, :] *= 0.05
    aff[:, :, y // 2, :] *= 0.05

    root = Path(tempfile.mkdtemp(prefix="gpu_waterz_zarr_"))
    aff_path = root / "aff.zarr"
    lab_path = root / "labels.zarr"
    zarr.save_array(str(aff_path), aff)
    print("wrote affinities", aff.shape, "->", aff_path)

    loaded = np.asarray(zarr.open_array(str(aff_path), mode="r"))
    if loaded.shape != (3, z, y, x):
        raise SystemExit(f"unexpected aff shape {loaded.shape}")

    if not wz.cuda_libs_ready():
        print("CUDA .so missing; run: bash scripts/build_cuda.sh")
        print("skipping segment; affinity zarr is ready at", aff_path)
        return 0

    labs = wz.segment(loaded, [0.3], threshold_mode="affinity")
    zarr.save_array(str(lab_path), labs[0])
    print("wrote labels", labs[0].shape, labs[0].dtype, "->", lab_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
