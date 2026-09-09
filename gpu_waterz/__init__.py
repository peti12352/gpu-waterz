"""gpu_waterz: CUDA mean-affinity agglomeration with a waterz-shaped API."""
from __future__ import annotations

from gpu_waterz.api import (
    DevBuf,
    affinity_to_scores,
    agglomerate,
    cuda_libs_ready,
    fragments,
    from_torch,
    region_graph,
    scores_to_affinity,
    segment,
    segment_d,
    to_torch,
)

__all__ = [
    "DevBuf",
    "affinity_to_scores",
    "agglomerate",
    "cuda_libs_ready",
    "fragments",
    "from_torch",
    "region_graph",
    "scores_to_affinity",
    "segment",
    "segment_d",
    "to_torch",
]

__version__ = "0.1.0"
