"""gpu_waterz: CUDA mean-affinity agglomeration with a waterz-shaped API."""
from __future__ import annotations

from gpu_waterz.api import (
    DevBuf,
    affinity_to_scores,
    agglomerate,
    cuda_libs_ready,
    fragments,
    from_torch,
    labels_from_fragments,
    missing_cuda_libs,
    region_graph,
    require_cuda_libs,
    resolve_thresholds,
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
    "labels_from_fragments",
    "missing_cuda_libs",
    "region_graph",
    "require_cuda_libs",
    "resolve_thresholds",
    "scores_to_affinity",
    "segment",
    "segment_d",
    "to_torch",
]

__version__ = "0.1.0"
