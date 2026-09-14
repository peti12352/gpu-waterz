"""Thin public API wrappers around ``src/segment.py``.

Naming:

- ``segment`` / ``agglomerate``: end-to-end affinities -> labels (stock
  ``waterz.agglomerate`` role). Returns a list, not a generator.
- ``labels_from_fragments``: LSD agglomerate-worker role (merge existing
  fragments with contact-mean ParHAC).
- ``fragments`` / ``region_graph``: LSD/daisy-style stage hooks.
- Thresholds default to **affinity** (merge while mean_aff > thr). Stock
  waterz default scoring is ``OneMinus<MeanAffinity>`` so its thresholds are
  scores; pass ``threshold_mode="score"`` or use ``scores_to_affinity``.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from gpu_waterz._backend import REQUIRED_SOS, ROOT, seg


DevBuf = seg.DevBuf

THRESHOLD_MODES = ("affinity", "score")


def cuda_libs_ready() -> bool:
    """True when the shared libraries needed by the GPU path are present."""
    return all(p.is_file() for p in REQUIRED_SOS)


def missing_cuda_libs() -> list[str]:
    return [p.name for p in REQUIRED_SOS if not p.is_file()]


def require_cuda_libs() -> None:
    """Raise with a build hint instead of a ctypes crash."""
    missing = missing_cuda_libs()
    if missing:
        raise RuntimeError(
            "CUDA libraries missing: "
            + ", ".join(missing)
            + ". Run: bash scripts/build_cuda.sh"
        )


def scores_to_affinity(scores: Sequence[float]) -> list[float]:
    """Map stock waterz heap scores to this API's affinity thresholds.

    With default ``OneMinus<MeanAffinity>``, score ~= 1 - mean_affinity.
    """
    return [float(1.0 - float(s)) for s in scores]


def affinity_to_scores(thresholds: Sequence[float]) -> list[float]:
    """Inverse of ``scores_to_affinity``."""
    return [1.0 - float(t) for t in thresholds]


def resolve_thresholds(
    thresholds: Sequence[float],
    threshold_mode: str = "affinity",
) -> list[float]:
    """Return affinity thresholds. Never infers units from magnitude."""
    mode = str(threshold_mode)
    if mode not in THRESHOLD_MODES:
        raise ValueError(
            f"threshold_mode must be 'affinity' or 'score', got {threshold_mode!r}"
        )
    vals = [float(t) for t in thresholds]
    if mode == "score":
        return scores_to_affinity(vals)
    return vals


def _is_cuda_torch(aff: Any) -> bool:
    return bool(hasattr(aff, "is_cuda") and aff.is_cuda)


def segment(
    aff: Any,
    thresholds: Sequence[float],
    *,
    aff_low: float = 1e-4,
    aff_high: float = 0.9999,
    eps: float | None = None,
    return_device: bool = False,
    threshold_mode: str = "affinity",
) -> list[Any]:
    """End-to-end affinity -> labels.

    Thresholds are affinity unless ``threshold_mode="score"``. ``eps`` sets
    ParHAC (1+eps). None uses ``WATERZ_AGG_EPS`` or dual-eps defaults
    (0.08 multi-T / 0.40 single T=0.3). Host numpy and CUDA CAI inputs are
    accepted. CUDA torch input with ``return_device=True`` returns torch
    tensors; otherwise DevBuf when ``return_device=True``.
    """
    aff_thr = resolve_thresholds(thresholds, threshold_mode)
    require_cuda_libs()
    use_dev = return_device or getattr(aff, "__cuda_array_interface__", None) is not None
    if use_dev:
        out = seg.segment_d(
            aff, aff_thr, aff_low=aff_low, aff_high=aff_high,
            return_device=return_device, eps=eps,
        )
        if return_device and _is_cuda_torch(aff):
            return [to_torch(b) for b in out]
        return out
    return seg.segment(
        aff, aff_thr, aff_low=aff_low, aff_high=aff_high, eps=eps,
    )


def labels_from_fragments(
    aff: Any,
    frag: Any,
    thresholds: Sequence[float],
    *,
    eps: float | None = None,
    threshold_mode: str = "affinity",
) -> list[np.ndarray]:
    """Merge existing fragments with contact-mean ParHAC (no watershed).

    LSD agglomerate-worker shape: affinities + fragment volume in, labels out.
    GPU RAG + ParHAC + extract; numpy in/out (not ``segment_d``). Still the
    waterz mean statistic, not mutex/Kruskal. Four-T VOI numbers in the docs
    are for *our* watershed fragments on CREMI-A val, not this entry.
    """
    aff_thr = resolve_thresholds(thresholds, threshold_mode)
    require_cuda_libs()
    frag = np.ascontiguousarray(frag, dtype=np.uint32)
    return seg.segment_from_fragments(aff, frag, aff_thr, eps=eps)


def agglomerate(
    aff: Any,
    thresholds: Sequence[float],
    **kwargs: Any,
) -> list[Any]:
    """Alias of ``segment`` for stock-waterz name familiarity.

    Unlike ``waterz.agglomerate``, this returns a list of copied/owned label
    volumes (not an in-place generator). Default thresholds are affinity;
    pass ``threshold_mode="score"`` for stock heap scores.
    """
    return segment(aff, thresholds, **kwargs)


def segment_d(
    aff: Any,
    thresholds: Sequence[float],
    aff_low: float = 1e-4,
    aff_high: float = 0.9999,
    return_device: bool = False,
    eps: float | None = None,
    threshold_mode: str = "affinity",
) -> list[Any]:
    """Device-resident end-to-end path (see ``src.segment.segment_d``)."""
    aff_thr = resolve_thresholds(thresholds, threshold_mode)
    require_cuda_libs()
    out = seg.segment_d(
        aff, aff_thr, aff_low=aff_low, aff_high=aff_high,
        return_device=return_device, eps=eps,
    )
    if return_device and _is_cuda_torch(aff):
        return [to_torch(b) for b in out]
    return out


def fragments(
    aff: Any,
    *,
    aff_low: float = 1e-4,
    aff_high: float = 0.9999,
) -> np.ndarray:
    """Watershed fragments only (uint32 [Z,Y,X]). LSD fragment-worker shape."""
    aff_u8 = seg._as_u8(aff)
    if aff_u8.ndim != 4 or aff_u8.shape[0] != 3:
        raise ValueError(f"aff must be [3,Z,Y,X], got {aff_u8.shape}")
    require_cuda_libs()
    return seg._watershed(aff_u8, aff_low, aff_high)


def region_graph(
    aff: Any,
    frag: np.ndarray,
) -> dict[str, np.ndarray]:
    """Contact-mean RAG from affinities + fragments.

    Returns dict with ``u``, ``v`` (uint32 endpoints), ``sum`` (float64),
    ``count`` (int64), and ``mean`` (float64 = sum/count).
    """
    aff_u8 = seg._as_u8(aff)
    if aff_u8.ndim != 4 or aff_u8.shape[0] != 3:
        raise ValueError(f"aff must be [3,Z,Y,X], got {aff_u8.shape}")
    frag = np.ascontiguousarray(frag, dtype=np.uint32)
    if frag.shape != aff_u8.shape[1:]:
        raise ValueError(
            f"frag shape {frag.shape} != aff spatial {aff_u8.shape[1:]}")
    require_cuda_libs()
    u, v, sm, ct = seg._rag(aff_u8, frag)
    mean = sm / np.maximum(ct.astype(np.float64), 1.0)
    return {"u": u, "v": v, "sum": sm, "count": ct, "mean": mean}


def from_torch(t: Any) -> Any:
    """Accept a torch tensor for the CUDA path (CAI / host fallback).

    CUDA tensors are passed through (``segment_d`` uses CAI). CPU tensors are
    converted to contiguous numpy.
    """
    if hasattr(t, "is_cuda") and bool(t.is_cuda):
        return t.contiguous()
    if hasattr(t, "detach"):
        return np.ascontiguousarray(t.detach().cpu().numpy())
    return t


def to_torch(buf: Any, device: str | None = None) -> Any:
    """Wrap labels as a torch tensor. CAI stays on CUDA (do not CPU-copy).

    ``torch.as_tensor`` on a CUDA Array Interface object can land on CPU
    (pytorch issue 54139). Always pass device=cuda for CAI unless overridden.
    """
    import torch

    if hasattr(buf, "__cuda_array_interface__"):
        dev = device if device is not None else "cuda"
        return torch.asarray(buf, device=dev)
    arr = np.ascontiguousarray(buf)
    if device is None:
        return torch.as_tensor(arr)
    return torch.as_tensor(arr, device=device)


def repo_root():
    """Absolute path to the repository root (for build scripts / docs)."""
    return ROOT
