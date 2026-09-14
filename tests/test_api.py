"""Unit tests that do not require CUDA .so files."""
from __future__ import annotations

import numpy as np
import pytest

import gpu_waterz as wz


def test_scores_to_affinity():
    assert wz.scores_to_affinity([0.7]) == [pytest.approx(0.3)]
    assert wz.affinity_to_scores([0.2, 0.5]) == [pytest.approx(0.8), pytest.approx(0.5)]


def test_threshold_mode_score():
    assert wz.resolve_thresholds([0.7], "score") == [pytest.approx(0.3)]
    assert wz.resolve_thresholds([0.3], "affinity") == [pytest.approx(0.3)]


def test_invalid_threshold_mode():
    with pytest.raises(ValueError, match="threshold_mode"):
        wz.resolve_thresholds([0.3], "nope")


def test_missing_lib_message(monkeypatch):
    monkeypatch.setattr(wz, "missing_cuda_libs", lambda: ["libws_gpu.so"])
    monkeypatch.setattr("gpu_waterz.api.missing_cuda_libs", lambda: ["libws_gpu.so"])
    with pytest.raises(RuntimeError, match="build_cuda.sh"):
        wz.require_cuda_libs()
    aff = np.zeros((3, 4, 4, 4), np.float32)
    with pytest.raises(RuntimeError, match="build_cuda.sh"):
        wz.segment(aff, [0.3])


def test_import_surface():
    for name in (
        "segment", "agglomerate", "segment_d", "fragments", "region_graph",
        "labels_from_fragments",
        "scores_to_affinity", "from_torch", "to_torch", "cuda_libs_ready",
        "resolve_thresholds", "require_cuda_libs", "voi_limits", "grade_voi",
        "AFF_THRESHOLDS",
    ):
        assert hasattr(wz, name)


def test_bad_aff_shape_fragments():
    aff = np.zeros((2, 4, 4, 4), np.float32)
    with pytest.raises(ValueError, match=r"\[3,Z,Y,X\]"):
        wz.fragments(aff)


@pytest.mark.skipif(not wz.cuda_libs_ready(), reason="CUDA libs not built")
def test_segment_synthetic_shape():
    rng = np.random.default_rng(1)
    aff = rng.random((3, 8, 16, 16), dtype=np.float32)
    a = wz.segment(aff, [0.3])
    b = wz.segment(aff, [0.3])
    assert len(a) == 1
    assert a[0].shape == (8, 16, 16)
    assert a[0].dtype == np.uint32
    assert np.array_equal(a[0], b[0])
    c = wz.agglomerate(aff, [0.3])
    assert np.array_equal(a[0], c[0])
    d = wz.segment(aff, [0.7], threshold_mode="score")
    assert np.array_equal(a[0], d[0])
    fr = wz.fragments(aff)
    from_fr = wz.labels_from_fragments(aff, fr, [0.3])
    assert np.array_equal(a[0], from_fr[0])
