"""Published CREMI-A val VOI limits for the shipped contact-mean stack.

Split and merge must each stay within +0.02 of stock waterz at every
affinity threshold. Fragment IDs need not match waterz; the partition is
what is graded.
"""
from __future__ import annotations

AFF_THRESHOLDS = (0.2, 0.3, 0.4, 0.5)
BASE_SPLIT = {0.2: 0.3779, 0.3: 0.4538, 0.4: 0.5178, 0.5: 0.6109}
BASE_MERGE = {0.2: 0.3325, 0.3: 0.2411, 0.4: 0.2181, 0.5: 0.2093}
SLACK = 0.02
FRAGMENTS_VAL = 2_175_400
EDGES_VAL = 7_505_458
AFF_LOW = 1e-4
AFF_HIGH = 0.9999
BG_VAL = 506_568


def voi_limits() -> dict[float, tuple[float, float]]:
    """Map affinity threshold -> (max split, max merge)."""
    return {t: (BASE_SPLIT[t] + SLACK, BASE_MERGE[t] + SLACK) for t in AFF_THRESHOLDS}


def grade_voi(split: float, merge: float, threshold: float) -> tuple[bool, float, float]:
    sl, ml = voi_limits()[float(threshold)]
    return split <= sl and merge <= ml, sl, ml
