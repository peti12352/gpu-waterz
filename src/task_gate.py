"""TASK.md numbers. Do not soften."""
from __future__ import annotations

AFF_THRESHOLDS = (0.2, 0.3, 0.4, 0.5)
BASE_SPLIT = {0.2: 0.3779, 0.3: 0.4538, 0.4: 0.5178, 0.5: 0.6109}
BASE_MERGE = {0.2: 0.3325, 0.3: 0.2411, 0.4: 0.2181, 0.5: 0.2093}
SLACK = 0.02
FRAGMENTS_VAL = 2_175_400
EDGES_VAL = 7_505_458
AFF_LOW = 1e-4
AFF_HIGH = 0.9999
BG_VAL_MEASURED = 506_568


def limits():
    return {t: (BASE_SPLIT[t] + SLACK, BASE_MERGE[t] + SLACK) for t in AFF_THRESHOLDS}


def print_contract():
    print("TASK.md accuracy: voi_split <= base+0.02 AND voi_merge <= base+0.02 at every T")
    for t in AFF_THRESHOLDS:
        sl, ml = limits()[t]
        print(
            f"  aff {t}: split<= {sl:.4f} (base {BASE_SPLIT[t]})  "
            f"merge<= {ml:.4f} (base {BASE_MERGE[t]})"
        )
    print(f"TASK.md fragments@180Mvox={FRAGMENTS_VAL} edges={EDGES_VAL}")
