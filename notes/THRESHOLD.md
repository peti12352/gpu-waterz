# Threshold unit: locked from shipped script

API thresholds are affinity; see [docs/usage.md](../docs/usage.md). This
file is the lock against the shipped baseline script.

Source: `data/ws_bounty/baseline/run_baseline.py` (read 2026-08-30 on greengoblin).

```
GATE_THRESHOLDS = [0.2, 0.3, 0.4, 0.5]
ALL_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
# reproduce():
scores = sorted(1.0 - t for t in ALL_THRESHOLDS)
# then waterz.agglomerate(affs, scores, gt=gt)
# aff = 1.0 - s
```

**Hypothesis PLAN §4 is FACT.** waterz is called with **scores**:

| aff_thr (our API / TASK.md / filename) | waterz `thresholds` (score) |
|---|---|
| 0.2 | 0.8 |
| 0.3 | 0.7 |
| 0.4 | 0.6 |
| 0.5 | 0.5 |

Merge while `score < waterz_threshold`, i.e. while `mean_affinity > aff_thr`.

`voi.csv` columns: `aff_threshold,waterz_score_threshold,...` agree.
