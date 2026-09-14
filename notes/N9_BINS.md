# N9 Track D: MEAN + BinQueue T=0.3

Not a 2 Gvox/s number. Stock `OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>`.
Serial FIFO (`BinQueue`). Not hist-q, not X1. Fresh `wz_fragments.npy` copy per N
(waterz mutates the buffer). Limits: split <= 0.4738, merge <= 0.2611.

Each N: 2175401 nodes, 7505458 edges, merged 1853086, min score 0.00392157.

| N | voi_split | voi_merge | nseg | sec | gate |
|---|---|---|---|---|---|
| 256 | 0.455129 | 0.241600 | 322314 | 25.8 | PASS |
| 1024 | 0.455129 | 0.241600 | 322314 | 25.3 | PASS |
| 4096 | 0.455129 | 0.241600 | 322314 | 25.0 | PASS |

`voi.csv` T=0.3 is split 0.45379 / merge 0.24106 / nseg 322549. Same VOI and nseg
across N: at T=0.3 the binned FIFO did not change the scored partition between
256 and 4096 bins.

**No GPU bucket.** PASS is an order approximation of the exact heap, not a 12x
device path. No propose-visit count was taken on this run, so the >=5x visit
gate is unmet. N7 already showed E2 visits != wall. `gpu_bucket=false`.
