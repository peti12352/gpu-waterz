# gpu-waterz

GPU affinity-flow watershed + contact-mean agglomeration matching stock `waterz` quality.

**Not a 2 Gvox/s number. Not a 3090 Ti number.** Develop/report card here is an idle RTX 5090; TASK grades a 3090 Ti. Do not substitute.

**Contract:** [TASK.md](TASK.md). **Campaign atlas:** [notes/ATLAS.md](notes/ATLAS.md). **VOI negatives:** [data/cache/voi_atlas.csv](data/cache/voi_atlas.csv). **Remaining problem:** [notes/PROBLEM.md](notes/PROBLEM.md). **Log:** [notes/LOG.md](notes/LOG.md). **Sources:** [papers/SOURCES.md](papers/SOURCES.md).

---

## What this problem is

Connectomics / EM segmentation pipelines predict 3D affinities on GPU (fast), then turn them into a segmentation with `waterz` on CPU. That CPU step is the bottleneck by orders of magnitude (stock waterz ~6.7 Mvox/s, single-threaded).

The deliverable is the **whole** affinity-to-labels path: watershed fragments **and** agglomeration on top. A bare over-segmentation is not enough. Output is final uint32 labels, graded against ground truth.

Reference scoring string (stock waterz): `OneMinus<MeanAffinity<...>>` -- contact-area-weighted mean affinity, exact min-heap merge order when `discretize_queue=0`. Thresholds in this API are **affinity**; waterz scores are `1 - aff` (see [notes/THRESHOLD.md](notes/THRESHOLD.md)).

## Why these constraints (and not others)

| Constraint | Meaning | Why reasonable |
|---|---|---|
| VOI split and merge within +0.02 of waterz at aff 0.2, 0.3, 0.4, 0.5 | Both halves of variation of information vs shipped CREMI-A GT | Partition quality; trading split against merge is not a pass |
| Contact-mean agglomeration (or any algo that still PASSes that VOI gate) | Same statistic as production waterz | Frozen-weight Kruskal / mutex / AbsMax are a **different** partition class; they fail this gate on this RAG (see atlas) |
| >= 2 Gvox/s e2e on one RTX 3090 Ti | Official `make_big` volume `[3,375,2400,2400]` = 2.16 Gvox @ T=0.3; median of 5; CUDA events; aff already in VRAM | One card, one volume, comparable timing; fits the graded hardware leftover budget |
| Fit in 24 GB | Aff + labels + scratch on 3090 Ti | Real deployment constraint; listing leftover math is not a substitute for measured stage peaks |
| Run-to-run byte-identical labels | Determinism | Stock waterz is not self-identical (plateau tie order); a GPU impl can be |

Out of scope (TASK): training the affinity net, meshing, multi-node stitch, CPU fallback as the product, claiming 5090 numbers as 3090 Ti grades.

## Best measured result (idle RTX 5090)

Legal stack **N17 / N19 I0_REPRO** (four-T PASS, parks off, CUDA events, aff in VRAM):

| stage | ms |
|---|---|
| WS | ~1309 |
| RAG | ~85 |
| agg | ~1680 |
| extract | ~18 |
| **e2e** | **~3093 (~0.70 Gvox/s)** |

Pin: `data/cache/N19_I0_REPRO.json` (within 2% of N18 A1 3104 ms). 2-run 8 GiB label sha identical. **Not** median-of-5 on a 3090 Ti. **Not** 2 Gvox/s. WS alone (~1310 ms) already exceeds the ~1080 ms TASK e2e budget.

Env floor (process-cached; C++ product defaults stay **0** until a real 3090 Ti peak is measured):

```
WATERZ_UF_ALGO=3
WATERZ_HOST_PARK=0
WATERZ_AFF_PARK=0
WATERZ_AGG_LEVERS=15
WATERZ_FOLD_FLATTEN=1
WATERZ_SHARE_OFF=1
WATERZ_HOOK_ROOT=1
WATERZ_FUSE_DIRTY=1
WATERZ_NLIVE_ARITH=1
WATERZ_EMIT_HOLES=1
```

Dual-eps (TASK-legal):

| Path | eps | Gate |
|---|---|---|
| four-T accuracy | **0.08** | VOI at aff 0.2/0.3/0.4/0.5 |
| T=0.3 speed | **0.40** | single-threshold VOI + timed 2.16 |

Override with `WATERZ_AGG_EPS`. eps in (0.40, 0.5) and eps >= 0.5 are **dead** (merge FAIL). See atlas.

---

## API

```
from segment import segment
labs = segment(aff, [0.2, 0.3, 0.4, 0.5])  # list of uint32 [Z,Y,X]
```

`aff` is uint8 or float32 `[3,Z,Y,X]`, numpy. uint8 scale is `/255` in fp32 at the point of use.

Device-resident path (what TASK times):

```
from segment import segment_d
labs = segment_d(aff_d, [0.3], return_device=True)  # DevBufs, still in VRAM
```

CLI:

```
python src/segment.py cremiA_val/affinity.h5 --out-dir . --thresholds 0.2 0.3 0.4 0.5
python baseline/run_baseline.py --candidate mine_thr0.2.h5 mine_thr0.3.h5 mine_thr0.4.h5 mine_thr0.5.h5
```

## One-command gates

```
# identity diagnostic + T=0.3 VOI 2-run + four-T eps=0.08 (no 2.16)
bash scripts/legal_eval.sh

# same + official 2.16 timing if make_big volume exists (still not 3090 Ti)
bash scripts/legal_eval.sh --216

# full segment -> shipped grader -> det -> val bench
bash scripts/eval.sh
```

`legal_eval.sh` refuses if the GPU is busy (`card_busy`). Parks stay off.

Dataset: TASK Drive link / shipped 532 MB tarball (not in this repo; `data/` is gitignored except thin atlas pins).

## Algorithm

Same statistic as waterz `OneMinus<MeanAffinity>` (`funkey/waterz` `a0184d2`):

1. **Flow (GPU).** 6-neighbour affinities, OOB=`low`, bits where `aff==m || aff>=high`. Background iff `m<=low`.
2. **Plateau + basins (GPU).** Corner BFS rewrite, then basin / list-UF (W5). Dir order `(-z,-y,-x,+z,+y,+x)`. Deterministic min-index roots. Extra closed-plateau CCs fail VOI -- S1 plateau semantics are load-bearing.
3. **RAG (GPU atomic hash).** Three negative dirs, drop bg. `key=(min<<32)|max`, `atomicAdd` on `(sum,count)` of raw uint8 bytes; mean=`sum/count`.
4. **Agglomeration (GPU paper-ParHAC).** Matching of (1+eps)-heavy waterz-mean edges, S3-contract. Dual-path above. Rejected classes: frozen CC, mutex/AbsMax, Kruskal SDSL, union-all-in-band, NNG filter, GASP Average, RAMA -- see [voi_atlas.csv](data/cache/voi_atlas.csv).

## Where it diverges from waterz

- Flow bits on GPU; plateau/basin match vendored C++ semantics with fixed dir/index ties (not waterz enqueue-order jitter). Two `segment()` / `segment_d()` calls are byte-identical.
- Fragment IDs need not match waterz. The partition is graded.
- Agglomeration is (1+eps) ParHAC, not the serial S4 heap. eps=0.08 four-T / eps=0.40 T=0.3. Historical Y2 lock was eps=0.01 (also PASS, slower). RAC also PASSes and is too serial.

## Accuracy (CREMI-A val, shipped grader)

`ACCURACY GATE: PASS` at aff 0.2/0.3/0.4/0.5 (N16 deep four-T eps=0.08; N17 partition unchanged):

| aff | VOI split | limit | VOI merge | limit |
|-----|-----------|-------|-----------|-------|
| 0.2 | 0.3707 | 0.3979 | 0.3350 | 0.3525 |
| 0.3 | 0.4512 | 0.4738 | 0.2505 | 0.2611 |
| 0.4 | 0.5162 | 0.5378 | 0.2268 | 0.2381 |
| 0.5 | 0.6129 | 0.6309 | 0.2184 | 0.2293 |

Pins: [notes/N16_DEEP.md](notes/N16_DEEP.md), `data/cache/N19_I0_REPRO.json`. Waterz commit: `a0184d2`. Val: `[3,125,1200,1200]`. Bench: `make_big.py` 3x2x2 -> `[3,375,2400,2400]`.

## Memory

Stage peaks are maxima, not sums. Fused 2.16 WS without slabs was ~42 GiB. Legal path uses **z-slabs** N=3 (Z=125, aff=0 seams). Measured WS peak on legal stack: **13.22 GiB** ([notes/N17_DEEP.md](notes/N17_DEEP.md)).

`WATERZ_SHARE_OFF=1` is +4 B/vox. **C++ product defaults stay 0** until a 3090 Ti 24 GB peak is measured (`scripts/n19_3090_grade.py` refuses non-3090).

## Speed

**No graded speed number.** TASK requires median-of-5 >= 2 Gvox/s on a **3090 Ti**. Best idle-5090 number here is ~0.70 Gvox/s. Co-tenant VRAM invalidates timing; `card_busy` / `scripts/n18_free_gpu.sh` refuse or clear before 2.16.

Owners (nsys `--force-export`, [notes/N19_I0_NSYS.md](notes/N19_I0_NSYS.md)): `k_w5_compress_list` ~508 ms; ParHAC hash_rewrite + rebuild_active + rewrite_dirty_fuse ~838 ms. Honest ceiling and ruled-out attacks: [notes/PROBLEM.md](notes/PROBLEM.md).

### G9 (3090 Ti) - not run here

```
python src/segment.py big/affinity.h5 --out-dir /tmp/wz --thresholds 0.3
python scripts/n19_3090_grade.py   # refuses unless nvidia-smi name is RTX 3090 Ti
nvidia-smi --query-gpu=name,driver_version --format=csv
```

Report median Gvox/s only from that card.

## Known failure modes

- Partition-class agglomerators (mutex, Kruskal, frozen CC, GASP Average, NNG, ...) fail VOI in different ways -- cite the atlas, do not re-tune once.
- Extra closed-plateau CCs (nfrag +1.6%) fail merge VOI; not dust.
- HOST_PARK / AFF_PARK on the speed path book PCIe into stage times (N8 false stop; N10). Parks stay off.
- Empty nsys without `--force-export` (fixed in N19 I0).

Dev: greengoblin RTX 5090, driver 580, nvcc 12.8, `sm_120`. Graded card: RTX 3090 Ti (**not here**).
