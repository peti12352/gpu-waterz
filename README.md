# gpu-waterz

GPU implementation of affinity-flow watershed fragments plus contact-mean
agglomeration, matching stock [`waterz`](https://github.com/funkey/waterz)
segmentation quality on CREMI-A affinities.

**Docs:** [notes/ATLAS.md](notes/ATLAS.md) (campaign report) |
[data/cache/voi_atlas.csv](data/cache/voi_atlas.csv) (algorithm-class VOI table) |
[notes/PROBLEM.md](notes/PROBLEM.md) (open questions) |
[papers/SOURCES.md](papers/SOURCES.md) (pinned literature) |
[notes/LOG.md](notes/LOG.md) (lab notebook).
Evaluation protocol used in this tree: [TASK.md](TASK.md).

---

## Motivation

In connectomics, a CNN predicts 3D affinities on GPU. Turning those affinities
into a segmentation is still often done on CPU with `waterz`. That step is
slow relative to inference: stock waterz is about 6.7 Mvox/s single-threaded
on this workload.

This project implements the full affinity-to-labels pipeline on GPU:

1. affinity-flow watershed fragments (same semantics as waterz S1)
2. region adjacency graph with contact-mean affinities
3. hierarchical agglomeration under the waterz mean statistic
4. final uint32 labels (background 0)

A fragment over-segmentation alone is not the goal. The graded object is the
**partition after agglomeration**.

Reference scoring string: `OneMinus<MeanAffinity<...>>` (contact-area-weighted
mean). API thresholds are affinity; waterz heap scores are `1 - aff`
([notes/THRESHOLD.md](notes/THRESHOLD.md)).

## Why contact-mean (and what fails instead)

Mean affinity updates after every merge. That is not Kruskal on frozen weights,
not mutex / AbsMax, and not single-linkage MST. On the CREMI-A contact-mean RAG
those other classes produce systematically different partitions (giant
components, under-merge, or both). The measured VOI atlas is in
[data/cache/voi_atlas.csv](data/cache/voi_atlas.csv).

Exact average-linkage HAC is P-complete / CC-hard in the literature (ParHAC
2022; Abboud et al. ICALP 2024). Practical GPU work therefore uses a
**(1+eps)-approximate** matching schedule on the same contact-mean statistic,
and checks VOI rather than bit-identity with the serial heap.

## Quality bar used here

On shipped CREMI-A val affinities `[3,125,1200,1200]` with GT:

- VOI split and VOI merge each within +0.02 of stock waterz at affinity
  thresholds **0.2, 0.3, 0.4, 0.5** (both halves, every threshold)
- Run-to-run byte-identical labels (stock waterz is not self-identical;
  plateau tie-breaking can be fixed)

That bar is about **partition quality**, not about matching waterz label IDs.

## Algorithm (current stack)

Vendored waterz pin: `funkey/waterz` `a0184d2`.

1. **Flow (GPU).** 6-neighbour affinities, OOB = `low`, bits where
   `aff == m || aff >= high`. Background iff `m <= low`.
2. **Plateau + basins (GPU).** Corner BFS rewrite, then list union-find.
   Dir order `(-z, -y, -x, +z, +y, +x)`, min-index roots. Extra closed-plateau
   components fail VOI; S1 plateau semantics are load-bearing.
3. **RAG (GPU).** Three negative dirs, drop background. Atomic hash of
   `(sum, count)` on raw uint8 affinity bytes; mean = `sum/count`.
4. **Agglomeration (GPU).** Paper-style ParHAC matching on (1+eps)-heavy
   waterz-mean edges, S3 contract. Dual-eps schedule:
   - four-threshold VOI path: `eps = 0.08`
   - single-threshold (aff 0.3) path: `eps = 0.40`
   - larger eps on the speed path fails merge VOI (atlas / N18 B2)

Large volumes use z-slab decomposition (aff=0 seams on the mirror-tiled
benchmark volume). Host parking of corners/affinity on the timed path is
off: those copies were PCIe, not watershed work.

### Recommended env

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

`WATERZ_AGG_EPS` overrides the default eps for a run.

## Measured performance

Idle RTX 5090, CUDA events, affinity already in VRAM, official mirror-tiled
volume `[3,375,2400,2400]` = 2.16 Gvox at affinity 0.3
(`data/cache/N19_I0_REPRO.json`):

| stage | ms |
|---|---|
| watershed | ~1309 |
| RAG | ~85 |
| agglomeration | ~1680 |
| extract | ~18 |
| end-to-end | ~3093 (~0.70 Gvox/s) |

Four-threshold VOI: PASS. Two full label volumes: byte-identical.
Watershed device peak on this stack: ~13.2 GiB ([notes/N17_DEEP.md](notes/N17_DEEP.md)).

Dominant kernels (nsys force-export, [notes/N19_I0_NSYS.md](notes/N19_I0_NSYS.md)):
`k_w5_compress_list` ~508 ms; ParHAC hash rewrite / rebuild / dirty-fuse
together ~838 ms.

Throughput numbers are card- and contention-dependent. Refuse timing if
another process holds the GPU (`card_busy`).

## Accuracy table (CREMI-A val)

| aff | VOI split | limit | VOI merge | limit |
|-----|-----------|-------|-----------|-------|
| 0.2 | 0.3707 | 0.3979 | 0.3350 | 0.3525 |
| 0.3 | 0.4512 | 0.4738 | 0.2505 | 0.2611 |
| 0.4 | 0.5162 | 0.5378 | 0.2268 | 0.2381 |
| 0.5 | 0.6129 | 0.6309 | 0.2184 | 0.2293 |

## API

```
from segment import segment
labs = segment(aff, [0.2, 0.3, 0.4, 0.5])  # list of uint32 [Z,Y,X]
```

`aff`: uint8 or float32 `[3,Z,Y,X]`. Scale uint8 by `/255` in fp32.

Device-resident path:

```
from segment import segment_d
labs = segment_d(aff_d, [0.3], return_device=True)
```

```
python src/segment.py cremiA_val/affinity.h5 --out-dir . --thresholds 0.2 0.3 0.4 0.5
python baseline/run_baseline.py --candidate mine_thr0.2.h5 mine_thr0.3.h5 mine_thr0.4.h5 mine_thr0.5.h5
bash scripts/legal_eval.sh          # VOI + four-T + identity diagnostic
bash scripts/legal_eval.sh --216    # also time the 2.16 Gvox volume if present
```

CREMI tarball / large caches are not in git (`data/` ignored except thin pins).

## Findings worth reading first

1. **Mean affinity is not Kruskal.** Partition-class swaps fail VOI in
   structurally different ways (atlas).
2. **(1+eps) contact-mean ParHAC** is the parallel class that cleared the
   four-threshold VOI gate here; eps has a sharp empirical boundary near 0.40
   on the aff-0.3 path.
3. **Timing must exclude host parking.** D2H of affinity or hundreds of
   millions of corner indices inside the timed window is not a watershed
   result (N8/N10).
4. **Plateau semantics matter more than BFS wall time.** Extra closed-plateau
   CCs fail VOI; BFS itself is a small slice of watershed time vs list
   compress / UF.

Open algorithmic questions: [notes/PROBLEM.md](notes/PROBLEM.md).

## Where it diverges from stock waterz

- Deterministic plateau/basin ties (fixed dir + min index), not enqueue order.
- Fragment IDs need not match waterz; the partition is what is graded.
- Agglomeration is (1+eps) ParHAC, not the serial exact heap. Exact RAC and
  Funke quantile+BinQueue also pass VOI and remain serial-shaped.

## Known failure modes

- Mutex / frozen CC / Kruskal / FH / SRM / Soille / GASP Average / NNG: see atlas.
- Raising eps past the measured boundary fails merge VOI.
- Co-tenant GPU use makes e2e times meaningless.

Dev measurements above: RTX 5090, driver 580, nvcc 12.8, `sm_120`.
