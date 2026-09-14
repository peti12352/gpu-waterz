# GPU watershed + agglomeration bounty: dataset & baseline

Everything here is hdf5, so no zarr/tensorstore needed.

```
cremiA_val/affinity.h5   ["affinity"] uint8  [3, 125, 1200, 1200]   z/y/x nearest-neighbour affinity, scale /255
cremiA_val/gt.h5         ["gt"]       uint32 [125, 1200, 1200]      CREMI-A ground-truth segmentation
cremiA_val/raw.h5        ["raw"]      uint8  [125, 1200, 1200]      EM image, visualization only
baseline/labels_thr<t>.h5["labels"]   uint32 [125, 1200, 1200]      stock-waterz output per threshold
baseline/voi.csv                                                    the numbers below
baseline/run_baseline.py                                            reproduces them / grades your labels
make_big.py                                                         builds the speed-benchmark volume locally
viz/                                                                renders of affinity, GT and baseline
```

## Provenance

* EM + GT: CREMI sample A, validation crop (z 100:225 of the training volume, xy 0:1200).
* `cremiA_val/affinity.h5`: predicted by the **released CAD checkpoint** (`CremiA.ckpt`,
  Liu et al., *Cross-Dimension Affinity Distillation for 3D EM Neuron Segmentation*, CVPR 2024),
  verified bit-exact against the published weights, run through our inference path and
  quantized to uint8.

## Baseline: stock waterz

`pip install waterz` (github.com/funkey/waterz), then `python baseline/run_baseline.py`.

Defaults, unchanged: scoring `OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>`,
`aff_threshold_low = 0.0001`, `aff_threshold_high = 0.9999`, `discretize_queue = 0`
(= an exact `std::priority_queue` min-heap, **not** the bucketed queue: merge order is exact).
`waterz.agglomerate()` does **both** stages: watershed fragments *and* the
mean-affinity agglomeration on top. The bounty target is that whole thing.

**Threshold convention.** waterz thresholds are *scores*, `score = 1 - mean_affinity`.
We quote **affinity** thresholds; `waterz_score_threshold = 1 - aff_threshold`. Both
columns are in `voi.csv`. Thresholds are passed ascending in score, so one pass yields
all of them.

### Numbers (CREMI-A val, 180 Mvox, this exact affinity)

| aff_thr | waterz score | VOI split | VOI merge | **VOI total** | #segments |
|--------:|-------------:|----------:|----------:|--------------:|----------:|
| 0.9 | 0.1 | 1.8581 | 0.1810 | 2.0390 | 824 626 |
| 0.7 | 0.3 | 1.1009 | 0.1933 | 1.2942 | 561 007 |
| 0.5 | 0.5 | 0.6109 | 0.2093 | 0.8202 | 380 709 |
| 0.4 | 0.6 | 0.5178 | 0.2181 | 0.7359 | 345 425 |
| **0.3** | **0.7** | **0.4538** | **0.2411** | **0.6949** | 322 549 |
| 0.2 | 0.8 | 0.3779 | 0.3325 | 0.7104 | 294 587 |
| 0.1 | 0.9 | 0.3081 | 0.6894 | 0.9975 | 241 027 |

Best total VOI = **0.6949 at affinity threshold 0.3**: matches the accuracy the CAD paper
reports for this checkpoint. The watershed alone produces **2 175 400 fragments** and a region
graph with **7 505 458 edges** on this volume.

Metric: `waterz.evaluate(labels.astype(np.uint64), gt.astype(np.uint64))` ->
`voi_split = H(seg|gt)` (over-segmentation) and `voi_merge = H(gt|seg)` (under-segmentation),
in **bits** (log2), lower is better. Voxels with `gt == 0` are excluded from the metric
(only 125 voxels here, immaterial); a *predicted* label 0 is treated as an ordinary label,
not as "unassigned".

### Run-to-run jitter: the reference is NOT deterministic

Stock waterz does not reproduce itself bit-for-bit. Two identical `agglomerate()` calls in
one process return different label volumes: raw watershed fragment counts vary by ~20 in
824 000, and VOI moves in the 5th decimal. Measured over 5 runs at affinity threshold 0.3:

```
voi_split  spread 1.7e-05      voi_merge  spread 1.5e-05      #segments  322545 (identical all 5)
```

At the ungraded aff_thr 0.9 (essentially the raw watershed) the spread is larger, ~5e-04.
Either way it is 3 orders of magnitude below the 0.02 gate, so it does not affect grading.
If `run_baseline.py` gives you numbers that differ from `voi.csv` in the 4th or 5th
decimal, that is expected, not a broken install.

Your implementation is still required to be deterministic (same input -> byte-identical
labels). You control your own tie-breaking; waterz's is an artefact, not a spec.

### Reference CPU cost

Stock waterz on this 180 Mvox volume, single threshold, whole pipeline: **26.9 s = 6.7 Mvox/s**
: watershed 3.8 s (direction bits 0.34 s, plateau BFS 2.1 s, basin labelling 1.0 s), region
graph ~14 s, merge loop 0.5-4 s depending on threshold, relabel ~1 s. Note this build is
effectively **single-threaded**: it links `-fopenmp` but contains no `#pragma omp` at all.

On the 2.16 Gvox benchmark volume, watershed + region graph alone take **145 s**
(watershed 45 s), producing **26 023 852 fragments and 90 323 139 region-graph edges**
: measured, not extrapolated. Peak host RAM for that run was ~60 GB.

## Exact semantics (verified against the waterz source, not the paper)

Read this before implementing: several details are easy to get wrong.

**Affinity indexing.** `aff[c][z,y,x]` is the affinity of the edge between voxel `(z,y,x)`
and its neighbour **one step back** along axis `c`; channel 0 pairs with z, 1 with y, 2
with x (`basic_watershed.hpp:58-63`). So each voxel sees **6** neighbour affinities: its
own three channels (-z, -y, -x) plus the three channels of its forward neighbours
(`aff[0][z+1][y][x]` for +z, etc.). Only 3 channels are stored; the neighbourhood is 6.

**Volume boundary.** Out-of-range affinities are substituted with `low`, not 0
(`negz = (z>0) ? aff[0][z][y][x] : low`). Since the flow test is `m > low` strict, a voxel
can never flow out of the volume.

**Direction bits.** If `m > low`, set a bit for every direction `d` with
`aff_d == m || aff_d >= high`. So `aff_threshold_high` adds directions that are *not* the
maximum. If `m <= low` no bits are set and the voxel ends up label 0 (background).

**Plateaus.** Voxels that mutually point at each other form plateaus. waterz seeds a BFS
from "plateau corners" (a voxel that points at a neighbour which does not point back),
walks inward, and rewrites each plateau voxel's bits to the single direction that leads off
the plateau. Getting this wrong is the usual source of VOI drift: arbitrary tie-breaking
changes fragment boundaries.

**Basins.** A serial scan follows flow chains; a chain that reaches an already-labelled
voxel inherits its ID, otherwise it becomes a new fragment. Fragment IDs start at 1;
0 is background.

**Region graph.** Built by scanning only the **three negative** directions per voxel
(`region_graph.hpp:53-67`): that covers every adjacent pair once. Edge weight = mean of
the affinities on the contact faces, accumulated as a running mean in fp32. Edges whose
smaller endpoint is 0 (i.e. fragment-to-background) are accumulated but then **dropped**:
the graph-building loop starts at `id1 = 1`. Background does not participate in
agglomeration.

**Agglomeration.** Pop the globally cheapest edge; stop when `score >= threshold`, i.e.
merge strictly while `score < threshold`, i.e. while `mean_affinity > affinity_threshold`
(strict). Merging `b` into `a`: neighbours exclusive to `b` have their edge moved to `a`;
neighbours shared with `a` have the two edges combined by contact-area-weighted mean into
the **cheaper** of the two, with the other deleted, and the survivor marked stale. A stale
edge popped later is rescored and re-pushed. Scores are monotone non-decreasing, which is
what makes the lazy staleness correct.

**Extraction.** Union-find chains are path-compressed once, then every voxel is relabelled
by a single table lookup.

**Multi-threshold.** Thresholds are processed in ascending score order in one pass; each
resumes where the previous stopped (`_mergedUntil`).

**Tolerance.** The +0.02 epsilon exists because exact tie-breaking (plateau BFS order,
equal-score edges, fp32 running means) is not reproducible on a parallel machine. Do not
try to be bit-identical to waterz; be statistically equivalent.

## Accuracy gate

At each of affinity thresholds **0.2 / 0.3 / 0.4 / 0.5**:

```
voi_split(yours) <= voi_split(baseline) + 0.02
voi_merge(yours) <= voi_merge(baseline) + 0.02
```

Both must hold. Trading split for merge is not a pass: the pair is graded, not the total.
No post-processing beyond a min-size filter.

Self-check:

```
python baseline/run_baseline.py --candidate mine_thr0.2.h5 mine_thr0.3.h5 \
                                mine_thr0.4.h5 mine_thr0.5.h5
# must print:  ACCURACY GATE: PASS
```

## Speed gate

**>= 2 Gvox/s, end-to-end, on a single NVIDIA RTX 3090 Ti** (24 GB, ~1008 GB/s).

* **The benchmark volume is exactly `[3, 375, 2400, 2400]` = 2.16 Gvox**, produced by
  `python make_big.py` with its default settings (see below). Not a crop of it, not a
  different tiling: that volume. At the target rate one run takes ~1.1 s, long enough
  that launch overhead and clock ramp don't distort the number.
* End-to-end = affinity already in VRAM -> final uint32 labels in VRAM, for one
  threshold (use 0.3). Host↔device transfer and disk I/O are excluded; everything
  else: watershed, region graph, agglomeration, relabel: is included.
* Report **median, min and max of 5 timed runs** after one warm-up run, timed with
  CUDA events, plus a per-stage breakdown (watershed / region graph / agglomeration /
  relabel). Median is what's graded.
* Also report the same measurement on the 1.44 Gvox volume
  (`make_big.py --factors 2 2 2`) so the scaling is visible. It is not graded.
* It fits: 2.16 Gvox is 6.5 GB of uint8 affinity plus 8.6 GB of uint32 labels = 15.1 GB,
  leaving ~9 GB of the 3090 Ti's 24 GB for fragment and region-graph structures. That volume carries
  **26.0 M fragments and 90.3 M region-graph edges** (measured with stock waterz). If your
  design needs more, chunk internally and say so: but the timing still covers the whole
  volume.
* Develop on any GPU; the reported number must come from a 3090 Ti. State the exact card
  and driver version.

## Building the speed volume

`cremiA_val/affinity.h5` is only 180 Mvox: too small to time reliably. `make_big.py`
mirror-tiles it. Default settings produce the graded benchmark volume:

```
python make_big.py               # 3x2x2 -> [3,375,2400,2400] = 2.16 Gvox, ~3 min, 3.05 GB on disk   <-- THE benchmark
python make_big.py --factors 2 2 2 --out big/affinity_1.44gvox.h5   # scaling reference
python make_big.py --factors 4 3 3 --out big/affinity_6gvox.h5      # 6.5 Gvox, forces chunking
python make_big.py --verify      # prove the mirroring is edge-correct
```

Affinity is an **edge** quantity: `affinity[c][z,y,x]` is the affinity between voxel
(z,y,x) and its neighbour one step back along axis c: so a plain `np.flip` produces a
volume whose edges are off by one. `make_big.py` flips every channel along the mirrored
axis *and* shifts that axis's own channel by one voxel, zeroing the seam plane. The
`--verify` mode proves this: mirrored crops yield **exactly** the same watershed fragment
count as the original (54936 vs 54936 on each of z, y, x), whereas naive `np.flip` is off
by 8-83 %.

Because tiles mirror rather than repeat, texture stays continuous across junctions and
fragment count / region-graph size scale roughly linearly with volume: a fair
throughput benchmark, not a degenerate one. No ground truth for this volume: it grades
speed only.

## Visualizations

`viz/` (from `make_viz.py`):
* `slice_z*.png`: full slice: raw EM | affinity RGB (R=x, G=y, B=z) | GT | baseline at every threshold
* `zoom_z*.png`: 500x500 crop, segmentation alpha-blended over EM
* `affinity_channels_z062.png`: the three channels separately; z is the weak, anisotropic one
* `section_xz_y600.png`: xz cross-section (z upsampled 6x) showing the 4x4x40 nm anisotropy
