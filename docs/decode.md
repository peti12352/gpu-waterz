# Decode: affinity graph to objects

Published VOI numbers are this algorithm on the shipped CREMI-A val
volume, timed on an idle RTX 5090. Call sites and stages:
[usage.md](usage.md). Other GPUs: [porting.md](porting.md). Papers:
[citations.md](citations.md).

## In a brain segmentation pipeline

```
EM volume
  -> CNN affinities  [3,Z,Y,X]   same neuron as +z / +y / +x neighbor?
  -> gpu_waterz.segment(...)     fragments, then mean merge
  -> uint32 labels               one ID per object, 0 = background
  -> optional proofreading / ChunkedGraph / synapses / skeletons
```

Electron microscopy of neuropil is a packed 3D volume. A CNN emits three
affinities per voxel in `[0, 1]`. High means stay together; low means
membrane. Funke et al. (MALA / waterz) made that the usual decode in
connectomics. The network already runs on GPU. Turning affinities into
labels is still often a CPU `waterz` call inside an LSD/daisy worker.

Waterz **cannot unmerge**. Watershed must almost never cross a true
membrane: one axon in many pieces is recoverable; two axons glued is a
connectome error you do not get back. Extra closed-plateau components
fail VOI.

Fragments are supervoxels. The object you count synapses on or load into a
proofreading graph is **after** mean merge. `fragments()` is a stage hook;
the quality bar is `segment()`.

## CREMI-A val (the files we grade)

Quality is `cremiA_val/{affinity,gt}.h5` from the
[dataset tarball](https://drive.google.com/file/d/1zbGpyr9M5Pvhgfy96V9erQwAeRZo23hW/view?usp=drive_link).
Shapes: affinity uint8 `[3,125,1200,1200]`, GT uint32 `[125,1200,1200]`.
The tarball README is the provenance we use: EM + GT are CREMI sample A,
crop z 100:225 of the training volume, xy 0:1200; affinities from released
CAD `CremiA.ckpt` (Liu et al., CVPR 2024). The gate is VOI vs stock waterz
on those arrays. The 3.1 s figure is `scripts/make_big.py` default: a 3x2x2
mirror-tile of that affinity (`[3,375,2400,2400]` = 2.16 Gvox).

## Over-segment, then merge

Thresholding the affinity graph ("cut every edge below 0.3, take connected
components") is brittle. Thin errors punch holes through a neurite or glue
two axons. Waterz grows conservative pieces, then glues them.

**Watershed.** Each voxel flows toward the locally strongest of its six
neighbors. Below `aff_low` you are background; above `aff_high` you are
definitely connected. A plateau (several voxels sharing the same max) is
**one** basin, not one basin per pixel. That plateau rule is load-bearing:
a GPU shortcut that emits extra closed-plateau CCs fails VOI. The BFS
that rewrites plateaus is cheap (~14 ms on 2.16 Gvox). The expensive part
is compressing the union-find (`k_w5_compress_list`, ~508 ms). Speeding up
BFS does not speed up this watershed.

**Agglomeration.** Fragments become nodes of a region adjacency graph
(RAG). An edge exists when two fragments touch. The weight is the
contact-area-weighted **mean** of the affinities on the shared face
(`sum/count`). Merge while that mean stays above T. On CREMI-A, T = 0.3 is
a reasonable operating point; production often keeps several cuts of the
same dendrogram (0.2, 0.3, 0.4, 0.5).

## Contact-mean is not Kruskal

When A and B merge into C, every neighbor of A or B now touches C. The mean
on C-D is not `min(mean(A,D), mean(B,D))`. It is
`(sum_AD + sum_BD) / (count_AD + count_BD)`: average linkage on the contact
graph, recomputed after every glue.

Kruskal, mutex, single-linkage, and GASP AbsMax treat the original
voxel-face affinities as frozen. Mean agglomeration re-asks: now that these
two lumps are one object, how strongly does the combined lump stick to its
neighbors? A weak sliver can be diluted by a large strong contact; a large
weak contact can drag a previously strong pair below T.

Same cached CREMI-A RAG, same VOI grader (`data/cache/voi_atlas.csv`):

- Frozen connected components: giant object. Merge VOI around 7.5.
- Mutex / GASP AbsMax: under-merge. Split VOI around 0.9-2.1.
- Kruskal SDSL: essentially no merges at the cuts we grade.

## Thresholds: affinity vs score

Waterz's default C++ scoring is `OneMinus<MeanAffinity>`. The heap pops on
score `~ 1 - mean_affinity`. Merge while `score < T_score`, i.e. while
`mean_affinity > T_aff`. If 0.3 is a reasonable membrane, you want
`T_aff = 0.3` (`T_score = 0.7`). This API is affinity unless you pass
`threshold_mode="score"`. Mixing them cuts the dendrogram at the wrong
height.

## VOI

Variation of Information:

- **split** `H(seg|gt)`: one true neuron, many labels (still chopped)
- **merge** `H(gt|seg)`: two true neurons, one label (fused)

Lower is better. Fusion is the expensive connectome mistake. The CREMI-A
gate requires **both** numbers, at all four affinity cuts, within +0.02 of
stock waterz on the whole val block `[3,125,1200,1200]`. Trading split
against merge is a fail. +0.02 is about 1000x waterz's own run-to-run
jitter (~1.7e-05).

IDs need not match waterz. Waterz is not even self-identical on plateaus.
This code is run-to-run byte-identical because we control ties.

Metric: `scripts/voi_numpy.py`, same formula as `waterz.evaluate`
(skip `gt == 0`, predicted 0 is a label, bits). `scripts/eval.sh` does
not import waterz. Set `WATERZ_USE_WATERZ_EVAL=1` to grade with
`waterz.evaluate` when that package is installed.

## Why ParHAC, and why two eps values

Exact average-linkage HAC has no friendly poly-log parallel algorithm
(ParHAC 2022; Bateni et al., [arXiv:2404.14730](https://arxiv.org/abs/2404.14730)).
The serial waterz heap is global min edge, merge, reweight, repeat. Mapping
that to CUDA as one pop per kernel keeps the serial chain.

Approximate the **order of merges**; keep the **mean statistic**. ParHAC
does (1+eps)-heavy matching: many disjoint merges in a round if they are
close enough to locally heaviest, then contract, repeat. Product defaults
on this RAG:

- Four cuts (0.2-0.5): `eps = 0.08`. 0.09 still passes four-T (aff 0.2
  merge 0.3441, limit 0.3525).
- Single cut at 0.3: `eps = 0.40`. Every 0.01 step from 0.41 to 0.49 fails
  merge VOI (N18 B2).

Lu, Zlateski, and Seung ([arXiv:2106.10795](https://arxiv.org/abs/2106.10795))
distribute exact mean clustering by freezing anything that touches a fake
chunk boundary. A freeze that is not their Algorithm 2 produces different
parents than ParHAC. We stopped.

[funkey/waterz PR 24](https://github.com/funkey/waterz/pull/24) took CPU RAG
52s -> 18s and agglomeration 71s -> 28s on 1024^3-class volumes. That is the
CPU class this GPU path replaces.

## What the CUDA path implements

The README claims, with the knobs:

- Watershed: affinity flow; a plateau (tied max) is one basin. Extra
  closed-plateau components fail VOI.
- Region graph: each edge stores integer affinity-byte sum and contact
  count. Float `atomicAdd` disagreed across runs on 7.4e5 of 7.5e6 val
  edges.
- Agglomeration: ParHAC (1+eps)-heavy matching, not one heap pop per
  kernel. Dual-eps: **0.08** for four cuts, **0.40** for a single T=0.3.
  Clustered-graph "small merge" rounds are false here (layer 0 is most of
  the merges). ParHAC's published code is CPU (CPAM), not CUDA.
- 2.16 Gvox fits by z-slab N=3, peak 13.22 GiB. Naive fused working set is
  ~42 GiB. See [porting.md](porting.md).
- Identity: two full runs, byte-identical labels. CREMI-A val
  nfrag=2175400, bg=506568. Gate: `bash scripts/eval.sh`.
  Pin: `data/cache/N19_I0_REPRO.json`.
- Public API does not yet have `min_size` or merge-from-a-precomputed-RAG.
