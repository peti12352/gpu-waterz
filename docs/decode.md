# Decode: affinity graph to objects

Published VOI and speed numbers are this algorithm on CREMI-A, timed on an
idle RTX 5090. Change the clustering class, the dataset, or the VOI rule
and those numbers do not apply. The library still runs; that is a different
experiment.

Call sites and stages: [usage.md](usage.md). Other GPUs: [porting.md](porting.md).
Papers: [citations.md](citations.md).

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

This repo is that second step. It does not train the CNN, mesh, skeletonize,
or proofread.

Waterz **cannot unmerge**. Watershed must almost never cross a true
membrane: one axon in many pieces is recoverable; two axons glued is a
connectome error you do not get back. Extra plateau components are not
dust. They are a different fragment class and fail VOI.

Fragments are supervoxels. The object you count synapses on or load into a
proofreading graph is **after** mean merge. `fragments()` is a stage hook;
the quality bar is `segment()`.

## Over-segment, then merge

Thresholding the affinity graph ("cut every edge below 0.3, take connected
components") is brittle. Thin errors punch holes through a neurite or glue
two axons. Waterz grows conservative pieces, then glues them.

**Watershed.** Each voxel flows toward the locally strongest of its six
neighbors. Below `aff_low` you are background; above `aff_high` you are
definitely connected. A plateau (several voxels sharing the same max) is
**one** basin, not one basin per pixel. That plateau rule is load-bearing.
A tempting GPU shortcut (extra closed-plateau CCs) fails VOI. The BFS
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

Those are other products. Do not loosen +0.02 to "make mutex pass."

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
jitter (~1.7e-05), not "VOI is fuzzy."

IDs need not match waterz. Waterz is not even self-identical on plateaus.
This code is run-to-run byte-identical because we control ties.

Metric: `waterz.evaluate(labels.uint64, gt.uint64)`.

## Why ParHAC, and why two eps values

Exact average-linkage HAC has no friendly poly-log parallel algorithm
(ParHAC 2022; Abboud et al., [arXiv:2404.14730](https://arxiv.org/abs/2404.14730)).
The serial waterz heap is global min edge, merge, reweight, repeat. Mapping
that to CUDA as one pop per kernel keeps the serial chain.

Approximate the **order of merges**; keep the **mean statistic**. ParHAC
does (1+eps)-heavy matching: many disjoint merges in a round if they are
close enough to locally heaviest, then contract, repeat. On this RAG `eps`
is a phase boundary, not a quality-vs-speed slider.

- Four cuts (0.2-0.5): `eps = 0.08`. 0.09 already fails at 0.2.
- Single cut at 0.3: `eps = 0.40`. Every 0.01 step from 0.41 to 0.49 fails
  merge VOI (N18 B2).

Lu, Zlateski, and Seung ([arXiv:2106.10795](https://arxiv.org/abs/2106.10795))
distribute exact mean clustering by freezing anything that touches a fake
chunk boundary. A freeze that is not their Algorithm 2 produces different
parents than ParHAC. We stopped.

## If you wanted a different product

The API is flexible on *how you call this pipeline*. It is not a menu of
clustering algorithms.

| You wanted | This repo | What to do |
|---|---|---|
| Mutex / GASP / Kruskal / long-range channels | 3-channel contact-mean only | Other stack. Atlas is the proof. |
| Fragments / connected components as the deliverable | Quality bar is after merge | `fragments()`; do not cite four-T |
| Exact serial heap | GPU path is (1+eps) ParHAC | CPU `waterz` (`WATERZ_AGG_CPU=1` is slow) |
| One threshold, usually 0.3 | Four-T is the published gate | Use eps=0.40 on that single cut; do not advertise four-T |
| Daisy/LSD **production** stitch | Block loop only | `examples/zarr_block.py`; not their freeze/ChunkedGraph |
| Proofreading / CAVE / AGQ | We emit uint32 labels | This is the decode before those |
| Same Gvox/s on another GPU | Pin is idle RTX 5090 | Time that card. [porting.md](porting.md) |
| Another dataset / "close enough" total VOI | Pin is this CREMI-A checkpoint | Re-run `legal_eval.sh` on yours |
| Bit-identity with stock waterz | Waterz is not self-identical | Grade partitions |

## Timing hygiene (not algorithm law)

These are how we avoided fooling ourselves. Dropping them is a different
timing or quality story.

| Rule | Why |
|---|---|
| Dual-eps 0.08 / 0.40 | Measured so ParHAC still passes four-T / T=0.3 |
| Fragment identity vs `wz_fragments.npy` | N18 A4: T=0.3 VOI looked OK after nfrag 2.175M -> 3.14M |
| Ignore val cuts under 100 ms before a 2.16 claim | N18 B3: a val "win" was slower on 2.16 |
| Parks off (`HOST_PARK=0`, `AFF_PARK=0`) | Parked affinity timed PCIe and doubled e2e with the same labels |

[funkey/waterz PR 24](https://github.com/funkey/waterz/pull/24) took CPU RAG
52s -> 18s and agglomeration 71s -> 28s on 1024³-class volumes. That is the
CPU class this GPU path replaces.
