# Watershed, contact-mean, ParHAC

Stock waterz is a serial CPU decode: affinity-flow watershed, then
agglomerate fragments with contact-mean (average linkage on the region
graph). Merge while the mean stays above T. Heap, reweight, repeat.

We kept the statistic. We approximated the order, because exact
average-linkage HAC has no friendly parallel algorithm. Three CUDA
stages:

## 1. Watershed

Each voxel flows to the strongest of its six neighbors. Below `aff_low`
you are background; above `aff_high` you are definitely connected. A
plateau (several voxels sharing the same max) is **one** basin, not one
basin per pixel. That rule is not optional: extra closed-plateau pieces
fail VOI.

You cannot unmerge later. One axon in many fragments is recoverable. Two
axons glued is a connectome error you do not get back. `fragments()` is a
hook. The quality bar is `segment()`.

On 2.16 Gvox the BFS that rewrites plateaus is ~14 ms. The cost is
compressing the union-find (`k_w5_compress_list`, ~508 ms). Speeding up
BFS does not speed up this watershed.

## 2. Region graph

Fragments become nodes. An edge exists when two fragments touch. The
weight is the contact-area-weighted **mean** of the affinities on the
shared face: integer byte-sum / contact-count.

Not a float `atomicAdd`. Float addition is not associative. Two runs on
the same val RAG disagreed on ~7.4e5 of 7.5e6 edges. Integers commute, so
a second run is byte-identical.

When A and B merge into C, the edge C-D is
`(sum_AD + sum_BD) / (count_AD + count_BD)`. Kruskal, mutex, and GASP
AbsMax freeze the original face weights. Mean re-asks after every glue: a
weak sliver can be diluted by a large strong contact; a large weak
contact can drag a previously strong pair below T. Same cached CREMI-A
RAG, those algorithms miss the VOI box. Table:
[voi_atlas.csv](../data/cache/voi_atlas.csv).

## 3. ParHAC on that RAG

The serial waterz heap is global min edge, merge, reweight, repeat. One
pop per CUDA kernel keeps that chain.

ParHAC does (1+eps)-heavy matching: many disjoint merges in a round if
they are close enough to locally heaviest, then contract, repeat. Paper
matching, waterz means. Official ParHAC is CPU (CPAM). This is a CUDA
port of the order approximation, not of a different linkage.

Two eps values:

- several cuts 0.2 / 0.3 / 0.4 / 0.5: **eps = 0.08**
- a single cut at T = 0.3: **eps = 0.40**
- 0.41 already fails **merge** VOI at 0.3

ParHAC papers talk about "small merge" clustered-graph rounds. False
here. Layer 0 is most of the merges (~71% at eps 0.08). A StarMerge-style
clustered rewrite is the wrong picture of this RAG.

Thresholds in this API are **affinity** (merge while mean > T). Stock
waterz heaps on score `1 - affinity`. Mixing them cuts the dendrogram at
the wrong height.

## Fit

2.16 Gvox is three z-slabs of Z=125, not a fused 42 GiB working set.
Peak **13.22 GiB**. An 8-tile serial fallback was 8.3x slower. Dead.

## The numbers (idle RTX 5090)

Quality is the shipped CREMI-A val block `[3,125,1200,1200]`. VOI split
**and** merge, each within +0.02 of stock waterz, at all four cuts.
Trading one against the other is a fail. Fragment IDs need not match
waterz. Waterz is not even self-identical on plateaus. We control ties,
so we are run-to-run identical. nfrag = 2,175,400, bg = 506,568.

| aff | VOI split | VOI merge |
|-----|-----------|-----------|
| 0.2 | 0.3707 | 0.3350 |
| 0.3 | 0.4512 | 0.2505 |
| 0.4 | 0.5162 | 0.2268 |
| 0.5 | 0.6129 | 0.2184 |

Single T=0.3 at eps 0.40: split 0.440823, merge 0.254275.

Speed is a 3x2x2 tile of that affinity, `[3,375,2400,2400]` = 2.16 Gvox,
affinity 0.3, CUDA events, affinity already in VRAM, GPU idle:

| stage | ms |
|---|---|
| watershed | 1309 |
| RAG | 85 |
| agglomeration | 1680 |
| extract | 18 |
| end-to-end | 3093 (~0.70 Gvox/s) |

Pin: `data/cache/N19_I0_REPRO.json`. How to time, and what owns those
milliseconds: [porting.md](porting.md). Pipeline and VOI definition:
[decode.md](decode.md).

## What lost

Different merge rule (Kruskal, mutex, GASP, complete-link, WPGMA, Ward,
...): different partition, or not contact-mean.

Exact mean heap / RNN / NN-chain as a way to close 1680 ms: dendrogram
height ~12,500; RNN hit a 5,000-round cap on every threshold; four-T
fail.

Lu spatial freeze (their Algorithm 2): four-T can pass, still ~6645 s on
CPU with millions of residual edges.

Linked-list CSR rewrite of the dirty fuse: identity-true, then 23 s of
agglomeration on 2.16 from pointer-chase.

A listed-insert / slot-emit stack: four-T pass, ~56 ms off 2.16
agglomeration. Not the pin. Still the 1600 ms class.

If the heavy agglomeration kernels vanished you would still have hundreds
of ms of watershed plus roughly 690-1130 ms of agglomeration. The leftover
owners on the pin are list-compress, rebuild-active, dirty-fuse,
hash-insert, emit-holes.

Gate: `bash scripts/eval.sh`. Plus the 2.16 timing: `bash scripts/eval.sh --216`.
