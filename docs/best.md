# Watershed, contact-mean, ParHAC

Stock waterz decodes CNN affinities on the CPU: an affinity-flow
watershed produces fragments, then those fragments agglomerate under
contact-mean (average linkage on the region adjacency graph). Merges
continue while the mean stays above a threshold T. The serial
implementation is a heap: pop the current min edge, merge, reweight
neighbors, repeat.

The GPU port keeps that contact-mean statistic and approximates only the
order of merges. Exact average-linkage HAC has no friendly parallel
algorithm, so a literal heap-pop per CUDA kernel would recreate the
serial chain. The implementation is three stages.

## 1. Watershed

Each voxel flows toward the strongest of its six neighbors. Affinity
below `aff_low` is background; above `aff_high` the voxels are treated as
definitely connected. A plateau, meaning several voxels that share the
same maximum, is one basin rather than one basin per pixel. Extra
closed-plateau components fail VOI, so that rule is part of the quality
gate rather than a visualization detail.

Waterz cannot unmerge. An axon split into many fragments can still be
glued; two axons fused into one label is a connectome error that later
stages do not recover. `fragments()` exposes the watershed as a stage
hook. Quality is graded on `segment()`, after agglomeration.

On 2.16 Gvox the BFS that rewrites plateaus is about 14 ms. The expensive
step is compressing the union-find (`k_w5_compress_list`, about 508 ms),
so making BFS faster does not make this watershed faster.

## 2. Region graph

Fragments become nodes of a region adjacency graph. An edge exists when
two fragments touch, and its weight is the contact-area-weighted mean of
the affinities on the shared face, stored as an integer byte-sum and a
contact count.

A float `atomicAdd` of those sums is not associative. Two runs on the
same val RAG disagreed on about 7.4e5 of 7.5e6 edges. Integer
accumulation commutes, which is why a second run is byte-identical.

When fragments A and B merge into C, the edge from C to a neighbor D is
`(sum_AD + sum_BD) / (count_AD + count_BD)`. Kruskal, mutex, and GASP
AbsMax freeze the original voxel-face weights. Contact-mean recomputes
after every glue, so a weak sliver can be diluted by a large strong
contact, and a large weak contact can pull a previously strong pair
below T. On the same cached CREMI-A RAG those frozen-weight algorithms
miss the VOI box; the table is
[voi_atlas.csv](../data/cache/voi_atlas.csv).

## 3. ParHAC on that RAG

The serial waterz heap is still global min edge, merge, reweight, repeat.
Issuing one pop per CUDA kernel preserves that dependence.

ParHAC instead does (1+eps)-heavy matching: many disjoint merges in a
round if they are close enough to locally heaviest, then contract, then
repeat. The matching is the paper algorithm; the edge statistic is still
waterz contact-mean. Official ParHAC is CPU (CPAM). This CUDA path is
that order approximation on the contact-mean RAG, not a different
linkage.

Two eps values are required on this graph. Several cuts at 0.2, 0.3,
0.4, and 0.5 use **eps = 0.08**. A single cut at T = 0.3 uses **eps =
0.40**. Raising that single-cut eps to 0.41 already fails merge VOI at
T = 0.3.

ParHAC papers describe clustered-graph rounds in which most merges are
small. On this RAG that picture is wrong: layer 0 is about 71% of the
merges at eps 0.08, so a StarMerge-style clustered rewrite does not
match the work.

Thresholds in the public API are affinity: merge while the mean stays
above T. Stock waterz heaps on score `1 - affinity`. Mixing the two
conventions cuts the dendrogram at the wrong height.

## Working set

2.16 Gvox fits as three z-slabs of Z=125 rather than a fused working set
of about 42 GiB. Peak is **13.22 GiB**. An 8-tile serial fallback was
8.3x slower and is not used.

## Quality and speed (idle RTX 5090)

Quality is the shipped CREMI-A val block `[3,125,1200,1200]`. VOI split
and VOI merge must each stay within +0.02 of stock waterz at all four
affinity cuts; improving one by worsening the other is a fail. Fragment
IDs do not need to match waterz. Stock waterz is not self-identical on
plateaus. Ties are resolved so that two full runs of this code produce
byte-identical labels. nfrag = 2,175,400, bg = 506,568.

| aff | VOI split | VOI merge |
|-----|-----------|-----------|
| 0.2 | 0.3707 | 0.3350 |
| 0.3 | 0.4512 | 0.2505 |
| 0.4 | 0.5162 | 0.2268 |
| 0.5 | 0.6129 | 0.2184 |

A single cut at T=0.3 with eps 0.40 gives split 0.440823 and merge
0.254275.

Speed is a 3x2x2 tile of that affinity, `[3,375,2400,2400]` = 2.16 Gvox,
at affinity 0.3, timed with CUDA events, affinity already in VRAM, GPU
idle:

| stage | ms |
|---|---|
| watershed | 1309 |
| RAG | 85 |
| agglomeration | 1680 |
| extract | 18 |
| end-to-end | 3093 (~0.70 Gvox/s) |

The pin is `data/cache/N19_I0_REPRO.json`. How those milliseconds were
measured, and which kernels own them, is in [porting.md](porting.md).
The pipeline and the VOI definition are in [decode.md](decode.md).

## Other clustering

A different merge rule (Kruskal, mutex, GASP, complete-link, WPGMA, Ward,
and similar) produces a different partition, or else it is not
contact-mean.

An exact mean heap, RNN, or NN-chain is not a way to close the 1680 ms
agglomeration: the dendrogram height is about 12,500, RNN hit a
5,000-round cap on every threshold, and four-T VOI fails.

Lu spatial freeze (Algorithm 2 in that paper) can pass four-T and still
takes about 6645 s on CPU with millions of residual edges.

A linked-list CSR rewrite of the dirty fuse is identity-true and then
spends 23 s of agglomeration on 2.16 Gvox in pointer-chasing.

A listed-insert / slot-emit stack passes four-T and cuts about 56 ms off
2.16 agglomeration. It is not the pin, and it remains in the 1600 ms
class.

If the heavy agglomeration kernels disappeared, hundreds of milliseconds
of watershed would remain, plus roughly 690-1130 ms of agglomeration. The
kernels that still dominate the pin are list-compress, rebuild-active,
dirty-fuse, hash-insert, and emit-holes.

The quality gate is `bash scripts/eval.sh`. The same gate plus the 2.16
timing is `bash scripts/eval.sh --216`.
